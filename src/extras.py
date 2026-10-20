"""
extras.py
=========
Experimentos complementarios:

  R1 regularization   : min_samples_leaf en el árbol y tasa de aprendizaje x número de árboles en GB.
  S1 significance     : prueba de Friedman + diferencia crítica de Nemenyi (Demšar, 2006) sobre los 14
                        conjuntos; Wilcoxon por pares con corrección de Holm; t de remuestreo corregida
                        (Nadeau y Bengio, 2003) por conjunto, RF vs DT.
  I1 resampling       : Credit Card con submuestreo aleatorio, SMOTE y ajuste del umbral de decisión.

Uso:  python -m src.extras regularization significance resampling
"""
from __future__ import annotations

import sys
import time
from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import (balanced_accuracy_score, f1_score, matthews_corrcoef, precision_score,
                             recall_score, average_precision_score, accuracy_score)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.tree import DecisionTreeClassifier

from . import datasets as D
from .experiments import RES, RS, impute_fold, splits, _append, _done_keys

MODS = ["Árbol de Decisión", "Árbol podado (ccp)", "Random Forest", "Gradient Boosting"]


# ----------------------------------------------------------------------------- R1
def regularization(names=("Concrete", "Moons", "BCW (Original)", "Glass"), leaves=(1, 2, 5, 10, 20, 50)):
    allds = {**D.REAL, **D.SYNTH}
    rows = []
    for name in names:
        d = allds[name]()
        cv = StratifiedKFold(10, shuffle=True, random_state=RS)
        for msl in leaves:
            tr_a, va_a, va_b, nl = [], [], [], []
            for tr, te in cv.split(d["X"], d["y"]):
                Xa, Xb = impute_fold(d["X"][tr], d["X"][te])
                m = DecisionTreeClassifier(min_samples_leaf=msl, random_state=RS).fit(Xa, d["y"][tr])
                tr_a.append(m.score(Xa, d["y"][tr])); va_a.append(m.score(Xb, d["y"][te]))
                va_b.append(balanced_accuracy_score(d["y"][te], m.predict(Xb))); nl.append(m.get_n_leaves())
            rows.append(dict(dataset=name, min_samples_leaf=msl, train_acc=np.mean(tr_a), cv_acc=np.mean(va_a),
                             cv_bal_acc=np.mean(va_b), cv_std=np.std(va_a), leaves=np.mean(nl)))
    pd.DataFrame(rows).to_csv(RES / "reg_min_samples_leaf.csv", index=False)
    # Gradient Boosting: curvas por etapas (staged_predict) para tres tasas de aprendizaje
    rows = []
    for name in ("Concrete", "Moons"):
        d = allds[name]()
        for seed in range(3):
            Xtr, Xte, ytr, yte = train_test_split(d["X"], d["y"], test_size=0.3, stratify=d["y"], random_state=RS + seed)
            for lr in (0.01, 0.1, 0.5):
                gb = GradientBoostingClassifier(n_estimators=500, learning_rate=lr, max_depth=3, random_state=RS).fit(Xtr, ytr)
                for k, (ptr, pte) in enumerate(zip(gb.staged_predict(Xtr), gb.staged_predict(Xte)), start=1):
                    if k in (1, 2, 5, 10, 20, 50, 100, 150, 200, 300, 400, 500):
                        rows.append(dict(dataset=name, seed=seed, learning_rate=lr, n_estimators=k,
                                         train_acc=np.mean(ptr == ytr), test_acc=np.mean(pte == yte)))
    pd.DataFrame(rows).to_csv(RES / "reg_gb_staged.csv", index=False)
    print("  regularización lista", flush=True)


# ----------------------------------------------------------------------------- S1
def _bench():
    small = [pd.read_csv(RES / "benchmark_small.csv")]
    big = pd.read_csv(RES / "benchmark_big.csv")
    return pd.concat(small + [big[big.model.isin(MODS)]], ignore_index=True)


def nadeau_bengio(a, b, n_train, n_test):
    """t de remuestreo corregida: varianza inflada por (1/J + n_test/n_train)."""
    d = np.asarray(a) - np.asarray(b)
    J = len(d)
    if J < 2 or np.var(d, ddof=1) == 0:
        return np.nan, np.nan
    t = d.mean() / np.sqrt((1.0 / J + n_test / n_train) * np.var(d, ddof=1))
    return t, 2 * stats.t.sf(abs(t), J - 1)


def significance(alpha=0.05):
    df = _bench()
    M = df.groupby(["dataset", "model"]).bal_acc.mean().unstack()[MODS]
    N, k = M.shape
    ranks = M.rank(axis=1, ascending=False, method="average")
    chi2, p_f = stats.friedmanchisquare(*[M[m] for m in MODS])
    # Iman-Davenport
    Ff = (N - 1) * chi2 / (N * (k - 1) - chi2)
    p_id = stats.f.sf(Ff, k - 1, (k - 1) * (N - 1))
    q_alpha = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728}[k]   # Nemenyi, alfa = 0.05 (Demšar, 2006)
    CD = q_alpha * np.sqrt(k * (k + 1) / (6.0 * N))
    summ = pd.DataFrame({"model": MODS, "avg_rank": ranks.mean().values})
    summ.to_csv(RES / "sig_ranks.csv", index=False)
    pd.Series(dict(N=N, k=k, chi2=chi2, p_friedman=p_f, F_iman_davenport=Ff, p_iman_davenport=p_id, CD=CD)
              ).to_csv(RES / "sig_friedman.csv")
    # Wilcoxon por pares + Holm
    pr = []
    for a, b in combinations(MODS, 2):
        w = stats.wilcoxon(M[a], M[b], zero_method="zsplit")
        pr.append(dict(A=a, B=b, mean_diff=float((M[a] - M[b]).mean()), W=w.statistic, p=w.pvalue,
                       rank_diff=float(ranks[a].mean() - ranks[b].mean())))
    pr = pd.DataFrame(pr).sort_values("p").reset_index(drop=True)
    m = len(pr)
    adj, run = [], 0.0
    for i, p in enumerate(pr.p):
        run = max(run, min(1.0, (m - i) * p)); adj.append(run)
    pr["p_holm"] = adj
    pr["nemenyi_sig"] = pr.rank_diff.abs() > CD
    pr.to_csv(RES / "sig_pairs.csv", index=False)
    # por conjunto: RF vs DT y GB vs DT (t corregida; sólo conjuntos con >= 3 pliegues)
    rows = []
    for ds, g in df.groupby("dataset"):
        n = int(g.n.iloc[0]); J = int(g.fold.nunique())
        if J < 3:
            rows.append(dict(dataset=ds, comparison="—", J=J, note="partición única (HAR)"))
            continue
        for other in ("Random Forest", "Gradient Boosting"):
            a = g[g.model == other].sort_values("fold").bal_acc.values
            b = g[g.model == "Árbol de Decisión"].sort_values("fold").bal_acc.values
            t, p = nadeau_bengio(a, b, n * (J - 1) / J, n / J)
            rows.append(dict(dataset=ds, comparison=f"{other} vs DT", J=J, diff=float(np.mean(a - b)), t=t, p=p,
                             sig=bool(p < alpha) if not np.isnan(p) else False))
    pd.DataFrame(rows).to_csv(RES / "sig_per_dataset.csv", index=False)
    print(f"  Friedman chi2={chi2:.2f} p={p_f:.4f}; CD={CD:.3f}; rangos={summ.round(2).to_dict('records')}", flush=True)


# ----------------------------------------------------------------------------- I1
def resampling(budget_s=10**9):
    from imblearn.over_sampling import SMOTE
    from imblearn.under_sampling import RandomUnderSampler
    t0 = time.perf_counter()
    out = RES / "imbalance_resampling.csv"
    done = _done_keys(out, ["model", "strategy", "fold"])
    d = D.credit_card()
    base = {"Árbol de Decisión": DecisionTreeClassifier(random_state=RS),
            "Random Forest": RandomForestClassifier(200, random_state=RS, n_jobs=-1),
            "Gradient Boosting": GradientBoostingClassifier(random_state=RS)}
    for fold, (tr, te) in enumerate(splits(d, 10)):
        Xtr, ytr, Xte, yte = d["X"][tr], d["y"][tr], d["X"][te], d["y"][te]
        for strat in ("submuestreo 1:10", "SMOTE 1:10", "umbral óptimo (F1)"):
            for mname, mdl in base.items():
                if (mname, strat, str(fold)) in done:
                    continue
                if time.perf_counter() - t0 > budget_s:
                    print("  presupuesto agotado; reanudar", flush=True)
                    return False
                m = clone(mdl); thr = 0.5; t1 = time.perf_counter()
                if strat.startswith("submuestreo"):
                    Xs, ys = RandomUnderSampler(sampling_strategy=0.1, random_state=RS).fit_resample(Xtr, ytr)
                    m.fit(Xs, ys)
                elif strat.startswith("SMOTE"):
                    Xs, ys = SMOTE(sampling_strategy=0.1, random_state=RS).fit_resample(Xtr, ytr)
                    m.fit(Xs, ys)
                else:  # umbral elegido en una validación interna (25% del entrenamiento)
                    Xa, Xv, ya, yv = train_test_split(Xtr, ytr, test_size=0.25, stratify=ytr, random_state=RS)
                    m.fit(Xa, ya)
                    pv = m.predict_proba(Xv)[:, 1]
                    cand = np.unique(np.quantile(pv, np.linspace(0.5, 1.0, 400)))
                    f1s = [f1_score(yv, (pv >= c).astype(int), zero_division=0) for c in cand]
                    thr = float(cand[int(np.argmax(f1s))])
                    m.fit(Xtr, ytr)
                t_fit = time.perf_counter() - t1
                p = m.predict_proba(Xte)[:, 1]
                yhat = (p >= thr).astype(int) if strat.startswith("umbral") else m.predict(Xte)
                _append(out, dict(model=mname, strategy=strat, fold=fold, threshold=thr, t_fit_s=t_fit,
                                  test_acc=accuracy_score(yte, yhat), bal_acc=balanced_accuracy_score(yte, yhat),
                                  recall_pos=recall_score(yte, yhat), precision_pos=precision_score(yte, yhat, zero_division=0),
                                  f1=f1_score(yte, yhat), mcc=matthews_corrcoef(yte, yhat), pr_auc=average_precision_score(yte, p)))
                print(f"  {strat} | {mname} | fold {fold} ({t_fit:.0f}s)", flush=True)
    return True


EXP = {"regularization": regularization, "significance": significance, "resampling": resampling}


# ----------------------------------------------------------------------------- T1 / D2
def threshold_sensitivity():
    """Sensibilidad de las conclusiones al umbral de binarización (Concrete y Forest Fires)."""
    from .experiments import models, fit_eval
    out = RES / "threshold_sensitivity.csv"
    done = _done_keys(out, ["dataset", "threshold", "model", "fold"])
    cfg = [("Concrete", t) for t in (25.0, 30.0, 35.0, 40.0, 45.0)] + [("Forest Fires", t) for t in (0.0, 5.0)]
    for name, t in cfg:
        if name == "Concrete":
            d = D.concrete(threshold=t)
        else:
            df = pd.read_csv(D.DATA / "forestfires.csv")
            base = D.forest_fires()
            d = dict(base, y=(df["area"].to_numpy() > t).astype(int))
        for fold, (tr, te) in enumerate(splits(d, 10)):
            for mname, mdl in models().items():
                if (name, str(t), mname, str(fold)) in done:
                    continue
                m = fit_eval(clone(mdl), d["X"][tr], d["y"][tr], d["X"][te], d["y"][te])
                _append(out, dict(dataset=name, threshold=t, pos_rate=float(d["y"].mean()), model=mname, fold=fold,
                                  bal_acc=m["bal_acc"], mcc=m["mcc"], gap=m["gap"], leaves=m.get("leaves")))
        print(f"  {name} umbral {t} listo", flush=True)


def dim_1000(model_name=None):
    """Agrega el caso de 1000 atributos de ruido a q_dimensionality.csv (por modelo, reanudable)."""
    from sklearn.impute import SimpleImputer
    from sklearn.model_selection import cross_val_score
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    d = D.bcw()
    rng = np.random.default_rng(RS)
    # misma secuencia aleatoria que q_dimensionality: se consumen los bloques previos (0, 20, 100, 300)
    for k in (0, 20, 100, 300):
        rng.normal(0, 1, (len(d["y"]), k))
    k = 1000
    X = np.hstack([d["X"], rng.normal(0, 1, (len(d["y"]), k)) * np.nanstd(d["X"])])
    imp = lambda: SimpleImputer(strategy="median")
    mods = {"Árbol de Decisión": make_pipeline(imp(), DecisionTreeClassifier(random_state=RS)),
            "Random Forest": make_pipeline(imp(), RandomForestClassifier(200, random_state=RS, n_jobs=-1)),
            "Gradient Boosting": make_pipeline(imp(), GradientBoostingClassifier(random_state=RS)),
            "KNN (k=5, estandarizado)": make_pipeline(imp(), StandardScaler(), KNeighborsClassifier(5))}
    path = RES / "q_dimensionality.csv"
    cur = pd.read_csv(path)
    for mname, mdl in mods.items():
        if model_name and mname != model_name:
            continue
        if ((cur.noise_dims == k) & (cur.model == mname)).any():
            continue
        t0 = time.perf_counter()
        s = cross_val_score(mdl, X, d["y"], cv=StratifiedKFold(10, shuffle=True, random_state=RS), scoring="balanced_accuracy")
        cur = pd.concat([cur, pd.DataFrame([dict(noise_dims=k, p=X.shape[1], model=mname, bal_acc=s.mean(), std=s.std())])])
        cur.to_csv(path, index=False)
        print(f"  +1000 {mname}: {s.mean():.3f} ({time.perf_counter() - t0:.0f}s)", flush=True)


# ----------------------------------------------------------------------------- N1
def nested():
    """Búsqueda anidada de hiperparámetros de RF y GB (CV externa 10, interna 3) en los 12 conjuntos pequeños.

    Compara contra los hiperparámetros fijos de la comparativa principal. Reanudable (guarda fila por fila).
    """
    grids = {"Random Forest": (RandomForestClassifier(random_state=RS, n_jobs=-1),
                               {"n_estimators": [200, 500], "max_features": ["sqrt", 0.5], "min_samples_leaf": [1, 3]}),
             "Gradient Boosting": (GradientBoostingClassifier(random_state=RS),
                                   {"learning_rate": [0.05, 0.1], "n_estimators": [100, 300], "max_depth": [2, 3]})}
    out = RES / "nested_search.csv"
    done = _done_keys(out, ["dataset", "model", "fold"])
    for name, loader in {**D.REAL, **D.SYNTH}.items():
        if name in ("Credit Card", "HAR"):
            continue
        d = loader()
        for fold, (tr, te) in enumerate(splits(d, 10)):
            Xtr, Xte = impute_fold(d["X"][tr], d["X"][te])
            for mname, (est, grid) in grids.items():
                key = mname + " (anidado)"
                if (name, key, str(fold)) in done:
                    continue
                gs = GridSearchCV(est, grid, cv=StratifiedKFold(3, shuffle=True, random_state=RS),
                                  scoring="balanced_accuracy", n_jobs=-1).fit(Xtr, d["y"][tr])
                _append(out, dict(dataset=name, model=key, fold=fold,
                                  bal_acc=balanced_accuracy_score(d["y"][te], gs.predict(Xte)),
                                  **{f"best_{k}": str(v) for k, v in gs.best_params_.items()}))
        print(f"  anidado: {name} listo", flush=True)


EXP["nested"] = nested
EXP["thresholds"] = threshold_sensitivity
EXP["dim1000"] = dim_1000

if __name__ == "__main__":
    for n in sys.argv[1:] or list(EXP):
        t0 = time.perf_counter(); print("==", n, flush=True); EXP[n](); print(f"   ({time.perf_counter() - t0:.1f}s)", flush=True)
