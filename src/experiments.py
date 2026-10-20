"""
experiments.py
==============
Experimentos del ABP (todos con random_state = 42):

  E1  benchmark       : DT (sin poda), DT podado (ccp_alpha por CV interna), Random Forest y
                        Gradient Boosting sobre los 8 conjuntos reales y los sintéticos.
  E2  rotation_sweep  : profundidad, hojas y exactitud vs. ángulo de rotación (0°-45°).
  E3  depth_sweep     : curvas entrenamiento/validación vs. profundidad máxima (sobreajuste).
  E4  ccp_path        : camino de poda por complejidad de costo (Breiman et al., 1984).
  E5  preguntas clave : correlación (Wine), no normalidad (Forest Fires), desbalance (Credit Card),
                        dimensionalidad (BCW + ruido, HAR) y escalado/atípicos (Concrete).

Uso:  python -m src.experiments [nombre_experimento ...]
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, skew
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score, f1_score,
                             matthews_corrcoef, precision_score, recall_score)
from sklearn.model_selection import (GridSearchCV, StratifiedGroupKFold, StratifiedKFold, cross_val_score,
                                     train_test_split)
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.class_weight import compute_sample_weight

from . import datasets as D

RS = D.RANDOM_STATE
# Protocolo completo, sin reducciones por cómputo.
# Si la poda de Credit Card completo tarda demasiado, cambie a False (búsqueda de alfa corta: 10 alfas, CV interna de 3).
FULL_PRUNING = True
RES = Path(os.environ.get("ABP_RESULTS_DIR", D.ROOT / "results"))  # ABP_RESULTS_DIR: carpeta alternativa (prueba rápida)
RES.mkdir(parents=True, exist_ok=True)


# ----------------------------------------------------------------------------- modelos
class PrunedTree(DecisionTreeClassifier):
    """Árbol con ccp_alpha elegido por validación cruzada interna sobre el camino de poda."""

    def __init__(self, n_alphas=25, inner_cv=5, criterion="gini", class_weight=None, random_state=RS):
        super().__init__(criterion=criterion, class_weight=class_weight, random_state=random_state)
        self.n_alphas, self.inner_cv = n_alphas, inner_cv

    def fit(self, X, y, sample_weight=None):
        base = DecisionTreeClassifier(criterion=self.criterion, class_weight=self.class_weight,
                                      random_state=self.random_state)
        path = base.cost_complexity_pruning_path(X, y)
        alphas = np.unique(np.clip(path.ccp_alphas[:-1], 0.0, None))  # el último deja sólo la raíz; recorta -1e-18 numérico
        if len(alphas) > self.n_alphas:
            alphas = np.quantile(alphas, np.linspace(0, 1, self.n_alphas))
        cv = StratifiedKFold(self.inner_cv, shuffle=True, random_state=self.random_state)
        gs = GridSearchCV(base, {"ccp_alpha": alphas}, cv=cv, scoring="balanced_accuracy").fit(X, y)
        self.ccp_alpha = float(gs.best_params_["ccp_alpha"])
        self.cv_alpha_ = self.ccp_alpha
        return super().fit(X, y, sample_weight=sample_weight)


def models(n_estimators_rf=200, n_estimators_gb=100, fast=False):
    return {
        "Árbol de Decisión": DecisionTreeClassifier(random_state=RS),
        "Árbol podado (ccp)": PrunedTree(n_alphas=10 if fast else 25, inner_cv=3 if fast else 5),
        "Random Forest": RandomForestClassifier(n_estimators=n_estimators_rf, random_state=RS, n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=n_estimators_gb, learning_rate=0.1,
                                                        max_depth=3, random_state=RS),
    }


def _scores(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)
    return None


def evaluate(y, yhat, proba, ytr, yhat_tr):
    binary = len(np.unique(np.concatenate([y, ytr]))) == 2
    m = dict(train_acc=accuracy_score(ytr, yhat_tr), test_acc=accuracy_score(y, yhat),
             bal_acc=balanced_accuracy_score(y, yhat), f1_macro=f1_score(y, yhat, average="macro"),
             mcc=matthews_corrcoef(y, yhat))
    if binary:
        m.update(recall_pos=recall_score(y, yhat, zero_division=0),
                 precision_pos=precision_score(y, yhat, zero_division=0),
                 pr_auc=average_precision_score(y, proba[:, 1]) if proba is not None else np.nan)
    m["gap"] = m["train_acc"] - m["test_acc"]
    return m


def tree_stats(model):
    if isinstance(model, DecisionTreeClassifier):
        return dict(depth=model.get_depth(), leaves=model.get_n_leaves())
    if isinstance(model, RandomForestClassifier):
        return dict(depth=np.mean([t.get_depth() for t in model.estimators_]),
                    leaves=np.mean([t.get_n_leaves() for t in model.estimators_]))
    if isinstance(model, GradientBoostingClassifier):
        return dict(depth=3, leaves=np.mean([t.get_n_leaves() for t in model.estimators_.ravel()]))
    return {}


def impute_fold(Xtr, Xte):
    """Imputa faltantes con la mediana del entrenamiento del pliegue (sin fuga hacia la prueba)."""
    if not (np.isnan(Xtr).any() or np.isnan(Xte).any()):
        return Xtr, Xte
    med = np.nanmedian(Xtr, axis=0)
    Xtr, Xte = Xtr.copy(), Xte.copy()
    for X in (Xtr, Xte):
        r, c = np.where(np.isnan(X)); X[r, c] = med[c]
    return Xtr, Xte


def fit_eval(model, Xtr, ytr, Xte, yte, sample_weight=None):
    Xtr, Xte = impute_fold(Xtr, Xte)
    t0 = time.perf_counter()
    if sample_weight is not None:
        model.fit(Xtr, ytr, sample_weight=sample_weight)
    else:
        model.fit(Xtr, ytr)
    t_fit = time.perf_counter() - t0
    t0 = time.perf_counter(); yhat = model.predict(Xte); t_pred = time.perf_counter() - t0
    m = evaluate(yte, yhat, _scores(model, Xte), ytr, model.predict(Xtr))
    m.update(t_fit_s=t_fit, t_pred_ms=1e3 * t_pred / len(yte), **tree_stats(model))
    return m


def splits(d, n_splits):
    if "groups" in d:  # HAR: CV estratificada y agrupada por sujeto
        cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=RS)
        return list(cv.split(d["X"], d["y"], d["groups"]))
    if "split" in d:  # partición única (sólo si se usa el espejo de GitHub sin archivos de sujetos)
        return [d["split"]]
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RS)
    return list(skf.split(d["X"], d["y"]))


# ----------------------------------------------------------------------------- E1
def benchmark(names=None, out_name=None):
    catalog = {**{k: (f, 1) for k, f in D.REAL.items()},
               **{k: (f, 2 if k not in ("Ortogonal IF-THEN", "XOR 0°") else 1) for k, f in D.SYNTH.items()}}
    rows = []
    for name, (loader, cat) in catalog.items():
        if names and name not in names:
            continue
        d = loader()
        big = len(d["y"]) > 5000
        n_splits = 5 if big else 10
        for fold, (tr, te) in enumerate(splits(d, n_splits)):
            for mname, mdl in models(fast=big).items():
                m = fit_eval(clone(mdl), d["X"][tr], d["y"][tr], d["X"][te], d["y"][te])
                m.update(dataset=name, category=cat, model=mname, fold=fold, n=len(d["y"]),
                         p=d["X"].shape[1], n_classes=len(np.unique(d["y"])))
                rows.append(m)
        print(f"  benchmark: {name} listo", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(RES / (out_name or ("benchmark.csv" if not names else f"benchmark_{'_'.join(n[:4] for n in names)}.csv")),
              index=False)
    return df


# ----------------------------------------------------------------------------- E2
def rotation_sweep(angles=range(0, 50, 5), seeds=range(3)):
    rows = []
    for kind, gen in [("Lineal", D.rotated_linear), ("XOR", D.rotated_xor)]:
        for th in angles:
            for s in seeds:
                d = gen(th, random_state=RS + s)
                Xtr, Xte, ytr, yte = train_test_split(d["X"], d["y"], test_size=0.3, stratify=d["y"],
                                                      random_state=RS + s)
                for mname, mdl in [("DT sin poda", DecisionTreeClassifier(random_state=RS)),
                                   ("DT profundidad 2" if kind == "XOR" else "DT profundidad 1",
                                    DecisionTreeClassifier(max_depth=2 if kind == "XOR" else 1, random_state=RS)),
                                   ("DT profundidad 4", DecisionTreeClassifier(max_depth=4, random_state=RS)),
                                   ("Random Forest", RandomForestClassifier(200, random_state=RS, n_jobs=-1)),
                                   ("Gradient Boosting", GradientBoostingClassifier(random_state=RS))]:
                    m = fit_eval(mdl, Xtr, ytr, Xte, yte)
                    rows.append(dict(kind=kind, theta=th, seed=s, model=mname, test_acc=m["test_acc"],
                                     depth=m.get("depth"), leaves=m.get("leaves")))
        print(f"  rotación {kind} lista", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(RES / "rotation_sweep.csv", index=False)
    return df


# ----------------------------------------------------------------------------- E3
def depth_sweep(names=("Concrete", "Moons", "BCW (Original)"), depths=range(1, 21)):
    rows = []
    allds = {**D.REAL, **D.SYNTH}
    for name in names:
        d = allds[name]()
        cv = StratifiedKFold(10, shuffle=True, random_state=RS)
        for dep in depths:
            tr_acc, va_acc = [], []
            for tr, te in cv.split(d["X"], d["y"]):
                Xa, Xb = impute_fold(d["X"][tr], d["X"][te])
                m = DecisionTreeClassifier(max_depth=dep, random_state=RS).fit(Xa, d["y"][tr])
                tr_acc.append(m.score(Xa, d["y"][tr])); va_acc.append(m.score(Xb, d["y"][te]))
            rows.append(dict(dataset=name, max_depth=dep, train_acc=np.mean(tr_acc), cv_acc=np.mean(va_acc),
                             cv_std=np.std(va_acc)))
    df = pd.DataFrame(rows)
    df.to_csv(RES / "depth_sweep.csv", index=False)
    return df


# ----------------------------------------------------------------------------- E4
def ccp_path(names=("Concrete", "Moons", "BCW (Original)")):
    rows, chosen = [], []
    allds = {**D.REAL, **D.SYNTH}
    for name in names:
        d = allds[name]()
        Xtr, Xte, ytr, yte = train_test_split(d["X"], d["y"], test_size=0.3, stratify=d["y"], random_state=RS)
        Xtr, Xte = impute_fold(Xtr, Xte)
        path = DecisionTreeClassifier(random_state=RS).cost_complexity_pruning_path(Xtr, ytr)
        for a in path.ccp_alphas[:-1]:
            m = DecisionTreeClassifier(ccp_alpha=a, random_state=RS).fit(Xtr, ytr)
            rows.append(dict(dataset=name, alpha=a, train_acc=m.score(Xtr, ytr), test_acc=m.score(Xte, yte),
                             leaves=m.get_n_leaves(), depth=m.get_depth()))
        pt = PrunedTree().fit(Xtr, ytr)
        full = DecisionTreeClassifier(random_state=RS).fit(Xtr, ytr)
        chosen.append(dict(dataset=name, alpha_cv=pt.cv_alpha_, leaves_full=full.get_n_leaves(),
                           depth_full=full.get_depth(), test_full=full.score(Xte, yte),
                           train_full=full.score(Xtr, ytr), leaves_pruned=pt.get_n_leaves(),
                           depth_pruned=pt.get_depth(), test_pruned=pt.score(Xte, yte),
                           train_pruned=pt.score(Xtr, ytr)))
    pd.DataFrame(rows).to_csv(RES / "ccp_path.csv", index=False)
    pd.DataFrame(chosen).to_csv(RES / "ccp_chosen.csv", index=False)
    return pd.DataFrame(rows), pd.DataFrame(chosen)


# ----------------------------------------------------------------------------- E5
def q_correlation(n_boot=200):
    """Wine: correlación media, importancias e inestabilidad del atributo raíz bajo bootstrap."""
    d = D.wine()
    X, y, names = d["X"], d["y"], d["feature_names"]
    C = np.corrcoef(X.T)
    iu = np.triu_indices_from(C, 1)
    rng = np.random.default_rng(RS)
    roots = []
    for _ in range(n_boot):
        i = rng.choice(len(y), len(y), replace=True)
        roots.append(names[DecisionTreeClassifier(random_state=RS).fit(X[i], y[i]).tree_.feature[0]])
    imp = pd.DataFrame({"feature": names,
                        "DT": DecisionTreeClassifier(random_state=RS).fit(X, y).feature_importances_,
                        "RF": RandomForestClassifier(500, random_state=RS, n_jobs=-1).fit(X, y).feature_importances_,
                        "GB": GradientBoostingClassifier(random_state=RS).fit(X, y).feature_importances_})
    root_freq = pd.Series(roots).value_counts(normalize=True).rename_axis("feature").reset_index(name="freq_raiz")
    pd.DataFrame(C, index=names, columns=names).to_csv(RES / "wine_corr.csv")
    imp.merge(root_freq, on="feature", how="left").fillna(0).to_csv(RES / "q_correlation.csv", index=False)
    summary = dict(mean_abs_corr=float(np.mean(np.abs(C[iu]))), max_abs_corr=float(np.max(np.abs(C[iu]))),
                   n_pairs_gt_0_6=int(np.sum(np.abs(C[iu]) > 0.6)))
    pd.Series(summary).to_csv(RES / "q_correlation_summary.csv")
    return imp, root_freq, summary


class RankTransformer:
    """Rango empírico de cada atributo respecto al entrenamiento (empates promediados), sin fuga."""

    def fit(self, X, y=None):
        self.sorted_ = np.sort(X, axis=0)
        return self

    def transform(self, X):
        n = self.sorted_.shape[0]
        lo = np.column_stack([np.searchsorted(self.sorted_[:, j], X[:, j], side="left") for j in range(X.shape[1])])
        hi = np.column_stack([np.searchsorted(self.sorted_[:, j], X[:, j], side="right") for j in range(X.shape[1])])
        return (lo + hi) / (2.0 * n)

    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)

    def get_params(self, deep=True):
        return {}

    def set_params(self, **p):
        return self


def q_distribution():
    """Forest Fires: sesgo de los atributos e invariancia del árbol a transformaciones monótonas."""
    d = D.forest_fires()
    X, y, names = d["X"], d["y"], d["feature_names"]
    sk = pd.Series(skew(X, axis=0), index=names)
    Xlog = X.copy()
    pos = (X >= 0).all(axis=0)
    Xlog[:, pos] = np.log1p(X[:, pos])
    cv = StratifiedKFold(10, shuffle=True, random_state=RS)
    rows = []
    for tname, XX in [("original", X), ("log1p", Xlog), ("rangos", X)]:
        pre = [RankTransformer()] if tname == "rangos" else []  # rangos ajustados dentro de cada pliegue
        for mname, mdl in [("Árbol de Decisión", make_pipeline(*pre, DecisionTreeClassifier(random_state=RS))),
                           ("KNN (k=5, estandarizado)", make_pipeline(*pre, StandardScaler(), KNeighborsClassifier(5)))]:
            s = cross_val_score(mdl, XX, y, cv=cv, scoring="balanced_accuracy")
            rows.append(dict(transform=tname, model=mname, bal_acc=s.mean(), std=s.std()))
    # igualdad exacta de predicciones del árbol (misma partición) entre original y log1p
    tr, te = next(cv.split(X, y))
    p1 = DecisionTreeClassifier(random_state=RS).fit(X[tr], y[tr]).predict(X[te])
    p2 = DecisionTreeClassifier(random_state=RS).fit(Xlog[tr], y[tr]).predict(Xlog[te])
    p3 = DecisionTreeClassifier(random_state=RS).fit(Xlog[tr], y[tr])
    p1m = DecisionTreeClassifier(random_state=RS).fit(X[tr], y[tr])
    out = pd.DataFrame(rows)
    out["pred_identicas_DT_log"] = float(np.mean(p1 == p2))
    out["pred_train_identicas"] = float(np.mean(p1m.predict(X[tr]) == p3.predict(Xlog[tr])))
    out["misma_estructura"] = bool(p1m.get_n_leaves() == p3.get_n_leaves() and
                                   np.array_equal(p1m.tree_.feature, p3.tree_.feature))
    out.to_csv(RES / "q_distribution.csv", index=False)
    sk.rename("skewness").to_csv(RES / "forest_skew.csv")
    return out, sk


def q_imbalance():
    """Credit Card: efecto del desbalance y de la ponderación de clases."""
    d = D.credit_card()
    X, y = d["X"], d["y"]
    cv = StratifiedKFold(5, shuffle=True, random_state=RS)
    rows = []
    for tr, te in cv.split(X, y):
        sw = compute_sample_weight("balanced", y[tr])
        variants = [("Árbol de Decisión", DecisionTreeClassifier(random_state=RS), None),
                    ("Árbol de Decisión", DecisionTreeClassifier(class_weight="balanced", random_state=RS), "bal"),
                    ("Árbol podado (ccp)", PrunedTree(n_alphas=10, inner_cv=3), None),
                    ("Random Forest", RandomForestClassifier(200, random_state=RS, n_jobs=-1), None),
                    ("Random Forest", RandomForestClassifier(200, class_weight="balanced_subsample",
                                                             random_state=RS, n_jobs=-1), "bal"),
                    ("Gradient Boosting", GradientBoostingClassifier(random_state=RS), None),
                    ("Gradient Boosting", GradientBoostingClassifier(random_state=RS), "bal_sw")]
        for mname, mdl, w in variants:
            m = fit_eval(mdl, X[tr], y[tr], X[te], y[te], sample_weight=sw if w == "bal_sw" else None)
            m.update(model=mname, weighting="balanceada" if w else "ninguna")
            rows.append(m)
    df = pd.DataFrame(rows)
    df["trivial_acc"] = 1 - y.mean()
    df.to_csv(RES / "q_imbalance.csv", index=False)
    return df


def q_dimensionality(noise_dims=(0, 20, 100, 300, 1000)):
    """BCW + atributos de ruido gaussiano irrelevantes: degradación por dimensionalidad."""
    d = D.bcw()
    rng = np.random.default_rng(RS)
    cv = StratifiedKFold(10, shuffle=True, random_state=RS)
    rows = []
    for k in noise_dims:
        X = np.hstack([d["X"], rng.normal(0, 1, (len(d["y"]), k)) * np.nanstd(d["X"])])
        imp = lambda: SimpleImputer(strategy="median")
        for mname, mdl in [("Árbol de Decisión", make_pipeline(imp(), DecisionTreeClassifier(random_state=RS))),
                           ("Random Forest", make_pipeline(imp(), RandomForestClassifier(200, random_state=RS, n_jobs=-1))),
                           ("Gradient Boosting", make_pipeline(imp(), GradientBoostingClassifier(random_state=RS))),
                           ("KNN (k=5, estandarizado)", make_pipeline(imp(), StandardScaler(), KNeighborsClassifier(5)))]:
            s = cross_val_score(mdl, X, d["y"], cv=cv, scoring="balanced_accuracy")
            rows.append(dict(noise_dims=k, p=X.shape[1], model=mname, bal_acc=s.mean(), std=s.std()))
        pd.DataFrame(rows).to_csv(RES / "q_dimensionality.csv", index=False)  # guardado incremental
        print(f"  dimensionalidad +{k} listo", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(RES / "q_dimensionality.csv", index=False)
    return df


def q_scaling_outliers(frac=0.05, factor=10.0):
    """Concrete: invariancia al escalado y robustez ante atípicos (KNN como control)."""
    d = D.concrete()
    X, y = d["X"], d["y"]
    cv = StratifiedKFold(10, shuffle=True, random_state=RS)
    rng = np.random.default_rng(RS)
    Xo = X.copy()
    mask = rng.random(X.shape) < frac
    Xo[mask] = Xo[mask] * factor  # valores atípicos multiplicativos (errores de captura)
    rows = []
    for cond, XX in [("original (sin escalar)", X), (f"{int(frac*100)}% de celdas atípicas (x{int(factor)})", Xo)]:
        for mname, mdl in [("Árbol de Decisión", DecisionTreeClassifier(random_state=RS)),
                           ("Árbol + StandardScaler", make_pipeline(StandardScaler(), DecisionTreeClassifier(random_state=RS))),
                           ("Random Forest", RandomForestClassifier(200, random_state=RS, n_jobs=-1)),
                           ("Gradient Boosting", GradientBoostingClassifier(random_state=RS)),
                           ("KNN (k=5) sin escalar", KNeighborsClassifier(5)),
                           ("KNN (k=5) + StandardScaler", make_pipeline(StandardScaler(), KNeighborsClassifier(5))),
                           ("KNN (k=5) + MinMaxScaler", make_pipeline(MinMaxScaler(), KNeighborsClassifier(5)))]:
            s = cross_val_score(mdl, XX, y, cv=cv, scoring="accuracy")
            rows.append(dict(condition=cond, model=mname, acc=s.mean(), std=s.std()))
    df = pd.DataFrame(rows)
    rng_ = pd.DataFrame({"feature": d["feature_names"], "min": X.min(0), "max": X.max(0),
                         "n_outliers_iqr": [int(np.sum((c < np.percentile(c, 25) - 1.5 * (np.percentile(c, 75) - np.percentile(c, 25))) |
                                                       (c > np.percentile(c, 75) + 1.5 * (np.percentile(c, 75) - np.percentile(c, 25))))) for c in X.T]})
    df.to_csv(RES / "q_scaling_outliers.csv", index=False)
    rng_.to_csv(RES / "concrete_ranges.csv", index=False)
    return df


EXPERIMENTS = {"benchmark_small": lambda: benchmark([k for k in list(D.REAL) + list(D.SYNTH)
                                                     if k not in ("Credit Card", "HAR")], out_name="benchmark_small.csv"),
               "benchmark_credit": lambda: benchmark(["Credit Card"]),
               "benchmark_har": lambda: benchmark(["HAR"]),
               "rotation": rotation_sweep, "depth": depth_sweep, "ccp": ccp_path,
               "correlation": q_correlation, "distribution": q_distribution, "imbalance": q_imbalance,
               "dimensionality": q_dimensionality, "scaling": q_scaling_outliers}

# ----------------------------------------------------------------------------- ejecución reanudable
def _done_keys(path, keys):
    if not path.exists():
        return set()
    df = pd.read_csv(path)
    return set(map(tuple, df[keys].astype(str).to_numpy()))


def _append(path, row):
    """Agrega una fila alineando columnas con el archivo existente (las métricas binarias pueden faltar)."""
    new = pd.DataFrame([row])
    if path.exists():
        new = pd.concat([pd.read_csv(path), new], ignore_index=True)
    new.to_csv(path, index=False)


def big_runs(budget_s=10**9):
    """Credit Card y HAR (benchmark + variantes ponderadas), guardando fila por fila.

    Versión Colab: Credit Card completo con CV estratificada de 10 pliegues, HAR con CV de 5 pliegues
    agrupada por sujeto y RF de 200 árboles (igual que en los conjuntos pequeños). Guarda fila por fila:
    si la sesión se cae, basta con volver a ejecutar y continúa donde se quedó.
    """
    t0 = time.perf_counter()
    out = RES / "benchmark_big.csv"
    done = _done_keys(out, ["dataset", "model", "fold"])
    for name in ["Credit Card", "HAR"]:
        d = D.REAL[name]()
        for fold, (tr, te) in enumerate(splits(d, 10 if name == "Credit Card" else 5)):
            sw = compute_sample_weight("balanced", d["y"][tr])
            tasks = {**{k: (v, None) for k, v in models(n_estimators_rf=200, fast=not FULL_PRUNING).items()}}
            if name == "Credit Card":
                tasks.update({
                    "Árbol de Decisión (bal)": (DecisionTreeClassifier(class_weight="balanced", random_state=RS), None),
                    "Random Forest (bal)": (RandomForestClassifier(200, class_weight="balanced_subsample",
                                                                   random_state=RS, n_jobs=-1), None),
                    "Gradient Boosting (bal)": (GradientBoostingClassifier(random_state=RS), sw)})
            for mname, (mdl, w) in tasks.items():
                if (name, mname, str(fold)) in done:
                    continue
                if time.perf_counter() - t0 > budget_s:
                    print("  presupuesto agotado; reanudar después", flush=True)
                    return False
                m = fit_eval(clone(mdl), d["X"][tr], d["y"][tr], d["X"][te], d["y"][te], sample_weight=w)
                m.update(dataset=name, category=1, model=mname, fold=fold, n=len(d["y"]), p=d["X"].shape[1],
                         n_classes=len(np.unique(d["y"])))
                _append(out, m)
                print(f"  {name} | {mname} | fold {fold}: bal_acc={m['bal_acc']:.3f} ({m['t_fit_s']:.0f}s)", flush=True)
    return True


EXPERIMENTS["big"] = big_runs


if __name__ == "__main__":
    todo = sys.argv[1:] or list(EXPERIMENTS)
    for name in todo:
        t0 = time.perf_counter()
        print(f"== {name}", flush=True)
        EXPERIMENTS[name]()
        print(f"   ({time.perf_counter() - t0:.1f} s)", flush=True)
