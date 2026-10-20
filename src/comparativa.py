"""
comparativa.py
==============
Comparativa de los 13 clasificadores del ABP. Uso:

    python -m src.comparativa <paso> [<paso> ...]

Pasos:
  verify_folds   Comprueba que los pliegues de este equipo son idénticos a los del estudio base
                 (results/benchmark_small.csv y benchmark_big.csv): re-entrena el árbol Gini sin poda, que es
                 determinista, y compara pliegue por pliegue.
  benchmark      Comparativa de los 13 clasificadores con la misma validación cruzada del estudio base
                 (10 pliegues estratificados; HAR: 5 pliegues agrupados por sujeto). Los resultados de CART, RF
                 y GB se importan del estudio base si existen (son los mismos modelos); use benchmark_full para
                 recalcularlos. Reanudable: guarda fila por fila.
  benchmark_full Igual, pero recalcula todos los modelos.
  benchmark_new  Sólo los 10 modelos que no evalúa el estudio base (parte 1 del flujo Colab + Anaconda).
  resampling_baseline3  Línea base "sin tratamiento" de Credit Card en 3 pliegues (sólo si el remuestreo de
                 extras.py se ejecutó con 3 pliegues).
  separability   Prueba exacta de separabilidad lineal por programación lineal (uno contra uno).
  cv_schemes     Justificación del k-fold estratificado: holdout 90/10 repetido vs. 10-fold estratificado,
                 10-fold sin estratificar y 5-fold estratificado (variabilidad de la estimación).
  timing         Prueba de tiempos controlada: partición 90/10 fija, mismo equipo, 1 hilo, repeticiones;
                 tiempo real, tiempo de CPU y memoria pico.
  significance   Friedman + Nemenyi + comparación contra el mejor modelo con corrección de Holm.
  summary        Tablas resumen (CSV).
"""
from __future__ import annotations

import gc
import json
import os
import platform
import sys
import time
import tracemalloc
import warnings
from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import linprog
from sklearn.base import clone
from sklearn.metrics import balanced_accuracy_score, matthews_corrcoef
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

from . import config as C
from . import datasets as D
from .c50 import C50Classifier, check_c50
from .experiments import RES, RS, _append, _done_keys, evaluate, impute_fold, splits
from .knn_family import stratified_subsample
from .models import FAMILY, MODEL_ORDER, BASE_EQUIV, final_estimator, make_models, predict_both

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

ALL = {**D.REAL, **D.SYNTH}
SMALL = [k for k in ALL if k not in ("Credit Card", "HAR")]
METRICS = ["train_acc", "test_acc", "bal_acc", "f1_macro", "mcc", "recall_pos", "precision_pos", "pr_auc", "gap"]


def datasets():
    names = C.DATASETS or list(ALL)
    return [(n, ALL[n]) for n in names]


def n_splits_for(name):
    if C.SMOKE:
        return 3
    return 5 if name == "HAR" else 10


def _c50_available():
    if not C.USE_C50:
        return False, "C5.0 desactivado en config.py"
    return check_c50()


def fit_eval_comp(model, Xtr, ytr, Xte, yte, train_cap):
    """Entrena, predice y evalúa. La exactitud de entrenamiento usa una submuestra si n_train > train_cap."""
    Xtr, Xte = impute_fold(Xtr, Xte)
    t0 = time.perf_counter(); model.fit(Xtr, ytr); t_fit = time.perf_counter() - t0
    idx = stratified_subsample(np.unique(ytr, return_inverse=True)[1], train_cap, RS)
    if isinstance(final_estimator(model), C50Classifier):
        # C5.0 corre en R: prueba y entrenamiento en una sola llamada (mismo resultado, un arranque de R menos)
        t0 = time.perf_counter(); P, yall = predict_both(model, np.vstack([Xte, Xtr[idx]])); t_pred = time.perf_counter() - t0
        proba, yhat, yhat_tr = P[:len(yte)], yall[:len(yte)], yall[len(yte):]
    else:
        t0 = time.perf_counter(); proba, yhat = predict_both(model, Xte); t_pred = time.perf_counter() - t0
        _, yhat_tr = predict_both(model, Xtr[idx])
    m = evaluate(yte, yhat, proba, ytr[idx], yhat_tr)
    m.update(t_fit_s=t_fit, t_pred_ms=1e3 * t_pred / len(yte), n_train_eval=len(idx))
    est = final_estimator(model)
    for attr, col in (("k", "k_used"), ("n_train_used_", "n_train_used"), ("tree_size_", "tree_size"),
                      ("n_leaves_", "leaves")):
        if hasattr(est, attr):
            m[col] = getattr(est, attr)
    if isinstance(est, DecisionTreeClassifier):
        m.update(leaves=est.get_n_leaves(), depth=est.get_depth())
    return m


# ----------------------------------------------------------------------------- ejecución en paralelo
def run_parallel(func, tasks, on_result):
    """Ejecuta func(*task) para cada tarea; on_result(resultado) se llama en el proceso principal en cuanto
    termina cada tarea (así lo terminado se guarda aunque la corrida se interrumpa). Con N_WORKERS = 1 corre
    en serie. El paralelismo no cambia resultados: cada tarea es independiente y determinista."""
    if C.N_WORKERS <= 1 or len(tasks) <= 1:
        for t in tasks:
            on_result(func(*t))
        return
    from joblib import Parallel, delayed
    try:
        gen = Parallel(n_jobs=C.N_WORKERS, return_as="generator_unordered")(delayed(func)(*t) for t in tasks)
    except TypeError:                                   # joblib antiguo
        gen = Parallel(n_jobs=C.N_WORKERS)(delayed(func)(*t) for t in tasks)
    for r in gen:
        on_result(r)


def _single_thread():
    from threadpoolctl import threadpool_limits
    return threadpool_limits(limits=1)


# ----------------------------------------------------------------------------- verificación de pliegues
def load_base():
    parts = [pd.read_csv(RES / f) for f in ("benchmark_small.csv", "benchmark_big.csv") if (RES / f).exists()]
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def verify_folds():
    """Re-entrena el árbol Gini sin poda ('Árbol de Decisión' del estudio base) y compara su BA pliegue por pliegue."""
    bench = load_base()
    if bench.empty:
        print("  No hay resultados del estudio base en results/ (benchmark_small.csv / benchmark_big.csv).")
        return False
    rows = []
    for name, loader in datasets():
        ref = bench[(bench.dataset == name) & (bench.model == "Árbol de Decisión")].set_index("fold")["bal_acc"]
        if ref.empty:
            rows.append(dict(dataset=name, pliegues=0, coinciden=0, dif_max=np.nan, estado="sin datos del estudio base"))
            print(f"  {name}: sin resultados del estudio base"); continue
        d = loader()
        folds = splits(d, n_splits_for(name))
        diffs = []
        for f, (tr, te) in enumerate(folds):
            if f not in ref.index:
                diffs.append(np.inf); continue
            m = fit_eval_comp(DecisionTreeClassifier(random_state=RS), d["X"][tr], d["y"][tr], d["X"][te], d["y"][te],
                            C.TRAIN_EVAL_MAX)
            diffs.append(abs(m["bal_acc"] - ref.loc[f]))
        ok = int(np.sum(np.array(diffs) < 1e-9))
        estado = "OK" if ok == len(folds) == len(ref) else "NO COINCIDE"
        rows.append(dict(dataset=name, pliegues=len(folds), coinciden=ok, dif_max=float(np.max(diffs)), estado=estado))
        print(f"  {name}: {ok}/{len(folds)} pliegues idénticos -> {estado}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(RES / "comp_verificacion_pliegues.csv", index=False)
    all_ok = bool((df.estado == "OK").all())
    print("  RESULTADO:", "se pueden reutilizar los resultados del estudio base" if all_ok else
          "hay diferencias: use 'python -m src.comparativa benchmark_full' (recalcula CART, RF y GB)")
    return all_ok


# ----------------------------------------------------------------------------- comparativa de 13 modelos
def _fold_task(name, fold, base, X, y, tr, te, pending, large):
    """Tarea de un pliegue (corre en un proceso aparte): entrena y evalúa los modelos pendientes."""
    rows = []
    with _single_thread():
        fac = make_models(n_jobs=1, large=large)
        for mname in pending:
            cap = C.SLOW_TRAIN_EVAL_MAX if mname == "CM-KNN" else C.TRAIN_EVAL_MAX
            m = fit_eval_comp(fac[mname](), X[tr], y[tr], X[te], y[te], cap)
            rows.append({**base, "model": mname, "family": FAMILY[mname], **m, "estado": "ok"})
    return rows


def benchmark(reuse_base=True, only_extra=False):
    out = RES / "benchmark_comp.csv"
    done = _done_keys(out, ["dataset", "model", "fold"])
    bench = load_base() if reuse_base else pd.DataFrame()
    no_reuse = set()                      # conjuntos cuyos pliegues no coinciden con los del estudio base
    vf = RES / "comp_verificacion_pliegues.csv"
    if reuse_base and vf.exists():
        ver = pd.read_csv(vf)
        no_reuse = set(ver.loc[ver.estado != "OK", "dataset"])
        if no_reuse:
            print("  Pliegues distintos a los del estudio base en:", ", ".join(sorted(no_reuse)),
                  "-> CART, RF y GB se recalculan ahí (en los demás se importan)", flush=True)
    c50_ok, c50_msg = _c50_available()
    if not c50_ok:
        print("  AVISO C5.0:", c50_msg)
    print(f"  procesos en paralelo: {C.N_WORKERS}", flush=True)
    tasks = []
    for name, loader in datasets():
        d = loader()
        folds = splits(d, n_splits_for(name))
        large = len(d["y"]) > C.LARGE_N
        for fold, (tr, te) in enumerate(folds):
            base = dict(dataset=name, fold=fold, n=len(d["y"]), p=d["X"].shape[1], n_classes=len(np.unique(d["y"])))
            pending = []
            for mname in MODEL_ORDER:
                if (name, mname, str(fold)) in done or (only_extra and mname in BASE_EQUIV):
                    continue
                info = {**base, "model": mname, "family": FAMILY[mname]}
                if reuse_base and mname in BASE_EQUIV and not bench.empty and name not in no_reuse:
                    r = bench[(bench.dataset == name) & (bench.model == BASE_EQUIV[mname]) & (bench.fold == fold)]
                    if len(r):
                        r = r.iloc[0]
                        _append(out, {**info, **{k: r.get(k, np.nan) for k in METRICS + ["leaves", "depth"]},
                                      "estado": "importado del estudio base"})
                        continue
                if mname == "C5.0" and not c50_ok:
                    _append(out, {**info, "estado": "no disponible (falta R/C50)"}); continue
                if mname == "CM-KNN" and len(te) > C.CMKNN_MAX_TEST:
                    _append(out, {**info, "estado": "omitido (costo O(m n^3))"}); continue
                pending.append(mname)
            if pending:
                tasks.append((name, fold, base, d["X"], d["y"], tr, te, pending, large))
    print(f"  pliegues por calcular: {len(tasks)}", flush=True)

    def save(rows):
        for row in rows:
            _append(out, row)
        if rows:
            r0 = rows[0]
            print(f"  {r0['dataset']} | pliegue {r0['fold']}: " +
                  ", ".join(f"{r['model']} {r['bal_acc']:.3f}" for r in rows), flush=True)

    run_parallel(_fold_task, tasks, save)


def benchmark_new():
    benchmark(reuse_base=False, only_extra=True)


def benchmark_full():
    benchmark(reuse_base=False)


# ----------------------------------------------------------------------------- separabilidad lineal
def lp_separable(X, y01, time_limit):
    """¿Existe (w, b) con y_i (w.x_i + b) >= 1 para todo i? Programa lineal de factibilidad (HiGHS)."""
    s = np.where(y01 == 1, 1.0, -1.0)
    A = -s[:, None] * np.hstack([X, np.ones((len(X), 1))])
    res = linprog(np.zeros(X.shape[1] + 1), A_ub=A, b_ub=-np.ones(len(X)), bounds=(None, None),
                  method="highs", options={"time_limit": float(time_limit)})
    return {0: "separable", 2: "no separable"}.get(res.status, "indeterminado")


def _sep_task(name):
    with _single_thread():
        d = ALL[name]()
        X = d["X"].copy()
        if np.isnan(X).any():
            med = np.nanmedian(X, axis=0); r, c = np.where(np.isnan(X)); X[r, c] = med[c]
        X = StandardScaler().fit_transform(X)
        y = d["y"]; classes = np.unique(y)
        t0 = time.perf_counter()
        res = [lp_separable(X[(y == a) | (y == b)], (y[(y == a) | (y == b)] == b).astype(int), C.LP_TIME_LIMIT)
               for a, b in combinations(classes, 2)]
        t_lp = time.perf_counter() - t0
        svm = LinearSVC(C=100.0, dual=False, max_iter=5000).fit(X, y)
    return dict(dataset=name, n=len(y), p=X.shape[1], clases=len(classes), pares=len(res),
                pares_separables=res.count("separable"), pares_no_separables=res.count("no separable"),
                pares_indeterminados=res.count("indeterminado"),
                linealmente_separable=("sí" if res.count("separable") == len(res) else
                                       "no" if "no separable" in res else "indeterminado"),
                exactitud_lineal_entrenamiento=float(svm.score(X, y)), t_lp_s=t_lp)


def separability():
    rows = []

    def save(r):
        rows.append(r)
        print(f"  {r['dataset']}: {r['linealmente_separable']} ({r['pares_separables']}/{r['pares']} pares), "
              f"SVM lineal entren. = {r['exactitud_lineal_entrenamiento']:.3f}", flush=True)

    run_parallel(_sep_task, [(n,) for n, _ in datasets()], save)
    order = [n for n, _ in datasets()]
    pd.DataFrame(rows).set_index("dataset").loc[order].reset_index().to_csv(RES / "comp_separabilidad.csv", index=False)


# ----------------------------------------------------------------------------- justificación del k-fold
def _schemes():
    return [("Holdout 90/10", C.HOLDOUT_REPS,
             lambda r, X, y: [train_test_split(np.arange(len(y)), test_size=0.1, stratify=y, random_state=RS + r)]),
            ("10-fold estratificado", C.CV_REPS,
             lambda r, X, y: list(StratifiedKFold(10, shuffle=True, random_state=RS + r).split(X, y))),
            ("10-fold sin estratificar", C.CV_REPS,
             lambda r, X, y: list(KFold(10, shuffle=True, random_state=RS + r).split(X))),
            ("5-fold estratificado", C.CV_REPS,
             lambda r, X, y: list(StratifiedKFold(5, shuffle=True, random_state=RS + r).split(X, y)))]


def _cv_task(name, mname, scheme_idx, reps_todo):
    scheme, _, gen = _schemes()[scheme_idx]
    rows = []
    with _single_thread():
        d = ALL[name]()
        X, y = d["X"], d["y"]
        K = len(np.unique(y))
        factory = make_models(n_jobs=1)[mname]
        for r in reps_todo:
            folds = gen(r, X, y)
            scores, missing = [], 0
            for tr, te in folds:
                missing += int(len(np.unique(y[te])) < K)
                Xa, Xb = impute_fold(X[tr], X[te])
                mdl = factory().fit(Xa, y[tr])
                scores.append(balanced_accuracy_score(y[te], predict_both(mdl, Xb)[1]))
            rows.append(dict(dataset=name, model=mname, scheme=scheme, rep=r, k=len(folds),
                             estimate=float(np.mean(scores)), fold_std=float(np.std(scores)),
                             folds_missing_class=missing))
    return rows


def cv_schemes():
    out = RES / "comp_cv_esquemas.csv"
    done = _done_keys(out, ["dataset", "model", "scheme", "rep"])
    tasks = []
    for name in (C.CV_SCHEME_DATASETS or SMALL):
        for mname in C.CV_SCHEME_MODELS:
            for i, (scheme, reps, _) in enumerate(_schemes()):
                todo = [r for r in range(reps) if (name, mname, scheme, str(r)) not in done]
                if todo:
                    tasks.append((name, mname, i, todo))

    def save(rows):
        for row in rows:
            _append(out, row)
        if rows:
            print(f"  esquemas de validación: {rows[0]['dataset']} | {rows[0]['model']} | {rows[0]['scheme']} listo",
                  flush=True)

    run_parallel(_cv_task, tasks, save)


# ----------------------------------------------------------------------------- línea base del remuestreo (3 pliegues)
def _base3_task(fold, X, y, tr, te):
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
    from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score
    rows = []
    with _single_thread():
        for mname, mdl in (("Árbol de Decisión", DecisionTreeClassifier(random_state=RS)),
                           ("Random Forest", RandomForestClassifier(200, random_state=RS, n_jobs=1)),
                           ("Gradient Boosting", GradientBoostingClassifier(random_state=RS))):
            Xtr, Xte = impute_fold(X[tr], X[te])
            t1 = time.perf_counter(); mdl.fit(Xtr, y[tr]); t_fit = time.perf_counter() - t1
            p = mdl.predict_proba(Xte)[:, 1]; yh = mdl.predict(Xte)
            rows.append(dict(model=mname, strategy="sin tratamiento", fold=fold, threshold=0.5, t_fit_s=t_fit,
                             test_acc=float(np.mean(yh == y[te])), bal_acc=balanced_accuracy_score(y[te], yh), recall_pos=recall_score(y[te], yh),
                             precision_pos=precision_score(y[te], yh, zero_division=0), f1=f1_score(y[te], yh),
                             mcc=matthews_corrcoef(y[te], yh), pr_auc=average_precision_score(y[te], p)))
    return rows


def resampling_baseline3():
    """Credit Card sin tratamiento sobre los mismos 3 pliegues que el remuestreo reducido de extras.py."""
    out = RES / "comp_remuestreo_linea_base_3pliegues.csv"
    done = _done_keys(out, ["model", "fold"])
    d = D.REAL["Credit Card"]()
    tasks = [(f, d["X"], d["y"], tr, te) for f, (tr, te) in enumerate(splits(d, 3))
             if any((m, str(f)) not in done for m in ("Árbol de Decisión", "Random Forest", "Gradient Boosting"))]

    def save(rows):
        for row in rows:
            if (row["model"], str(row["fold"])) not in done:
                _append(out, row)
        print(f"  línea base 3 pliegues | pliegue {rows[0]['fold']} listo", flush=True)

    run_parallel(_base3_task, tasks, save)


# ----------------------------------------------------------------------------- prueba de tiempos
def system_info():
    info = dict(fecha=time.strftime("%Y-%m-%d %H:%M"), sistema=platform.platform(), procesador=platform.processor(),
                maquina=platform.machine(), nucleos_logicos=os.cpu_count(), python=sys.version.split()[0],
                entorno_conda=os.environ.get("CONDA_DEFAULT_ENV"), hilos_medicion=C.TIMING_THREADS)
    try:
        import psutil
        info["ram_gb"] = round(psutil.virtual_memory().total / 2**30, 1)
        info["nucleos_fisicos"] = psutil.cpu_count(logical=False)
    except ImportError:
        pass
    import importlib
    for lib in ("numpy", "scipy", "sklearn", "pandas", "imblearn"):
        try:
            info[lib] = importlib.import_module(lib).__version__
        except Exception:
            pass
    try:
        from threadpoolctl import threadpool_info
        info["blas"] = sorted({f"{t.get('internal_api')} {t.get('version')}" for t in threadpool_info()})
    except Exception:
        pass
    info["c50"] = check_c50()[1]
    with open(RES / "system_info.json", "w", encoding="utf-8") as fh:
        json.dump(info, fh, indent=2, ensure_ascii=False)
    return info


def timing():
    from threadpoolctl import threadpool_limits
    info = system_info()
    print("  Equipo:", info.get("procesador") or info.get("maquina"), "| núcleos:", info.get("nucleos_logicos"),
          "| RAM:", info.get("ram_gb"), "GB | entorno:", info.get("entorno_conda"), flush=True)
    out = RES / "comp_tiempos.csv"
    done = _done_keys(out, ["dataset", "model", "rep"])
    c50_ok, _ = _c50_available()
    for name, loader in datasets():
        d = loader()
        tr, te = train_test_split(np.arange(len(d["y"])), test_size=0.1, stratify=d["y"], random_state=RS)
        Xtr, Xte = impute_fold(d["X"][tr], d["X"][te]); ytr, yte = d["y"][tr], d["y"][te]
        large = len(d["y"]) > C.LARGE_N
        reps = C.TIMING_REPS_LARGE if len(d["y"]) > C.TIMING_LARGE_N else C.TIMING_REPS
        mem_on = C.TIMING_MEMORY and len(d["y"]) <= C.TIMING_MEMORY_MAX_N
        for mname, factory in make_models(n_jobs=C.TIMING_THREADS, large=large).items():
            base = dict(dataset=name, model=mname, family=FAMILY[mname], n_train=len(tr), n_test=len(te),
                        p=d["X"].shape[1])
            if mname == "C5.0" and not c50_ok:
                if (name, mname, "0") not in done:
                    _append(out, {**base, "rep": 0, "estado": "no disponible (falta R/C50)"})
                continue
            if mname == "CM-KNN" and len(te) > C.CMKNN_MAX_TEST:
                if (name, mname, "0") not in done:
                    _append(out, {**base, "rep": 0, "estado": "omitido (costo O(m n^3))"})
                continue
            with threadpool_limits(limits=C.TIMING_THREADS):
                peak = np.nan
                # C5.0 corre en un proceso de R: tracemalloc sólo vería Python, así que su memoria no se mide
                if mem_on and mname != "C5.0" and not all((name, mname, str(r)) in done for r in range(reps)):
                    gc.collect(); tracemalloc.start()
                    mdl = factory().fit(Xtr, ytr); predict_both(mdl, Xte)
                    peak = tracemalloc.get_traced_memory()[1] / 2**20
                    tracemalloc.stop(); del mdl
                for r in range(reps):
                    if (name, mname, str(r)) in done:
                        continue
                    gc.collect()
                    mdl = factory()
                    w0, c0 = time.perf_counter(), time.process_time(); mdl.fit(Xtr, ytr)
                    w1, c1 = time.perf_counter(), time.process_time(); proba, yhat = predict_both(mdl, Xte)
                    w2, c2 = time.perf_counter(), time.process_time()
                    row = {**base, "rep": r, "estado": "ok", "fit_wall_s": w1 - w0, "fit_cpu_s": c1 - c0,
                           "pred_wall_s": w2 - w1, "pred_cpu_s": c2 - c1,
                           "pred_ms_por_muestra": 1e3 * (w2 - w1) / len(te), "mem_pico_mb": peak if r == 0 else np.nan,
                           "bal_acc_holdout": balanced_accuracy_score(yte, yhat), "mcc_holdout": matthews_corrcoef(yte, yhat)}
                    est = final_estimator(mdl)
                    if hasattr(est, "fit_time_r_"):     # C5.0: tiempos medidos dentro de R (sin E/S ni arranque)
                        row.update(fit_wall_r_s=est.fit_time_r_, fit_cpu_r_s=est.fit_cpu_r_,
                                   pred_wall_r_s=est.predict_time_r_, pred_cpu_r_s=est.predict_cpu_r_)
                    _append(out, row)
                    print(f"  tiempos | {name} | {mname} | rep {r}: entrenamiento {w1 - w0:.2f}s, "
                          f"predicción {w2 - w1:.2f}s", flush=True)
                    del mdl


# ----------------------------------------------------------------------------- significancia
Q_NEMENYI = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164,
             11: 3.219, 12: 3.268, 13: 3.313, 14: 3.354, 15: 3.391}


def _rank_tests(M, label):
    N, k = M.shape
    ranks = M.rank(axis=1, ascending=False, method="average")
    R = ranks.mean()
    chi2, p_f = stats.friedmanchisquare(*[M[c] for c in M.columns])
    Ff = (N - 1) * chi2 / (N * (k - 1) - chi2) if N * (k - 1) != chi2 else np.inf
    p_id = stats.f.sf(Ff, k - 1, (k - 1) * (N - 1))
    CD = Q_NEMENYI.get(k, np.nan) * np.sqrt(k * (k + 1) / (6.0 * N))
    ctrl = R.idxmin()
    se = np.sqrt(k * (k + 1) / (6.0 * N))
    comp = pd.DataFrame({"model": [c for c in M.columns if c != ctrl]})
    comp["rank_diff"] = [R[c] - R[ctrl] for c in comp.model]
    comp["z"] = comp.rank_diff / se
    comp["p"] = 2 * stats.norm.sf(np.abs(comp.z))
    comp = comp.sort_values("p").reset_index(drop=True)
    m, run, adj = len(comp), 0.0, []
    for i, p in enumerate(comp.p):
        run = max(run, min(1.0, (m - i) * p)); adj.append(run)
    comp["p_holm"] = adj
    comp["significativo_vs_control"] = comp.p_holm < 0.05
    comp["nemenyi_vs_control"] = comp.rank_diff > CD
    comp.insert(0, "control", ctrl); comp.insert(0, "analisis", label)
    summ = dict(analisis=label, N=N, k=k, chi2=chi2, p_friedman=p_f, F_iman_davenport=Ff, p_iman_davenport=p_id,
                CD_nemenyi=CD, control=ctrl, datasets=";".join(M.index))
    rk = pd.DataFrame({"analisis": label, "model": R.index, "rango_promedio": R.values,
                       "ba_media": M.mean().values}).sort_values("rango_promedio")
    return summ, comp, rk


def significance():
    df = pd.read_csv(RES / "benchmark_comp.csv")
    df = df[df.estado.isin(["ok", "importado del estudio base"])]
    M = df.groupby(["dataset", "model"]).bal_acc.mean().unstack()
    models = [m for m in MODEL_ORDER if m in M.columns]
    M = M[models]
    analyses = [("A: conjuntos con los 13 modelos", M.dropna(axis=0)),
                ("B: los 14 conjuntos, modelos completos", M.dropna(axis=1))]
    S, Cmp, Rk = [], [], []
    for label, MM in analyses:
        if MM.shape[0] < 2 or MM.shape[1] < 2:
            continue
        s, c, r = _rank_tests(MM, label); S.append(s); Cmp.append(c); Rk.append(r)
        print(f"  {label}: N={s['N']} k={s['k']} Friedman p={s['p_friedman']:.2e}, CD={s['CD_nemenyi']:.2f}, "
              f"mejor={s['control']}", flush=True)
    pd.DataFrame(S).to_csv(RES / "comp_friedman.csv", index=False)
    pd.concat(Cmp).to_csv(RES / "comp_vs_control_holm.csv", index=False)
    pd.concat(Rk).to_csv(RES / "comp_rangos.csv", index=False)


# ----------------------------------------------------------------------------- resúmenes
def summary():
    df = pd.read_csv(RES / "benchmark_comp.csv")
    ok = df[df.estado.isin(["ok", "importado del estudio base"])]
    g = ok.groupby(["dataset", "model"])
    t = g[["bal_acc", "mcc", "f1_macro", "gap", "train_acc", "test_acc", "pr_auc"]].mean()
    t["bal_acc_std"] = g.bal_acc.std()
    t.reset_index().to_csv(RES / "comp_resumen_dataset_modelo.csv", index=False)
    per = ok.groupby("model").agg(ba_media=("bal_acc", "mean"), mcc_media=("mcc", "mean"), brecha_media=("gap", "mean"))
    per["familia"] = [FAMILY[m] for m in per.index]
    per.reindex([m for m in MODEL_ORDER if m in per.index]).to_csv(RES / "comp_resumen_modelos.csv")
    status = df.groupby(["model", "estado"]).size().unstack(fill_value=0)
    status.to_csv(RES / "comp_estado_corridas.csv")
    if (RES / "comp_cv_esquemas.csv").exists():
        cv = pd.read_csv(RES / "comp_cv_esquemas.csv")
        s = cv.groupby(["dataset", "model", "scheme"]).agg(estimacion_media=("estimate", "mean"),
                                                          variabilidad_std=("estimate", "std"),
                                                          rango=("estimate", lambda v: v.max() - v.min()),
                                                          pliegues_sin_alguna_clase=("folds_missing_class", "sum"))
        s.reset_index().to_csv(RES / "comp_cv_resumen.csv", index=False)
    if (RES / "comp_tiempos.csv").exists():
        tm = pd.read_csv(RES / "comp_tiempos.csv")
        tm = tm[tm.estado == "ok"].copy()
        # C5.0 corre en R: se usan los tiempos medidos dentro de R (sin arranque de R ni escritura de archivos)
        for a, b in (("fit_wall_s", "fit_wall_r_s"), ("fit_cpu_s", "fit_cpu_r_s"), ("pred_wall_s", "pred_wall_r_s"),
                     ("pred_cpu_s", "pred_cpu_r_s")):
            if b in tm:
                tm[a + "_py"] = tm[a]
                tm[a] = tm[b].where(tm[b].notna(), tm[a])
        tm["pred_ms_por_muestra"] = 1e3 * tm.pred_wall_s / tm.n_test
        agg = tm.groupby(["dataset", "model"]).agg(fit_s=("fit_wall_s", "mean"), fit_s_std=("fit_wall_s", "std"),
                                                   fit_cpu_s=("fit_cpu_s", "mean"), pred_s=("pred_wall_s", "mean"),
                                                   pred_ms_muestra=("pred_ms_por_muestra", "mean"),
                                                   mem_pico_mb=("mem_pico_mb", "max"), repeticiones=("rep", "count"))
        agg.reset_index().to_csv(RES / "comp_tiempos_resumen.csv", index=False)
    print("  resúmenes guardados en", RES)


STEPS = {"verify_folds": verify_folds, "benchmark": benchmark, "benchmark_full": benchmark_full,
         "benchmark_new": benchmark_new, "resampling_baseline3": resampling_baseline3,
         "separability": separability, "cv_schemes": cv_schemes, "timing": timing,
         "significance": significance, "summary": summary}

if __name__ == "__main__":
    for step in sys.argv[1:] or ["benchmark", "separability", "cv_schemes", "timing", "significance", "summary"]:
        t0 = time.perf_counter()
        print(f"== comparativa {step}", flush=True)
        r = STEPS[step]()
        print(f"   ({time.perf_counter() - t0:.1f} s)", flush=True)
        if step == "verify_folds" and r is False:
            sys.exit(2)
