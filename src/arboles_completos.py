"""
arboles_completos.py
====================
Características de los árboles completos (DT, CART, C5.0 y CHAID) en los 14 conjuntos.

  * Hojas y profundidad medias en la validación cruzada (se leen de results/benchmark_comp.csv).
  * Árbol entrenado con todos los datos de cada conjunto: hojas, profundidad, nodos y corte de la raíz.
  * Figura del árbol CART completo de BCW (fig_arbol_cart_bcw).

Uso:  python -m src.arboles_completos
"""
from __future__ import annotations

import re
import subprocess

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree

from . import datasets as D
from .c50 import C50Classifier
from .chaid import CHAIDClassifier
from .experiments import RES, PrunedTree, RS
from .visualization import FIG

ORDER = ["Iris", "BCW (Original)", "Wine", "Glass", "Concrete", "Forest Fires", "Credit Card", "HAR",
         "Ortogonal IF-THEN", "XOR 0°", "Lineal rotado 45°", "XOR rotado 45°", "Circles", "Moons"]
CV_NAME = {"DT": "Árbol de Decisión (entropía)", "CART": "CART (Gini + poda)", "C5.0": "C5.0", "CHAID": "CHAID"}


def _impute(X):
    X = X.copy()
    med = np.nanmedian(X, axis=0)
    rows, cols = np.where(np.isnan(X))
    X[rows, cols] = med[cols]
    return X


def _loaders():
    return {**D.REAL, **D.SYNTH}


def _sk_root(m, names):
    t = m.tree_
    if t.node_count == 1:
        return "(sin división)", np.nan
    return names[t.feature[0]], float(t.threshold[0])


def _c50_root(m, names):
    """Lee el árbol C5.0 guardado por R (atributos f1..fp): corte de la raíz y profundidad."""
    code = f'suppressMessages(library(C50)); m <- readRDS("{m.model_path_.as_posix()}"); print(summary(m))'
    out = subprocess.run([m.rscript_, "-e", code], capture_output=True, text=True).stdout
    block = out.split("Decision tree:", 1)[-1].split("Evaluation on training data", 1)[0]
    lines = [ln for ln in block.splitlines() if ln.strip()]
    hit = re.search(r"\bf(\d+)\s*(<=|>|=|in)\s*([-+\d.eE]+)?", lines[0]) if lines else None
    if not hit:
        return "(sin división)", np.nan, 0
    depth = 1 + max((len(ln) - len(ln.lstrip(" :.")) ) // 4 for ln in lines)
    thr = float(hit.group(3)) if hit.group(3) else np.nan
    return names[int(hit.group(1)) - 1], thr, depth


def c50_only():
    """Recalcula sólo las filas de C5.0 de results/arboles_completos.csv."""
    out = pd.read_csv(RES / "arboles_completos.csv")
    loaders = _loaders()
    for name in ORDER:
        d = loaders[name]()
        c5 = C50Classifier().fit(_impute(d["X"]), d["y"])
        feat, thr, depth = _c50_root(c5, d["feature_names"])
        k = (out.dataset == name) & (out.model == "C5.0")
        out.loc[k, ["hojas", "raiz", "umbral", "profundidad"]] = [c5.tree_size_, feat, thr, depth]
        print(f"  {name}: C5.0 raíz = {feat}, profundidad {depth}, hojas {c5.tree_size_}", flush=True)
    out.to_csv(RES / "arboles_completos.csv", index=False)


def main():
    loaders = _loaders()
    rows = []
    for name in ORDER:
        d = loaders[name]()
        X, y, names = _impute(d["X"]), d["y"], d["feature_names"]
        dt = DecisionTreeClassifier(criterion="entropy", random_state=RS).fit(X, y)
        big = len(y) > 5000          # Credit Card y HAR: búsqueda de alfa corta (10 alfas, CV interna de 3)
        cart = (PrunedTree(n_alphas=10, inner_cv=3) if big else PrunedTree()).fit(X, y)
        for label, m in (("DT", dt), ("CART", cart)):
            feat, thr = _sk_root(m, names)
            rows.append(dict(dataset=name, model=label, hojas=m.get_n_leaves(), profundidad=m.get_depth(),
                             nodos=m.tree_.node_count, raiz=feat, umbral=thr))
        ch = CHAIDClassifier().fit(X, y)
        root = ch.tree_
        feat = names[root["feature"]] if not root.get("leaf", True) else "(sin división)"
        ramas = len(root["children"]) if not root.get("leaf", True) else 0
        rows.append(dict(dataset=name, model="CHAID", hojas=ch.n_leaves_, profundidad=ch.depth_,
                         nodos=ch.n_nodes_, raiz=feat, umbral=np.nan, ramas_raiz=ramas))
        c5 = C50Classifier().fit(X, y)
        feat, thr, c5_depth = _c50_root(c5, names)
        rows.append(dict(dataset=name, model="C5.0", hojas=c5.tree_size_, profundidad=c5_depth, nodos=np.nan,
                         raiz=feat, umbral=thr))
        print(f"  {name}: raíz DT = {rows[-4]['raiz']} | CART = {rows[-3]['raiz']} | "
              f"CHAID = {rows[-2]['raiz']} | C5.0 = {rows[-1]['raiz']}", flush=True)
    full = pd.DataFrame(rows)
    # hojas y profundidad medias en la validación cruzada
    b = pd.read_csv(RES / "benchmark_comp.csv")
    b = b[b.estado != "omitido (costo O(m n^3))"]
    cv = []
    for label, mname in CV_NAME.items():
        s = b[b.model == mname]
        hojas = s.groupby("dataset")["tree_size" if label == "C5.0" else "leaves"].mean()
        prof = s.groupby("dataset")["depth"].mean()
        for ds in ORDER:
            cv.append(dict(dataset=ds, model=label, hojas_cv=hojas.get(ds, np.nan), prof_cv=prof.get(ds, np.nan)))
    out = full.merge(pd.DataFrame(cv), on=["dataset", "model"], how="left")
    out.to_csv(RES / "arboles_completos.csv", index=False)
    print("  guardado", RES / "arboles_completos.csv")
    figure_cart_bcw()


def figure_cart_bcw():
    d = D.bcw()
    X, y = _impute(d["X"]), d["y"]
    m = PrunedTree().fit(X, y)
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    arts = plot_tree(m, feature_names=d["feature_names"], class_names=["benigno", "maligno"], filled=False,
                     impurity=False, proportion=False, rounded=True, fontsize=6, ax=ax)
    leaves = m.tree_.children_left == -1
    for a in list(arts) + list(ax.texts):   # rótulos en español
        t = a.get_text().replace("samples", "muestras").replace("value", "[ben., mal.]").replace("class", "clase")
        a.set_text({"True": "sí", "False": "no"}.get(t.strip(), t))
    boxes = [a for a in arts if a.get_bbox_patch() is not None]
    for i, a in enumerate(boxes):
        v = m.tree_.value[i][0]
        frac = v[1] / v.sum()
        col = "#F2D9E0" if frac > 0.5 else "#DCE3EE"
        a.get_bbox_patch().set_facecolor(col if leaves[i] else "#FFFFFF")
        a.get_bbox_patch().set_edgecolor("#9D2449" if frac > 0.5 else "#1B396A")
    ax.set_title(f"CART podado en BCW con todos los datos: {m.get_n_leaves()} hojas, profundidad {m.get_depth()} "
                 f"(α = {m.ccp_alpha:.4f})", fontsize=8)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"fig_arbol_cart_bcw.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  figura", FIG / "fig_arbol_cart_bcw.png")


if __name__ == "__main__":
    import sys
    c50_only() if "c50" in sys.argv[1:] else main()
