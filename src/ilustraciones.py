"""
ilustraciones.py
================
Figuras ilustrativas del marco teórico. Uso: python -m src.ilustraciones <carpeta_salida>

  fig_ilus_arboles      Partición ortogonal, funciones de impureza, efecto escalera, XOR sin ganancia marginal,
                        fusión de categorías de CHAID y reducción de varianza de RF.
  fig_ilus_fronteras13  Fronteras de decisión de los 13 clasificadores sobre el mismo conjunto 2D desbalanceado.
  fig_ilus_metodos      Validación cruzada estratificada vs holdout y separabilidad lineal por programación lineal.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle
from scipy.optimize import linprog
from sklearn.datasets import make_moons
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier

from . import datasets as D
from .chaid import CHAIDClassifier, chi2_pvalue
from .models import FAMILY, MODEL_ORDER, SHORT, final_estimator, make_models, predict_both

AZUL, GUINDA, GRIS, DORADO = "#1B396A", "#9D2449", "#807E82", "#B38E5D"
CM2 = ListedColormap(["#DCE3EE", "#F2D9E0"])
plt.rcParams.update({"font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "xtick.labelsize": 7,
                     "ytick.labelsize": 7, "legend.fontsize": 6.5, "axes.spines.top": False,
                     "axes.spines.right": False, "savefig.dpi": 300})


def _save(fig, out, name):
    for ext in ("pdf", "png"):
        fig.savefig(out / f"{name}.{ext}", bbox_inches="tight")
    plt.close(fig)
    print("  ", name)


def _scatter(ax, X, y, s=7):
    ax.scatter(X[y == 0, 0], X[y == 0, 1], s=s, c=AZUL, lw=0, alpha=.75)
    ax.scatter(X[y == 1, 0], X[y == 1, 1], s=s, c=GUINDA, lw=0, alpha=.85)


def _boundary(ax, model, X, n=160, pad=.3):
    x0, x1 = X[:, 0].min() - pad, X[:, 0].max() + pad
    y0, y1 = X[:, 1].min() - pad, X[:, 1].max() + pad
    xx, yy = np.meshgrid(np.linspace(x0, x1, n), np.linspace(y0, y1, n))
    _, z = predict_both(model, np.c_[xx.ravel(), yy.ravel()])
    z = z.reshape(xx.shape)
    ax.contourf(xx, yy, z, levels=[-.5, .5, 1.5], cmap=CM2)
    ax.contour(xx, yy, z, levels=[.5], colors="k", linewidths=.7)
    ax.set_xlim(x0, x1); ax.set_ylim(y0, y1)


# ----------------------------------------------------------------------------- figura 1: árboles
def fig_arboles(out):
    fig, axs = plt.subplots(2, 3, figsize=(7.2, 4.6))
    # (a) partición ortogonal
    d = D.SYNTH["Ortogonal IF-THEN"]()
    X, y = d["X"], d["y"]
    dt = DecisionTreeClassifier(max_depth=2, random_state=42).fit(X, y)
    ax = axs[0, 0]
    _boundary(ax, dt, X); _scatter(ax, X, y, 4)
    t = dt.tree_
    ax.set_title("(a) Partición ortogonal: 2 cortes, 3 hojas")
    ax.text(.02, .02, f"x1 > {t.threshold[0]:.2f}  y  x2 ≤ {t.threshold[t.children_right[0]]:.2f}",
            transform=ax.transAxes, fontsize=6.5, bbox=dict(fc="white", ec=GRIS, lw=.4))
    ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$")
    # (b) impurezas
    ax = axs[0, 1]
    p = np.linspace(1e-4, 1 - 1e-4, 300)
    H = -(p * np.log2(p) + (1 - p) * np.log2(1 - p))
    ax.plot(p, H / 2, c=AZUL, label="Entropía / 2")
    ax.plot(p, 1 - p ** 2 - (1 - p) ** 2, c=GUINDA, label="Gini")
    ax.plot(p, np.minimum(p, 1 - p), c=DORADO, ls="--", label="Error de clasificación")
    ax.set_title("(b) Impureza de un nodo (2 clases)"); ax.set_xlabel("proporción de la clase 1, $p$")
    ax.legend(frameon=False, loc="lower center")
    # (c) efecto escalera
    d = D.SYNTH["Lineal rotado 45°"]()
    X, y = d["X"], d["y"]
    ax = axs[0, 2]
    dt = DecisionTreeClassifier(random_state=42).fit(X, y)
    _boundary(ax, dt, X, n=300); _scatter(ax, X, y, 3)
    g = np.linspace(-4, 4, 2)
    ax.plot(g, g, c=DORADO, lw=1.2, ls="--")
    ax.set_title(f"(c) Efecto escalera: {dt.get_n_leaves()} hojas a 45°"); ax.set_xlabel("$x_1$")
    # (d) XOR: ganancia marginal nula
    d = D.SYNTH["XOR 0°"]()
    X, y = d["X"], d["y"]
    ax = axs[1, 0]

    def ig(mask):
        def h(v):
            q = v.mean() if len(v) else 0
            return 0 if q in (0, 1) else -(q * np.log2(q) + (1 - q) * np.log2(1 - q))
        n = len(y)
        return h(y) - mask.sum() / n * h(y[mask]) - (~mask).sum() / n * h(y[~mask])
    ts = np.linspace(-2.8, 2.8, 57)
    ax.plot(ts, [ig(X[:, 0] <= t) for t in ts], c=AZUL, label="cortes en $x_1$")
    ax.plot(ts, [ig(X[:, 1] <= t) for t in ts], c=GUINDA, label="cortes en $x_2$")
    ax.set_title("(d) XOR 0°: ganancia del corte raíz"); ax.set_xlabel("umbral $t$")
    ax.set_ylabel("ganancia de información (bits)"); ax.set_ylim(0, .06); ax.legend(frameon=False)
    ax.text(.03, .80, "ningún corte informa por sí solo:\nC5.0 y CHAID no dividen la raíz",
            transform=ax.transAxes, fontsize=6.3)
    # (e) CHAID: fusión de categorías (Iris, largo del pétalo discretizado)
    ax = axs[1, 1]
    di = D.REAL["Iris"]()
    Xi, yi = di["X"], di["y"]
    ch = CHAIDClassifier(max_depth=1).fit(Xi, yi)
    j = ch.tree_["feature"]
    codes = ch._codes(Xi)[:, j]
    nc = ch.n_codes_[j]
    T = np.array([[np.sum((codes == c) & (yi == k)) for k in range(3)] for c in range(nc)])
    bottom = np.zeros(nc)
    cols = [AZUL, DORADO, GUINDA]
    for k in range(3):
        ax.bar(range(nc), T[:, k], bottom=bottom, color=cols[k], width=.8, label=f"clase {k}")
        bottom += T[:, k]
    cmap = ch.tree_["cmap"]
    for g in np.unique(cmap):
        idx = np.flatnonzero(cmap == g)
        ax.annotate("", xy=(idx.min() - .4, bottom.max() + 2), xytext=(idx.max() + .4, bottom.max() + 2),
                    arrowprops=dict(arrowstyle="-", color="k", lw=.8))
        ax.text(idx.mean(), bottom.max() + 3, f"G{g + 1}", ha="center", fontsize=6.3)
    ax.set_ylim(0, bottom.max() + 9)
    ax.set_title("(e) CHAID: fusión de intervalos")
    ax.set_xlabel(f"intervalo de {di['feature_names'][j].replace(' (cm)', '')} (cuantiles)"); ax.set_ylabel("muestras")
    ax.legend(frameon=False, loc="center right", fontsize=5.8)
    # (f) RF: varianza del promedio
    ax = axs[1, 2]
    B = np.arange(1, 201)
    for rho, c in ((0.0, AZUL), (0.2, DORADO), (0.5, GUINDA)):
        ax.plot(B, rho + (1 - rho) / B, c=c, label=fr"$\rho$ = {rho}")
    ax.set_xscale("log"); ax.set_ylim(0, 1.05)
    ax.set_title(r"(f) RF: Var$(\bar f)/\sigma^2=\rho+(1-\rho)/B$"); ax.set_xlabel("número de árboles $B$")
    ax.legend(frameon=False)
    fig.tight_layout()
    _save(fig, out, "fig_ilus_arboles")


# ----------------------------------------------------------------------------- figura 2: 13 fronteras
def fig_fronteras13(out):
    X, y = make_moons(600, noise=.25, random_state=7)
    rng = np.random.default_rng(3)
    keep = (y == 0) | (rng.random(len(y)) < .25)          # clase 1 minoritaria (~20 %)
    X, y = X[keep], y[keep]
    fac = make_models(n_jobs=1)
    fig, axs = plt.subplots(3, 5, figsize=(7.2, 4.5))
    axs = axs.ravel()
    for i, name in enumerate(MODEL_ORDER):
        ax = axs[i]
        m = fac[name]()
        if name == "CM-KNN":
            final_estimator(m).set_params(n_max=200)
        m.fit(X, y)
        _boundary(ax, m, X, n=60 if name == "CM-KNN" else 140)
        _scatter(ax, X, y, 2.5)
        _, yh = predict_both(m, X)
        tpr = [(yh[y == k] == k).mean() for k in (0, 1)]
        ax.set_title(f"{SHORT[name]}  (BA entr. {np.mean(tpr):.2f})", fontsize=7,
                     color={"Árboles": "#5F5E63", "Ensambles": AZUL, "Mínima Distancia": DORADO,
                            "Variantes de KNN": GUINDA}[FAMILY[name]])
        ax.set_xticks([]); ax.set_yticks([])
        print("     ", name)
    for ax in axs[13:]:
        ax.axis("off")
    axs[13].text(0, .7, "Moons con la clase 1 (guinda)\nreducida al 25 %:\n"
                 f"{(y == 0).sum()} vs {(y == 1).sum()} muestras", fontsize=7)
    axs[13].text(0, .15, "Árboles: cortes ortogonales\nEnsambles: fronteras suavizadas\n"
                 "Mínima Distancia: recta (centroides)\nVariantes de KNN: fronteras locales;\n"
                 "CS-KNN favorecen la minoritaria", fontsize=6.5, color=GRIS)
    fig.tight_layout()
    _save(fig, out, "fig_ilus_fronteras13")


# ----------------------------------------------------------------------------- figura 3: métodos
def fig_metodos(out):
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.4), gridspec_kw=dict(width_ratios=[1.45, 1, 1]))
    # (a) esquema: 10-fold estratificado vs holdout
    ax = axs[0]
    rng = np.random.default_rng(0)
    y = np.r_[np.zeros(40, int), np.ones(10, int)]
    skf = StratifiedKFold(10, shuffle=True, random_state=0)
    order = np.argsort(y, kind="stable")
    for f, (_, te) in enumerate(skf.split(np.zeros(len(y)), y)):
        for i in range(len(y)):
            col = GUINDA if order[i] in te else "#DCE3EE"
            ax.add_patch(Rectangle((i, 9 - f), .9, .8, color=col, lw=0))
    for i in range(len(y)):
        ax.add_patch(Rectangle((i, 10.4), .9, .8, color=AZUL if y[order][i] == 0 else DORADO, lw=0))
    hold = np.array([3, 11, 19, 27, 36])          # un holdout al azar puede no tomar ninguna minoritaria
    for i in range(len(y)):
        ax.add_patch(Rectangle((i, -1.6), .9, .8, color=GUINDA if i in hold else "#DCE3EE", lw=0))
    ax.set_xlim(-1, 51); ax.set_ylim(-3.2, 11.6); ax.axis("off")
    ax.text(25, 11.35, "clases (azul: mayoritaria, dorado: minoritaria)", ha="center", fontsize=6)
    ax.text(-1, 4.6, "10 pliegues", rotation=90, ha="right", va="center", fontsize=6.5)
    ax.text(-1, -1.2, "holdout", ha="right", va="center", fontsize=6.5)
    ax.text(25, -2.6, "cada pliegue: 4 mayoritarias + 1 minoritaria; este holdout: 0 minoritarias",
            ha="center", fontsize=6)
    ax.set_title("(a) 10-fold estratificado vs. holdout 90/10")
    # (b) separable linealmente (factible)
    for k, (ax, sep) in enumerate(((axs[1], True), (axs[2], False))):
        r = np.random.default_rng(5 + k)
        if sep:
            A = r.normal([-1.2, -1], .45, (40, 2)); Bm = r.normal([1.2, 1], .45, (40, 2))
        else:
            A = r.normal([-.4, -.3], .7, (40, 2)); Bm = r.normal([.4, .3], .7, (40, 2))
        Xs = np.r_[A, Bm]; ys = np.r_[-np.ones(40), np.ones(40)]
        # LP de factibilidad: y_i (w.x_i + b) >= 1
        Aub = -ys[:, None] * np.c_[Xs, np.ones(len(Xs))]
        res = linprog(np.zeros(3), A_ub=Aub, b_ub=-np.ones(len(Xs)), bounds=[(None, None)] * 3, method="highs")
        ax.scatter(A[:, 0], A[:, 1], s=6, c=AZUL, lw=0); ax.scatter(Bm[:, 0], Bm[:, 1], s=6, c=GUINDA, lw=0)
        if res.status == 0:
            w1, w2, b = res.x
            gx, gy = np.meshgrid(np.linspace(-3, 3, 200), np.linspace(-3, 3, 200))
            ax.contour(gx, gy, w1 * gx + w2 * gy + b, levels=[0], colors="k", linewidths=.8)
            ax.set_title("(b) Programa lineal factible:\nlinealmente separable")
        else:
            ax.set_title("(c) Programa lineal infactible:\nno separable")
        ax.set_xlim(-3, 3); ax.set_ylim(-3, 3); ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    _save(fig, out, "fig_ilus_metodos")


if __name__ == "__main__":
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "figures")
    out.mkdir(parents=True, exist_ok=True)
    which = sys.argv[2:] or ["arboles", "metodos", "fronteras13"]
    for w in which:
        globals()["fig_" + w](out)
