"""
sobreajuste.py
==============
E12. Condiciones que favorecen el sobreajuste: pocos datos, desbalance, instancias atípicas y datos faltantes.

Sobre BCW (Original), con CV estratificada de 10 pliegues (semilla 42), se degrada una condición a la vez y se
mide la brecha de exactitud balanceada (BA de entrenamiento - BA de prueba) y la BA de prueba de DT (entropía,
sin poda), CART (poda por costo-complejidad), Random Forest y Gradient Boosting.

  pocos datos   se entrena con una submuestra estratificada de n muestras del pliegue de entrenamiento.
  desbalance    se eliminan malignos del entrenamiento hasta que representen el porcentaje indicado.
  atípicas      una fracción de las muestras de entrenamiento recibe la etiqueta contraria (instancias
                inconsistentes con su región); la prueba no se altera.
  faltantes     una fracción de celdas (entrenamiento y prueba) se borra al azar y se imputa con la mediana
                del entrenamiento.

Uso:  python -m src.sobreajuste
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier

from . import datasets as D
from .experiments import RES, PrunedTree, RS
from .visualization import FIG

MODELS = {"DT (sin poda)": DecisionTreeClassifier(criterion="entropy", random_state=RS),
          "CART (podado)": PrunedTree(),
          "Random Forest": RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=RS),
          "Gradient Boosting": GradientBoostingClassifier(random_state=RS)}
LEVELS = {"pocos datos": [25, 50, 100, 200, 400, None],
          "desbalance": [None, 0.20, 0.10, 0.05, 0.02],
          "atípicas": [0.0, 0.05, 0.10, 0.20, 0.30],
          "faltantes": [0.0, 0.10, 0.20, 0.30, 0.40, 0.50]}


def _impute(Xtr, Xte):
    med = np.nanmedian(Xtr, axis=0)
    med = np.where(np.isnan(med), 0.0, med)
    out = []
    for X in (Xtr, Xte):
        X = X.copy()
        r, c = np.where(np.isnan(X))
        X[r, c] = med[c]
        out.append(X)
    return out


def _subsample(y, n, rng):
    """Índices de una submuestra estratificada de tamaño n (al menos 2 por clase)."""
    idx = []
    for c in np.unique(y):
        pos = np.where(y == c)[0]
        k = max(2, int(round(n * len(pos) / len(y))))
        idx.append(rng.choice(pos, min(k, len(pos)), replace=False))
    return np.sort(np.concatenate(idx))


def _degrade(cond, level, Xtr, ytr, Xte, rng):
    if cond == "pocos datos" and level is not None:
        keep = _subsample(ytr, level, rng)
        Xtr, ytr = Xtr[keep], ytr[keep]
    elif cond == "desbalance" and level is not None:
        neg, pos = np.where(ytr == 0)[0], np.where(ytr == 1)[0]
        n_pos = max(2, int(round(level * len(neg) / (1 - level))))
        keep = np.sort(np.concatenate([neg, rng.choice(pos, min(n_pos, len(pos)), replace=False)]))
        Xtr, ytr = Xtr[keep], ytr[keep]
    elif cond == "atípicas" and level:
        ytr = ytr.copy()
        flip = rng.choice(len(ytr), int(round(level * len(ytr))), replace=False)
        ytr[flip] = 1 - ytr[flip]
    elif cond == "faltantes" and level:
        Xtr, Xte = Xtr.copy(), Xte.copy()
        Xtr[rng.random(Xtr.shape) < level] = np.nan
        Xte[rng.random(Xte.shape) < level] = np.nan
    return Xtr, ytr, Xte


def main():
    d = D.bcw()
    X, y = d["X"], d["y"]
    cv = StratifiedKFold(10, shuffle=True, random_state=RS)
    rows = []
    for cond, levels in LEVELS.items():
        for level in levels:
            for fold, (tr, te) in enumerate(cv.split(X, y)):
                rng = np.random.default_rng(RS + fold)
                Xtr, ytr, Xte = _degrade(cond, level, X[tr], y[tr], X[te], rng)
                Xtr, Xte = _impute(Xtr, Xte)
                for mname, base in MODELS.items():
                    m = clone(base).fit(Xtr, ytr)
                    ba_tr = balanced_accuracy_score(ytr, m.predict(Xtr))
                    ba_te = balanced_accuracy_score(y[te], m.predict(Xte))
                    rows.append(dict(condicion=cond, nivel=level, fold=fold, model=mname, n_train=len(ytr),
                                     ba_train=ba_tr, ba_test=ba_te, brecha=ba_tr - ba_te))
            s = pd.DataFrame([r for r in rows if r["condicion"] == cond and r["nivel"] == level])
            print(f"  {cond} = {level}: " + ", ".join(f"{k} {v:.3f}" for k, v in
                                                     s.groupby("model").brecha.mean().items()), flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(RES / "sobreajuste_condiciones.csv", index=False)
    figure(df)


def figure(df=None):
    if df is None:
        df = pd.read_csv(RES / "sobreajuste_condiciones.csv")
    df = df.copy()
    df["x"] = df.nivel
    df.loc[(df.condicion == "pocos datos") & df.nivel.isna(), "x"] = 629
    df.loc[(df.condicion == "desbalance") & df.nivel.isna(), "x"] = 0.345
    style = {"DT (sin poda)": ("#807E82", "-", "o"), "CART (podado)": ("#5F5E63", "--", "s"),
             "Random Forest": ("#1B396A", "-", "o"), "Gradient Boosting": ("#9D2449", "-", "^")}
    titles = {"pocos datos": "Pocos datos", "desbalance": "Desbalance", "atípicas": "Instancias atípicas",
              "faltantes": "Datos faltantes"}
    xlabels = {"pocos datos": "muestras de entrenamiento", "desbalance": "fracción de malignos (orig. 0.345)",
               "atípicas": "fracción con etiqueta contraria", "faltantes": "fracción de celdas faltantes"}
    plt.rcParams.update({"font.size": 7, "axes.spines.top": False, "axes.spines.right": False})
    fig, axs = plt.subplots(2, 4, figsize=(7.2, 3.9), sharey="row")
    for j, cond in enumerate(titles):
        s = df[df.condicion == cond].groupby(["model", "x"])[["brecha", "ba_test"]].mean().reset_index()
        for mname, (c, ls, mk) in style.items():
            t = s[s.model == mname].sort_values("x")
            axs[0, j].plot(t.x, t.brecha, color=c, ls=ls, marker=mk, ms=3, lw=1.2, label=mname)
            axs[1, j].plot(t.x, t.ba_test, color=c, ls=ls, marker=mk, ms=3, lw=1.2)
        axs[0, j].set_title(titles[cond], fontsize=8)
        axs[1, j].set_xlabel(xlabels[cond], fontsize=6.5)
        if cond == "pocos datos":
            for a in axs[:, j]:
                a.set_xscale("log")
                a.set_xticks([25, 50, 100, 200, 629])
                a.set_xticklabels(["25", "50", "100", "200", "629"])
                a.minorticks_off()
        if cond == "desbalance":
            for a in axs[:, j]:
                a.set_xscale("log")
                a.set_xticks([0.345, 0.2, 0.1, 0.05, 0.02])
                a.set_xticklabels(["orig.", "0.20", "0.10", "0.05", "0.02"])
                a.minorticks_off()
                a.invert_xaxis()
        for a in axs[:, j]:
            a.grid(lw=0.3, alpha=0.5)
        axs[0, j].axhline(0, color="#5F5E63", lw=0.6, ls=":")
    axs[0, 0].set_ylabel("brecha de BA\n(entrenamiento − prueba)")
    axs[1, 0].set_ylabel("BA de prueba")
    h, l = axs[0, 0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, fontsize=7, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.03))
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"fig_sobreajuste_condiciones.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  figura", FIG / "fig_sobreajuste_condiciones.png")


if __name__ == "__main__":
    main()
