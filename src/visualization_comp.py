"""
visualization_comp.py
=====================
Figuras del estudio comparativo (paleta del Manual de Identidad Gráfica del TecNM). Uso: python -m src.visualization_comp

  fig_comp_mapa_ba        Exactitud balanceada media: conjuntos x 13 clasificadores.
  fig_comp_rangos         Rango promedio de Friedman con significancia contra el mejor modelo (Holm).
  fig_comp_sobreajuste    Brecha entrenamiento - prueba por clasificador (distribución sobre los conjuntos).
  fig_comp_validacion     Variabilidad de la estimación según el esquema de validación (holdout vs k-fold).
  fig_comp_tiempos        Tiempo de entrenamiento y de predicción por clasificador (escala logarítmica).
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

from .experiments import RES
from .models import FAMILY, MODEL_ORDER, SHORT
from .visualization import AZUL, DORADO, GRIS, GUINDA, FIG as _FIG

import os
FIG = (RES / "figuras") if os.environ.get("ABP_RESULTS_DIR") else _FIG   # prueba rápida: no toca figures/

FAM_COL = {"Árboles": GRIS, "Ensambles": AZUL, "Mínima Distancia": DORADO, "Variantes de KNN": GUINDA}
plt.rcParams.update({"font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8, "xtick.labelsize": 7,
                     "ytick.labelsize": 7, "legend.fontsize": 7, "axes.spines.top": False,
                     "axes.spines.right": False, "savefig.dpi": 300})


def _save(fig, name):
    FIG.mkdir(exist_ok=True)
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(FIG / f"{name}.png", bbox_inches="tight")
    plt.close(fig)


def _ok():
    df = pd.read_csv(RES / "benchmark_comp.csv")
    return df[df.estado.isin(["ok", "importado del estudio base"])]


def heatmap():
    df = _ok()
    M = df.groupby(["dataset", "model"]).bal_acc.mean().unstack()
    M = M[[m for m in MODEL_ORDER if m in M.columns]]
    order = [d for d in ["Iris", "BCW (Original)", "Wine", "Glass", "Concrete", "Forest Fires", "Credit Card", "HAR",
                         "Ortogonal IF-THEN", "XOR 0°", "Lineal rotado 45°", "XOR rotado 45°", "Circles", "Moons"]
             if d in M.index]
    M = M.loc[order]
    cmap = LinearSegmentedColormap.from_list("t", ["#F4F4F6", "#C9D3E3", AZUL])
    fig, ax = plt.subplots(figsize=(7.16, 0.32 * len(M) + 1.4))
    im = ax.imshow(M.values, cmap=cmap, vmin=max(0.4, np.nanmin(M.values)), vmax=1.0, aspect="auto")
    for i in range(M.shape[0]):
        best = np.nanmax(M.values[i])
        for j in range(M.shape[1]):
            v = M.values[i, j]
            if np.isnan(v):
                ax.text(j, i, "—", ha="center", va="center", fontsize=6, color=GRIS); continue
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=5.8,
                    color="white" if v > 0.8 else "black", fontweight="bold" if abs(v - best) < 5e-4 else "normal")
    ax.set_xticks(range(M.shape[1])); ax.set_xticklabels([SHORT[m] for m in M.columns], rotation=45, ha="right")
    ax.set_yticks(range(M.shape[0])); ax.set_yticklabels(M.index)
    for lab, m in zip(ax.get_xticklabels(), M.columns):
        lab.set_color(FAM_COL[FAMILY[m]])
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01, label="Exactitud balanceada")
    ax.set_title("Exactitud balanceada media por conjunto (negritas: mejor del conjunto; —: no aplica)")
    _save(fig, "fig_comp_mapa_ba")


def ranks():
    rk = pd.read_csv(RES / "comp_rangos.csv")
    hc = pd.read_csv(RES / "comp_vs_control_holm.csv")
    fr = pd.read_csv(RES / "comp_friedman.csv")
    label = fr.analisis.iloc[0]
    r = rk[rk.analisis == label].sort_values("rango_promedio")
    h = hc[hc.analisis == label].set_index("model")
    f = fr[fr.analisis == label].iloc[0]
    fig, ax = plt.subplots(figsize=(3.5, 0.22 * len(r) + 0.9))
    y = np.arange(len(r))[::-1]
    for yi, (m, v) in zip(y, zip(r.model, r.rango_promedio)):
        sig = m in h.index and bool(h.loc[m, "significativo_vs_control"])
        ax.barh(yi, v, color=FAM_COL[FAMILY[m]], alpha=1.0 if sig or m == f.control else 0.45)
        ax.text(v + 0.1, yi, f"{v:.2f}" + (" *" if sig else ("  (mejor)" if m == f.control else "")),
                va="center", fontsize=6.5)
    ax.set_yticks(y); ax.set_yticklabels([SHORT[m] for m in r.model])
    ax.set_xlabel("Rango promedio (1 = mejor)")
    ax.set_title(f"Friedman p = {f.p_friedman:.1e} | N = {int(f.N)} conjuntos | CD Nemenyi = {f.CD_nemenyi:.2f}",
                 fontsize=7.5)
    ax.text(0.99, 0.01, "* peor que el mejor (Holm, α = 0.05)", transform=ax.transAxes, ha="right", va="bottom",
            fontsize=6, color=GRIS)
    _save(fig, "fig_comp_rangos")


def overfitting():
    df = _ok()
    g = df.groupby(["dataset", "model"]).gap.mean().unstack()
    models = [m for m in MODEL_ORDER if m in g.columns]
    fig, ax = plt.subplots(figsize=(7.16, 2.3))
    data = [g[m].dropna().values for m in models]
    bp = ax.boxplot(data, widths=0.6, patch_artist=True, medianprops=dict(color="k"))
    for patch, m in zip(bp["boxes"], models):
        patch.set_facecolor(FAM_COL[FAMILY[m]]); patch.set_alpha(0.55)
    ax.axhline(0, color=GRIS, lw=0.6, ls="--")
    ax.set_xticks(range(1, len(models) + 1)); ax.set_xticklabels([SHORT[m] for m in models], rotation=35, ha="right")
    ax.set_ylabel("Brecha entrenamiento − prueba")
    ax.set_title("Sobreajuste por clasificador (cada punto de la caja es un conjunto de datos)")
    _save(fig, "fig_comp_sobreajuste")


def validation():
    s = pd.read_csv(RES / "comp_cv_resumen.csv")
    schemes = ["Holdout 90/10", "5-fold estratificado", "10-fold sin estratificar", "10-fold estratificado"]
    schemes = [x for x in schemes if x in s.scheme.unique()]
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.4))
    data = [s[s.scheme == x].variabilidad_std.dropna().values for x in schemes]
    bp = axes[0].boxplot(data, widths=0.55, patch_artist=True, medianprops=dict(color="k"))
    for patch, c in zip(bp["boxes"], [GUINDA, DORADO, GRIS, AZUL]):
        patch.set_facecolor(c); patch.set_alpha(0.6)
    axes[0].set_xticks(range(1, len(schemes) + 1)); axes[0].set_xticklabels(schemes, rotation=20, ha="right")
    axes[0].set_ylabel("Desv. est. de la estimación\nentre repeticiones")
    axes[0].set_title("Estabilidad de la estimación de la BA")
    miss = s.groupby("scheme").pliegues_sin_alguna_clase.sum().reindex(schemes).fillna(0)
    axes[1].bar(range(len(schemes)), miss.values, color=[GUINDA, DORADO, GRIS, AZUL][:len(schemes)])
    axes[1].set_xticks(range(len(schemes))); axes[1].set_xticklabels(schemes, rotation=20, ha="right")
    axes[1].set_ylabel("Pliegues de prueba sin\nalguna clase (total)")
    axes[1].set_title("Pliegues a los que les falta una clase")
    fig.tight_layout()
    _save(fig, "fig_comp_validacion")


def timing():
    t = pd.read_csv(RES / "comp_tiempos_resumen.csv")
    models = [m for m in MODEL_ORDER if m in t.model.unique()]
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.5), sharey=True)
    for ax, col, title in [(axes[0], "fit_s", "Entrenamiento (s)"), (axes[1], "pred_ms_muestra", "Predicción (ms por muestra)")]:
        for i, m in enumerate(models):
            v = t[t.model == m][col].dropna().values
            ax.scatter(np.clip(v, 1e-5, None), np.full(len(v), i) + np.random.default_rng(i).uniform(-0.15, 0.15, len(v)),
                       s=8, color=FAM_COL[FAMILY[m]], alpha=0.7, lw=0)
            if len(v):
                ax.plot([np.median(v)] * 2, [i - 0.3, i + 0.3], color="k", lw=1)
        ax.set_xscale("log"); ax.set_title(title); ax.grid(axis="x", lw=0.3, alpha=0.5)
    axes[0].set_yticks(range(len(models))); axes[0].set_yticklabels([SHORT[m] for m in models]); axes[0].invert_yaxis()
    fig.suptitle("Costo computacional (un punto por conjunto de datos; línea = mediana; 1 hilo, mismo equipo)", fontsize=8)
    fig.tight_layout()
    _save(fig, "fig_comp_tiempos")


def main():
    for f, need in [(heatmap, "benchmark_comp.csv"), (overfitting, "benchmark_comp.csv"), (ranks, "comp_rangos.csv"),
                    (validation, "comp_cv_resumen.csv"), (timing, "comp_tiempos_resumen.csv")]:
        if (RES / need).exists():
            f()
        else:
            print(f"  (omitida {f.__name__}: falta results/{need})")
    print("  figuras en", FIG)


if __name__ == "__main__":
    main()
