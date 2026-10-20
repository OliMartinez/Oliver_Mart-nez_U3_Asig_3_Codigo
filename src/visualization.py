"""
visualization.py
================
Figuras del ABP con la paleta del Manual de Identidad Gráfica del TecNM:
azul Pantone 294 C (#1B396A), guinda (#9D2449), gris Cool Gray 10 C (#807E82), dorado (#B38E5D).

Uso:  python -m src.visualization
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from . import datasets as D
from .experiments import RES, RS, PrunedTree

AZUL, GUINDA, GRIS, DORADO = "#1B396A", "#9D2449", "#807E82", "#B38E5D"
AZUL_CL, GUINDA_CL = "#C9D3E3", "#EBC6D2"
COL = {"Árbol de Decisión": GRIS, "Árbol podado (ccp)": DORADO, "Random Forest": AZUL, "Gradient Boosting": GUINDA}
SHORT = {"Árbol de Decisión": "DT", "Árbol podado (ccp)": "DT podado", "Random Forest": "RF", "Gradient Boosting": "GB"}
FIG = D.ROOT / "figures"
plt.rcParams.update({"font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "legend.fontsize": 7,
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.spines.top": False,
                     "axes.spines.right": False, "savefig.dpi": 300})
CMAP = LinearSegmentedColormap.from_list("tecnm", [AZUL_CL, "#F4F4F6", GUINDA_CL])


def _save(fig, name):
    FIG.mkdir(exist_ok=True)
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(FIG / f"{name}.png", bbox_inches="tight")
    plt.close(fig)


def decision_maps(grid=250):
    """Efecto escalera: DT, DT podado, RF y GB sobre escenarios ortogonales y no ortogonales."""
    sets = [("Ortogonal IF-THEN", D.orthogonal()), ("XOR 0°", D.rotated_xor(0)),
            ("Lineal rotado 45°", D.rotated_linear(45)), ("XOR rotado 45°", D.rotated_xor(45)),
            ("Circles", D.circles()), ("Moons", D.moons())]
    mods = [("Árbol de Decisión", lambda: DecisionTreeClassifier(random_state=RS)),
            ("Árbol podado (ccp)", lambda: PrunedTree()),
            ("Random Forest", lambda: RandomForestClassifier(200, random_state=RS, n_jobs=-1)),
            ("Gradient Boosting", lambda: GradientBoostingClassifier(random_state=RS))]
    fig, axes = plt.subplots(len(sets), len(mods), figsize=(7.16, 10.2))
    for i, (sname, d) in enumerate(sets):
        X, y = d["X"], d["y"]
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=RS)
        lo, hi = X.min(0) - 0.3, X.max(0) + 0.3
        gx, gy = np.meshgrid(np.linspace(lo[0], hi[0], grid), np.linspace(lo[1], hi[1], grid))
        G = np.column_stack([gx.ravel(), gy.ravel()])
        for j, (mname, mk) in enumerate(mods):
            ax = axes[i, j]
            m = mk().fit(Xtr, ytr)
            P = m.predict_proba(G)[:, 1].reshape(gx.shape)
            ax.contourf(gx, gy, P, levels=np.linspace(0, 1, 11), cmap=CMAP)
            ax.contour(gx, gy, P, levels=[0.5], colors="k", linewidths=0.6)
            ax.scatter(Xte[yte == 0, 0], Xte[yte == 0, 1], s=2.5, c=AZUL, lw=0)
            ax.scatter(Xte[yte == 1, 0], Xte[yte == 1, 1], s=2.5, c=GUINDA, lw=0)
            info = f"acc={m.score(Xte, yte):.3f}"
            if isinstance(m, DecisionTreeClassifier):
                info += f"\nprof={m.get_depth()}, hojas={m.get_n_leaves()}"
            ax.text(0.02, 0.02, info, transform=ax.transAxes, fontsize=5.8,
                    bbox=dict(fc="white", ec="none", alpha=0.75, pad=1))
            ax.set_xticks([]); ax.set_yticks([])
            if i == 0:
                ax.set_title(mname)
            if j == 0:
                ax.set_ylabel(sname)
    fig.tight_layout(h_pad=0.3, w_pad=0.3)
    _save(fig, "fig_fronteras")


def rotation_plot():
    df = pd.read_csv(RES / "rotation_sweep.csv")
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.2))
    for kind, ls in [("Lineal", "-"), ("XOR", "--")]:
        d = df[(df.kind == kind) & (df.model == "DT sin poda")].groupby("theta")[["depth", "leaves"]].mean()
        axes[0].plot(d.index, d.depth, ls, marker="o", ms=3, color=GRIS, label=f"{kind}: profundidad")
        axes[0].plot(d.index, d.leaves, ls, marker="s", ms=3, color=GUINDA, label=f"{kind}: hojas")
    axes[0].set_yscale("log"); axes[0].set_title("Complejidad del árbol sin poda")
    axes[0].set_xlabel("Ángulo de rotación (°)"); axes[0].legend(frameon=False, fontsize=5.8)
    for ax, kind in zip(axes[1:], ["Lineal", "XOR"]):
        d = df[df.kind == kind].groupby(["model", "theta"]).test_acc.mean().unstack(0)
        for mname in d.columns:
            col = {"DT sin poda": GRIS, "DT profundidad 4": DORADO, "Random Forest": AZUL,
                   "Gradient Boosting": GUINDA}.get(mname, "k")
            ax.plot(d.index, d[mname], marker="o", ms=2.5, color=col, label=mname,
                    ls=":" if "profundidad 1" in mname or "profundidad 2" in mname else "-")
        ax.set_title(f"{kind}: exactitud de prueba"); ax.set_xlabel("Ángulo de rotación (°)")
        ax.grid(lw=0.3, alpha=0.5)
    axes[1].legend(frameon=False, fontsize=5.8, loc="lower left")
    axes[2].legend(frameon=False, fontsize=5.8, loc="lower right")
    fig.tight_layout()
    _save(fig, "fig_rotacion")


def overfitting_plots():
    ds = pd.read_csv(RES / "depth_sweep.csv")
    cp = pd.read_csv(RES / "ccp_path.csv")
    ch = pd.read_csv(RES / "ccp_chosen.csv").set_index("dataset")
    names = ds.dataset.unique()
    fig, axes = plt.subplots(2, len(names), figsize=(7.16, 3.9))
    for j, n in enumerate(names):
        d = ds[ds.dataset == n]
        ax = axes[0, j]
        ax.plot(d.max_depth, d.train_acc, "-o", ms=2.5, color=GUINDA, label="Entrenamiento")
        ax.plot(d.max_depth, d.cv_acc, "-s", ms=2.5, color=AZUL, label="Validación (10-fold)")
        ax.fill_between(d.max_depth, d.cv_acc - d.cv_std, d.cv_acc + d.cv_std, color=AZUL, alpha=0.12)
        best = d.loc[d.cv_acc.idxmax()]
        ax.axvline(best.max_depth, color=DORADO, lw=1)
        ax.set_title(n); ax.set_xlabel("max_depth"); ax.grid(lw=0.3, alpha=0.5)
        c = cp[cp.dataset == n]
        ax = axes[1, j]
        ax.plot(c.alpha, c.train_acc, "-", drawstyle="steps-post", color=GUINDA, label="Entrenamiento")
        ax.plot(c.alpha, c.test_acc, "-", drawstyle="steps-post", color=AZUL, label="Prueba")
        ax.axvline(ch.loc[n, "alpha_cv"], color=DORADO, lw=1, label=r"$\alpha$ por CV")
        ax.set_xscale("symlog", linthresh=1e-4); ax.set_xlabel(r"ccp_alpha ($\alpha$)")
        ax2 = ax.twinx(); ax2.plot(c.alpha, c.leaves, color=GRIS, lw=0.8, ls=":"); ax2.set_yscale("log")
        ax2.spines["top"].set_visible(False); ax2.tick_params(labelsize=6)
        if j == len(names) - 1:
            ax2.set_ylabel("hojas (gris)", fontsize=7)
        ax.grid(lw=0.3, alpha=0.5)
    axes[0, 0].set_ylabel("Exactitud"); axes[1, 0].set_ylabel("Exactitud")
    axes[0, 0].legend(frameon=False); axes[1, 0].legend(frameon=False, loc="lower left")
    fig.tight_layout()
    _save(fig, "fig_sobreajuste_poda")


def benchmark_plot():
    df = pd.read_csv(RES / "benchmark_all.csv")
    order = ["Iris", "BCW (Original)", "Wine", "Glass", "Concrete", "Forest Fires", "Credit Card", "HAR",
             "Ortogonal IF-THEN", "XOR 0°", "Lineal rotado 45°", "XOR rotado 45°", "Circles", "Moons"]
    order = [o for o in order if o in df.dataset.unique()]
    mods = list(COL)
    g = df.groupby(["dataset", "model"])[["bal_acc", "gap"]].mean()
    fig, axes = plt.subplots(2, 1, figsize=(7.16, 3.9), sharex=True)
    x = np.arange(len(order)); w = 0.2
    for k, m in enumerate(mods):
        v = [g.loc[(o, m)] if (o, m) in g.index else pd.Series({"bal_acc": np.nan, "gap": np.nan}) for o in order]
        axes[0].bar(x + (k - 1.5) * w, [r["bal_acc"] for r in v], w, color=COL[m], label=SHORT[m])
        axes[1].bar(x + (k - 1.5) * w, [r["gap"] for r in v], w, color=COL[m])
    axes[0].set_ylabel("Exactitud balanceada"); axes[0].set_ylim(0.4, 1.02)
    axes[1].set_ylabel("Brecha entren. − prueba")
    for ax in axes:
        ax.axvline(7.5, color="k", lw=0.6, ls="--"); ax.grid(axis="y", lw=0.3, alpha=0.5)
    axes[0].text(3.5, 1.0, "Datos reales (UCI)", ha="center", fontsize=7)
    axes[0].text(10.5, 1.0, "Sintéticos", ha="center", fontsize=7)
    axes[1].set_xticks(x); axes[1].set_xticklabels(order, rotation=35, ha="right")
    axes[0].legend(frameon=False, ncol=4, loc="lower center", bbox_to_anchor=(0.5, 1.06))
    fig.tight_layout()
    _save(fig, "fig_benchmark")


def key_questions_plot():
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.3))
    # Correlación: inestabilidad de la raíz vs importancias
    qc = pd.read_csv(RES / "q_correlation.csv").sort_values("RF", ascending=True).tail(8)
    y = np.arange(len(qc))
    axes[0].barh(y - 0.27, qc.DT, 0.27, color=GRIS, label="DT")
    axes[0].barh(y, qc.RF, 0.27, color=AZUL, label="RF")
    axes[0].barh(y + 0.27, qc.freq_raiz, 0.27, color=DORADO, label="Freq. raíz (bootstrap)")
    axes[0].set_yticks(y); axes[0].set_yticklabels([f[:16] for f in qc.feature], fontsize=6)
    axes[0].set_title("Wine: importancia y raíz"); axes[0].legend(frameon=False, fontsize=5.5, loc="lower right")
    # Dimensionalidad
    qd = pd.read_csv(RES / "q_dimensionality.csv")
    for m, col in [("Árbol de Decisión", GRIS), ("Random Forest", AZUL), ("Gradient Boosting", GUINDA),
                   ("KNN (k=5, estandarizado)", DORADO)]:
        d = qd[qd.model == m]
        axes[1].plot(d.p, d.bal_acc, "-o", ms=3, color=col, label=SHORT.get(m, "KNN-5"))
    axes[1].set_xscale("log"); axes[1].set_xlabel("Dimensión (9 reales + ruido)")
    axes[1].set_title("BCW + atributos irrelevantes"); axes[1].legend(frameon=False, fontsize=6)
    axes[1].grid(lw=0.3, alpha=0.5)
    # Desbalance
    qi = pd.read_csv(RES / "benchmark_big.csv")
    qi = qi[qi.dataset == "Credit Card"]
    order = ["Árbol de Decisión", "Árbol de Decisión (bal)", "Árbol podado (ccp)", "Random Forest",
             "Random Forest (bal)", "Gradient Boosting", "Gradient Boosting (bal)"]
    g = qi.groupby("model")[["test_acc", "recall_pos", "pr_auc"]].mean().reindex(order).reset_index()
    lab = [SHORT.get(m.replace(" (bal)", ""), m) + (" bal" if "(bal)" in m else "") for m in g.model]
    trivial = 1 - 492 / 284807  # Credit Card completo
    x = np.arange(len(g))
    axes[2].bar(x - 0.27, g.test_acc, 0.27, color=GRIS, label="Exactitud")
    axes[2].bar(x, g.recall_pos, 0.27, color=GUINDA, label="Recall fraude")
    axes[2].bar(x + 0.27, g.pr_auc, 0.27, color=AZUL, label="PR-AUC")
    axes[2].axhline(trivial, color="k", ls="--", lw=0.7)
    axes[2].text(len(g) - 0.5, trivial, "exactitud trivial", fontsize=5.5, ha="right", va="bottom")
    axes[2].set_xticks(x); axes[2].set_xticklabels(lab, rotation=40, ha="right", fontsize=6)
    axes[2].set_ylim(0.5, 1.02); axes[2].set_title("Credit Card: desbalance")
    axes[2].legend(frameon=False, fontsize=5.5, loc="upper center", bbox_to_anchor=(0.5, -0.32), ncol=3)
    fig.tight_layout()
    _save(fig, "fig_preguntas")


def collect():
    """Une la comparativa: conjuntos pequeños (CV 10) + Credit Card y HAR (benchmark_big, 4 modelos base)."""
    small = pd.read_csv(RES / "benchmark_small.csv")
    big = pd.read_csv(RES / "benchmark_big.csv")
    big = big[big.model.isin(list(COL))]
    pd.concat([small, big], ignore_index=True).to_csv(RES / "benchmark_all.csv", index=False)


def main(parts=("collect", "benchmark", "rotation", "overfit", "questions", "maps")):
    fn = {"collect": collect, "benchmark": benchmark_plot, "rotation": rotation_plot, "overfit": overfitting_plots,
          "questions": key_questions_plot, "maps": decision_maps}
    for p in parts:
        fn[p]()
    print("figuras listas:", parts)


if __name__ == "__main__":
    import sys
    main(tuple(sys.argv[1:]) or ("collect", "benchmark", "rotation", "overfit", "questions", "maps"))


def significance_plot():
    """Diagrama de diferencia crítica (Nemenyi) con los rangos promedio de Friedman."""
    r = pd.read_csv(RES / "sig_ranks.csv").sort_values("avg_rank")
    fr = pd.read_csv(RES / "sig_friedman.csv", index_col=0).iloc[:, 0]
    CD, k = fr["CD"], int(fr["k"])
    fig, ax = plt.subplots(figsize=(3.5, 1.55))
    ax.set_xlim(k + 0.2, 0.8); ax.set_ylim(0, 1.6); ax.axis("off")
    ax.hlines(1.2, 1, k, color="k", lw=0.8)
    for t in range(1, k + 1):
        ax.vlines(t, 1.15, 1.25, color="k", lw=0.8); ax.text(t, 1.32, str(t), ha="center", fontsize=7)
    for i, (m, v) in enumerate(zip(r.model, r.avg_rank)):
        y = 0.95 - 0.22 * i
        ax.plot([v, v], [1.2, y], color=COL[m], lw=1.2); ax.plot(v, 1.2, "o", ms=4, color=COL[m])
        left = v < (k + 1) / 2
        ax.text(v + (-0.08 if left else 0.08), y, f"{SHORT[m]} ({v:.2f})", ha="left" if left else "right",
                va="center", fontsize=7, color=COL[m])
    ax.hlines(1.48, 1, 1 + CD, color=GUINDA, lw=1.5); ax.text(1 + CD / 2, 1.53, f"CD = {CD:.2f}", ha="center", fontsize=6.5, color=GUINDA)
    # barras que unen modelos sin diferencia significativa
    vals = r.avg_rank.to_numpy(); yb = 1.05
    for i in range(len(vals)):
        j = max(jj for jj in range(len(vals)) if vals[jj] - vals[i] <= CD)
        if j > i:
            ax.hlines(yb, vals[i] - 0.03, vals[j] + 0.03, color="k", lw=2.2); yb -= 0.06
    _save(fig, "fig_significancia")


def regularization_plot():
    m = pd.read_csv(RES / "reg_min_samples_leaf.csv")
    g = pd.read_csv(RES / "reg_gb_staged.csv")
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.2))
    for name, col in [("Concrete", GRIS), ("Moons", AZUL), ("BCW (Original)", GUINDA), ("Glass", DORADO)]:
        d = m[m.dataset == name]
        axes[0].plot(d.min_samples_leaf, d.cv_acc, "-o", ms=3, color=col, label=name)
        axes[0].plot(d.min_samples_leaf, d.train_acc, ":", color=col, lw=0.8)
    axes[0].set_xscale("log"); axes[0].set_xlabel("min_samples_leaf"); axes[0].set_ylabel("Exactitud")
    axes[0].set_title("Árbol: hoja mínima (línea punteada = entren.)"); axes[0].legend(frameon=False, fontsize=5.8)
    for ax, name in zip(axes[1:], ["Concrete", "Moons"]):
        d = g[g.dataset == name].groupby(["learning_rate", "n_estimators"])[["train_acc", "test_acc"]].mean().reset_index()
        for lr, col in [(0.01, GRIS), (0.1, AZUL), (0.5, GUINDA)]:
            e = d[d.learning_rate == lr]
            ax.plot(e.n_estimators, e.test_acc, "-o", ms=2.5, color=col, label=f"ν = {lr}")
            ax.plot(e.n_estimators, e.train_acc, ":", color=col, lw=0.8)
        ax.set_xscale("log"); ax.set_xlabel("Número de árboles"); ax.set_title(f"GB en {name}")
        ax.grid(lw=0.3, alpha=0.5)
    axes[1].legend(frameon=False, fontsize=6); axes[0].grid(lw=0.3, alpha=0.5)
    fig.tight_layout()
    _save(fig, "fig_regularizacion")
