"""Genera las tablas LaTeX del reporte a partir de results/*.csv (evita errores de transcripción)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import manual_split
from .experiments import RES

ORDER = ["Iris", "BCW (Original)", "Wine", "Glass", "Concrete", "Forest Fires", "Credit Card", "HAR",
         "Ortogonal IF-THEN", "XOR 0°", "Lineal rotado 45°", "XOR rotado 45°", "Circles", "Moons"]
MODS = ["Árbol de Decisión", "Árbol podado (ccp)", "Random Forest", "Gradient Boosting"]


def tex(s):
    return s.replace("°", r"$^\circ$").replace("%", r"\%")


def benchmark_table(df):
    g = df.groupby(["dataset", "model"])
    mean, std = g.mean(numeric_only=True), g.std(numeric_only=True)
    lines = []
    for i, ds in enumerate(ORDER):
        if i == 8:
            lines.append(r"\midrule")
        best = max(mean.loc[(ds, m), "bal_acc"] for m in MODS)
        cells = []
        for m in MODS:
            v, s = mean.loc[(ds, m), "bal_acc"], std.loc[(ds, m), "bal_acc"]
            txt = f"{v:.3f}" + (f"$\\pm${s:.2f}" if not np.isnan(s) else "")
            cells.append(r"\textbf{" + txt + "}" if abs(v - best) < 5e-4 else txt)
        gaps = [f"{mean.loc[(ds, m), 'gap']:.2f}" for m in MODS]
        leaves = f"{mean.loc[(ds, 'Árbol de Decisión'), 'leaves']:.0f}/{mean.loc[(ds, 'Árbol podado (ccp)'), 'leaves']:.0f}"
        lines.append(f"{tex(ds)} & " + " & ".join(cells) + " & " + " / ".join(gaps) + f" & {leaves} \\\\")
    return "\n".join(lines)


def manual_tables():
    r = manual_split.run()
    d = r["datos"]
    data = " \\\\\n".join(f"{i+1} & {a:.1f} & {b:.1f} & {int(c)}" for i, (a, b, c) in enumerate(d.to_numpy()))
    t = r["tabla"].head(6)
    cand = " \\\\\n".join(
        f"{row.atributo} $\\le$ {row.umbral:.2f} & {row.n_izq} ({row.pos_izq}) & {row.n_der} ({row.pos_der}) & "
        f"{row.H_izq:.3f} & {row.H_der:.3f} & {row.ganancia:.3f} & {row.delta_gini:.3f}" for row in t.itertuples())
    t2 = r["tabla_nivel2"].head(3)
    cand2 = " \\\\\n".join(
        f"{row.atributo} $\\le$ {row.umbral:.2f} & {row.n_izq} ({row.pos_izq}) & {row.n_der} ({row.pos_der}) & "
        f"{row.H_izq:.3f} & {row.H_der:.3f} & {row.ganancia:.3f} & {row.delta_gini:.3f}" for row in t2.itertuples())
    return data, cand, cand2, r


def credit_table(big):
    c = big[big.dataset == "Credit Card"]
    order = ["Árbol de Decisión", "Árbol de Decisión (bal)", "Árbol podado (ccp)", "Random Forest",
             "Random Forest (bal)", "Gradient Boosting", "Gradient Boosting (bal)"]
    g = c.groupby("model").mean(numeric_only=True).reindex(order)
    out = []
    for m, r in g.iterrows():
        name = m.replace("Árbol de Decisión", "DT").replace("Árbol podado (ccp)", "DT podado").replace(
            "Random Forest", "RF").replace("Gradient Boosting", "GB").replace("(bal)", "balanceado")
        out.append(f"{name} & {r.test_acc:.4f} & {r.bal_acc:.3f} & {r.recall_pos:.3f} & {r.precision_pos:.3f} & "
                   f"{r.mcc:.3f} & {r.pr_auc:.3f} \\\\")
    return "\n".join(out)


def main():
    small = [pd.read_csv(RES / "benchmark_small.csv")]
    big = pd.read_csv(RES / "benchmark_big.csv")
    df = pd.concat(small + [big[big.model.isin(MODS)]], ignore_index=True)
    data, cand, cand2, r = manual_tables()
    ch = pd.read_csv(RES / "ccp_chosen.csv")
    ccp = "\n".join(f"{tex(x.dataset)} & {x.alpha_cv:.4f} & {x.leaves_full}/{x.depth_full} & {x.leaves_pruned}/{x.depth_pruned} & "
                    f"{x.train_full:.3f}/{x.test_full:.3f} & {x.train_pruned:.3f}/{x.test_pruned:.3f} \\\\" for x in ch.itertuples())
    sc = pd.read_csv(RES / "q_scaling_outliers.csv")
    p = sc.pivot(index="model", columns="condition", values="acc")
    p = p[[c for c in sc.condition.unique()]].reindex(sc.model.unique())
    scal = "\n".join(f"{m} & {a:.3f} & {b:.3f} & {b - a:+.3f} \\\\" for m, (a, b) in p.iterrows())
    sp = pd.read_csv(RES / "sig_pairs.csv")
    ab = {"Árbol de Decisión": "DT", "Árbol podado (ccp)": "DT podado", "Random Forest": "RF", "Gradient Boosting": "GB"}
    sig = "\n".join(f"{ab[r.B]} vs {ab[r.A]} & {-r.mean_diff:+.3f} & {abs(r.rank_diff):.2f} & {r.p:.4f} & {r.p_holm:.4f} & "
                     f"{'sí' if r.nemenyi_sig else 'no'} \\\\" for r in sp.itertuples())
    rs = pd.read_csv(RES / "imbalance_resampling.csv").groupby(["strategy", "model"]).mean(numeric_only=True).reset_index()
    order_s = ["submuestreo 1:10", "SMOTE 1:10", "umbral óptimo (F1)"]
    lines = []
    for st in order_s:
        for mname in ["Árbol de Decisión", "Random Forest", "Gradient Boosting"]:
            q = rs[(rs.strategy == st) & (rs.model == mname)].iloc[0]
            lines.append(f"{st.replace('umbral óptimo (F1)', 'Umbral (F1)').replace('submuestreo', 'Submuestreo')} & {ab[mname]} & "
                         f"{q.bal_acc:.3f} & {q.recall_pos:.3f} & {q.precision_pos:.3f} & {q.mcc:.3f} \\\\")
    resamp = "\n".join(lines)
    th = pd.read_csv(RES / "threshold_sensitivity.csv").groupby(["dataset", "threshold", "model"], sort=False).mean(numeric_only=True)
    tl = []
    for (ds, t), g in th.groupby(level=[0, 1], sort=False):
        v = {m: g.loc[(ds, t, m), "bal_acc"] for m in MODS}
        best = max(v.values())
        cells = " & ".join((r"\textbf{%.3f}" % v[m]) if abs(v[m] - best) < 5e-4 else "%.3f" % v[m] for m in MODS)
        lab = f"{t:.0f} MPa" if ds == "Concrete" else f"área $>$ {t:.0f} ha"
        tl.append(f"{ds} & {lab} & {100 * g['pos_rate'].iloc[0]:.1f}\\% & {cells} \\\\")
    umbral = "\n".join(tl)
    with open(RES / "tables.tex", "w") as fh:
        for k, v in dict(TABbenchmark=benchmark_table(df), TABmanualdata=data, TABmanualcand=cand,
                         TABmanualcandII=cand2, TABcredit=credit_table(big), TABccp=ccp, TABscaling=scal, TABsig=sig, TABresamp=resamp, TABumbral=umbral).items():
            fh.write(f"\\newcommand{{\\{k}}}{{%\n{v}%\n}}\n")
    print(open(RES / "tables.tex").read()[:3000])
    print("H raiz", r["H_raiz"], "Gini", r["Gini_raiz"], r["sklearn_entropy"])


if __name__ == "__main__":
    main()
