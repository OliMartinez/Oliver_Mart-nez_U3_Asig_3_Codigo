"""
manual_split.py
===============
Cálculo "a mano" (explícito, sin scikit-learn) de entropía, índice de Gini y ganancia de
información para todas las particiones candidatas de un subconjunto reducido, y validación
contra la primera partición elegida por DecisionTreeClassifier.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

from .datasets import orthogonal


def entropy(y):
    p = np.bincount(y, minlength=2) / len(y)
    p = p[p > 0]
    return float(abs(-(p * np.log2(p)).sum()))


def gini(y):
    p = np.bincount(y, minlength=2) / len(y)
    return float(1.0 - (p ** 2).sum())


def candidate_table(X, y, names):
    """Ganancia de información y reducción de Gini para cada umbral (puntos medios)."""
    H0, G0, n = entropy(y), gini(y), len(y)
    rows = []
    for j, name in enumerate(names):
        v = np.unique(X[:, j])
        for t in (v[:-1] + v[1:]) / 2:
            L, R = y[X[:, j] <= t], y[X[:, j] > t]
            HL, HR, GL, GR = entropy(L), entropy(R), gini(L), gini(R)
            rows.append(dict(atributo=name, umbral=round(float(t), 3), n_izq=len(L), n_der=len(R),
                             pos_izq=int(L.sum()), pos_der=int(R.sum()), H_izq=HL, H_der=HR,
                             ganancia=H0 - (len(L) * HL + len(R) * HR) / n,
                             gini_izq=GL, gini_der=GR, delta_gini=G0 - (len(L) * GL + len(R) * GR) / n))
    return pd.DataFrame(rows).sort_values("ganancia", ascending=False).reset_index(drop=True)


def manual_subset(n=12, random_state=7):
    """Subconjunto reducido del dataset ortogonal, redondeado a 1 decimal para cálculo a mano."""
    d = orthogonal(n=400, random_state=1)
    rng = np.random.default_rng(random_state)
    i1 = rng.choice(np.flatnonzero(d["y"] == 1), 4, replace=False)
    i0 = rng.choice(np.flatnonzero(d["y"] == 0), n - 4, replace=False)
    idx = np.concatenate([i0, i1])
    X = np.round(d["X"][idx], 1)
    y = ((X[:, 0] > 3.5) & (X[:, 1] < 1.0)).astype(int)
    order = np.argsort(X[:, 0])
    return X[order], y[order]


def run():
    X, y = manual_subset()
    names = ["X1", "X2"]
    tab = candidate_table(X, y, names)
    out = {"datos": pd.DataFrame({"X1": X[:, 0], "X2": X[:, 1], "y": y}),
           "H_raiz": entropy(y), "Gini_raiz": gini(y), "tabla": tab}
    for crit in ("entropy", "gini"):
        clf = DecisionTreeClassifier(criterion=crit, max_depth=1, random_state=0).fit(X, y)
        out[f"sklearn_{crit}"] = (names[clf.tree_.feature[0]], float(clf.tree_.threshold[0]))
    # Segundo nivel (rama con mezcla) para completar el árbol a mano
    best = tab.iloc[0]
    j = names.index(best.atributo)
    mask = X[:, j] > best.umbral if best.pos_der > 0 and best.H_der > 0 else X[:, j] <= best.umbral
    out["tabla_nivel2"] = candidate_table(X[mask], y[mask], names)
    clf = DecisionTreeClassifier(criterion="entropy", random_state=0).fit(X, y)
    out["sklearn_tree_depth"], out["sklearn_tree_leaves"] = clf.get_depth(), clf.get_n_leaves()
    return out


if __name__ == "__main__":
    r = run()
    print(r["datos"].to_string())
    print("H raíz = %.4f bits, Gini raíz = %.4f" % (r["H_raiz"], r["Gini_raiz"]))
    print(r["tabla"].head(8).round(4).to_string())
    print("sklearn:", r["sklearn_entropy"], r["sklearn_gini"])
    print(r["tabla_nivel2"].head(5).round(4).to_string())
    print("árbol completo sklearn: profundidad", r["sklearn_tree_depth"], "hojas", r["sklearn_tree_leaves"])
