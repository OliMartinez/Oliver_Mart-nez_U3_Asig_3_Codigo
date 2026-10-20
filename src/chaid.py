"""
chaid.py
========
CHAID (CHi-squared Automatic Interaction Detection; Kass, 1980) implementado desde cero.

Diferencias con CART/C4.5:
  * Usa la prueba chi-cuadrada de independencia (no entropía ni Gini) para elegir el atributo.
  * Produce divisiones múltiples (un hijo por grupo de categorías), no sólo binarias.
  * Detiene el crecimiento por significancia estadística (no requiere poda posterior).

CHAID trabaja con predictores categóricos. Los atributos continuos se discretizan en n_bins
intervalos por cuantiles calculados sólo con el entrenamiento (sin fuga); las categorías resultantes
son ordinales, así que sólo se fusionan categorías adyacentes.

Algoritmo en cada nodo, para cada atributo j:
  1. Fusión: mientras exista un par de grupos adyacentes cuya tabla 2 x K tenga p > alpha_merge
     (distribución de clases indistinguible), fusionar el par con el mayor p.
  2. Prueba del atributo: chi-cuadrada de la tabla (grupos x clases) y corrección de Bonferroni
     para predictores ordinales: multiplicador C(c - 1, g - 1) (c categorías originales, g grupos).
  3. Se elige el atributo con menor p ajustado; se divide si p_adj < alpha_split, el nodo tiene al
     menos min_parent muestras y cada hijo al menos min_child (los grupos pequeños se fusionan con
     su vecino más parecido).
Simplificación declarada: se omite el paso de re-división de grupos fusionados (CHAID exhaustivo).
"""
from __future__ import annotations

from math import comb

import numpy as np
from scipy.stats import chi2 as chi2_dist
from sklearn.base import BaseEstimator, ClassifierMixin


def chi2_pvalue(table: np.ndarray) -> float:
    """p-valor de la prueba chi-cuadrada de independencia (sin corrección de Yates)."""
    t = table[table.sum(axis=1) > 0][:, table.sum(axis=0) > 0]
    if t.shape[0] < 2 or t.shape[1] < 2:
        return 1.0
    n = t.sum()
    E = t.sum(axis=1, keepdims=True) * t.sum(axis=0, keepdims=True) / n
    stat = float(((t - E) ** 2 / E).sum())
    dof = (t.shape[0] - 1) * (t.shape[1] - 1)
    return float(chi2_dist.sf(stat, dof))


def adjacent_pvalues(R: np.ndarray) -> np.ndarray:
    """p-valores chi-cuadrada de TODOS los pares adyacentes (R[i], R[i+1]) a la vez (vectorizado).

    Para cada par se ignoran las clases ausentes en ambos grupos (dof = clases presentes - 1).
    """
    P = np.stack([R[:-1], R[1:]], axis=1)                # (G-1, 2, K)
    rt = P.sum(axis=2, keepdims=True)                    # totales por grupo
    ct = P.sum(axis=1, keepdims=True)                    # totales por clase
    n = rt.sum(axis=1, keepdims=True)
    E = rt * ct / np.maximum(n, 1e-300)
    with np.errstate(divide="ignore", invalid="ignore"):
        stat = np.where(E > 0, (P - E) ** 2 / E, 0.0).sum(axis=(1, 2))
    dof = (ct[:, 0, :] > 0).sum(axis=1) - 1
    ok = (dof > 0) & (rt[:, :, 0] > 0).all(axis=1)
    p = np.ones(len(P))
    p[ok] = chi2_dist.sf(stat[ok], dof[ok])
    return p


class CHAIDClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, max_depth=5, alpha_merge=0.05, alpha_split=0.05, min_parent=20, min_child=10, n_bins=10):
        self.max_depth, self.alpha_merge, self.alpha_split = max_depth, alpha_merge, alpha_split
        self.min_parent, self.min_child, self.n_bins = min_parent, min_child, n_bins

    # ------------------------------------------------------------------ discretización
    def _codes(self, X):
        return np.column_stack([np.searchsorted(e, X[:, j], side="right") for j, e in enumerate(self.edges_)])

    def fit(self, X, y):
        X = np.asarray(X, float)
        self.classes_, yi = np.unique(np.asarray(y), return_inverse=True)
        self.K_ = len(self.classes_)
        qs = np.linspace(0, 1, self.n_bins + 1)[1:-1]
        self.edges_ = [np.unique(np.quantile(X[:, j], qs)) for j in range(X.shape[1])]
        self.n_codes_ = [len(e) + 1 for e in self.edges_]
        C = self._codes(X)
        self.n_nodes_, self.n_leaves_, self.depth_ = 0, 0, 0
        self.tree_ = self._grow(C, yi, np.arange(len(yi)), 0)
        return self

    # ------------------------------------------------------------------ crecimiento
    def _table(self, codes, y, nc):
        return np.bincount(codes * self.K_ + y, minlength=nc * self.K_).reshape(nc, self.K_).astype(float)

    def _merge_feature(self, codes, y, nc):
        """Devuelve (p_ajustado, grupos) con grupos = lista de (lo, hi) contiguos sobre códigos presentes."""
        T = self._table(codes, y, nc)
        present = np.flatnonzero(T.sum(axis=1) > 0)
        c = len(present)
        if c < 2:
            return 1.0, None
        groups = [[int(k), int(k)] for k in present]
        rows = [T[k].copy() for k in present]
        while len(groups) > 1:
            ps = adjacent_pvalues(np.vstack(rows))
            i = int(np.argmax(ps))
            if ps[i] <= self.alpha_merge:
                break
            groups[i] = [groups[i][0], groups[i + 1][1]]; rows[i] = rows[i] + rows[i + 1]
            del groups[i + 1], rows[i + 1]
        if len(groups) < 2:
            return 1.0, None
        # tamaño mínimo de los hijos: fusionar el grupo más pequeño con su vecino más parecido
        while len(groups) > 1 and min(r.sum() for r in rows) < self.min_child:
            i = int(np.argmin([r.sum() for r in rows]))
            nb = [j for j in (i - 1, i + 1) if 0 <= j < len(groups)]
            j = max(nb, key=lambda j: chi2_pvalue(np.vstack([rows[i], rows[j]])))
            a, b = sorted((i, j))
            groups[a] = [groups[a][0], groups[b][1]]; rows[a] = rows[a] + rows[b]
            del groups[b], rows[b]
        if len(groups) < 2:
            return 1.0, None
        g = len(groups)
        p = chi2_pvalue(np.vstack(rows))
        return min(1.0, p * comb(c - 1, g - 1)), groups

    def _leaf(self, y):
        self.n_leaves_ += 1
        return {"leaf": True, "dist": np.bincount(y, minlength=self.K_) / max(len(y), 1)}

    def _grow(self, C, yi, idx, depth):
        self.n_nodes_ += 1
        self.depth_ = max(self.depth_, depth)
        y = yi[idx]
        if depth >= self.max_depth or len(idx) < self.min_parent or len(np.unique(y)) < 2:
            return self._leaf(y)
        best = (1.0, None, None)
        for j in range(C.shape[1]):
            p, groups = self._merge_feature(C[idx, j], y, self.n_codes_[j])
            if groups is not None and p < best[0]:
                best = (p, j, groups)
        p, j, groups = best
        if j is None or p >= self.alpha_split:
            return self._leaf(y)
        # mapa código -> hijo (códigos no vistos en el nodo: grupo más cercano)
        nc = self.n_codes_[j]
        cmap = np.empty(nc, dtype=int)
        for code in range(nc):
            dists = [0 if lo <= code <= hi else min(abs(code - lo), abs(code - hi)) for lo, hi in groups]
            cmap[code] = int(np.argmin(dists))
        child_of = cmap[C[idx, j]]
        children = [self._grow(C, yi, idx[child_of == g], depth + 1) for g in range(len(groups))]
        return {"leaf": False, "feature": j, "cmap": cmap, "children": children, "p_adj": p,
                "dist": np.bincount(y, minlength=self.K_) / len(y)}

    # ------------------------------------------------------------------ predicción
    def predict_proba(self, X):
        C = self._codes(np.asarray(X, float))
        out = np.empty((len(C), self.K_))
        stack = [(self.tree_, np.arange(len(C)))]
        while stack:
            node, idx = stack.pop()
            if node["leaf"] or len(idx) == 0:
                out[idx] = node["dist"]
                continue
            child = node["cmap"][C[idx, node["feature"]]]
            for g, ch in enumerate(node["children"]):
                stack.append((ch, idx[child == g]))
        return out

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]
