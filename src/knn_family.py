"""
knn_family.py
=============
Clasificador de Mínima Distancia y variantes de KNN, multiclase y compatibles con scikit-learn
(fit / predict / predict_proba). Mínima Distancia no es una variante de KNN (no consulta vecinos: usa
un centroide por clase); se incluye en este módulo porque comparte la interfaz y el preprocesamiento. Todos esperan datos ya estandarizados (se usan dentro
de un Pipeline con StandardScaler; ver models.py).

* MinimumDistance   Mínima Distancia Euclidiana: un centroide por clase; asigna la clase del centroide
                    más cercano (frontera = mediatrices entre centroides; clasificador lineal).
* KNNClassifier     KNN estándar: voto por mayoría entre los k vecinos más cercanos.
* CVKNN             KNN con k elegido por validación cruzada estratificada interna (exactitud balanceada).
* DirectCSKNN       Direct Cost-Sensitive KNN (Qin et al., 2013): estima P(c|x) en la vecindad con
                    m-estimate y elige la clase de mínimo riesgo. Con costo w_c por errar la clase c:
                    R(i|x) = sum_{j != i} w_j P(j|x)  =>  argmin_i R(i|x) = argmax_i w_i P(i|x).
* DistancesCSKNN    Distance-based Cost-Sensitive KNN: d_cs(x, x_n) = d(x, x_n) / w_{y_n}^gamma; las
                    muestras de clases costosas se vuelven "más cercanas". gamma='auto' = 1/p, que hace
                    que la deformación reproduzca la regla de Bayes con costos (estimador k-NN de densidad).
* CMKNN             Correlation-Matrix KNN (Zhang et al., 2017): reconstruye cada muestra de prueba como
                    combinación dispersa de las de entrenamiento (l1 + l2,1 + Laplaciano); el soporte
                    de la reconstrucción define un k distinto por muestra. Resuelto por IRLS.
* OneStepKNN        One-Step KNN (Zhang y Li, 2023): aprende en entrenamiento la matriz de correlación
                    entre muestras y un operador P; en prueba w(y) = P y en un solo producto matricial.

Costos (Direct-CS y Distances-CS): no hay un costo "natural" en conjuntos balanceados, así que se usa
w_c = n_max / n_c (inverso a la frecuencia de la clase en entrenamiento); en clases balanceadas w_c ~ 1
y ambos métodos se reducen a KNN.

Escalabilidad: CM-KNN y One-Step KNN construyen sistemas de n x n (n = muestras de entrenamiento), por
lo que entrenan con una submuestra estratificada de tamaño n_max (500 en CM-KNN, cuyo costo está en la
predicción, y 300 en One-Step KNN, cuyo entrenamiento es O(n^4) por iteración), declarada en el reporte.
CM-KNN agrega una coordenada constante antes de normalizar (augment=True); One-Step KNN no la usa.
Iteraciones de IRLS: 8 en CM-KNN y 20 en One-Step KNN.
"""
from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import NearestNeighbors

EPS = 1e-12


# ----------------------------------------------------------------------------- utilidades
def class_costs(yi: np.ndarray, K: int, mode: str = "inverse_freq") -> np.ndarray:
    """Costo w_c de errar la clase c. 'inverse_freq': n_max / n_c ; 'uniform': 1."""
    counts = np.bincount(yi, minlength=K).astype(float)
    if mode == "uniform":
        return np.ones(K)
    return counts.max() / np.maximum(counts, 1.0)


def stratified_subsample(yi: np.ndarray, n_max: int, random_state: int = 42) -> np.ndarray:
    """Índices de una submuestra estratificada de tamaño <= n_max (al menos 2 por clase si existen)."""
    n = len(yi)
    if n <= n_max:
        return np.arange(n)
    rng = np.random.default_rng(random_state)
    out = []
    classes, counts = np.unique(yi, return_counts=True)
    quota = np.maximum(np.round(counts / n * n_max).astype(int), np.minimum(counts, 2))
    for c, q in zip(classes, quota):
        idx = np.flatnonzero(yi == c)
        out.append(rng.choice(idx, size=min(q, len(idx)), replace=False))
    return np.sort(np.concatenate(out))


class _Base(BaseEstimator, ClassifierMixin):
    """Codifica las etiquetas a 0..K-1 y define predict a partir de _decision (argmax)."""

    def _encode(self, y):
        self.classes_, yi = np.unique(np.asarray(y), return_inverse=True)
        self.K_ = len(self.classes_)
        return yi

    def _decision(self, X):
        return self.predict_proba(X)

    def predict(self, X):
        return self.classes_[np.argmax(self._decision(np.asarray(X, float)), axis=1)]


# ----------------------------------------------------------------------------- Mínima Distancia
class MinimumDistance(_Base):
    """Clasificador de Mínima Distancia Euclidiana (caso límite de Bayes con covarianzas sigma^2 I)."""

    def fit(self, X, y):
        X = np.asarray(X, float)
        yi = self._encode(y)
        self.centroids_ = np.vstack([X[yi == c].mean(axis=0) for c in range(self.K_)])
        return self

    def _dist(self, X):
        D2 = (np.einsum("ij,ij->i", X, X)[:, None] + np.einsum("ij,ij->i", self.centroids_, self.centroids_)[None, :]
              - 2.0 * X @ self.centroids_.T)
        return np.sqrt(np.maximum(D2, 0.0))

    def predict_proba(self, X):
        inv = 1.0 / (self._dist(np.asarray(X, float)) + 1e-9)
        return inv / inv.sum(axis=1, keepdims=True)


# ----------------------------------------------------------------------------- KNN
class KNNClassifier(_Base):
    """KNN estándar (voto por mayoría). La búsqueda de vecinos usa NearestNeighbors de scikit-learn."""

    def __init__(self, k=5, n_jobs=None):
        self.k, self.n_jobs = k, n_jobs

    def fit(self, X, y):
        self.X_ = np.asarray(X, float)
        self.y_ = self._encode(y)
        self.nn_ = NearestNeighbors(n_neighbors=min(self.k, len(self.y_)), n_jobs=self.n_jobs).fit(self.X_)
        return self

    def _neighbors(self, X):
        return self.nn_.kneighbors(np.asarray(X, float), return_distance=True)

    def predict_proba(self, X):
        _, idx = self._neighbors(X)
        lab = self.y_[idx]
        P = np.zeros((len(lab), self.K_))
        for c in range(self.K_):
            P[:, c] = (lab == c).mean(axis=1)
        return P


class CVKNN(KNNClassifier):
    """KNN con k elegido por CV estratificada interna maximizando la exactitud balanceada.

    Si el entrenamiento supera inner_max muestras, k se elige sobre una submuestra estratificada
    de ese tamaño (el modelo final usa todo el entrenamiento).
    """

    def __init__(self, ks=(1, 3, 5, 7, 9, 11, 15), n_splits=3, inner_max=20000, random_state=42, n_jobs=None):
        self.ks, self.n_splits, self.inner_max, self.random_state, self.n_jobs = ks, n_splits, inner_max, random_state, n_jobs
        self.k = ks[0]

    def fit(self, X, y):
        X = np.asarray(X, float)
        yi = np.unique(np.asarray(y), return_inverse=True)[1]
        sel = stratified_subsample(yi, self.inner_max, self.random_state)
        Xs, ys = X[sel], yi[sel]
        n_min = int(np.bincount(ys).min())
        n_splits = max(2, min(self.n_splits, n_min))
        scores = np.zeros(len(self.ks))
        kmax = max(self.ks)
        for tr, va in StratifiedKFold(n_splits, shuffle=True, random_state=self.random_state).split(Xs, ys):
            nn = NearestNeighbors(n_neighbors=min(kmax, len(tr)), n_jobs=self.n_jobs).fit(Xs[tr])
            idx = nn.kneighbors(Xs[va], return_distance=False)
            lab = ys[tr][idx]
            for i, k in enumerate(self.ks):
                kk = min(k, lab.shape[1])
                counts = np.stack([(lab[:, :kk] == c).sum(axis=1) for c in range(ys.max() + 1)], axis=1)
                scores[i] += balanced_accuracy_score(ys[va], counts.argmax(axis=1))
        self.cv_scores_ = scores / n_splits
        self.k = int(self.ks[int(np.argmax(self.cv_scores_))])   # empates -> el k menor
        return super().fit(X, y)


# ----------------------------------------------------------------------------- KNN sensibles al costo
class DirectCSKNN(KNNClassifier):
    """Direct-CS-KNN: posterior por m-estimate en la vecindad y decisión de mínimo riesgo."""

    def __init__(self, k=5, m=2.0, cost_mode="inverse_freq", n_jobs=None):
        self.k, self.m, self.cost_mode, self.n_jobs = k, m, cost_mode, n_jobs

    def fit(self, X, y):
        super().fit(X, y)
        self.prior_ = np.bincount(self.y_, minlength=self.K_) / len(self.y_)
        self.costs_ = class_costs(self.y_, self.K_, self.cost_mode)
        return self

    def predict_proba(self, X):
        _, idx = self._neighbors(X)
        lab = self.y_[idx]
        k = lab.shape[1]
        counts = np.stack([(lab == c).sum(axis=1) for c in range(self.K_)], axis=1)
        return (counts + self.m * self.prior_[None, :]) / (k + self.m)      # P(c|x) por m-estimate

    def _decision(self, X):
        return self.predict_proba(X) * self.costs_[None, :]                 # argmax w_c P(c|x) = min riesgo


class DistancesCSKNN(_Base):
    """Distances-CS-KNN: métrica deformada d / w_c^gamma (búsqueda exacta por clase y fusión)."""

    def __init__(self, k=5, gamma="auto", cost_mode="inverse_freq", n_jobs=None):
        self.k, self.gamma, self.cost_mode, self.n_jobs = k, gamma, cost_mode, n_jobs

    def fit(self, X, y):
        X = np.asarray(X, float)
        self.y_ = self._encode(y)
        self.gamma_ = 1.0 / X.shape[1] if self.gamma == "auto" else float(self.gamma)
        self.costs_ = class_costs(self.y_, self.K_, self.cost_mode)
        self.factor_ = self.costs_ ** self.gamma_
        self.nn_ = []
        for c in range(self.K_):
            Xc = X[self.y_ == c]
            self.nn_.append(NearestNeighbors(n_neighbors=min(self.k, len(Xc)), n_jobs=self.n_jobs).fit(Xc))
        return self

    def predict_proba(self, X):
        X = np.asarray(X, float)
        D, L = [], []
        for c in range(self.K_):
            d, _ = self.nn_[c].kneighbors(X, return_distance=True)
            D.append(d / self.factor_[c]); L.append(np.full(d.shape, c))
        D, L = np.hstack(D), np.hstack(L)
        order = np.argsort(D, axis=1)[:, : self.k]
        lab = np.take_along_axis(L, order, axis=1)
        dist = np.take_along_axis(D, order, axis=1)
        w = 1.0 / (dist + 1e-9)
        votes = np.stack([(lab == c).mean(axis=1) for c in range(self.K_)], axis=1)
        wvotes = np.stack([(w * (lab == c)).sum(axis=1) for c in range(self.K_)], axis=1) / w.sum(axis=1, keepdims=True)
        return 0.999 * votes + 0.001 * wvotes        # mayoría; desempate por distancia deformada


# ----------------------------------------------------------------------------- reconstrucción matricial
def feature_laplacian(X: np.ndarray, n_neighbors: int = 5) -> np.ndarray:
    """Laplaciano L = D - S del grafo k-NN entre CARACTERÍSTICAS (kernel de calor)."""
    F = X.T
    d = F.shape[0]
    D2 = np.maximum(np.einsum("ij,ij->i", F, F)[:, None] + np.einsum("ij,ij->i", F, F)[None, :] - 2 * F @ F.T, 0)
    pos = D2[D2 > 0]
    S = np.exp(-D2 / ((np.median(pos) if pos.size else 1.0) + EPS))
    k = min(n_neighbors, d - 1)
    if k < 1:
        return np.zeros((d, d))
    idx = np.argsort(D2, axis=1)[:, 1: k + 1]
    mask = np.zeros((d, d), dtype=bool)
    mask[np.arange(d)[:, None], idx] = True
    mask |= mask.T
    S = S * mask
    np.fill_diagonal(S, 0.0)
    return np.diag(S.sum(axis=1)) - S


def _batched_solve(A_base, diag_add, B, mem_bytes=256 * 2**20):
    """Resuelve (A_base + diag(diag_add[:, j])) w_j = B[:, j] para todas las columnas, en lotes acotados en memoria."""
    n, m = B.shape
    chunk = max(1, int(mem_bytes // (8 * n * n)))
    W = np.empty((n, m))
    ii = np.arange(n)
    for s in range(0, m, chunk):
        e = min(m, s + chunk)
        A = np.broadcast_to(A_base, (e - s, n, n)).copy()
        A[:, ii, ii] += diag_add[:, s:e].T
        W[:, s:e] = np.linalg.solve(A, B[:, s:e].T[..., None])[..., 0].T
    return W


def _unit(X, augment=False):
    """Normaliza cada muestra a norma l2 unitaria (estándar en representación dispersa).

    augment=True agrega antes una coordenada constante (coordenadas homogéneas) para no perder la
    información de magnitud que la normalización elimina."""
    if augment:
        X = np.hstack([X, np.ones((len(X), 1))])
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + EPS)


def _support(W, tau_rel):
    A = np.abs(W)
    return A >= tau_rel * (A.max(axis=0, keepdims=True) + EPS)


def _class_scores(A, y, K):
    """Suma de pesos por clase normalizada; columnas sin soporte -> distribución uniforme."""
    S = np.stack([(A * (y[:, None] == c)).sum(axis=0) for c in range(K)], axis=1)
    tot = S.sum(axis=1, keepdims=True)
    return np.where(tot > 0, S / np.maximum(tot, EPS), 1.0 / K)


class CMKNN(_Base):
    """CM-KNN: min ||X^T W - Y||^2 + rho1 ||W||_1 + rho2 ||W||_{2,1} + rho3 Tr(W^T X L X^T W)."""

    def __init__(self, rho1=0.3, rho2=0.01, rho3=1e-3, n_neighbors_graph=5, max_iter=8, tol=1e-3,
                 tau_rel=0.1, eps=1e-6, n_max=500, augment=True, random_state=42):
        self.rho1, self.rho2, self.rho3, self.augment = rho1, rho2, rho3, augment
        self.n_neighbors_graph, self.max_iter, self.tol = n_neighbors_graph, max_iter, tol
        self.tau_rel, self.eps, self.n_max, self.random_state = tau_rel, eps, n_max, random_state

    def fit(self, X, y):
        yi = self._encode(y)
        sel = stratified_subsample(yi, self.n_max, self.random_state)
        self.n_train_used_ = len(sel)
        self.X_ = _unit(np.asarray(X, float)[sel], self.augment)
        self.y_ = yi[sel]
        L = feature_laplacian(self.X_, self.n_neighbors_graph)
        self.G_ = self.X_ @ self.X_.T
        self.A_base_ = self.G_ + self.rho3 * (self.X_ @ L @ self.X_.T)
        return self

    def reconstruct(self, Xtest):
        B = self.X_ @ _unit(np.asarray(Xtest, float), self.augment).T
        n, m = B.shape
        scale = np.trace(self.G_) / n
        W = _batched_solve(self.A_base_, np.full((n, m), 1e-3 * scale), B)
        for _ in range(self.max_iter):
            D1 = 1.0 / (np.abs(W) + self.eps)
            D2 = 1.0 / (np.linalg.norm(W, axis=1) + self.eps)
            W_new = _batched_solve(self.A_base_, 0.5 * self.rho1 * D1 + 0.5 * self.rho2 * D2[:, None], B)
            rel = np.linalg.norm(W_new - W) / (np.linalg.norm(W) + EPS)
            W = W_new
            if rel < self.tol:
                break
        return W

    def predict_proba(self, X):
        A = np.clip(self.reconstruct(X), 0.0, None)
        A = A * _support(A, self.tau_rel)
        self.k_selected_ = (A > 0).sum(axis=0)
        return _class_scores(A, self.y_, self.K_)


class OneStepKNN(_Base):
    """One-Step KNN: min ||X^T - X^T W||^2 + alpha ||W||_{2,1}^2 + beta ||W||_1^2, diag(W)=0."""

    def __init__(self, alpha=1e-3, beta=1e-4, max_iter=20, tol=1e-3, tau_rel=0.7, eps=1e-6, d1_cap=1e6,
                 n_max=300, augment=False, random_state=42):
        self.alpha, self.beta, self.max_iter, self.tol, self.augment = alpha, beta, max_iter, tol, augment
        self.tau_rel, self.eps, self.d1_cap, self.n_max, self.random_state = tau_rel, eps, d1_cap, n_max, random_state

    def fit(self, X, y):
        yi = self._encode(y)
        sel = stratified_subsample(yi, self.n_max, self.random_state)
        self.n_train_used_ = len(sel)
        X = _unit(np.asarray(X, float)[sel], self.augment)
        self.y_ = yi[sel]
        n = X.shape[0]
        G = X @ X.T
        scale = np.trace(G) / n
        self_pen = np.zeros((n, n)); np.fill_diagonal(self_pen, 1e8 * scale)
        W = _batched_solve(G, 1e-3 * scale + self_pen, G)
        for _ in range(self.max_iter):
            rn = np.linalg.norm(W, axis=1)
            Dr = rn.sum() / (rn + self.eps)
            D1 = np.minimum(np.abs(W).sum() / (np.abs(W) + self.eps), self.d1_cap * scale)
            W_new = _batched_solve(G, self.alpha * Dr[:, None] + self.beta * D1 + self_pen, G)
            rel = np.linalg.norm(W_new - W) / (np.linalg.norm(W) + EPS)
            W = W_new
            if rel < self.tol:
                break
        rn = np.linalg.norm(W, axis=1)
        self.theta_ = rn / (rn.max() + EPS)                       # confiabilidad (outliers -> ~0)
        Dr = rn.sum() / (rn + self.eps)
        D1 = np.minimum(np.abs(W).sum() / (np.abs(W) + self.eps), self.d1_cap * scale)
        np.fill_diagonal(D1, np.nan)
        A = G + np.diag(self.alpha * Dr + self.beta * np.nanmedian(D1, axis=1))
        self.P_ = np.linalg.solve(A, X)                           # operador de una sola pasada
        return self

    def predict_proba(self, X):
        Wp = np.clip(self.P_ @ _unit(np.asarray(X, float), self.augment).T, 0.0, None)
        Wp = Wp * _support(Wp, self.tau_rel) * self.theta_[:, None]
        self.k_selected_ = (Wp > 0).sum(axis=0)
        return _class_scores(Wp, self.y_, self.K_)
