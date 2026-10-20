"""
datasets.py
===========
Carga de los conjuntos reales (copias del UCI ML Repository) y generación de los sintéticos.

Conjuntos indicados en la guía del ABP:
Categoría 1 (estructuras ortogonales / efectividad)
    wine, forest_fires, credit_card, har, concrete, glass, bcw, orthogonal (IF-THEN) e iris (prueba de efectividad)
Categoría 2 (fronteras no orientadas a los ejes / limitación)
    circles y moons (Make Circles / Make Moons) y el conjunto de rotación ("XOR continuo") en sus dos lecturas:
    rotated_linear(45) (dos clases linealmente separables rotadas) y rotated_xor(45). rotated_xor(0) es el punto
    de referencia sin rotar del mismo conjunto (la guía compara el desempeño antes y después de rotar).

Cada loader devuelve un dict: X, y, feature_names, description, task_note.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris, load_wine, make_circles, make_moons

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RANDOM_STATE = 42


def _pack(X, y, names, desc, note=""):
    return dict(X=np.asarray(X, float), y=np.asarray(y, int), feature_names=list(names),
                description=desc, task_note=note)


# ----------------------------------------------------------------------------- reales
def wine():
    d = load_wine()
    return _pack(d.data, d.target, d.feature_names, "Wine (UCI): 178 x 13, 3 clases",
                 "Atributos químicos altamente correlacionados")


def forest_fires():
    df = pd.read_csv(DATA / "forestfires.csv")
    months = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
    days = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]
    df["month"] = df["month"].map({m: i + 1 for i, m in enumerate(months)})
    df["day"] = df["day"].map({d: i + 1 for i, d in enumerate(days)})
    y = (df["area"] > 0).astype(int)
    X = df.drop(columns="area")
    return _pack(X, y, X.columns, "Forest Fires (UCI): 517 x 12, binaria (área quemada > 0)",
                 "Distribuciones muy sesgadas (area, rain, FFMC); objetivo binarizado")


def credit_card(legit_frac=1.0, random_state=RANDOM_STATE):
    """Credit Card Fraud completo (legit_frac=1.0). Con legit_frac<1 se submuestrean las legítimas."""
    df = pd.read_csv(DATA / "creditcard.csv")
    rng = np.random.default_rng(random_state)
    fraud = df[df.Class == 1]
    legit = df[df.Class == 0]
    legit = legit.iloc[rng.choice(len(legit), int(legit_frac * len(legit)), replace=False)]
    df = pd.concat([fraud, legit]).sample(frac=1.0, random_state=random_state)
    X = df.drop(columns="Class")
    return _pack(X, df["Class"], X.columns,
                 f"Credit Card Fraud (ULB/Kaggle): {len(df)} x 30, fraude = {df.Class.mean():.2%}",
                 "Desbalance extremo; se conserva el 100% de los fraudes")


def har():
    """Human Activity Recognition (smartphones). CV agrupada por sujeto (requiere subject_*.txt del UCI)."""
    def read(name):
        return np.loadtxt(DATA / "har" / name)
    Xtr, Xte = read("X_train.txt"), read("X_test.txt")
    ytr, yte = read("y_train.txt").astype(int) - 1, read("y_test.txt").astype(int) - 1
    d = _pack(np.vstack([Xtr, Xte]), np.concatenate([ytr, yte]), [f"f{i}" for i in range(Xtr.shape[1])],
              "HAR (UCI): 10 299 x 561, 6 actividades", "Alta dimensionalidad; CV agrupada por sujeto")
    # Sujeto de cada muestra: la CV agrupada impide que una persona esté en entrenamiento y prueba a la vez.
    d["groups"] = np.concatenate([read("subject_train.txt"), read("subject_test.txt")]).astype(int)
    d["classes"] = ["caminar", "subir", "bajar", "sentado", "de pie", "acostado"]
    return d


def concrete(threshold=35.0):
    df = pd.read_csv(DATA / "concrete.csv")
    y = (df["strength"] >= threshold).astype(int)
    X = df.drop(columns="strength")
    return _pack(X, y, X.columns, f"Concrete (UCI): 1030 x 8, binaria (resistencia >= {threshold} MPa)",
                 "Atributos sin escalar (kg/m3, días) con valores atípicos")


def glass():
    df = pd.read_csv(DATA / "glass.csv", header=None)
    names = ["RI", "Na", "Mg", "Al", "Si", "K", "Ca", "Ba", "Fe"]
    y_raw = df.iloc[:, 9].to_numpy()
    classes = np.unique(y_raw)
    y = np.searchsorted(classes, y_raw)
    return _pack(df.iloc[:, :9], y, names, "Glass Identification (UCI): 214 x 9, 6 clases",
                 "Clases pequeñas y desbalanceadas")


def bcw():
    df = pd.read_csv(DATA / "bcw.csv", header=None, na_values="?")
    names = ["clump", "unif_size", "unif_shape", "adhesion", "epith_size", "bare_nuclei",
             "chromatin", "nucleoli", "mitoses"]
    X = df.iloc[:, :9]  # 16 faltantes ('?') se dejan como NaN: se imputan dentro de cada pliegue
    y = (df.iloc[:, 9] == 4).astype(int)
    return _pack(X, y, names, "Breast Cancer Wisconsin Original (UCI): 699 x 9, binaria (maligno)",
                 "Atributos ordinales 1-10: umbrales naturales")


def iris():
    d = load_iris()
    return _pack(d.data, d.target, d.feature_names, "Iris: 150 x 4, 3 clases", "Prueba de efectividad")


# ----------------------------------------------------------------------------- sintéticos
def orthogonal(n=1000, label_noise=0.0, random_state=RANDOM_STATE):
    """Regla lógica IF X1 > 3.5 AND X2 < 1.0 THEN clase 1."""
    rng = np.random.default_rng(random_state)
    X = np.column_stack([rng.uniform(0, 6, n), rng.uniform(-2, 4, n)])
    y = ((X[:, 0] > 3.5) & (X[:, 1] < 1.0)).astype(int)
    if label_noise > 0:
        flip = rng.random(n) < label_noise
        y[flip] = 1 - y[flip]
    return _pack(X, y, ["X1", "X2"], "Sintético ortogonal: IF X1>3.5 AND X2<1.0", "Frontera paralela a ejes")


def circles(n=1000, noise=0.08, random_state=RANDOM_STATE):
    X, y = make_circles(n, noise=noise, factor=0.5, random_state=random_state)
    return _pack(X, y, ["X1", "X2"], "make_circles", "Frontera circular")


def moons(n=1000, noise=0.20, random_state=RANDOM_STATE):
    X, y = make_moons(n, noise=noise, random_state=random_state)
    return _pack(X, y, ["X1", "X2"], "make_moons", "Fronteras entrelazadas")


def rotate(X, theta_deg):
    t = np.deg2rad(theta_deg)
    R = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
    return X @ R.T


def rotated_linear(theta_deg=45.0, n=1000, margin=0.25, random_state=RANDOM_STATE):
    """Dos clases linealmente separables por X1 = 0 (margen 2*margin), rotadas theta grados."""
    rng = np.random.default_rng(random_state)
    X = rng.uniform(-3, 3, size=(n, 2))
    X = X[np.abs(X[:, 0]) > margin]
    y = (X[:, 0] > 0).astype(int)
    return _pack(rotate(X, theta_deg), y, ["X1", "X2"], f"Lineal rotado {theta_deg:.0f} grados",
                 "Separable linealmente; sólo cambia la orientación")


def rotated_xor(theta_deg=45.0, n=1000, margin=0.15, random_state=RANDOM_STATE):
    """XOR continuo: clase = signo(X1) XOR signo(X2), con margen, rotado theta grados."""
    rng = np.random.default_rng(random_state)
    X = rng.uniform(-3, 3, size=(int(n * 1.3), 2))
    X = X[(np.abs(X[:, 0]) > margin) & (np.abs(X[:, 1]) > margin)][:n]
    y = ((X[:, 0] > 0) ^ (X[:, 1] > 0)).astype(int)
    return _pack(rotate(X, theta_deg), y, ["X1", "X2"], f"XOR continuo rotado {theta_deg:.0f} grados",
                 "Tablero de ajedrez; ortogonal a 0 grados, diagonal a 45")


REAL = {"Wine": wine, "Forest Fires": forest_fires, "Credit Card": credit_card, "HAR": har,
        "Concrete": concrete, "Glass": glass, "BCW (Original)": bcw, "Iris": iris}
SYNTH = {"Ortogonal IF-THEN": orthogonal, "Circles": circles, "Moons": moons,
         "Lineal rotado 45°": lambda: rotated_linear(45), "XOR rotado 45°": lambda: rotated_xor(45),
         "XOR 0°": lambda: rotated_xor(0)}
