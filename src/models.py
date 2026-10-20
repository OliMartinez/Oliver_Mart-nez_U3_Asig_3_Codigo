"""
models.py
=========
Catálogo de los 13 clasificadores y su preprocesamiento.

  Árboles:            Árbol de Decisión (entropía), CART (Gini + poda), C5.0, CHAID
  Ensambles:          Random Forest, Gradient Boosting
  Mínima Distancia:   clasificador por centroides (no es un método de vecinos)
  Variantes de KNN:   KNN (k=5), CV-KNN, Direct-CS-KNN, Distances-CS-KNN, CM-KNN, One-Step KNN

Preprocesamiento (siempre dentro de cada pliegue, sin fuga): imputación por mediana del entrenamiento
para todos; además, Mínima Distancia y las variantes de KNN se envuelven en un Pipeline con StandardScaler, porque la
distancia euclidiana depende de la escala. Los árboles no necesitan escalado (sólo usan el orden).
"""
from __future__ import annotations

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from . import config as C
from .c50 import C50Classifier
from .chaid import CHAIDClassifier
from .experiments import RS, FULL_PRUNING, PrunedTree
from .knn_family import (CMKNN, CVKNN, DirectCSKNN, DistancesCSKNN, KNNClassifier, MinimumDistance,
                         OneStepKNN)

MODEL_ORDER = ["Árbol de Decisión (entropía)", "CART (Gini + poda)", "C5.0", "CHAID",
               "Random Forest", "Gradient Boosting",
               "Mínima Distancia", "KNN (k=5)", "CV-KNN", "Direct-CS-KNN", "Distances-CS-KNN",
               "CM-KNN", "One-Step KNN"]

FAMILY = {**{m: "Árboles" for m in MODEL_ORDER[:4]}, **{m: "Ensambles" for m in MODEL_ORDER[4:6]},
          "Mínima Distancia": "Mínima Distancia", **{m: "Variantes de KNN" for m in MODEL_ORDER[7:]}}

SHORT = {"Árbol de Decisión (entropía)": "DT-entropía", "CART (Gini + poda)": "CART", "C5.0": "C5.0",
         "CHAID": "CHAID", "Random Forest": "RF", "Gradient Boosting": "GB", "KNN (k=5)": "KNN",
         "Mínima Distancia": "MinDist", "CV-KNN": "CV-KNN", "Direct-CS-KNN": "Direct-CS",
         "Distances-CS-KNN": "Dist-CS", "CM-KNN": "CM-KNN", "One-Step KNN": "One-Step"}

# Modelos que también evalúa el estudio base (experiments.py), con los mismos hiperparámetros y semilla:
BASE_EQUIV = {"CART (Gini + poda)": "Árbol podado (ccp)", "Random Forest": "Random Forest",
              "Gradient Boosting": "Gradient Boosting"}


def _scaled(model):
    return make_pipeline(StandardScaler(), model)


def make_models(n_jobs=None, large=False):
    """Fábrica: nombre -> función sin argumentos que crea un modelo nuevo (sin entrenar).

    n_jobs: núcleos para RF y búsqueda de vecinos (None = config). large: conjunto grande (poda rápida
    si FULL_PRUNING es False, como en experiments.py).
    """
    nj = C.N_JOBS if n_jobs is None else n_jobs
    fast = large and not FULL_PRUNING
    return {
        "Árbol de Decisión (entropía)": lambda: DecisionTreeClassifier(criterion=C.DT_CRITERION, random_state=RS),
        "CART (Gini + poda)": lambda: PrunedTree(n_alphas=10 if fast else 25, inner_cv=3 if fast else 5),
        "C5.0": lambda: C50Classifier(trials=C.C50_TRIALS),
        "CHAID": lambda: CHAIDClassifier(),
        "Random Forest": lambda: RandomForestClassifier(n_estimators=200, random_state=RS, n_jobs=nj),
        "Gradient Boosting": lambda: GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3,
                                                                random_state=RS),
        "KNN (k=5)": lambda: _scaled(KNNClassifier(k=5, n_jobs=nj)),
        "Mínima Distancia": lambda: _scaled(MinimumDistance()),
        "CV-KNN": lambda: _scaled(CVKNN(n_jobs=nj)),
        "Direct-CS-KNN": lambda: _scaled(DirectCSKNN(k=5, cost_mode=C.COST_MODE, n_jobs=nj)),
        "Distances-CS-KNN": lambda: _scaled(DistancesCSKNN(k=5, cost_mode=C.COST_MODE, n_jobs=nj)),
        "CM-KNN": lambda: _scaled(CMKNN(n_max=C.CMKNN_N_MAX)),
        "One-Step KNN": lambda: _scaled(OneStepKNN(n_max=C.ONESTEP_N_MAX)),
    }


def final_estimator(model):
    return model.steps[-1][1] if hasattr(model, "steps") else model


def predict_both(model, X):
    """Probabilidades y etiquetas con una sola pasada de predicción (importante en CM-KNN y C5.0).

    Direct-CS-KNN decide por mínimo riesgo (argmax w_c P(c|x)), no por la probabilidad máxima.
    """
    import numpy as np
    proba = model.predict_proba(X)
    est = final_estimator(model)
    score = proba * est.costs_[None, :] if isinstance(est, DirectCSKNN) else proba
    return proba, est.classes_[np.argmax(score, axis=1)]
