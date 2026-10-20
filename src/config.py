"""
config.py
=========
Decisiones y parámetros de la comparativa de 13 clasificadores. Todos los módulos de la comparativa
leen estos valores, por lo que cualquier cambio de diseño se hace sólo en este archivo.

Decisiones de diseño:
  1. "Árbol de Decisión" y "CART" son modelos distintos:
       - Árbol de Decisión: criterio de entropía (familia ID3/C4.5), sin poda.
       - CART: criterio de Gini con poda por complejidad de costo (alfa elegido por CV interna).
         Es el mismo modelo "Árbol podado (ccp)" del estudio base (experiments.py), por eso se reutilizan sus resultados.
  2. C5.0 se ejecuta con el paquete oficial C50 de R (no existe una versión fiel en Python).
  3. Direct-CS-KNN y Distances-CS-KNN se extienden a multiclase con costos inversos a la frecuencia
     de cada clase (w_c = n_max / n_c); en clases balanceadas equivalen a KNN.
  4. Se usan los 14 conjuntos. CM-KNN y One-Step KNN entrenan con una submuestra estratificada
     (CMKNN_N_MAX / ONESTEP_N_MAX) y CM-KNN se omite donde el pliegue de prueba supera CMKNN_MAX_TEST
     muestras (Credit Card), por su costo O(m * n^3).
"""
from __future__ import annotations

import os

SMOKE = os.environ.get("ABP_SMOKE") == "1"          # prueba rápida (python correr.py --prueba)

# ----------------------------------------------------------------------------- decisiones
DT_CRITERION = "entropy"          # criterio del "Árbol de Decisión" ("entropy" o "gini")
USE_C50 = True                    # False: se omite C5.0 (por ejemplo, si no se puede instalar R)
C50_TRIALS = 1                    # 1 = un solo árbol C5.0; >1 activa el boosting de C5.0
COST_MODE = "inverse_freq"        # costos de los CS-KNN: "inverse_freq" o "uniform"
CMKNN_N_MAX = 500                 # muestras de entrenamiento (submuestra estratificada) de CM-KNN
ONESTEP_N_MAX = 300               # ídem para One-Step KNN (entrenamiento O(n^4) por iteración)
CMKNN_MAX_TEST = 5000             # CM-KNN se omite si el pliegue de prueba es mayor (Credit Card)

# ----------------------------------------------------------------------------- evaluación
N_JOBS = -1                       # núcleos para RF y búsqueda de vecinos en la validación cruzada
# Procesos en paralelo (un pliegue por proceso) en la comparativa, la justificación del k-fold y la
# separabilidad. No cambia ningún resultado: cada pliegue es independiente y tiene semilla fija.
# Por defecto, la mitad de los procesadores lógicos (laptop de 8 hilos -> 4 procesos).
N_WORKERS = int(os.environ.get("ABP_N_WORKERS", max(1, (os.cpu_count() or 2) // 2)))
TRAIN_EVAL_MAX = 10000            # exactitud de entrenamiento (brecha) con submuestra si el entrenamiento es mayor
SLOW_TRAIN_EVAL_MAX = 300         # ídem para CM-KNN (su predicción es la parte costosa)
DATASETS = None                   # None = los 14 conjuntos; o una lista de nombres

# ----------------------------------------------------------------------------- justificación del k-fold
CV_SCHEME_MODELS = ["Árbol de Decisión (entropía)", "Random Forest", "KNN (k=5)", "Mínima Distancia"]
HOLDOUT_REPS = 30                 # repeticiones del holdout 90/10
CV_REPS = 10                      # repeticiones de cada esquema k-fold (con distinta semilla)
CV_SCHEME_DATASETS = None         # None = los 12 conjuntos pequeños

# ----------------------------------------------------------------------------- separabilidad lineal
LP_TIME_LIMIT = 600               # segundos máximos por programa lineal

# ----------------------------------------------------------------------------- prueba de tiempos
TIMING_THREADS = 1                # hilos de BLAS/OpenMP y n_jobs durante la medición (comparación justa)
TIMING_REPS = 3                   # repeticiones por modelo (se reporta media y desviación)
TIMING_REPS_LARGE = 1             # repeticiones en conjuntos con n > TIMING_LARGE_N (HAR y Credit Card)
TIMING_LARGE_N = 5000
TIMING_MEMORY = False              # medir memoria pico (tracemalloc) en una corrida aparte
TIMING_MEMORY_MAX_N = 50000       # sin medición de memoria si n es mayor (Credit Card: duplicaría el tiempo)
LARGE_N = 5000                    # Credit Card y HAR (mismo umbral que experiments.py): poda rápida de CART si FULL_PRUNING es False

if SMOKE:
    DATASETS = ["Iris", "Wine", "Moons"]
    CV_SCHEME_DATASETS = ["Iris", "Wine"]
    HOLDOUT_REPS, CV_REPS, TIMING_REPS = 3, 2, 1
    CMKNN_N_MAX, ONESTEP_N_MAX = 120, 120
    LP_TIME_LIMIT = 60
