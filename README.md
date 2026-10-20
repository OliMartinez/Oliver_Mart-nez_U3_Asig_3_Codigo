# Evaluación comparativa de árboles de decisión, ensambles y clasificadores basados en distancia en fronteras ortogonales y no ortogonales

Random Forest, Gradient Boosting, Mínima Distancia y variantes de KNN en fronteras de decisión ortogonales y no ortogonales.

Oliver de Jesús Martínez Pérez · Maestría en Ciencias de la Computación · TecNM / IT de León · Reconocimiento de Patrones

## Contenido de la comparativa
- **13 clasificadores** sobre los mismos pliegues: Árbol de Decisión (entropía), CART (Gini + poda), C5.0 (R), CHAID,
  Random Forest, Gradient Boosting, Mínima Distancia y seis variantes de KNN (KNN, CV-KNN, Direct-CS-KNN,
  Distances-CS-KNN, CM-KNN y One-Step KNN).
- **Sobreajuste** (brecha entrenamiento − prueba) de los 13 modelos.
- **Separabilidad lineal** exacta por programación lineal.
- **Justificación del k-fold estratificado**: holdout 90/10 repetido vs. 10-fold estratificado, sin estratificar y 5-fold.
- **Prueba de tiempos** controlada en un solo equipo (tiempo real y de CPU) dentro de un entorno de Anaconda.
- **Estadística para 13 modelos**: Friedman, Nemenyi y comparación contra el mejor con corrección de Holm.
- **Características de los árboles completos** (DT, CART, C5.0 y CHAID): hojas, profundidad y atributo de la raíz en los 14 conjuntos.
- **Condiciones de sobreajuste (E12)**: pocos datos, desbalance, instancias atípicas y datos faltantes en BCW.

## Archivos de la comparativa
| Archivo | Contenido |
|---|---|
| `src/config.py` | Decisiones de diseño y parámetros. |
| `src/models.py` | Catálogo de los 13 modelos y su preprocesamiento. |
| `src/knn_family.py` | Mínima Distancia y variantes de KNN (KNN, CV-KNN, Direct-CS-KNN, Distances-CS-KNN, CM-KNN, One-Step KNN), multiclase. |
| `src/chaid.py` | CHAID implementado desde cero. |
| `src/c50.py` | C5.0 mediante el paquete C50 de R. |
| `src/comparativa.py` | Experimentos de la comparativa. |
| `src/visualization_comp.py` | Figuras de resultados de la comparativa (`fig_comp_*`). |
| `src/ilustraciones.py` | Figuras ilustrativas del marco teórico (`fig_ilus_*`). |
| `src/arboles_completos.py` | Árboles completos de los 14 conjuntos (`results/arboles_completos.csv`) y figura del CART de BCW (`fig_arbol_cart_bcw`). |
| `src/sobreajuste.py` | E12, condiciones de sobreajuste (`results/sobreajuste_condiciones.csv`, `fig_sobreajuste_condiciones`). |
| `environment.yml` | Entorno de Anaconda (incluye R y C50). |
| `descargar_datos.py`, `verificar.py`, `correr.py` | Descarga de datos, verificación del entorno y ejecución con bitácora. |

El reporte técnico (fuentes LaTeX) se entrega por separado en `Reporte_fuentes_LaTeX.zip`.

Los módulos `src/experiments.py`, `src/extras.py`, `src/visualization.py` y `src/make_tables.py` forman el estudio
base de la guía (DT, DT podado, RF y GB); la comparativa reutiliza sus resultados de CART, RF y GB.

## Uso — flujo Colab + Anaconda
```bash
conda env create -f environment.yml
conda activate abp-arboles
python descargar_datos.py
python correr.py --prueba                 # prueba de ~5-10 minutos

# PARTE 1 (no requiere los resultados de Colab): ~5-7 h
python verificar.py --parte 1
python correr.py --parte 1                # agregar --remuestreo-3 si en Colab el remuestreo se redujo a 3 pliegues

# PARTE 2 (con results/ y figures/ de Colab copiados aquí): ~15-25 min
python verificar.py --parte 2
python correr.py --parte 2                # o --parte 2 --recalcular si los pliegues no coinciden

# Figuras del marco teórico
python -m src.ilustraciones figures

# Árboles completos y condiciones de sobreajuste (E12)
python -m src.arboles_completos
python -m src.sobreajuste
```
La salida en pantalla queda en `results/bitacora.log` (inicio, fin, duración y estado de cada paso).
Otros modos: `--modo desde-cero` (todo en este equipo) y `--modo con-colab` (la comparativa completa en una sola ejecución).
