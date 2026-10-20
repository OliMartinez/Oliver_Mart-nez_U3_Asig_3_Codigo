"""Comprueba que el proyecto esté listo para ejecutar la comparativa (se ejecuta desde la raíz del proyecto).

    python verificar.py --parte 1          # antes de la parte 1 (flujo Colab + Anaconda)
    python verificar.py --parte 2          # antes de la parte 2 (Colab terminado y resultados copiados)
    python verificar.py --modo desde-cero
    python verificar.py --modo con-colab
"""
import argparse
import importlib
import os
import sys
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--modo", choices=["desde-cero", "con-colab"])
ap.add_argument("--parte", choices=["1", "2"])
args = ap.parse_args()
if not (args.modo or args.parte):
    ap.error("indique --parte 1, --parte 2, --modo desde-cero o --modo con-colab")
modo = args.modo or ("parte1" if args.parte == "1" else "parte2")
ok = True


def mal(msg):
    global ok
    ok = False
    print("  [X] ", msg)


def bien(msg):
    print("  [OK]", msg)


print("1) Carpeta del proyecto")
faltan = [d for d in ("src", "results") if not Path(d).is_dir()]
if faltan:
    mal("No se encuentra: " + ", ".join(faltan) + " (el script debe ejecutarse desde la raíz de ABP_Arboles)")
else:
    bien(f"Raíz del proyecto: {Path.cwd()}")
for f in ("src/comparativa.py", "src/models.py", "src/knn_family.py", "src/chaid.py", "src/c50.py", "src/config.py",
          "src/visualization_comp.py"):
    if not Path(f).exists():
        mal(f"Falta {f}")

print("2) Librerías de Python")
for lib in ("numpy", "scipy", "pandas", "sklearn", "matplotlib", "imblearn", "threadpoolctl", "psutil"):
    try:
        importlib.import_module(lib)
    except ImportError:
        mal(f"Falta la librería {lib} (activar el entorno: conda activate abp-arboles)")
pref = os.environ.get("CONDA_PREFIX")
if pref and Path(sys.prefix).resolve() != Path(pref).resolve():
    mal(f"El intérprete en uso ({sys.prefix}) no es el del entorno activo ({pref}): el entorno está incompleto. "
        "Recrearlo con: conda env remove -n abp-arboles; conda env create -f environment.yml")
elif sys.version_info[:2] != (3, 11):
    mal(f"Se está usando Python {sys.version.split()[0]}, pero el entorno abp-arboles usa Python 3.11 "
        "(activar el entorno: conda activate abp-arboles)")
if ok:
    bien(f"Python {sys.version.split()[0]} | entorno: {os.environ.get('CONDA_DEFAULT_ENV', '(no es conda)')}")

print("3) R y el paquete C50 (para C5.0)")
try:
    sys.path.insert(0, str(Path.cwd()))
    from src.c50 import check_c50
    c_ok, msg = check_c50()
    (bien if c_ok else mal)(msg)
except Exception as e:
    mal(f"No se pudo comprobar C5.0: {e}")

print("4) Datos")
datos = ["glass.csv", "bcw.csv", "forestfires.csv", "concrete.csv", "creditcard.csv", "har/X_train.txt",
         "har/X_test.txt", "har/y_train.txt", "har/y_test.txt", "har/subject_train.txt", "har/subject_test.txt"]
faltan = [f for f in datos if not (Path("data") / f).exists()]
if faltan:
    mal("Faltan datos: " + ", ".join(faltan) + "  ->  python descargar_datos.py")
else:
    bien("Todos los datos están en data/")

R = Path("results")
base_files = ["benchmark_small.csv", "benchmark_big.csv"]
print("5) Resultados")
if modo == "desde-cero":
    viejos = sorted(p.name for p in R.glob("*.csv"))
    if viejos:
        mal(f"results/ tiene {len(viejos)} archivos de corridas anteriores; deben moverse a otra carpeta "
            "(por ejemplo results_previos/) para empezar desde cero.")
    else:
        bien("results/ está vacía: se ejecutará todo desde el inicio")
elif modo == "parte1":
    bien("La parte 1 no necesita los resultados de Colab (puede ejecutarse en paralelo con Colab)")
    if (R / "benchmark_comp.csv").exists():
        print("  · Existe un benchmark_comp.csv previo: la ejecución continuará desde el último resultado")
else:
    import pandas as pd
    if "ABP_RESULTS_DIR" not in Path("src/experiments.py").read_text(encoding="utf-8"):
        mal("src/experiments.py corresponde a la copia de Colab; de Colab sólo deben copiarse results/ y figures/")
    for f in base_files:
        if not (R / f).exists():
            mal(f"Falta results/{f} de la corrida de Colab (copiar ahí el contenido de results/ de Google Drive)")
    if (R / "benchmark_big.csv").exists():
        b = pd.read_csv(R / "benchmark_big.csv")
        base = ["Árbol de Decisión", "Árbol podado (ccp)", "Random Forest", "Gradient Boosting"]
        for name, folds, n in (("Credit Card", 10, 284807), ("HAR", 5, None)):
            s = b[(b.dataset == name) & b.model.isin(base)]
            got = s.groupby("model").fold.nunique().reindex(base).fillna(0).astype(int)
            if (got < folds).any():
                mal(f"{name}: la corrida de Colab está incompleta (pliegues por modelo: "
                    f"{ {k: int(v) for k, v in got.items()} }; se esperaban {folds})")
            elif n and int(s.n.iloc[0]) != n:
                mal(f"{name}: la corrida de Colab usó {int(s.n.iloc[0])} filas (submuestra); se esperaban {n}")
            else:
                bien(f"{name}: {folds} pliegues completos del estudio base")
    if (R / "benchmark_small.csv").exists():
        s = pd.read_csv(R / "benchmark_small.csv")
        if s.dataset.nunique() < 12 or s.groupby(["dataset", "model"]).fold.nunique().min() < 10:
            mal("benchmark_small.csv está incompleto (se esperaban 12 conjuntos x 4 modelos x 10 pliegues)")
        else:
            bien("Conjuntos pequeños: 12 x 4 modelos x 10 pliegues")
    if modo == "parte2":
        print("6) Resultados de la parte 1 (este equipo)")
        from src import datasets as D
        from src.models import MODEL_ORDER, BASE_EQUIV
        extra = [m for m in MODEL_ORDER if m not in BASE_EQUIV]
        esperados = {n: (5 if n == "HAR" else 10) for n in {**D.REAL, **D.SYNTH}}
        f = R / "benchmark_comp.csv"
        if not f.exists():
            mal("Falta results/benchmark_comp.csv; primero debe ejecutarse: python correr.py --parte 1")
        else:
            bv = pd.read_csv(f)
            hechos = set(zip(bv.dataset, bv.model, bv.fold.astype(int)))
            faltan = [(n, m) for n, k in esperados.items() for m in extra
                      if any((n, m, i) not in hechos for i in range(k))]
            if faltan:
                mal(f"benchmark_comp.csv incompleto: {len(faltan)} combinaciones conjunto-modelo sin todos sus "
                    f"pliegues (p. ej. {faltan[:3]}). Falta terminar la parte 1: python correr.py --parte 1")
            else:
                bien(f"Comparativa de los {len(extra)} modelos de la ampliación: {len(esperados)} conjuntos completos")
        for archivo, que in (("comp_separabilidad.csv", "separabilidad lineal"), ("comp_cv_esquemas.csv",
                             "justificación del k-fold"), ("comp_tiempos.csv", "prueba de tiempos")):
            if (R / archivo).exists():
                bien(f"{que}: results/{archivo}")
            else:
                mal(f"Falta results/{archivo} ({que}). Falta terminar la parte 1: python correr.py --parte 1")
    elif (R / "benchmark_comp.csv").exists():
        print("  · Existe un benchmark_comp.csv previo: la ejecución continuará desde el último resultado "
              "(borrarlo para empezar de cero)")

cmd = f"--parte {args.parte}" if args.parte else f"--modo {modo}"
print("\n" + (f"Verificación correcta. Siguiente paso: python correr.py {cmd}" if ok else
              "Hay puntos marcados con [X]; deben corregirse antes de volver a ejecutar este script."))
sys.exit(0 if ok else 1)
