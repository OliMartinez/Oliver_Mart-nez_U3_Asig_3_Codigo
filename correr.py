"""Ejecuta los experimentos de la comparativa de 13 clasificadores, registra una bitácora y, opcionalmente,
envía notificaciones con ntfy. Se ejecuta desde la raíz del proyecto, con el entorno activado
(conda activate abp-arboles).

FLUJO COLAB + ANACONDA
    python correr.py --prueba                 # prueba rápida (~5-10 min) en results_prueba/, no modifica results/
    python correr.py --parte 1                # sin resultados de Colab: 10 modelos de la ampliación, separabilidad,
                                              #   justificación del k-fold y prueba de tiempos (~5-7 h)
    python correr.py --parte 1 --remuestreo-3 # ídem + línea base de Credit Card en 3 pliegues (sólo si en
                                              #   Colab el remuestreo se redujo a 3 pliegues)
    python correr.py --parte 2                # con los resultados de Colab en results/ (~15-25 min)
    python correr.py --parte 2 --recalcular   # recalcula CART, RF y GB en todos los conjuntos

OTROS MODOS
    python correr.py --modo desde-cero        # estudio base y comparativa, todo en este equipo
    python correr.py --modo con-colab         # comparativa completa en una sola ejecución (requiere Colab terminado)

Si la ejecución se interrumpe, basta con repetir el mismo comando: cada paso guarda fila por fila y continúa.
La salida en pantalla se guarda también en results/bitacora.log (hora de inicio y fin, duración y estado de
cada paso).
"""
import argparse
import os
import platform
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

TEMA = "abp-arboles-cambia-este-nombre"   # tema de ntfy para notificaciones; con este valor no se envían

BASE = [["src.experiments", "benchmark_small"], ["src.experiments", "rotation"], ["src.experiments", "depth"],
          ["src.experiments", "ccp"], ["src.experiments", "correlation"], ["src.experiments", "distribution"],
          ["src.experiments", "scaling"], ["src.experiments", "dimensionality"], ["src.experiments", "big"],
          ["src.extras", "regularization"], ["src.extras", "thresholds"], ["src.extras", "resampling"],
          ["src.extras", "nested"], ["src.extras", "significance"], ["src.visualization"], ["src.make_tables"]]
COMPARATIVA = [["src.comparativa", "benchmark"], ["src.comparativa", "separability"],
               ["src.comparativa", "cv_schemes"], ["src.comparativa", "timing"],
               ["src.comparativa", "significance"], ["src.comparativa", "summary"], ["src.visualization_comp"]]
# Parte 1: no depende de Colab. La prueba de tiempos va al final y sola (sin otros procesos compitiendo).
PARTE1 = [["src.comparativa", "benchmark_new"], ["src.comparativa", "separability"], ["src.comparativa", "cv_schemes"]]
TIEMPOS = [["src.comparativa", "timing"]]
# Parte 2: necesita los resultados del estudio base (Colab) en results/.
PARTE2_FIN = [["src.comparativa", "significance"], ["src.comparativa", "summary"], ["src.visualization_comp"]]

ap = argparse.ArgumentParser(description="Experimentos de la comparativa de 13 clasificadores")
ap.add_argument("--parte", choices=["1", "2"], help="flujo Colab + Anaconda")
ap.add_argument("--modo", choices=["desde-cero", "con-colab"])
ap.add_argument("--prueba", action="store_true", help="prueba rápida con 3 conjuntos pequeños")
ap.add_argument("--recalcular", action="store_true",
                help="recalcula CART, RF y GB en vez de importarlos del estudio base")
ap.add_argument("--remuestreo-3", action="store_true", dest="remuestreo3",
                help="(parte 1) línea base de Credit Card en 3 pliegues, si en Colab el remuestreo se redujo a 3")
a = ap.parse_args()
env = dict(os.environ, PYTHONUTF8="1", PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")

if a.prueba:
    env.update(ABP_SMOKE="1", ABP_RESULTS_DIR="results_prueba")
    pasos = PARTE1 + TIEMPOS + [["src.comparativa", "benchmark_full"]] + PARTE2_FIN
    etiqueta = "PRUEBA RÁPIDA"
elif a.parte == "1":
    pasos = PARTE1 + ([["src.comparativa", "resampling_baseline3"]] if a.remuestreo3 else []) + TIEMPOS
    etiqueta = "parte 1 (sin Colab)"
elif a.parte == "2":
    pasos = ([] if a.recalcular else [["src.comparativa", "verify_folds"]]) + \
            [["src.comparativa", "benchmark_full" if a.recalcular else "benchmark"]] + PARTE2_FIN
    etiqueta = "parte 2 (con resultados de Colab)" + (" - recalculando CART, RF y GB" if a.recalcular else "")
elif a.modo == "desde-cero":
    pasos = BASE + COMPARATIVA
    etiqueta = "desde cero"
elif a.modo == "con-colab":
    pasos = ([] if a.recalcular else [["src.comparativa", "verify_folds"]]) + \
            [["src.comparativa", "benchmark_full" if a.recalcular else "benchmark"]] + COMPARATIVA[1:]
    etiqueta = "con resultados de Colab"
else:
    ap.error("indique --parte 1, --parte 2, --modo desde-cero, --modo con-colab o --prueba")

RES = Path(env.get("ABP_RESULTS_DIR", "results"))
RES.mkdir(parents=True, exist_ok=True)
LOG = open(RES / "bitacora.log", "a", encoding="utf-8")


def ahora():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def log(msg, consola=True):
    if consola:
        print(msg, flush=True)
    LOG.write(msg + "\n"); LOG.flush()


def avisar(msg):
    log(f"[{ahora()}] {msg}")
    if "cambia-este-nombre" in TEMA:
        return
    try:
        urllib.request.urlopen(urllib.request.Request(f"https://ntfy.sh/{TEMA}", data=msg.encode("utf-8")), timeout=10)
    except Exception as e:
        print("  (no se pudo enviar la notificación:", e, ")")


def correr(paso):
    """Ejecuta un paso en un proceso aparte; su salida va a la pantalla y a la bitácora, línea por línea."""
    p = subprocess.Popen([sys.executable, "-m", *paso], env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         text=True, encoding="utf-8", errors="replace", bufsize=1)
    for linea in p.stdout:
        log(linea.rstrip("\n"))
    return p.wait()


log("\n" + "=" * 78)
log(f"[{ahora()}] INICIO: {etiqueta}  |  comando: python {' '.join(sys.argv)}")
log(f"  equipo: {platform.node()} | {platform.platform()} | Python {platform.python_version()} | "
    f"entorno: {os.environ.get('CONDA_DEFAULT_ENV', '(no es conda)')} | hilos lógicos: {os.cpu_count()} | "
    f"procesos en paralelo: {env.get('ABP_N_WORKERS', 'automático (la mitad de los hilos)')}")
log("  pasos: " + " -> ".join(" ".join(p[1:] or p) for p in pasos))
log("=" * 78)
avisar(f"Inicio de los experimentos ({etiqueta}): {len(pasos)} pasos")
inicio = time.time()
for i, p in enumerate(pasos, 1):
    nombre = " ".join(p)
    t0 = time.time()
    log(f"\n[{ahora()}] >>> PASO {i}/{len(pasos)}: {nombre}")
    code = correr(p)
    mins = (time.time() - t0) / 60
    estado = "OK" if code == 0 else f"ERROR (código {code})"
    log(f"[{ahora()}] <<< FIN PASO {i}/{len(pasos)}: {nombre} | {mins:.1f} min | {estado}")
    if code == 2 and p == ["src.comparativa", "verify_folds"]:
        avisar("AVISO: algunos conjuntos tienen pliegues distintos a los de Colab (ver "
               "results/comp_verificacion_pliegues.csv): CART, RF y GB se recalculan sólo en esos conjuntos; "
               "en los demás se importan de Colab.")
        continue
    if code != 0:
        if p == ["src.comparativa", "verify_folds"]:
            avisar("AVISO: no se pudieron verificar los pliegues. Ver results/bitacora.log o usar: "
                   "python correr.py --parte 2 --recalcular")
        else:
            avisar(f"ERROR: falló {nombre} tras {mins:.0f} min. Detalle en results/bitacora.log; "
                   "al repetir el mismo comando se continúa desde el último resultado guardado.")
        LOG.close()
        sys.exit(1)
    avisar(f"[{i}/{len(pasos)}] {nombre} terminó ({mins:.0f} min)")
avisar(f"Terminaron todos los experimentos ({etiqueta}, {(time.time() - inicio) / 3600:.1f} h).")
if a.prueba:
    log("\nPrueba rápida correcta. Resultados en results_prueba/ (carpeta desechable).")
elif a.parte == "1":
    log("\nParte 1 terminada. Siguiente paso, con los resultados de Colab en results/: "
        "python verificar.py --parte 2")
elif a.parte == "2":
    log("\nResultados: tablas en results/ (comp_*.csv) y figuras en figures/ (fig_comp_*.png).")
LOG.close()
