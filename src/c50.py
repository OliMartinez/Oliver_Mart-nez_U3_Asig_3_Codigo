"""
c50.py
======
C5.0 (Quinlan) a través del paquete oficial `C50` de R (Kuhn y Quinlan), envuelto como clasificador
compatible con scikit-learn. No existe una implementación fiel de C5.0 en Python, por eso se llama a R.

Funcionamiento:
  * fit():  escribe X (float64 binario) e y (int32) en una carpeta temporal y ejecuta
            `Rscript c50_runner.R fit ...`, que entrena C5.0() y guarda el modelo (.rds).
  * predict_proba(): ejecuta `Rscript c50_runner.R predict ...` con el modelo guardado.
  * Los tiempos de entrenamiento y predicción se miden dentro de R (proc.time), sin contar la
    escritura de archivos ni el arranque de R: atributos fit_time_r_, fit_cpu_r_, predict_time_r_, predict_cpu_r_.

Requisitos: R y el paquete C50 (conda install -c conda-forge r-base r-c50, o install.packages("C50")).
Si Rscript no está en el PATH, defina la variable de entorno RSCRIPT con su ruta completa.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

R_SCRIPT = r"""
args <- commandArgs(trailingOnly = TRUE)
mode <- args[1]
suppressPackageStartupMessages(library(C50))
readX <- function(path, p) {
  v <- readBin(path, "double", n = file.info(path)$size / 8)
  df <- as.data.frame(matrix(v, ncol = p, byrow = TRUE))
  names(df) <- paste0("f", seq_len(p))
  df
}
if (mode == "fit") {
  p <- as.integer(args[2]); X <- readX(args[3], p)
  yv <- readBin(args[4], "integer", n = file.info(args[4])$size / 4)
  y <- factor(paste0("c", yv), levels = paste0("c", sort(unique(yv))))
  t0 <- proc.time()
  m <- C5.0(x = X, y = y, trials = as.integer(args[6]),
            control = C5.0Control(CF = as.numeric(args[7]), minCases = as.integer(args[8])))
  dt <- proc.time() - t0
  saveRDS(m, args[5])
  cat(sprintf("%.6f %.6f %d\n", dt[["elapsed"]], dt[["user.self"]] + dt[["sys.self"]], as.integer(m$size[1])),
      file = args[9])
} else if (mode == "predict") {
  p <- as.integer(args[2]); m <- readRDS(args[3]); X <- readX(args[4], p)
  t0 <- proc.time()
  pr <- predict(m, newdata = X, type = "prob")
  dt <- proc.time() - t0
  write.table(pr, args[5], sep = ",", row.names = FALSE, col.names = TRUE, quote = FALSE)
  cat(sprintf("%.6f %.6f\n", dt[["elapsed"]], dt[["user.self"]] + dt[["sys.self"]]), file = args[6])
} else if (mode == "check") {
  cat("C50", as.character(packageVersion("C50")), "\n")
}
"""


def find_rscript() -> str | None:
    """Busca Rscript: variable RSCRIPT, PATH y ubicaciones típicas de un entorno conda."""
    env = os.environ.get("RSCRIPT")
    if env and Path(env).exists():
        return env
    found = shutil.which("Rscript")
    if found:
        return found
    prefix = Path(sys.prefix)
    for cand in (prefix / "Lib/R/bin/x64/Rscript.exe", prefix / "Lib/R/bin/Rscript.exe",
                 prefix / "Scripts/Rscript.exe", prefix / "bin/Rscript",
                 Path("C:/Program Files/R").glob("R-*/bin/x64/Rscript.exe") if os.name == "nt" else []):
        for c in (cand if not isinstance(cand, Path) else [cand]):
            if Path(c).exists():
                return str(c)
    return None


def check_c50() -> tuple[bool, str]:
    """Comprueba que R y el paquete C50 estén disponibles. Devuelve (ok, mensaje)."""
    rs = find_rscript()
    if rs is None:
        return False, "No se encontró Rscript (instale R o defina la variable RSCRIPT)."
    with tempfile.TemporaryDirectory() as tmp:
        script = Path(tmp) / "c50_runner.R"
        script.write_text(R_SCRIPT, encoding="utf-8")
        r = subprocess.run([rs, str(script), "check"], capture_output=True, text=True)
    if r.returncode != 0:
        return False, "R está instalado pero falta el paquete C50: " + (r.stderr.strip().splitlines() or [""])[-1]
    return True, r.stdout.strip() + f" (Rscript: {rs})"


class C50Classifier(BaseEstimator, ClassifierMixin):
    """Árbol C5.0 (trials=1: un solo árbol, sin boosting). CF = factor de confianza de la poda."""

    def __init__(self, trials=1, cf=0.25, min_cases=2):
        self.trials, self.cf, self.min_cases = trials, cf, min_cases

    def _run(self, args):
        r = subprocess.run([self.rscript_, str(self.script_)] + [str(a) for a in args], capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError("Error en C5.0 (R):\n" + r.stderr[-2000:])

    def fit(self, X, y):
        X = np.ascontiguousarray(np.asarray(X, dtype=np.float64))
        self.classes_, yi = np.unique(np.asarray(y), return_inverse=True)
        self.rscript_ = find_rscript()
        if self.rscript_ is None:
            raise RuntimeError("No se encontró Rscript; C5.0 requiere R con el paquete C50.")
        self.tmp_ = tempfile.mkdtemp(prefix="c50_")
        self.script_ = Path(self.tmp_) / "c50_runner.R"
        self.script_.write_text(R_SCRIPT, encoding="utf-8")
        self.p_ = X.shape[1]
        fx, fy = Path(self.tmp_) / "Xtr.bin", Path(self.tmp_) / "ytr.bin"
        X.tofile(fx); yi.astype(np.int32).tofile(fy)
        self.model_path_, ft = Path(self.tmp_) / "model.rds", Path(self.tmp_) / "fit_time.txt"
        self._run(["fit", self.p_, fx, fy, self.model_path_, self.trials, self.cf, self.min_cases, ft])
        vals = ft.read_text().split()
        self.fit_time_r_, self.fit_cpu_r_, self.tree_size_ = float(vals[0]), float(vals[1]), int(vals[2])
        fx.unlink(); fy.unlink()
        return self

    def predict_proba(self, X):
        X = np.ascontiguousarray(np.asarray(X, dtype=np.float64))
        fx, fo, ft = Path(self.tmp_) / "Xte.bin", Path(self.tmp_) / "prob.csv", Path(self.tmp_) / "pred_time.txt"
        X.tofile(fx)
        self._run(["predict", self.p_, self.model_path_, fx, fo, ft])
        vals = ft.read_text().split()
        self.predict_time_r_, self.predict_cpu_r_ = float(vals[0]), float(vals[1])
        with open(fo, encoding="utf-8") as fh:
            header = fh.readline().strip().split(",")
        P = np.loadtxt(fo, delimiter=",", skiprows=1, ndmin=2)
        out = np.zeros((len(X), len(self.classes_)))
        for col, name in enumerate(header):
            out[:, int(name.strip('"')[1:])] = P[:, col]
        return out

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]

    def __del__(self):
        tmp = getattr(self, "tmp_", None)
        if tmp:
            shutil.rmtree(tmp, ignore_errors=True)
