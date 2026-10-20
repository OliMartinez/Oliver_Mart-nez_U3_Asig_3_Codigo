"""Descarga los datos del ABP en data/ (ejecutar desde la raíz del proyecto ABP_Arboles)."""
import glob, os, shutil, urllib.request, zipfile

os.makedirs("data/har", exist_ok=True)
gh = "https://raw.githubusercontent.com/"
archivos = {
    "glass.csv": gh + "jbrownlee/Datasets/master/glass.csv",
    "bcw.csv": gh + "jbrownlee/Datasets/master/breast-cancer-wisconsin.csv",
    "forestfires.csv": gh + "nthanhtin/forestfires/master/forestfires.csv",
    "concrete.csv": gh + "stedy/Machine-Learning-with-R-datasets/master/concrete.csv",
    "creditcard.csv": gh + "nsethi31/Kaggle-Data-Credit-Card-Fraud-Detection/master/creditcard.csv",
}
for nombre, url in archivos.items():
    print("Descargando", nombre, "...")
    urllib.request.urlretrieve(url, os.path.join("data", nombre))

# HAR desde el UCI (incluye los archivos de sujetos para la CV agrupada)
urls = ["https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip",
        "https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip"]
for u in urls:
    try:
        print("Descargando HAR ...")
        urllib.request.urlretrieve(u, "har.zip")
        break
    except Exception as e:
        print("Falló:", u, "->", e)
with zipfile.ZipFile("har.zip") as z:
    z.extractall("har_tmp")
for inner in glob.glob("har_tmp/*.zip"):          # el zip del UCI trae otro zip adentro
    with zipfile.ZipFile(inner) as z:
        z.extractall("har_tmp")
base = [p for p in glob.glob("har_tmp/**/UCI HAR Dataset", recursive=True) if "__MACOSX" not in p][0]
for part in ("train", "test"):
    for f in (f"X_{part}.txt", f"y_{part}.txt", f"subject_{part}.txt"):
        shutil.copy(os.path.join(base, part, f), os.path.join("data", "har", f))
shutil.rmtree("har_tmp")
os.remove("har.zip")
print("Listo.")
print("data/    :", sorted(os.listdir("data")))
print("data/har :", sorted(os.listdir(os.path.join("data", "har"))))
