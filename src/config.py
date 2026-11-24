"""Configuración común del proyecto: rutas, semilla y nombres de columnas.

Todo el grupo importa de aquí para que los resultados sean reproducibles.
"""
from pathlib import Path

# Carpeta raíz del repositorio (la que contiene src/, data/, notebooks/...)
ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
SUBMISSIONS = ROOT / "submissions"

# Reproducibilidad y validación
RANDOM_STATE = 42
N_FOLDS = 5

# Columnas
ID_COL = "id"
TARGET = "price"
CATEGORICAL = ["category", "brand", "brand_tier", "material", "season", "channel", "country"]
NUMERIC = ["weight_kg", "rating", "reviews", "shipping_cost", "margin", "discount", "is_promo"]

# Formato de la submission. submission_ejemplo.csv usa ";" pero la web de Streamlit lee con ",".
SUBMISSION_SEP = ","
