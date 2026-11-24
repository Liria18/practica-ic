"""Mide el modelo de src/modelo.py. Es la prueba que decide si un modelo es mejor.

Uso (desde la raíz del repositorio, con el .venv activado):
    python -m src.evaluar

No cambies este fichero ni src/config.py (semilla, folds) en un PR de modelo:
así todos medimos igual.
"""
import pandas as pd

from src.config import DATA_RAW
from src.modelo import AUTOR, DESCRIPCION, build_final_model
from src.models import evaluar_cv
from src.preprocessing import add_features, split_xy


def main():
    train = pd.read_csv(DATA_RAW / "train.csv")
    X, y = split_xy(train)
    X = add_features(X)
    por_fold, _ = evaluar_cv(build_final_model(), X, y)
    print(f"Modelo: {DESCRIPCION} (autor: {AUTOR})")
    print("RMSE por fold:", " ".join(f"{v:.3f}" for v in por_fold))
    print(f"RMSE CV (5 folds): {por_fold.mean():.3f} ± {por_fold.std():.3f}")
    return por_fold.mean()


if __name__ == "__main__":
    main()
