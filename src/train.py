"""Entrena el modelo de src/modelo.py con todo train y genera la submission.

Uso (desde la raíz del repositorio, con el .venv activado):
    python -m src.train NombreAlumno

Guarda submissions/submission_NombreAlumno.csv con columnas id,prediction.
No hace falta tocar este fichero para cambiar de modelo: se cambia src/modelo.py.
"""
import sys

import pandas as pd

from src.config import DATA_RAW, ID_COL, SUBMISSION_SEP, SUBMISSIONS
from src.modelo import DESCRIPCION, build_final_model
from src.preprocessing import add_features, split_xy


def main(alumno: str = "Grupo"):
    train = pd.read_csv(DATA_RAW / "train.csv")
    test = pd.read_csv(DATA_RAW / "test.csv")
    X, y = split_xy(train)
    X = add_features(X)
    X_test = add_features(test.drop(columns=[ID_COL]))

    modelo = build_final_model()
    modelo.fit(X, y)
    pred = modelo.predict(X_test)

    SUBMISSIONS.mkdir(exist_ok=True)
    salida = SUBMISSIONS / f"submission_{alumno}.csv"
    sub = pd.DataFrame({"id": test[ID_COL], "prediction": pred})
    assert len(sub) == len(test) and sub["prediction"].notna().all()
    sub.to_csv(salida, index=False, sep=SUBMISSION_SEP)
    print(f"Modelo: {DESCRIPCION}")
    print(f"Guardado {salida} ({len(sub)} filas)")


if __name__ == "__main__":
    main(*sys.argv[1:])
