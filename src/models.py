"""Catálogo de modelos y utilidades de validación cruzada.

Todos los modelos son Pipelines completos (preprocesado + estimador), así el
preprocesado se ajusta solo con los datos de entrenamiento de cada fold y no
hay fuga de información (data leakage).
"""
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import Pipeline

from src.config import N_FOLDS, RANDOM_STATE
from src.preprocessing import build_preprocessor


class LogTargetRegressor(BaseEstimator, RegressorMixin):
    """Entrena el modelo sobre log(price) y devuelve predicciones en euros.

    Al deshacer el logaritmo con exp() se predice la mediana y no la media, y
    eso infravalora el precio. Se corrige con el factor de "smearing" de Duan:
    media de exp(residuos) en entrenamiento.
    """

    def __init__(self, regressor=None, smearing=True):
        self.regressor = regressor
        self.smearing = smearing

    def fit(self, X, y):
        y_log = np.log(np.asarray(y, dtype=float))
        self.regressor_ = clone(self.regressor).fit(X, y_log)
        if self.smearing:
            residuos = y_log - self.regressor_.predict(X)
            self.factor_ = float(np.mean(np.exp(residuos)))
        else:
            self.factor_ = 1.0
        return self

    def predict(self, X):
        return np.exp(self.regressor_.predict(X)) * self.factor_


class MezclaRegressor(BaseEstimator, RegressorMixin):
    """Media ponderada de dos modelos: peso*modelo_a + (1-peso)*modelo_b."""

    def __init__(self, modelo_a=None, modelo_b=None, peso=0.5):
        self.modelo_a = modelo_a
        self.modelo_b = modelo_b
        self.peso = peso

    def fit(self, X, y):
        self.a_ = clone(self.modelo_a).fit(X, y)
        self.b_ = clone(self.modelo_b).fit(X, y)
        return self

    def predict(self, X):
        return self.peso * self.a_.predict(X) + (1 - self.peso) * self.b_.predict(X)


def ridge_log(alpha=1.0):
    """Ridge con one-hot y escalado, sobre log(price) con corrección de smearing."""
    return LogTargetRegressor(Pipeline([("prep", build_preprocessor()), ("model", Ridge(alpha=alpha))]))


def lgbm_log(**params):
    """LightGBM sobre log(price). Los árboles no necesitan escalado ni one-hot."""
    base = dict(
        n_estimators=800, learning_rate=0.02, num_leaves=7, min_child_samples=40,
        subsample=0.8, subsample_freq=1, colsample_bytree=0.7,
        random_state=RANDOM_STATE, verbose=-1, n_jobs=1,
    )
    base.update(params)
    return LogTargetRegressor(
        Pipeline([("prep", build_preprocessor(escalar=False, onehot=False)), ("model", LGBMRegressor(**base))])
    )


def get_models() -> dict:
    """Modelos candidatos que se comparan en notebooks/03_modelos.ipynb."""
    arbol = build_preprocessor(escalar=False, onehot=False)
    return {
        "LinearRegression": Pipeline([("prep", build_preprocessor()), ("model", LinearRegression())]),
        "Ridge": Pipeline([("prep", build_preprocessor()), ("model", Ridge(alpha=1.0))]),
        "Ridge (log price)": ridge_log(),
        "RandomForest": Pipeline([("prep", arbol), ("model", RandomForestRegressor(
            n_estimators=300, min_samples_leaf=3, random_state=RANDOM_STATE, n_jobs=1))]),
        "HistGradientBoosting": Pipeline([("prep", arbol), ("model", HistGradientBoostingRegressor(
            random_state=RANDOM_STATE))]),
        "LightGBM (log price)": lgbm_log(),
    }


def get_cv() -> KFold:
    """KFold de 5 particiones con barajado y semilla fija (igual para todos)."""
    return KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def evaluar_cv(modelo, X, y, cv=None):
    """Devuelve (rmse_por_fold, predicciones_oof) con validación cruzada."""
    cv = cv or get_cv()
    oof = cross_val_predict(modelo, X, y, cv=cv)
    por_fold = [rmse(y.iloc[val], oof[val]) for _, val in cv.split(X)]
    return np.array(por_fold), oof


def comparar_modelos(modelos: dict, X, y) -> pd.DataFrame:
    """Tabla con RMSE medio y desviación en CV para cada modelo."""
    filas = []
    for nombre, modelo in modelos.items():
        por_fold, _ = evaluar_cv(modelo, X, y)
        filas.append({"modelo": nombre, "rmse_cv": por_fold.mean(), "std": por_fold.std()})
        print(f"{nombre:<25} RMSE CV = {por_fold.mean():.3f} ± {por_fold.std():.3f}")
    return pd.DataFrame(filas).sort_values("rmse_cv").reset_index(drop=True)
