"""Preprocesado y feature engineering común a todo el grupo.

- add_features(df): crea variables nuevas a partir de las originales.
- build_preprocessor(...): ColumnTransformer que imputa, escala y codifica.

Las decisiones están justificadas en notebooks/02_preprocesado.ipynb.
"""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, SplineTransformer, StandardScaler

from src.config import CATEGORICAL, ID_COL, NUMERIC, TARGET

TIER_ORDEN = {"Budget": 0, "Media": 1, "Premium": 2}

NUEVAS_NUMERICAS = [
    "log_weight",
    "log_reviews",
    "tier_num",
    "margen_neto",
    "factor_descuento",
    "envio_por_kg",
    "material_missing",
    "log_shipping",
    "log_factor_descuento",
    "log_factor_margen",
]


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Devuelve una copia de df con variables derivadas.

    Ideas que salen del EDA:
    - weight_kg y reviews tienen colas largas: su logaritmo se relaciona mejor con el precio.
    - brand_tier tiene orden natural (Budget < Media < Premium).
    - margin y discount actúan de forma multiplicativa sobre el precio.
    - material tiene valores perdidos: guardamos un indicador por si aporta información.
    """
    out = df.copy()
    out["log_weight"] = np.log1p(out["weight_kg"])
    out["log_reviews"] = np.log1p(out["reviews"])
    out["tier_num"] = out["brand_tier"].map(TIER_ORDEN)
    out["margen_neto"] = out["margin"] - out["discount"]
    out["factor_descuento"] = 1 - out["discount"]
    out["envio_por_kg"] = out["shipping_cost"] / (out["weight_kg"] + 0.1)
    out["material_missing"] = out["material"].isna().astype(int)
    # Versiones logarítmicas: si el precio se forma multiplicando factores,
    # en escala log esos factores se suman y un modelo lineal los captura bien.
    out["log_shipping"] = np.log1p(out["shipping_cost"])
    out["log_factor_descuento"] = np.log(1 - out["discount"])
    out["log_factor_margen"] = np.log1p(out["margin"])
    # log(peso) sin el +1: en log-log la relación peso-precio es casi una recta.
    # Solo la usa el modelo de splines (no está en NUEVAS_NUMERICAS).
    out["log_peso"] = np.log(out["weight_kg"])
    return out


def split_xy(train: pd.DataFrame):
    """Separa variables predictoras y objetivo (sin la columna id)."""
    X = train.drop(columns=[ID_COL, TARGET])
    y = train[TARGET]
    return X, y


def build_preprocessor(escalar: bool = True, onehot: bool = True) -> ColumnTransformer:
    """Construye el ColumnTransformer.

    escalar=True  -> StandardScaler en numéricas (necesario para modelos lineales).
    onehot=True   -> One-hot en categóricas; si es False usa OrdinalEncoder (árboles).
    """
    numericas = NUMERIC + NUEVAS_NUMERICAS
    pasos_num = [("imputer", SimpleImputer(strategy="median"))]
    if escalar:
        pasos_num.append(("scaler", StandardScaler()))

    if onehot:
        encoder = OneHotEncoder(handle_unknown="ignore")
    else:
        encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    pasos_cat = [
        ("imputer", SimpleImputer(strategy="constant", fill_value="Desconocido")),
        ("encoder", encoder),
    ]
    return ColumnTransformer(
        [("num", Pipeline(pasos_num), numericas), ("cat", Pipeline(pasos_cat), CATEGORICAL)]
    )


SPLINE_NUMERICAS = ["log_peso", "rating", "log_reviews", "shipping_cost", "margin", "discount"]


def build_spline_preprocessor(n_knots: int = 5) -> ColumnTransformer:
    """One-hot en categóricas + B-splines cúbicos en las numéricas.

    Los splines dejan que un modelo lineal aprenda curvas (p. ej. el efecto del
    margen o del rating no es una recta), sin perder la estructura aditiva en log.
    """
    pasos_cat = [
        ("imputer", SimpleImputer(strategy="constant", fill_value="Desconocido")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ]
    return ColumnTransformer([
        ("cat", Pipeline(pasos_cat), CATEGORICAL),
        ("num", SplineTransformer(n_knots=n_knots, degree=3), SPLINE_NUMERICAS),
        ("bin", "passthrough", ["is_promo"]),
    ])
