"""EL MODELO ACTUAL DEL GRUPO.

Este es el único fichero que hay que cambiar para proponer un modelo mejor.
Reglas (ver README.md):
- build_final_model() debe devolver un estimador de scikit-learn (fit/predict)
  que reciba X ya con add_features() aplicado y prediga `price` en euros.
- Se mide con `python -m src.evaluar` (validación cruzada de 5 folds, semilla 42).
- Si tu RMSE CV es más bajo que el de README.md, abre un PR que sustituya este
  fichero (y lo que necesites en preprocessing.py / models.py).
"""
from src.models import MezclaRegressor, lgbm_log, spline_log

# Nombre corto y autor, aparecen al ejecutar evaluar/train
DESCRIPCION = "Mezcla 0.85 Splines (log price, smearing por marca) + 0.15 LightGBM (log price)"
AUTOR = "Grupo"

# Splines: 5 nodos y alpha 0.3 (los mejores en CV entre 4-8 nodos y alpha 0.03-3)
PARAMS_SPLINES = dict(n_knots=5, alpha=0.3, smearing="brand")
# LightGBM: los mismos hiperparámetros que eligió Jose en notebooks/04_tuning.ipynb
MEJORES_PARAMS_LGBM = dict(
    n_estimators=800, learning_rate=0.04, num_leaves=3, min_child_samples=20,
    colsample_bytree=0.7, reg_lambda=5,
)
PESO_SPLINES = 0.85


def build_final_model():
    return MezclaRegressor(
        modelo_a=spline_log(**PARAMS_SPLINES),
        modelo_b=lgbm_log(**MEJORES_PARAMS_LGBM),
        peso=PESO_SPLINES,
    )
