# Historial de modelos

Cada vez que un modelo sustituye al anterior se añade una fila (la más reciente abajo).

| Fecha | Autor | Modelo | RMSE CV (5 folds) | Mejora |
|---|---|---|---|---|
| 2026-11-24 | Jose | Mezcla 0.7 Ridge (log price) + 0.3 LightGBM (log price) | 52.381 | versión base |
| 2026-11-25 | Grupo | Mezcla 0.85 Splines (log price, smearing por marca) + 0.15 LightGBM (log price) | 52.297 | -0,084 respecto a 52.381 |
