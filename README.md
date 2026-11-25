# Práctica 1 · Predicción del precio de productos

Ingeniería del Conocimiento, Grado en Ingeniería Informática.

Modelo de regresión que predice `price` de productos de e-commerce. Métrica: **RMSE** (a partir de 57 se obtiene un 5; cuanto más bajo, mejor).

## Modelo actual

| | |
|---|---|
| **Modelo** | Mezcla 0.85 Splines (log price, smearing por marca) + 0.15 LightGBM (log price) |
| **Autor** | Grupo |
| **RMSE CV (5 folds)** | **52.297** |

El historial de modelos que han ido sustituyendo al anterior está en [HISTORIAL.md](HISTORIAL.md).

## Forma de trabajo: el mejor modelo se queda

No hay reparto de tareas. En `main` siempre está **el mejor modelo que tenemos**. Cualquiera puede intentar mejorarlo, y si lo consigue, lo sustituye.

1. Haz `git pull` para tener la última versión de `main`.
2. Crea una rama: `git checkout -b modelo/tu-nombre-idea` (por ejemplo `modelo/irene-catboost`).
3. Cambia **`src/modelo.py`** (la función `build_final_model()`). Si te hace falta, también puedes tocar `src/preprocessing.py` o `src/models.py`. Para probar cosas usa los notebooks.
4. Mide tu modelo:
   ```powershell
   python -m src.evaluar
   ```
5. **Si tu RMSE CV es más bajo que el de la tabla de arriba**:
   - Actualiza la tabla "Modelo actual" de este README con tu modelo, tu nombre y tu RMSE.
   - Añade una fila a `HISTORIAL.md`.
   - Haz commit, push y abre un Pull Request a `main` pegando la salida de `python -m src.evaluar`.
   - Otra persona del grupo ejecuta `python -m src.evaluar` en tu rama; si le sale el mismo RMSE, hace merge.
6. Si no mejora, no pasa nada: no abras PR (o apunta la idea en un issue para que nadie la repita).

### Reglas para que la comparación sea justa

- Todos medimos con `python -m src.evaluar`: validación cruzada de 5 folds con semilla 42. **No cambies `src/evaluar.py` ni `src/config.py`** (semilla, folds) en un PR de modelo.
- Nada de usar `test.csv` para entrenar ni para elegir hiperparámetros.
- Si dos PRs llegan a la vez, se hace merge del que tenga menor RMSE y el otro actualiza su rama y vuelve a medir.
- Opcional: comprobar la submission en la web de prueba de la asignatura antes del PR.

## Estructura

```
data/raw/            train.csv, test.csv, submission_ejemplo.csv
src/modelo.py        EL MODELO ACTUAL (lo único que hay que cambiar para mejorar)
src/evaluar.py       mide el modelo con CV de 5 folds (no tocar)
src/train.py         entrena con todo train y genera la submission
src/preprocessing.py variables nuevas y ColumnTransformer
src/models.py        modelos base y utilidades de validación cruzada
src/config.py        rutas, semilla, columnas (no tocar)
notebooks/           EDA, preprocesado, comparación de modelos, tuning e informe
submissions/         aquí se generan los CSV (no se suben a GitHub)
```

## Instalación (Windows, PowerShell en VS Code)

```powershell
py -3.11 -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Generar tu submission

```powershell
python -m src.train NombreAlumno
```

Genera `submissions/submission_NombreAlumno.csv` con columnas `id,prediction` y 2000 filas, usando el modelo que haya en `main`. Cada alumno entrega la suya, con su informe (`notebooks/informe_final.ipynb` con su nombre) y su vídeo.
