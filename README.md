# Kaggle Playground S6E3 - Predict Customer Churn

Proyecto base para participar en la competencia:
`https://www.kaggle.com/competitions/playground-series-s6e3`

## Objetivo

Predecir la probabilidad de abandono (`Churn`) para cada `id` del set de prueba.

## Quickstart

```bash
kaggle competitions download -c playground-series-s6e3 -p data/raw
unzip data/raw/playground-series-s6e3.zip -d data/raw
```

## Estructura

```
.
|-- data/
|   `-- raw/
|       |-- playground-series-s6e3.zip
|       |-- sample_submission.csv
|       |-- test.csv
|       `-- train.csv
`-- docs/
    |-- competencia.md
    |-- setup.md
    `-- estado.md
```

## Estado inicial

- Repositorio remoto creado: `moffermann/kaggle-playground-s6e3-churn`
- Datos descargados y extraidos en `data/raw/`.

## Datos y reglas

- Los datos de competencia no se versionan en git.
- Antes de descargar o enviar, debes entrar a Kaggle y aceptar las reglas en `Join Competition`.

## Siguiente paso recomendado

Crear un baseline de clasificacion (por ejemplo, LightGBM o XGBoost) y subir el primer `submission.csv`.
