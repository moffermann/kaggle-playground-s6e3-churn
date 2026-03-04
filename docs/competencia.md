# Competencia

- Nombre: `Predict Customer Churn`
- URL: `https://www.kaggle.com/competitions/playground-series-s6e3`
- Tipo: clasificacion binaria tabular
- Metrica: `ROC AUC`
- Deadline: `2026-03-31 23:59 UTC`

## Que hay que predecir

La probabilidad de la variable `Churn` para cada registro del archivo `test.csv`.

## Archivos oficiales

- `train.csv`: datos de entrenamiento (incluye target `Churn`)
- `test.csv`: datos de prueba (sin target)
- `sample_submission.csv`: plantilla de envio

## Formato de envio

CSV con header:

```
id,Churn
594194,0.1
594195,0.3
...
```
