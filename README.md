# Kaggle Playground S6E3 - Predict Customer Churn

Competencia: `https://www.kaggle.com/competitions/playground-series-s6e3`

## Objetivo

Predecir la probabilidad de `Churn` para cada `id` del archivo `test.csv`.

## Estructura del proyecto

```
.
|-- src/
|   `-- churn_baseline/
|       |-- config.py
|       |-- data.py
|       |-- modeling.py
|       |-- evaluation.py
|       |-- artifacts.py
|       |-- kaggle_api.py
|       `-- pipeline.py
|-- scripts/
|   |-- train_baseline.py
|   |-- make_submission.py
|   |-- submit_kaggle.py
|   `-- run_baseline.py
|-- notebooks/
|   `-- baseline_walkthrough.ipynb
|-- artifacts/
|   |-- models/
|   |-- reports/
|   |-- submissions/
|   `-- logs/
|-- data/
|   `-- raw/
`-- docs/
```

## Quickstart

1. Descargar datos

```bash
kaggle competitions download -c playground-series-s6e3 -p data/raw
unzip data/raw/playground-series-s6e3.zip -d data/raw
```

2. Entrenar baseline y guardar modelo/metricas

```bash
python scripts/train_baseline.py
```

2b. Entrenar baseline con validacion robusta (Stratified K-Fold + OOF)

```bash
python scripts/train_cv.py --folds 5
```

2c. Entrenar multi-seed sobre CV (ensemble por promedio)

```bash
python scripts/train_cv_multiseed.py --folds 5 --seeds "42,2024,3407"
```

3. Generar submission

```bash
python scripts/make_submission.py
```

3b. Generar submission ensemble desde modelos multi-seed

```bash
python scripts/make_submission_ensemble.py
```

4. Enviar a Kaggle (opcional)

```bash
python scripts/submit_kaggle.py --message "playground-series-s6e3"
```

## Pipeline end-to-end (una sola llamada)

```bash
python scripts/run_baseline.py --submit --message "playground-series-s6e3"
```

## Notas

- `scripts/` contiene solo CLIs con argumentos.
- La logica de negocio vive en `src/churn_baseline/`.
- `artifacts/` guarda salidas locales y no se versiona (solo estructura).
- Antes de descargar/enviar, debes aceptar reglas en `Join Competition`.
