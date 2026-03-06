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

2d. Ejecutar experimento de feature engineering por bloques (A/B/C/O/P)

```bash
python scripts/experiment_features.py --feature-blocks "A"
```

Bloques relevantes para outliers:
- `O`: banderas `is_outlier_*` + `outlier_flag_count` usando umbrales p01/p99.
- `P`: features continuas recortadas p01/p99 (`pclip_*`) para mitigacion de colas.

2e. Ejecutar experimento DA R1 (segment representation fold-safe)

```bash
python scripts/experiment_da_r1.py --base-feature-blocks "none"
```

2f. Ejecutar experimento DA R2 (R1 + rare bucketing + frecuencias)

```bash
python scripts/experiment_da_r1.py --representation-mode r2 --base-feature-blocks "none"
```

2g. Ejecutar DA con reweighting por rareza de segmento (fold-safe)

```bash
python scripts/experiment_da_r1.py --representation-mode r2 --base-feature-blocks "none" --enable-segment-reweighting
```

2h. Ejecutar DA sintética conservadora (oversampling in-fold de segmentos raros)

```bash
python scripts/experiment_da_r1.py --representation-mode r2 --base-feature-blocks "none" --enable-segment-oversampling --oversample-max-multiplier 1.5 --oversample-max-added-rate 0.02
```

2i. Ejecutar oversampling calibrado por objetivo de varianza (sin adivinar fuerza)

```bash
python scripts/experiment_da_r1.py --representation-mode r2 --base-feature-blocks "none" --enable-segment-oversampling --oversample-policy variance_target --oversample-se-target 0.0125 --oversample-max-added-rate 0.02
```

2j. Calcular plan de oversampling (sin entrenar) para una grilla de `SE`

```bash
python scripts/experiment_da_r1.py --representation-mode r2 --base-feature-blocks "none" --enable-segment-oversampling --oversample-policy variance_target --oversample-plan-only --oversample-plan-se-grid "0.02,0.015,0.0125,0.01,0.009,0.008"
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
