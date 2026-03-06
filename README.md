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
|       |-- encoded_features.py
|       |-- modeling.py
|       |-- oof_tools.py
|       |-- evaluation.py
|       |-- artifacts.py
|       |-- kaggle_api.py
|       `-- pipeline.py
|-- scripts/
|   |-- analyze_oof_models.py
|   |-- blend_oof.py
|   |-- stack_oof.py
|   |-- train_baseline.py
|   |-- train_cv_lightgbm.py
|   |-- train_cv_xgboost.py
|   |-- make_submission.py
|   |-- make_submission_blend.py
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

2e. Entrenar baseline CV con LightGBM (OOF + modelo final)

```bash
python scripts/train_cv_lightgbm.py --folds 5
```

2f. Entrenar baseline CV con XGBoost (OOF + modelo final)

```bash
python scripts/train_cv_xgboost.py --folds 5
```

2g. Analizar OOFs de modelos para diversidad/correlacion

```bash
python scripts/analyze_oof_models.py --oof cb=artifacts/reports/train_cv_multiseed_full_hiiter_oof.csv#oof_ensemble --oof lgb=artifacts/reports/train_lightgbm_cv_oof.csv
```

2h. Optimizar blend sobre OOF (coordinate descent)

```bash
python scripts/blend_oof.py --oof cb=artifacts/reports/train_cv_multiseed_full_hiiter_oof.csv#oof_ensemble --oof lgb=artifacts/reports/train_lightgbm_cv_oof.csv --oof xgb=artifacts/reports/train_xgboost_cv_oof.csv --method coordinate
```

2i. Evaluar stacking lineal cross-fitted sobre OOF

```bash
python scripts/stack_oof.py --oof cb=artifacts/reports/train_cv_multiseed_full_hiiter_oof.csv#oof_ensemble --oof lgb=artifacts/reports/train_lightgbm_cv_oof.csv --oof xgb=artifacts/reports/train_xgboost_cv_oof.csv --stacker logistic
```

3. Generar submission

```bash
python scripts/make_submission.py
```

3b. Generar submission ensemble desde modelos multi-seed

```bash
python scripts/make_submission_ensemble.py
```

3c. Generar submission blend (CatBoost + LightGBM + XGBoost)

```bash
python scripts/make_submission_blend.py --weights-json-path artifacts/reports/model_diversity_blend_3model_full_grid002.json
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
