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
|       |-- diagnostics.py
|       |-- modeling.py
|       |-- evaluation.py
|       |-- artifacts.py
|       |-- kaggle_api.py
|       `-- pipeline.py
|-- scripts/
|   |-- audit_submission_parity.py
|   |-- analyze_train_test_drift.py
|   |-- evaluate_ensemble_robustness.py
|   |-- gate_submission_candidate.py
|   |-- snapshot_submission_artifacts.py
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

Con bloques de features:

```bash
python scripts/train_cv_multiseed.py --folds 5 --seeds "42,2024,3407" --feature-blocks "O,P"
```

2d. Ejecutar experimento de feature engineering por bloques (A/B/C/O/P)

```bash
python scripts/experiment_features.py --feature-blocks "A"
```

Bloques relevantes para outliers:
- `O`: banderas `is_outlier_*` + `outlier_flag_count` usando umbrales p01/p99.
- `P`: features continuas recortadas p01/p99 (`pclip_*`) para mitigacion de colas.

3. Generar submission

```bash
python scripts/make_submission.py
```

3b. Generar submission ensemble desde modelos multi-seed

```bash
python scripts/make_submission_ensemble.py
```

3c. Auditar paridad train/inferencia para una submission candidata

```bash
python scripts/audit_submission_parity.py --metrics-path artifacts/reports/train_cv_multiseed_metrics.json --submission-csv artifacts/submissions/playground-series-s6e3.csv
```

3d. Capturar snapshot reproducible de artefactos de submission

```bash
python scripts/snapshot_submission_artifacts.py --metrics-path artifacts/reports/train_cv_multiseed_metrics.json --submission-csv artifacts/submissions/playground-series-s6e3.csv --label "pre-submit-check"
```

3e. Analizar drift train/test y adversarial validation

```bash
python scripts/analyze_train_test_drift.py --feature-blocks none --adv-folds 3 --adv-sample-frac 0.35
```

3f. Medir robustez de ensemble (equal/rank/weighted) con validacion repetida

```bash
python scripts/evaluate_ensemble_robustness.py --oof cb=artifacts/reports/train_cv_multiseed_full_hiiter_oof.csv#oof_ensemble --oof lgb=artifacts/reports/train_lightgbm_cv_full_hiiter_oof.csv#oof_pred --oof xgb=artifacts/reports/train_xgboost_cv_full_hiiter_oof.csv#oof_pred --repeats 3 --folds 5
```

3g. Aplicar gate automatico GO/NO_GO para decidir submission

```bash
python scripts/gate_submission_candidate.py --candidate-name playground-series-s6e3 --parity-json artifacts/reports/diagnostic_submission_parity_issue5.json --drift-json artifacts/reports/diagnostic_train_test_drift_issue5.json --robustness-json artifacts/reports/diagnostic_ensemble_robustness_issue5.json
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
