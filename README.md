# Kaggle Playground S6E3 - Predict Customer Churn

Competencia: `https://www.kaggle.com/competitions/playground-series-s6e3`

## Objetivo

Predecir la probabilidad de `Churn` para cada `id` del archivo `test.csv`.

## Estructura del proyecto

Esquema resumido, no exhaustivo:

```
.
|-- src/
|   `-- churn_baseline/
|       |-- config.py
|       |-- data.py
|       |-- diagnostics.py
|       |-- noise_audit.py
|       |-- feature_engineering.py
|       |-- fm_probe.py
|       |-- gnn_probe.py
|       |-- hard_example_stability.py
|       |-- artifacts.py
|       |-- cleanroom_baseline.py
|       |-- evaluation.py
|       |-- kaggle_api.py
|       |-- linear_probe.py
|       |-- mlp_probe.py
|       |-- ngram_xgb.py
|       |-- modeling.py
|       |-- pseudo_labeling.py
|       |-- residual_ablation.py
|       |-- specialist.py
|       |-- submission_forensics.py
|       |-- telco_transfer.py
|       |-- target_priors.py
|       |-- uncertainty_band.py
|       |-- v3_dominance.py
|       `-- pipeline.py
|-- scripts/
|   |-- audit_submission_parity.py
|   |-- analyze_train_test_drift.py
|   |-- analyze_error_by_class.py
|   |-- analyze_family_generalization.py
|   |-- analyze_label_noise.py
|   |-- analyze_residual_hierarchy_ablation.py
|   |-- analyze_submission_forensics.py
|   |-- analyze_v3_dominance.py
|   |-- evaluate_against_v3.py
|   |-- evaluate_validation_protocol.py
|   |-- evaluate_ensemble_robustness.py
|   |-- experiment_fm_probe.py
|   |-- experiment_gnn_probe.py
|   |-- experiment_hierarchical_priors.py
|   |-- experiment_hard_example_stability.py
|   |-- experiment_linear_probe.py
|   |-- experiment_mlp_probe.py
|   |-- experiment_noise_mitigation.py
|   |-- experiment_cleanroom_baselines.py
|   |-- experiment_telco_transfer.py
|   |-- experiment_uncertainty_band.py
|   |-- run_ngram_xgb.py
|   |-- experiment_pseudo_label_family.py
|   |-- experiment_local_calibrator.py
|   |-- experiment_specialist_model.py
|   |-- experiment_telco_joint_training.py
|   |-- build_teacher_component_frame.py
|   |-- gate_submission_candidate.py
|   |-- make_submission_residual_hierarchical.py
|   |-- snapshot_submission_artifacts.py
|   |-- make_submission_hierarchical_priors.py
|   |-- materialize_residual_submission.py
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
python scripts/train_cv_multiseed.py --folds 5 --seeds "42,2024,3407" --feature-blocks "H,R,V"
```

Con estratificacion compuesta por target + familia:

```bash
python scripts/train_cv_multiseed.py --folds 5 --seeds "42,2024,3407" --feature-blocks "G,H,R,V" --stratify-mode composite
```

2d. Ejecutar experimento de feature engineering por bloques (A/B/C/F/G/T/H/R/S/V/O/P)

```bash
python scripts/experiment_features.py --feature-blocks "G,H,R,V" --stratify-mode composite
```

Notas:
- `experiment_features.py` compara contra un baseline leyendo `--baseline-metrics-path`.
- Ese JSON debe contener `ensemble_oof_auc`, `oof_auc` o `holdout_auc`.
- Si quieres un benchmark distinto, apunta `--baseline-metrics-path` al reporte correcto.
- `F` agrega una superficie target-free enfocada en `Electronic check + Month-to-month + Fiber optic`,
  con buckets de tenure mas finos, ranks y gaps internos dentro de esa macrofamilia.
- `T` agrega la version `fit-aware` de esa superficie: ajusta mapas solo sobre train,
  hace fallback `fine tenure + paperless -> tenure_bin + paperless -> paperless` y expone
  ranks aproximados respecto de la distribucion aprendida en train.
- Soporta un smoke monotónico experimental con:
  - `--monotonic-feature-set minimal`
  - `--monotonic-preset minimal`
- Si pasas `--monotonic-preset minimal` y dejas `--monotonic-feature-set none`, el CLI activa
  automaticamente `minimal` para no dejar constraints apuntando a columnas inexistentes.
- Valores soportados para ambos flags: `none`, `minimal`. Cualquier otro valor falla de forma explicita.
- Alias aceptados para desactivar: `off`, `false`, `0`.
- `minimal` agrega solo cuatro señales numéricas estables para este experimento:
  - `tenure`
  - `contract_commitment_ordinal`
  - `is_manual_payment`
  - `payment_friction_index`
- Para medir el efecto neto de la restricción, conviene comparar contra el mismo experimento con
  `--monotonic-feature-set minimal --monotonic-preset none`.
- Los reportes JSON de `train_baseline*` ahora incluyen `include_monotonic_features`; y
  `experiment_features.py` agrega `monotonic_feature_set` y `monotonic_preset`.
- Cuando hay constraints activas, `params_cv` y `params_full_train` también serializan
  `monotone_constraints` en el JSON.

2e. Ejecutar experimento con priors jerarquicos fold-safe

```bash
python scripts/experiment_hierarchical_priors.py --feature-mode raw_plus_priors --folds 5
```

Con desvio del cliente respecto de su cohorte:

```bash
python scripts/experiment_hierarchical_priors.py --feature-mode raw_plus_priors --include-deviation-features --folds 5
```

Con bloques base antes de construir priors:

```bash
python scripts/experiment_hierarchical_priors.py --base-feature-blocks "H,R" --feature-mode raw_plus_priors --folds 5
```

Bloques relevantes para outliers:
- `O`: banderas `is_outlier_*` + `outlier_flag_count` usando umbrales p01/p99.
- `P`: features continuas recortadas p01/p99 (`pclip_*`) para mitigacion de colas.

Bloques estructurales:
- `G`: cobertura y backoff por familia fit-aware (`segment3/segment5`, soporte train-only, unseen/low-support, familia colapsada al padre).
- `T`: superficie fit-aware enfocada en `Electronic check + Month-to-month + Fiber optic`, con fallback `fine tenure + paperless -> tenure_bin + paperless -> paperless` y ranks aproximados respecto del train.
- Si entrenas con `G` o `T`, en inferencia debes pasar `--train-csv` a los CLI que materializan submissions para reconstruir el estado train-only.
- `H`: contrastes intra-cohorte para bundles raros dentro de grupos duros (rareza y desvio respecto de la cohorte).
- `R`: lifecycle y renovacion (`tenure_mod_*`, distancia a borde de contrato, bins de ciclo).
- `S`: topologia del bundle y firma de servicios (support vs entertainment, archetype, signatures).
- `V`: presion de valor y friccion de pago (cargo efectivo, cohort del precio, service pressure).

3. Generar submission

```bash
python scripts/make_submission.py --feature-blocks "G,H,R,V" --train-csv data/raw/train.csv
```

Nota:
- `--train-csv` solo es obligatorio cuando usas bloques fit-aware como `G` o `T`.
- El resultado JSON de `make_submission.py` y `make_submission_ensemble.py` ahora tambien expone
  `include_monotonic_features` para mantener trazabilidad cuando la inferencia se hace por API
  usando esa ruta del pipeline.

3b. Generar submission ensemble desde modelos multi-seed

```bash
python scripts/make_submission_ensemble.py --feature-blocks "G,H,R,V" --train-csv data/raw/train.csv
```

Notas:
- `make_submission_ensemble.py` usa `--model-paths` si los pasas manualmente.
- Si no, lee `--metrics-path` y espera una clave `model_paths` en ese JSON.
- `--train-csv` es necesario cuando usas bloques fit-aware como `G` o `T`, para reconstruir estado aprendido solo en train.

3b2. Generar submission desde experimento de priors jerarquicos

```bash
python scripts/make_submission_hierarchical_priors.py --metrics-path artifacts/reports/hierarchical_priors_cv_metrics.json
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

3g. Analizar error por clase/cohorte entre referencia y challenger

```bash
python scripts/analyze_error_by_class.py --reference-weights-json artifacts/reports/submission_candidate_cb5_xgb3_lgb_r_weights.json --challenger-weights-json artifacts/reports/submission_candidate_cb5_xgb3_lgb_r_rvhi_weights.json
```

3g2. Construir una brujula de generalizacion por familia con leave-one-family-out

```bash
python scripts/analyze_family_generalization.py --family-level segment5 --feature-blocks "H,R,V" --top-k-families 12
```

Notas:
- Salidas por defecto:
  - JSON: `artifacts/reports/diagnostic_family_generalization_summary.json`
  - CSV: `artifacts/reports/diagnostic_family_generalization_families.csv`
- El script tambien imprime el JSON resumido a `stdout`.
- Por defecto usa como referencia el teacher blend `cb+xgb+lgb+r+rv` definido en `submission_candidate_cb5_xgb3_lgb_r_rvhi_weights.json`.
- Si quieres una referencia directa distinta, puedes pasar `--reference-oof <path>[#<prediction_column>]`.
- Si quieres reconstruir otra referencia ponderada, puedes pasar `--oof ...` y `--reference-weights-json ...`.
- `--family-level` soporta `segment3` y `segment5`.
- El rerun LOFO usa hiperparametros fijos y expone `--iterations`, `--learning-rate`, `--depth`, `--l2-leaf-reg`.
- El JSON resume familias de mayor dano y mayor gap de generalizacion; el CSV deja la tabla completa con:
  - `reference_auc`, `reference_logloss`, `reference_logloss_contribution`
  - `lofo_auc`, `lofo_logloss`
  - `generalization_gap_auc`, `generalization_gap_logloss`, `generalization_gap_logloss_contribution`
- El JSON tambien incluye bloques estructurados: `reference_source`, `selection`, `model_params`,
  `top_reference_risk_families` y `top_generalization_gap_families`.
- `min_train_rows` y `min_test_rows` filtran familias que no tienen soporte suficiente para ser accionables.
- El rerun LOFO no se ejecuta sobre todas las familias: primero se filtra por `min_train_rows` y
  `min_test_rows`, y luego se evalua solo el `top-k` ordenado por `reference_logloss_contribution`.
- En el CSV, las columnas `lofo_*` y `generalization_gap_*` quedan en `NaN` para familias que no
  entraron en ese rerun LOFO.

3g2b. Auditar `label noise`, `near-duplicates` y ejemplos duros contra `v3`

```bash
python scripts/analyze_label_noise.py \
  --train-csv data/raw/train.csv \
  --v3-oof artifacts/reports/validation_protocol_v3_chain_oof.csv#candidate_pred \
  --out-json artifacts/reports/label_noise_audit_summary.json \
  --out-rows-csv artifacts/reports/label_noise_audit_suspicious_rows.csv \
  --out-duplicate-csv artifacts/reports/label_noise_audit_duplicate_groups.csv
```

Notas:
- `--train-csv` default: `data/raw/train.csv`.
- `--v3-oof` debe apuntar a un CSV con `id`, `target` y una columna de prediccion continua; el sufijo `#candidate_pred` indica explicitamente esa columna.
- Usa `v3` como referencia fija y falla si el OOF no cubre exactamente `train.csv`.
- Si estan disponibles los OOF `cb/xgb/lgb/r/rv`, agrega desacuerdo del teacher:
  - `teacher_mean`
  - `teacher_std`
  - `teacher_range`
  - `teacher_unanimous_side`
- Distingue tres clases diagnosticas:
  - `label_noise_candidate`
  - `near_duplicate_conflict`
  - `hard_example_stable`
- El CSV de grupos de duplicados reporta firmas `exact` y `coarse`.
- El conflicto `coarse` solo cuenta grupos chicos (`--max-near-duplicate-group-size`, default `5`, minimo `2`) para no confundir cohortes grandes con near-duplicates.
- El JSON resume concentracion por `segment3`, `segment5` y por la macrofamilia dominante `Electronic check / Month-to-month / Fiber optic`.

3g2c. Hacer submission forensics sobre el historial de Kaggle y los artefactos locales

```bash
python scripts/analyze_submission_forensics.py \
  --competition playground-series-s6e3 \
  --reports-dir artifacts/reports \
  --submissions-dir artifacts/submissions
```

Notas:
- Usa la API autenticada de Kaggle y cruza el historial remoto con JSONs locales que referencian CSVs de `artifacts/submissions/`.
- El linking entre Kaggle y reportes locales se hace por nombre de CSV; `--submissions-dir` solo controla la verificacion de existencia del archivo local.
- Salidas por defecto:
  - `artifacts/reports/submission_forensics_summary.json`
  - `artifacts/reports/submission_forensics_ledger.csv`
  - `artifacts/reports/submission_forensics_report_links.csv`
- El resumen destaca:
  - mejor submission publica
  - clustering por `submission_family`
  - cantidad de submissions con artefactos locales ligados
  - correlacion local/public cuando existe una metrica local comparable suficiente
- Sirve para identificar familias que ya mostraron supervivencia publica antes de gastar CPU en `midcap`.

3g2d. Probar mitigacion minima de near-duplicates directamente contra `v3`

```bash
python scripts/experiment_noise_mitigation.py \
  --mode downweight \
  --feature-blocks R,V \
  --label noise_mitigation_downweight_smoke
```

Notas:
- El experimento entrena un challenger CatBoost en smoke y lo compara directamente contra `v3`.
- `--mode` soporta `downweight` o `drop`.
- La regla se deriva solo sobre el fold de train y marca filas minoritarias dentro de grupos `coarse` chicos y mixtos.
- La firma `coarse` usa:
  - categoricas crudas normalizadas (`gender`, `SeniorCitizen`, `Partner`, `Dependents`, `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`, `Contract`, `PaperlessBilling`, `PaymentMethod`)
  - `tenure` exacto
  - `MonthlyCharges` redondeado al entero mas cercano
  - `TotalCharges` redondeado a decenas
  - `segment3` y `segment5`
- Por default restringe la mitigacion a la macrofamilia dominante `Electronic check / Month-to-month / Fiber optic`; usa `--no-dominant-only` para desactivarlo.
- `--suspect-weight` solo aplica en `downweight`.
- `--min-group-size`, `--max-group-size` y `--majority-share-min` controlan la agresividad de la regla.
- El script siempre construye `analysis_oof` y corre el gate directo contra `v3`.
- Los artefactos principales son:
  - `<label>_metrics.json`
  - `<label>_oof.csv`
  - `<label>_analysis_oof.csv`
  - `<label>_candidate_metrics.json`
  - `<label>_reference_v3_metrics.json`
  - `validation_protocol_<label>_vs_v3_smoke.json`

3g2e. Correr el clean-room baseline rebuild minimo contra `v3`

```bash
python scripts/experiment_cleanroom_baselines.py \
  --train-csv data/raw/train.csv \
  --v3-oof artifacts/reports/validation_protocol_v3_chain_oof.csv#candidate_pred
```

Notas:
- Ejecuta tres reconstrucciones minimas del baseline con CatBoost:
  - `cb_raw`
  - `cb_r`
  - `cb_rv`
- Todas se comparan directamente contra `v3` desde `smoke` usando el mismo protocolo de validacion.
- Los labels efectivos del suite son:
  - `cleanroom_cb_raw_smoke`
  - `cleanroom_cb_r_smoke`
  - `cleanroom_cb_rv_smoke`
- Salidas por defecto:
  - `artifacts/reports/cleanroom_baseline_suite_summary.json`
  - `artifacts/reports/cleanroom_reference_v3_metrics.json`
  - por cada label (`cleanroom_cb_raw_smoke`, `cleanroom_cb_r_smoke`, `cleanroom_cb_rv_smoke`):
    - `<label>_metrics.json`
    - `<label>_oof.csv`
    - `<label>_analysis_oof.csv`
    - `<label>_candidate_metrics.json`
    - `validation_protocol_<label>_vs_v3_smoke.json`
- El objetivo no es encontrar un nuevo incumbent, sino medir cuanto del edge de `v3` sigue existiendo cuando se reconstruye el stack desde una base simple.

3g3. Evaluar un candidato OOF bajo el protocolo de `validation reset`

```bash
python scripts/evaluate_validation_protocol.py --stage smoke --analysis-oof artifacts/reports/family_gated_ec_mtm_fiber_any_teacher_smoke_oof.csv --target-family-level segment3 --target-family-value "Electronic check__Month-to-month__Fiber optic"
```

Notas:
- `--analysis-oof` acepta un CSV ya armado con columnas `id`, `target`, `reference_pred` y `candidate_pred`.
- Si no tienes ese artefacto, puedes usar `--candidate-oof <path>[#<prediction_column>]` y reconstruir la referencia con:
  - `--reference-oof <path>[#<prediction_column>]`, o
  - `--oof ...` + `--reference-weights-json`
- Defaults importantes:
  - `--train-csv data/raw/train.csv`
  - `--test-csv data/raw/test.csv`
  - `--reference-weights-json artifacts/reports/submission_candidate_cb5_xgb3_lgb_r_rvhi_weights.json`
  - si no pasas `--oof`, el script usa el teacher blend por defecto `cb+xgb+lgb+r+rv`
  - `--out-json artifacts/reports/validation_protocol_verdict.json`
- El JSON de salida deja:
  - metricas globales Split A
  - checks por macrofamilia dominante y top-3 por dano
  - checks adicionales por familia objetivo solo si pasas `--target-family-value`
  - guardrail de familia hermana solo en `segment5` y solo para `submission`; si no pasas `--sister-family-value`, el script intenta derivarla automaticamente desde la clave `segment5`
  - veredicto agregado `PASS/WARN/FAIL`
- Para `midcap` o `submission`, debes pasar tambien:
  - `--candidate-metrics-json`
  - `--reference-metrics-json`
- Sin esos JSON el script igual corre, pero el check `midcap_cv_std` queda en `FAIL`.
- Para `submission`, el gate exige `--submission-csv` para trazar el artefacto final.

3g3b. Diagnosticar por que `v3` domina challengers fallidos

```bash
python scripts/analyze_v3_dominance.py --label v3_dominance_v1
```

Notas:
- Usa `v3` fijo desde `artifacts/reports/validation_protocol_v3_chain_oof.csv#candidate_pred`.
- Por default compara un set curado de challengers que representan familias distintas de fracaso:
  - bloque global
  - challenger por familia
  - mitigacion data-centric
  - transfer externo
  - residual casi-vivo
- Produce:
  - `<label>_summary.json`
  - `<label>_challengers.csv`
  - `<label>_families.csv`
  - `<label>_slices.csv`
- Los cortes incluyen:
  - `segment3`
  - `segment5`
  - deciles de score `v3`
  - buckets de confianza `abs(v3 - 0.5)`
  - buckets de desacuerdo `abs(candidate - v3)`
- Sirve para decidir la siguiente hipotesis sin entrenar modelos nuevos.

3g4. Comparar una chain challenger directamente contra `v3`

```bash
python scripts/evaluate_against_v3.py --stage midcap --label "ec_any_best" --candidate-order "early_all_internet,ec_mtm_fiber_paperless_any,fiber_paperless_early" --candidate-step "ec_mtm_fiber_paperless_any|artifacts/reports/residual_reranker_ec_mtm_fiber_paperless_any_teacher_midcap_oof.csv"
```

Notas:
- Este CLI existe para no volver a comparar challengers solo contra `rvblend` cuando el incumbent operativo ya es `v3`.
- Usa `v3` como referencia fija:
  - `early_all_internet`
  - `fiber_paperless_early`
  - `late_mtm_fiber`
  - `late_mtm_fiber_paperless`
- Los OOF por defecto de `v3` son los artefactos `midcap`; `--stage` cambia los thresholds del protocolo, no esos paths por defecto.
- `--candidate-order` define la chain challenger completa; puede reutilizar pasos de `v3`, omitirlos o sobrescribirlos.
- `--candidate-step` usa formato `<preset>|<oof_path>` y solo hace falta para pasos que no pertenezcan al set fijo de `v3` o que quieras reemplazar.
- Los OOF usados por defecto para los pasos fijos de `v3` se leen desde `artifacts/reports`:
  - `residual_reranker_early_all_internet_midcap_oof.csv`
  - `residual_reranker_fiber_paperless_early_teacher_midcap_oof.csv`
  - `residual_reranker_late_mtm_fiber_teacher_midcap_oof.csv`
  - `residual_reranker_late_mtm_fiber_paperless_teacher_midcap_oof.csv`
- Todos los OOF de pasos deben contener estas columnas:
  - `id`
  - `oof_target`
  - `specialist_mask`
  - `candidate_pred`
  - `reference_pred`
- Todos los pasos deben compartir exactamente:
  - los mismos `id`
  - el mismo `oof_target`
  - el mismo `reference_pred` base dentro de una tolerancia numérica pequeña
- Si sobrescribes un paso existente de `v3`, el challenger cambia, pero la referencia `v3` sigue usando los OOF fijos originales.
- La union de `id` en los OOF debe cubrir exactamente `train.csv`; el script falla si detecta cobertura parcial.
- El script deja:
  - OOF de referencia `v3`
  - OOF del challenger
  - analysis OOF `candidate vs v3`
  - metrics JSON comparables (`cv_std_auc` incluido)
  - veredicto del protocolo aplicado directamente contra `v3`
  - summary JSON con el resumen del veredicto y los paths generados

3g4a. Auditar si la jerarquia residual `v3` es compresible

```bash
python scripts/analyze_residual_hierarchy_ablation.py \
  --stage midcap \
  --out-dir artifacts/reports/residual_ablation_v1
```

Notas:
- Evalua todas las subsecuencias no vacias y estrictas de la cadena `v3` manteniendo el orden original de pasos.
- Reutiliza `evaluate_candidate_chain_against_v3`, o sea:
  - no reentrena modelos
  - compara cada cadena comprimida directamente contra `v3`
  - corre el gate con los mismos thresholds del protocolo
- Flags utiles:
  - `--train-csv`
  - `--test-csv`
  - `--stage`
  - `--target-family-level`
  - `--target-family-value`
  - `--dominant-family-value`
  - `--out-dir`
- Salidas principales:
  - `artifacts/reports/residual_ablation_v1/residual_ablation_summary.json`
  - `artifacts/reports/residual_ablation_v1/residual_ablation_candidates.csv`
  - ademas, por cada cadena candidata, deja `analysis_oof`, `metrics` y `validation_protocol_*`.
- El resumen deja:
  - mejor cadena comprimida
  - ranking de ablation por paso individual
  - ranking de ablation por pares
  - thresholds de compresibilidad por tolerancia unilateral (`delta_vs_v3_oof_auc >= -tolerance`)
- Uso recomendado:
  - antes de abrir otra hipotesis sobre la familia residual
  - para distinguir entre edge real de la cadena y complejidad accidental

3g4aa. Destilar el delta total de `v3` sobre la referencia base

```bash
python scripts/experiment_residual_distillation.py \
  --label residual_distillation_smoke \
  --feature-blocks H,R,S,V \
  --alpha-grid 0.25,0.5,1.0,2.0,4.0
```

Notas:
- Esta linea aprende un solo target de regresion:
  - `distill_target = v3_pred - base_reference_pred`
- No reentrena la jerarquia residual original ni genera submissions.
- Si corres el CLI tal como esta, si usa `test.csv` para el gate del protocolo.
- Requiere los OOF `midcap` de la cadena `v3` ya presentes en `artifacts/reports/`:
  - `residual_reranker_early_all_internet_midcap_oof.csv`
  - `residual_reranker_fiber_paperless_early_teacher_midcap_oof.csv`
  - `residual_reranker_late_mtm_fiber_teacher_midcap_oof.csv`
  - `residual_reranker_late_mtm_fiber_paperless_teacher_midcap_oof.csv`
- El challenger se arma como:
  - `candidate_pred = clip(base_reference_pred + alpha * distilled_delta_pred)`
- `alpha` se escanea en smoke y el ganador se compara directo contra `v3`.
- Si el mejor `alpha` queda pegado al borde superior, la lectura correcta es que el regressor subestimó amplitud; no cierres la linea sin expandir ese grid.
- Artefactos principales:
  - `artifacts/reports/residual_distillation_smoke_analysis_oof.csv`
  - `artifacts/reports/residual_distillation_smoke_metrics.json`
  - `artifacts/reports/residual_distillation_smoke_reference_v3_metrics.json`
  - `artifacts/reports/residual_distillation_smoke_candidate_metrics.json`
  - `artifacts/reports/validation_protocol_residual_distillation_smoke_vs_v3_smoke.json`
  - `artifacts/models/residual_distillation_smoke.cbm`
- El `.cbm` que deja este smoke no es un camino de inferencia listo por si solo:
  - para usarlo despues tendrias que reconstruir `base_reference_pred` fuera de train
  - esa parte no se implemento en esta primera pasada

3g4b. Correr un `uncertainty-band reranker` local directamente contra `v3`

```bash
python scripts/experiment_uncertainty_band.py \
  --stage smoke \
  --feature-blocks "H,R,S,V" \
  --target-family-level segment3 \
  --target-family-value "Electronic check__Month-to-month__Fiber optic"
```

Notas:
- La referencia se fija en `v3` leyendo por defecto:
  - `artifacts/reports/validation_protocol_v3_chain_oof.csv#candidate_pred`
- El modelo no reemplaza al incumbent global:
  - solo corrige filas dentro de la familia objetivo
  - y solo si `abs(v3 - 0.5) <= --band-half-width`
- Por defecto agrega tambien las componentes `cb/xgb/lgb/r/rv` como features de desacuerdo del teacher.
- Guardrails relevantes:
  - valida alineacion exacta de `v3` contra `train.csv`
  - exige ambos labels dentro de la banda
  - exige minimo de filas en train y valid por fold
  - si pasas `--min-teacher-std`, aborta si el filtro adicional de desacuerdo distorsiona demasiado la tasa base del mask segun `--max-relative-mask-drift`
- Artefactos principales:
  - `artifacts/reports/uncertainty_band_smoke_metrics.json`
  - `artifacts/reports/uncertainty_band_smoke_oof.csv`
  - `artifacts/reports/validation_protocol_uncertainty_band_smoke_vs_v3.json`

3g4c. Correr la linea `hard-example stability` directamente contra `v3`

```bash
python scripts/experiment_hard_example_stability.py \
  --stage smoke \
  --stability-feature-blocks "R,V" \
  --reranker-feature-blocks "H,R,S,V" \
  --family-level segment3 \
  --family-value "Electronic check__Month-to-month__Fiber optic"
```

Notas:
- Esta linea construye primero un `hard_example_score` por fila usando OOF repetidos baratos.
- El score sale de:
  - `stability_pred_std`
  - `stability_flip_rate`
- Luego entrena un residual reranker local solo dentro de:
  - la familia objetivo
  - y el top cuantílico de `hard_example_score` dentro de esa familia
- `--reference-band-half-width` es opcional; sirve para intersectar el mask de dureza con la banda ambigua de `v3`.
- Artefactos principales:
  - `artifacts/reports/hard_example_stability_smoke_metrics.json`
  - `artifacts/reports/hard_example_stability_smoke_oof.csv`
  - `artifacts/reports/validation_protocol_hard_example_stability_smoke_vs_v3.json`

3g4d. Correr la linea `counterfactual teacher sensitivity` directamente contra `v3`

```bash
python scripts/experiment_counterfactual_sensitivity.py \
  --stage smoke \
  --target-family-level segment3 \
  --target-family-value "Electronic check__Month-to-month__Fiber optic"
```

Notas:
- Esta linea no entrena otro challenger grande. Parte desde `v3` y construye una correccion aditiva local.
- La correccion se basa en perturbaciones plausibles del mismo cliente, evaluadas con esta subfamilia del teacher disponible como modelos full-train:
  - `cb`
  - `r`
  - `rv`
  con pesos renormalizados desde `artifacts/reports/submission_candidate_cb5_xgb3_lgb_r_rvhi_weights.json`.
- Los escenarios contrafactuales se controlan con `--counterfactuals`; por defecto usa:
  - `auto_payment`
  - `paperless_off`
  - `contract_upgrade`
  - `stability_bundle`
- La señal no es una probabilidad nueva; es una medida de `counterfactual drop`:
  - cuanto baja el teacher si movemos al cliente a un estado mas estable
  - las subidas se recortan a `0`, o sea, el smoke usa solo `positive drop`
- `--signal-names` define como se agrega esa sensibilidad:
  - `stable_bundle_drop`
  - `mean_positive_drop`
  - `max_positive_drop`
- `--alpha-grid` escanea una correccion aditiva sobre `v3` y elige el mejor `candidate_pred` por AUC OOF.
- La correccion solo se aplica dentro del mask objetivo:
  - `target-family-level`
  - `target-family-value`
  y fuera de ese mask deja `candidate_pred == reference_pred`.
- `--reference-band-half-width` es opcional; si se pasa, la correccion solo se aplica tambien dentro de la banda ambigua de `v3`.
  Debe cumplir `0 < width < 0.5`.
- El mask final debe tener al menos `2000` filas; si no, el script falla de forma explicita.
- `--component-weights-json` debe contener:
  - `weights`
  - `components.cb_models`
  - `components.r_model`
  - `components.rv_model`
- El input crudo debe seguir teniendo estas columnas porque los contrafactuales las mutan directamente:
  - `PaymentMethod`
  - `PaperlessBilling`
  - `Contract`
- Artefactos principales:
  - `artifacts/reports/counterfactual_teacher_smoke_metrics.json`
  - `artifacts/reports/counterfactual_teacher_smoke_analysis_oof.csv`
  - `artifacts/reports/validation_protocol_counterfactual_teacher_smoke_vs_v3.json`

3g5. Correr la linea minima `bi-gram + target encoding + XGBoost`

```bash
python scripts/run_ngram_xgb.py \
  --train-csv data/raw/train.csv \
  --test-csv data/raw/test.csv \
  --original-csv artifacts/external/blastchar_telco/WA_Fn-UseC_-Telco-Customer-Churn.csv \
  --folds 2 \
  --inner-folds 3 \
  --metrics-path artifacts/reports/ngram_te_xgb_smoke_metrics.json \
  --oof-path artifacts/reports/ngram_te_xgb_smoke_oof.csv \
  --reference-v3-oof artifacts/reports/validation_protocol_v3_chain_oof.csv \
  --analysis-oof-path artifacts/reports/ngram_te_xgb_smoke_analysis_oof.csv
```

Notas:
- La linea vive separada del pipeline principal en `src/churn_baseline/ngram_xgb.py`.
- Usa:
  - numericas crudas (`SeniorCitizen`, `tenure`, `MonthlyCharges`, `TotalCharges`)
  - target encoding fold-safe para categoricas simples
  - bi-grams sobre `Contract`, `InternetService`, `PaymentMethod`, `OnlineSecurity`, `TechSupport`, `PaperlessBilling`
- `--include-trigrams` agrega tri-grams sobre las primeras 4 categoricas base.
- `--original-csv` es opcional; si se pasa, el CSV original se usa solo como apoyo para los mapas de target encoding.
  Debe incluir la columna objetivo `Churn`; si no existe, el CLI falla de forma explicita.
- El experimento elimina filas del dataset original que coincidan exactamente con filas de `train/test` para evitar solapamientos triviales.
- `--reference-v3-oof` y `--analysis-oof-path` tambien son opcionales; si ambos estan presentes, `--reference-v3-oof`
  debe apuntar a un CSV con columnas `id`, `target` y `candidate_pred`. El CLI deja listo un CSV con:
  - `id`
  - `target`
  - `reference_pred`
  - `candidate_pred`
  que se puede pasar directo a `evaluate_validation_protocol.py`.

3g5b. Correr la linea minima `external Telco transfer feature`

```bash
python scripts/experiment_telco_transfer.py \
  --train-csv data/raw/train.csv \
  --test-csv data/raw/test.csv \
  --original-csv artifacts/external/blastchar_telco/WA_Fn-UseC_-Telco-Customer-Churn.csv \
  --feature-blocks R,V \
  --label telco_transfer_smoke
```

Notas:
- Entrena un teacher CatBoost solo sobre el dataset original `blastchar` y proyecta una sola feature nueva:
  - `external_telco_pred`
- El teacher externo no usa etiquetas de la competencia.
- El script registra hash/metadata de:
  - `train.csv`
  - `test.csv`
  - `original.csv`
- Si hay filas del dataset original que coinciden exactamente con `train/test`, las elimina antes de entrenar el teacher externo.
- El challenger de la competencia se entrena en smoke con `feature_blocks` normales mas `external_telco_pred`.
- Los artefactos principales son:
  - `<label>_metrics.json`
  - `<label>_oof.csv`
  - `<label>_analysis_oof.csv`
  - `<label>_candidate_metrics.json`
  - `<label>_reference_v3_metrics.json`
  - `<label>_transfer_train.csv`
  - `<label>_transfer_test.csv`
  - `validation_protocol_<label>_vs_v3_smoke.json`
- El primer gate es siempre directo contra `v3`; no hay blend por default en esta linea.

3g5c. Correr el smoke minimo `source-aware joint training` con Telco original

```bash
python scripts/experiment_telco_joint_training.py \
  --train-csv data/raw/train.csv \
  --test-csv data/raw/test.csv \
  --original-csv artifacts/external/blastchar_telco/WA_Fn-UseC_-Telco-Customer-Churn.csv \
  --feature-blocks R,V \
  --external-weight 0.25 \
  --label telco_joint_training_smoke
```

Notas:
- Esta linea no proyecta un score externo; agrega filas reales del Telco original al fit de cada fold.
- El OOF y el gate se calculan solo sobre filas de la competencia.
- Agrega una categorica nueva:
  - `dataset_source` con valores `competition` y `telco_original`
- Las filas externas entran con peso configurable via `--external-weight`.
- El script elimina filas del Telco original que coincidan exactamente con `train/test` antes del joint training.
- El smoke nace comparado directamente contra `v3`; no se promueve si falla `validation_protocol_*_vs_v3_smoke.json`.
- Artefactos principales:
  - `<label>_metrics.json`
  - `<label>_oof.csv`
  - `<label>_analysis_oof.csv`
  - `<label>_candidate_metrics.json`
  - `<label>_reference_v3_metrics.json`
  - `validation_protocol_<label>_vs_v3_smoke.json`

3g6. Correr el smoke minimo `GraphSAGE + ANN graph`

```bash
python scripts/experiment_gnn_probe.py \
  --train-csv data/raw/train.csv \
  --test-csv data/raw/test.csv \
  --folds 2 \
  --device auto \
  --hidden-dim 64 \
  --epochs 12 \
  --patience 4 \
  --k-neighbors 8 \
  --metrics-path artifacts/reports/gnn_probe_smoke_v2_metrics.json \
  --oof-path artifacts/reports/gnn_probe_smoke_v2_oof.csv \
  --reference-v3-oof artifacts/reports/validation_protocol_v3_chain_oof.csv \
  --analysis-oof-path artifacts/reports/gnn_probe_smoke_v2_analysis_oof.csv
```

Notas:
- La linea vive separada del pipeline principal en `src/churn_baseline/gnn_probe.py`.
- Es una adaptacion minima del notebook externo de GNN:
  - `torch-geometric` para `GraphSAGE`
  - `pynndescent` para construir un grafo ANN sobre `train+test`
  - sin RAPIDS (`cuml/cupy`) y sin hill climbing en esta primera pasada
- Usa solo:
  - 16 categoricas base como embeddings de nodo
  - 3 numericas (`tenure`, `MonthlyCharges`, `TotalCharges`) estandarizadas
  - grafo KNN sobre `OHE(base_cats) + numericas escaladas`
- El entrenamiento es transductivo:
  - el grafo se construye sobre `train+test`
  - la perdida de cada fold usa solo los nodos de train del fold
- `--reference-v3-oof` y `--analysis-oof-path` son opcionales; si ambos estan presentes, el CLI deja listo un CSV con:
  - `id`
  - `target`
  - `reference_pred`
  - `candidate_pred`
  para pasarlo directo a `evaluate_validation_protocol.py`.

3h. Ejecutar experimento de especialista local sobre el incumbente

```bash
python scripts/experiment_specialist_model.py --preset early_manual_internet --feature-blocks "H,R,S,V"
```

Para la macrofamilia dominante de `issue #8`, el primer challenger supervisado se prueba con:

```bash
python scripts/experiment_specialist_model.py --approach classifier --preset ec_mtm_fiber_any --feature-blocks "H,R,S,V"
```

Requiere que exista el OOF del incumbente (`cb/xgb/lgb/r/rv`) y el archivo de pesos
`artifacts/reports/submission_candidate_cb5_xgb3_lgb_r_rvhi_weights.json`.
Por defecto escribe a `artifacts/models/`, `artifacts/reports/` y usa `--alpha-grid`
para escanear el override local. Si tus artefactos OOF tienen otros nombres o columnas,
debes pasarlos por `--oof` y hacerlos coincidir con las keys del JSON de pesos
(la mezcla usa columnas `pred_<name>` internamente).

Tambien soporta `--approach residual`, que aprende una correccion aditiva sobre el
score del incumbente para intentar mejorar ranking dentro de la cohorte dura.
La correccion final se aplica con clipping numericamente estable a
`[1e-6, 1 - 1e-6]`:

```bash
python scripts/experiment_specialist_model.py --approach residual --preset fiber_paperless_early --feature-blocks "H,R,S,V"
```

Con disagreement del teacher:

```bash
python scripts/experiment_specialist_model.py --approach residual --preset fiber_paperless_early --feature-blocks "H,R,S,V" --include-teacher-disagreement-features
```

Formulacion `prediction as feature` para una macrofamilia:

```bash
python scripts/experiment_specialist_model.py --approach feature --preset ec_mtm_fiber_any --feature-blocks "H,R,S,V" --include-teacher-disagreement-features
```

En `approach=feature`, el CLI:
- entrena un especialista local sobre el preset;
- materializa su score local y lo transforma en features apiladas (`family_specialist_pred_feature`, `family_specialist_available`, `family_specialist_delta_vs_reference`);
- entrena luego un challenger global con esas features apiladas;
- guarda dos modelos:
  - `--model-path`: challenger global
  - `--model-path` con sufijo `_family<suffix>`: especialista usado para construir la feature
- el JSON de metricas expone ambos artefactos como `global_model_path` y `family_model_path`
- el OOF resultante escribe `family_specialist_pred`, `challenger_pred` y `candidate_pred`

Formulacion `gated` para una macrofamilia:

```bash
python scripts/experiment_specialist_model.py --approach gated --preset ec_mtm_fiber_any --feature-blocks "H,R,S,V" --include-teacher-disagreement-features --family-weight 2.0
```

En `approach=gated`, el CLI:
- entrena un challenger global sobre todas las filas;
- agrega features de gating (`family_focus_mask`, `family_focus_reference_delta`) para la familia objetivo;
- repondera train y validacion dentro del `preset` con `--family-weight`;
- mantiene el mismo scan de blend contra `reference_pred`.
- `--family-weight` debe ser `> 0`, por defecto vale `2.0`, y se reutiliza tambien en el fit final sobre todo el train.
- guarda un solo modelo global en `--model-path`;
- el OOF escribe `specialist_mask`, `reference_pred`, `challenger_pred` y `candidate_pred`;
- el JSON de metricas expone `family_weight`, `family_gating_columns`, `best_alpha` y el delta global/on-mask contra la referencia.

Notas para `--include-teacher-disagreement-features`:
- agrega `teacher_component_*` y estadisticas derivadas (`std`, `range`, `top_gap`, deltas por pares) usando las columnas `pred_*` del OOF de referencia;
- requiere que los `--oof` del incumbente expongan componentes individuales (`pred_cb`, `pred_xgb`, `pred_lgb`, `pred_r`, `pred_rv`) y no solo la mezcla final;
- si el modelo especialista se promociona a inferencia, debes reconstruir el mismo `reference_component_frame` en test para mantener paridad train/test.

3i. Ejecutar calibracion local sobre el score del incumbente

```bash
python scripts/experiment_local_calibrator.py --preset early_manual_internet --method platt
```

Usa los mismos OOF y archivo de pesos del incumbente, y por defecto escribe a
`artifacts/models/` y `artifacts/reports/`. Metodos disponibles: `platt`, `isotonic`.
Los nombres de pesos del JSON deben coincidir con los nombres entregados en `--oof`,
y la calibracion solo es valida si la submission final usa exactamente el mismo incumbente
que genero ese `reference_pred`.

3i2. Generar una submission final aplicando una cadena residual ya entrenada

```bash
python scripts/make_submission_residual_hierarchical.py \
  --train-csv data/raw/train.csv \
  --reference-submission artifacts/submissions/playground-series-s6e3-rvblend.csv \
  --step "early_all_internet|artifacts/models/residual_reranker_early_all_internet_midcap.cbm|1.0|H,R,S,V" \
  --step "fiber_paperless_early|artifacts/models/residual_reranker_fiber_paperless_early_midcap.cbm|1.0|H,R,S,V"
```

Este CLI no entrena ni valida: solo aplica, en orden, modelos residuales ya entrenados
sobre un submission base y escribe CSV final + JSON de trazabilidad.

Si alguno de esos rerankers fue entrenado con disagreement del teacher, la inferencia
debe recibir ademas el `reference_component_frame` correspondiente al incumbente base.
Puedes construirlo con:

```bash
python scripts/build_teacher_component_frame.py \
  --cbrv-submission artifacts/submissions/playground-series-s6e3-rvblend.csv \
  --output-csv artifacts/reports/reference_component_frame_rvblend_test.csv
```

El builder CLI se llama `build_teacher_component_frame.py` y escribe un CSV con
`id + pred_cb/pred_xgb/pred_lgb/pred_r/pred_rv`, mas un JSON de paridad que
reconstruye numericamente las tres submissions base.

Y luego aplicarlo al reranker teacher-aware:

```bash
python scripts/make_submission_residual_hierarchical.py \
  --train-csv data/raw/train.csv \
  --reference-submission artifacts/submissions/playground-series-s6e3-rvblend.csv \
  --reference-component-csv artifacts/reports/reference_component_frame_rvblend_test.csv \
  --reference-mode base \
  --step "early_all_internet|artifacts/models/residual_reranker_early_all_internet_midcap.cbm|1.0|H,R,S,V" \
  --step "fiber_paperless_early|artifacts/models/residual_reranker_fiber_paperless_early_teacher_midcap.cbm|1.0|H,R,S,V"
```

Formato de cada `--step`:
- `<preset>|<model_path>|<alpha>|<feature_blocks>`
- `preset`: una key de la lista de presets soportados
- `model_path`: modelo residual CatBoost ya entrenado sobre 100% de su cohorte
- `alpha`: multiplicador aditivo en `[0, 1]`
- `feature_blocks`: bloques usados al entrenar ese reranker; deben coincidir al aplicar inferencia

Prerequisitos minimos:
- `test.csv`
- `train.csv` si algun step usa bloques fit-aware como `G` o `T`
- submission base con columnas `id, Churn`
- modelos `.cbm` residuales ya entrenados

El JSON de salida guarda la secuencia aplicada, `changed_rows`, `mae_vs_base`,
`max_abs_shift_vs_base` y rango final de predicciones. Cada paso aplica clipping
numericamente estable en `[1e-6, 1 - 1e-6]`.

`--reference-mode` controla contra que score corre cada reranker:
- `previous`: cada paso usa la salida del paso anterior.
- `base`: todos los pasos usan la submission base original; esto es util cuando
  validaste los rerankers de forma independiente contra el mismo teacher.

Si pasas `--reference-component-csv`, debes usar `--reference-mode base`.
Combinar `teacher components` con `previous` rompe la paridad entre
`reference_pred_feature` y las features `teacher_component_*` despues del primer paso.

3i3. Orquestar builder teacher-aware + submission residual en un solo comando

```bash
python scripts/materialize_residual_submission.py \
  --teacher-aware \
  --train-csv data/raw/train.csv \
  --reference-mode base \
  --reference-submission artifacts/submissions/playground-series-s6e3-rvblend.csv \
  --output-csv artifacts/submissions/playground-series-s6e3-residual-hier-v2-cli.csv \
  --report-json artifacts/reports/submission_candidate_residual_hierarchical_v2_cli.json \
  --step "early_all_internet|artifacts/models/residual_reranker_early_all_internet_midcap.cbm|1.0|H,R,S,V" \
  --step "fiber_paperless_early|artifacts/models/residual_reranker_fiber_paperless_early_teacher_midcap.cbm|1.0|H,R,S,V"
```

Este wrapper:
- construye el `reference_component_frame` si activas `--teacher-aware`;
- guarda el CSV/JSON del builder con nombres derivados del output final;
- aplica luego la cadena residual con los mismos parametros;
- imprime `run_config` resuelto antes de materializar la submission.

Flags utiles del wrapper:
- `--step` es obligatorio y puede repetirse una o mas veces.
- `--teacher-aware` exige `--reference-mode base`.
- `--train-csv` es necesario si algun paso o reconstruccion usa bloques fit-aware como `G` o `T`.
- `--reference-component-csv` permite reutilizar un frame `id + pred_*` ya construido.
- `--reference-component-report-json` fija la ruta del JSON del builder cuando usas `--teacher-aware`.
- `--test-csv` permite apuntar a otro `test.csv`.
- `--cb-metrics-json`, `--cb-feature-blocks`, `--r-model-path`, `--r-feature-blocks` controlan como se reconstruyen `cb` y `r`.
- `--cbxgblgb-submission`, `--cbxgblgb-weights-json`, `--cbr-submission`, `--cbr-weights-json`, `--cbrv-submission`, `--cbrv-weights-json` controlan las tres submissions base usadas para resolver `xgb`, `lgb` y `rv`.

Defaults derivados cuando usas `--teacher-aware`:
- CSV del builder: `<output_csv_stem>_teacher_components.csv`
- JSON del builder: `<report_json_stem>_teacher_components.json`

Presets soportados por la linea local (`scripts/experiment_specialist_model.py`):
- `early_manual_internet`
- `early_all_internet`
- `fiber_paperless_early`
- `late_mtm_fiber`
- `late_mtm_fiber_paperless`
- `long_fiber_any`
- `manual_no_internet`
- `one_year_fiber_any`
- `one_year_dsl_any`
- `one_year_dsl_paperless_49plus`: One year + DSL + PaperlessBilling=Yes + tenure >= 49
- `two_year_fiber_any`
- `two_year_dsl_paperless_49plus`: Two year + DSL + PaperlessBilling=Yes + tenure >= 49
- `one_year_fiber_paperless_49plus`
- `one_year_fiber_paperless_25_48`: One year + Fiber optic + PaperlessBilling=Yes + tenure 25-48 inclusive
- `two_year_fiber_paperless_49plus`
- `two_year_fiber_nopaperless_49plus`: Two year + Fiber optic + PaperlessBilling=No + tenure >= 49
- `mtm_dsl_paperless_25_48_manual`: Month-to-month + DSL + PaperlessBilling=Yes + tenure 25-48 + pago manual
- `mtm_dsl_paperless_25_48_any`: Month-to-month + DSL + PaperlessBilling=Yes + tenure 25-48 + cualquier pago
- `mtm_nointernet_mailed_0_24`: Month-to-month + InternetService=No + Mailed check + tenure <= 24
- `mtm_nointernet_no_0_6`: Month-to-month + InternetService=No + PaperlessBilling=No + tenure <= 6
- `ec_mtm_fiber_any`: Electronic check + Month-to-month + Fiber optic + cualquier PaperlessBilling + cualquier tenure
- `ec_mtm_fiber_paperless_0_6`: Electronic check + Month-to-month + Fiber optic + PaperlessBilling=Yes + tenure <= 6
- `ec_mtm_fiber_paperless_25_48`: Electronic check + Month-to-month + Fiber optic + PaperlessBilling=Yes + tenure 25-48
- `ec_mtm_fiber_paperless_any`: Electronic check + Month-to-month + Fiber optic + PaperlessBilling=Yes + cualquier tenure
- `ec_mtm_dsl_paperless_0_6`: Electronic check + Month-to-month + DSL + PaperlessBilling=Yes + tenure <= 6
- `ec_mtm_dsl_paperless_any`: Electronic check + Month-to-month + DSL + PaperlessBilling=Yes + cualquier tenure

3i4. Probar una senal ortogonal barata con modelo lineal disperso

```bash
python scripts/experiment_linear_probe.py \
  --model-family spline_logistic \
  --feature-blocks "H,R,S,V" \
  --reference-weights-json artifacts/reports/submission_candidate_cb5_xgb3_lgb_r_rvhi_weights.json \
  --oof cb=artifacts/reports/train_cv_multiseed_gate_s5_hiiter_oof.csv#oof_ensemble \
  --oof xgb=artifacts/reports/train_xgboost_cv_multiseed_hiiter_oof.csv#oof_ensemble \
  --oof lgb=artifacts/reports/train_lightgbm_cv_full_hiiter_oof.csv#oof_pred \
  --oof r=artifacts/reports/fe_blockR_hiiter_oof.csv#oof_pred \
  --oof rv=artifacts/reports/fe_blockRV_hiiter_oof.csv#oof_pred
```

El probe lineal usa `one-hot + escalado numerico + logistic regression` para buscar una
senal menos correlacionada con los boosters. Reporta OOF, correlacion contra el incumbente
y un scan de blend simple.

Familias soportadas:
- `logistic`
- `spline_logistic`: aplica spline basis solo sobre un subconjunto curado de variables
  numericas continuas y deja el resto en camino lineal
- `catboost_meta`: booster chico sobre el mismo `teacher_meta`, pensado para detectar
  interacciones no lineales con regularizacion fuerte

Modos de features soportados:
- `raw`: usa la matriz tabular normal construida desde `train.csv` y `--feature-blocks`
- `teacher_meta`: usa una matriz compacta con:
  - cohort keys (`PaymentMethod`, `Contract`, `InternetService`, `PaperlessBilling`, `tenure_bin`)
  - `reference_pred_feature`
  - `reference_logit_feature`
  - componentes `pred_*` del teacher
  - estadisticas de desacuerdo entre componentes del teacher
  - requiere que `--reference-weights-json` y `--oof` apunten a predicciones OOF
    alineadas por `id`
  - `catboost_meta` solo se soporta sobre este modo

Nota:
- en `teacher_meta`, `--feature-blocks` se ignora; dejarlo en `none` evita ambiguedad

Ejemplo `teacher_meta`:

```bash
python scripts/experiment_linear_probe.py \
  --model-family logistic \
  --feature-mode teacher_meta \
  --feature-blocks none \
  --reference-is-oof \
  --reference-weights-json artifacts/reports/submission_candidate_cb5_xgb3_lgb_r_rvhi_weights.json \
  --oof cb=artifacts/reports/train_cv_multiseed_gate_s5_hiiter_oof.csv#oof_ensemble \
  --oof xgb=artifacts/reports/train_xgboost_cv_multiseed_hiiter_oof.csv#oof_ensemble \
  --oof lgb=artifacts/reports/train_lightgbm_cv_full_hiiter_oof.csv#oof_pred \
  --oof r=artifacts/reports/fe_blockR_hiiter_oof.csv#oof_pred \
  --oof rv=artifacts/reports/fe_blockRV_hiiter_oof.csv#oof_pred
```

3i4b. Probar un reranker orientado a ranking sobre el teacher

```bash
python scripts/experiment_rank_reranker.py \
  --feature-blocks H,R,S,V \
  --preset global \
  --loss-function PairLogitPairwise \
  --folds 2 \
  --iterations 150 \
  --learning-rate 0.05 \
  --depth 6 \
  --l2-leaf-reg 5.0 \
  --max-pairs-per-group 1000 \
  --early-stopping-rounds 40 \
  --metrics-path artifacts/reports/rank_reranker_pairlogit_smoke_metrics.json \
  --oof-path artifacts/reports/rank_reranker_pairlogit_smoke_oof.csv \
  --model-path artifacts/models/rank_reranker_pairlogit_smoke.cbm
```

Notas del reranker:
- usa el blend incumbente como `reference_pred`
- puede correr en modo global o local con `--preset`
  - default: `global`
  - los valores locales validos son los mismos presets de `experiment_specialist_model.py`
  - `python scripts/experiment_rank_reranker.py --help` muestra la lista exacta
- agrega `pred_cb/xgb/lgb/r/rv` y estadisticas de desacuerdo del teacher
- construye queries jerarquicas con fallback:
  - `segment5`
  - `segment3`
  - `contract_internet_tenure`
  - `contract_tenure`
  - `contract`
  - `global`
- combina el score del ranker con el teacher mediante `rank-average` ponderado por `alpha`
- limita el costo pairwise muestreando hasta `max-pairs-per-group` pares por query
- flags operativos importantes:
  - `--reference-weights-json` y `--oof` controlan exactamente el teacher base
  - `--alpha-grid` controla la mezcla `teacher + ranker`
  - `--stratify-mode` controla solo el split externo de CV
  - `--min-query-rows`, `--min-query-positive-rows`, `--min-query-negative-rows` controlan el fallback jerarquico
  - `--train-csv`, `--random-state` y `--verbose` quedan expuestos para reproducibilidad
- losses soportadas:
  - `PairLogitPairwise`
  - `YetiRankPairwise`

Ejemplo `catboost_meta`:

```bash
python scripts/experiment_linear_probe.py \
  --model-family catboost_meta \
  --feature-mode teacher_meta \
  --feature-blocks none \
  --reference-is-oof \
  --tree-iterations 300 \
  --tree-depth 4 \
  --tree-learning-rate 0.05 \
  --tree-l2-leaf-reg 8.0 \
  --tree-early-stopping-rounds 50 \
  --reference-weights-json artifacts/reports/submission_candidate_cb5_xgb3_lgb_r_rvhi_weights.json \
  --oof cb=artifacts/reports/train_cv_multiseed_gate_s5_hiiter_oof.csv#oof_ensemble \
  --oof xgb=artifacts/reports/train_xgboost_cv_multiseed_hiiter_oof.csv#oof_ensemble \
  --oof lgb=artifacts/reports/train_lightgbm_cv_full_hiiter_oof.csv#oof_pred \
  --oof r=artifacts/reports/fe_blockR_hiiter_oof.csv#oof_pred \
  --oof rv=artifacts/reports/fe_blockRV_hiiter_oof.csv#oof_pred
```

Prerequisitos minimos del probe lineal:
- `train.csv`
- para `--feature-mode raw`: solo `train.csv`
- para `--feature-mode teacher_meta`:
  - `--reference-is-oof`
  - `--reference-weights-json` con objeto `{"weights": {...}}`
  - uno o mas `--oof` con formato `<name>=<path>[#<prediction_column>]`
  - las keys de `weights` deben coincidir con los nombres entregados en `--oof`
  - el merge OOF debe contener columnas `pred_*` y cubrir todos los `id` de train
- nota: incluso en `--feature-mode raw`, si entregas `--reference-weights-json` para
  medir contra una referencia, debes entregar tambien `--oof` y `--reference-is-oof`
  porque el script reconstruye la referencia OOF y valida esa cobertura por `id`

3i5. Probar una senal ortogonal con `FM/FFM` via `river`

```bash
python -m pip install river

python scripts/experiment_fm_probe.py \
  --model-family ffm \
  --feature-blocks none \
  --reference-weights-json artifacts/reports/submission_candidate_cb5_xgb3_lgb_r_rvhi_weights.json \
  --oof cb=artifacts/reports/train_cv_multiseed_gate_s5_hiiter_oof.csv#oof_ensemble \
  --oof xgb=artifacts/reports/train_xgboost_cv_multiseed_hiiter_oof.csv#oof_ensemble \
  --oof lgb=artifacts/reports/train_lightgbm_cv_full_hiiter_oof.csv#oof_pred \
  --oof r=artifacts/reports/fe_blockR_hiiter_oof.csv#oof_pred \
  --oof rv=artifacts/reports/fe_blockRV_hiiter_oof.csv#oof_pred
```

Familias soportadas:
- `fm`
- `ffm`

El probe FM/FFM usa entrenamiento online con `river`, OOF por folds, correlacion contra
el incumbente y scan de blend simple. Es mas lento que el probe lineal, asi que conviene
usarlo primero en smoke pequeño antes de pensar en promocion.

Prerequisitos minimos del probe FM/FFM:
- `train.csv` con la misma estructura del proyecto base
- `river>=0.23.0`
- `--reference-weights-json` con objeto `{"weights": {...}}` si quieres comparar contra el incumbente
- uno o mas `--oof` con formato `<name>=<path>[#<prediction_column>]`

Salidas esperadas:
- modelo full-train serializado a `--model-path`
- OOF a `--oof-path`
- metricas y blend scan a `--metrics-path`

Reproducibilidad:
- usa `--random-state` para fijar folds y semilla del modelo
- el smoke recomendado parte con `--folds 2 --epochs 1`

3i6. Probar una senal ortogonal con `MLP` tabular y embeddings

```bash
python scripts/experiment_mlp_probe.py \
  --train-csv artifacts/tmp/train_smoke_20pct.csv \
  --feature-blocks none \
  --hidden-dims 128,64 \
  --epochs 6 \
  --device auto \
  --reference-weights-json artifacts/reports/submission_candidate_cb5_xgb3_lgb_r_rvhi_weights.json \
  --oof cb=artifacts/reports/train_cv_multiseed_gate_s5_hiiter_oof.csv#oof_ensemble \
  --oof xgb=artifacts/reports/train_xgboost_cv_multiseed_hiiter_oof.csv#oof_ensemble \
  --oof lgb=artifacts/reports/train_lightgbm_cv_full_hiiter_oof.csv#oof_pred \
  --oof r=artifacts/reports/fe_blockR_hiiter_oof.csv#oof_pred \
  --oof rv=artifacts/reports/fe_blockRV_hiiter_oof.csv#oof_pred
```

El probe MLP usa embeddings por categorica, normalizacion numerica por fold, OOF,
correlacion contra el incumbente y scan de blend simple.

Prerequisitos minimos del probe MLP:
- `train.csv` o una muestra estratificada compatible
- `torch>=2`
- `--reference-weights-json` con objeto `{"weights": {...}}` si quieres comparar contra el incumbente
- uno o mas `--oof` con formato `<name>=<path>[#<prediction_column>]`
- `feature-blocks` acepta `none` o cualquier combinacion soportada por el proyecto
- `device=auto` usa `cuda` si esta disponible; si no, cae a `cpu`

Salidas esperadas:
- bundle `.pt` a `--model-path` (default: `artifacts/models/mlp_probe_smoke.pt`)
- OOF a `--oof-path` (default: `artifacts/reports/mlp_probe_smoke_oof.csv`)
- metricas y blend scan a `--metrics-path` (default: `artifacts/reports/mlp_probe_smoke_metrics.json`)

3i7. Simular pseudo-labeling offline por familia

```bash
python scripts/experiment_pseudo_label_family.py \
  --family-key "Electronic check__Month-to-month__Fiber optic__Yes__13_24" \
  --feature-blocks "H,R,S,V" \
  --label-mode soft \
  --scale-pseudo-weight-by-confidence \
  --upper-thresholds "0.88,0.85" \
  --lower-thresholds "0.15,0.12" \
  --pseudo-weights "0.05,0.10,0.20"
```

Este experimento no usa `test.csv`. Hace una simulacion offline dentro de `train.csv`:
- separa un `holdout` global real para medir AUC;
- esconde una fraccion de la familia objetivo como `pseudo_pool`;
- genera pseudo-labels solo sobre esas filas usando el teacher OOF (`cb/xgb/lgb/r/rv`);
- reentrena un `student` con pesos bajos en esas filas pseudo-etiquetadas;
- barre `thresholds + pseudo_weight` y reporta el delta contra un baseline sin pseudo-labels.

Contrato minimo:
- `--family-key`: key exacta `segment5 = PaymentMethod__Contract__InternetService__PaperlessBilling__tenure_bin`
- `--oof` y `--reference-weights-json` deben reconstruir el teacher OOF por `id`
- si usas `--require-teacher-agreement` o `--max-teacher-std`, esos `--oof` deben exponer componentes `pred_*`, no solo la mezcla final
- `--label-mode hard`: usa pseudo etiquetas binarias por threshold
- `--label-mode soft`: duplica cada pseudo fila en dos ejemplos ponderados (`y=1` y `y=0`) usando la probabilidad del teacher
- `--scale-pseudo-weight-by-confidence`: multiplica el peso base por `2 * |p - 0.5|`
- `--require-teacher-agreement` exige que todos los componentes `pred_*` queden del mismo lado de `0.5`
- `--max-teacher-std` permite filtrar solo filas con bajo desacuerdo entre teachers
- defaults operativos: `holdout_size=0.15`, `valid_size=0.15`, `pseudo_pool_fraction=0.50`, `repeats=3`, `min_selected_rows=100`
- la familia objetivo debe tener al menos `2000` filas en `train`, y cada repeticion debe dejar al menos `1000` filas de esa familia en el split de fit
- `--results-csv-path` guarda el barrido detallado por repeticion/configuracion
- `--metrics-path` resume la mejor configuracion y el ranking de configs

La primera familia sugerida para smoke es:
- `Electronic check__Month-to-month__Fiber optic__Yes__13_24`

3j. Aplicar gate automatico GO/NO_GO para decidir submission

```bash
python scripts/gate_submission_candidate.py --candidate-name playground-series-s6e3 --parity-json artifacts/reports/diagnostic_submission_parity_issue5.json --drift-json artifacts/reports/diagnostic_train_test_drift_issue5.json --robustness-json artifacts/reports/diagnostic_ensemble_robustness_issue5.json
```

4. Enviar a Kaggle (opcional)

```bash
python scripts/submit_kaggle.py --message "playground-series-s6e3"
```

## Pipeline end-to-end (una sola llamada)

```bash
python scripts/run_baseline.py --feature-blocks "G,H,R,V" --stratify-mode composite --submit --message "playground-series-s6e3"
```

## Notas

- `scripts/` contiene solo CLIs con argumentos.
- La logica de negocio vive en `src/churn_baseline/`.
- `artifacts/` guarda salidas locales y no se versiona (solo estructura).
- Antes de descargar/enviar, debes aceptar reglas en `Join Competition`.
- Los priors jerarquicos usan target encoding fold-safe con smoothing y fallback de cohortes.
- `--include-deviation-features` agrega medias de cohorte y desvio relativo del cliente respecto de su cohorte jerarquica.
