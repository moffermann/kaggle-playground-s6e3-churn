# Estado del proyecto

Fecha de inicializacion: `2026-03-04`
Ultima actualizacion: `2026-03-13`

Estado operativo actual:
- resumen ejecutivo: [phase-reset-summary.md](./phase-reset-summary.md)
- ledger de familias/modelos: [model-family-ledger.md](./model-family-ledger.md)

## Hecho

- Repositorio GitHub creado:
  - `https://github.com/moffermann/kaggle-playground-s6e3-churn`
- Carpeta local creada en:
  - `C:\\devel\\kaggle\\kaggle-playground-s6e3-churn`
- Datos descargados en `data/raw/`:
  - `playground-series-s6e3.zip`
  - `train.csv`
  - `test.csv`
  - `sample_submission.csv`

## Notas

- Los CSV y ZIP se ignoran en git por `.gitignore`.
- Esta competencia usa `ROC AUC` como metrica.
- Mejor incumbent publico vigente:
  - `v3`
  - Kaggle `ref 50828079`
  - `public score = 0.91421`
- La linea data-centric de `label noise / near-duplicates` quedo cerrada como `NO-GO` para mitigacion directa.
- La linea `external telco transfer feature` tambien quedo cerrada como `NO-GO`.
- La linea `source-aware joint training con Telco original` tambien quedo cerrada como `NO-GO`.
- La linea `uncertainty-band reranker` tambien quedo cerrada como `NO-GO`.
- La linea `hard-example stability score` tambien quedo cerrada como `NO-GO`.
- La linea `counterfactual teacher sensitivity` tambien quedo cerrada como `NO-GO`.
- `submission forensics` ya esta operativo y confirma que la familia con mejor supervivencia publica es `residual_hierarchy`.
- El `clean-room baseline rebuild` ya confirmo que `v3` sigue muy por encima de `cb_raw`, `cb_r` y `cb_rv` reconstruidos desde cero.
- La auditoria de `residual hierarchy ablation/compression` confirma que la familia residual sigue siendo el nucleo operativo de `v3`; la mejor compresion quita `late_mtm_fiber`, pero aun asi pierde y no pasa gate.
- La linea `total residual distillation` ya quedo cerrada como `NO-GO` para reemplazo:
  - pasa `smoke`, `midcap` y `submission gate` contra `v3`
  - pero la submission real `ref 50925374` hizo `0.91414`
  - queda `-0.00007` debajo del incumbent `v3`
  - traza local: `artifacts/reports/submission_result_residual_distillation_midcap.json`
- Siguiente direccion recomendada:
  - no abrir otra linea sin pasar primero por el filtro documentado en `phase-reset-summary.md` y `model-family-ledger.md`
  - toda hipotesis nueva debe nacer comparada directamente contra `v3` desde `smoke`
  - no reabrir `source-aware joint training` sin una formulacion materialmente distinta a la minima ya descartada
