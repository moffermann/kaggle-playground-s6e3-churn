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
- Siguiente direccion recomendada: `uncertainty-band reranker` focal sobre la banda ambigua de `v3`.
