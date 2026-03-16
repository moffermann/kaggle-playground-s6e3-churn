# Model Family Ledger

Fecha: `2026-03-16`

## Incumbent

- Submission operativa vigente: `v3`
- Kaggle `ref`: `50828079`
- `public score`: `0.91421`
- Working base para nuevas iteraciones: `v6`
  - `playground-series-s6e3-residual-hier-v6-compressed.csv`
  - Kaggle `ref 50974098`
  - `public score = 0.91421`
- Gate operativo: no promover ni subir candidatos con `delta local < 1e-05 en OOF AUC` contra `v3`, salvo fix de paridad o senal materialmente nueva.
- Referencias de terminologia:
  - `R` y `RV`: ver [README.md](../README.md)
  - `EC / MTM / Fiber`: macrofamilia `Electronic check / Month-to-month / Fiber optic`
  - protocolo operativo: [validation-reset-protocol.md](./validation-reset-protocol.md)

## Familias Que Si Produjeron Ganancia Durable

- Teacher robusto `CatBoost + XGB + LGB + R + RV`.
- Jerarquia residual `v3` sobre ese teacher.
- La mejora durable vino de:
  - diversidad moderada de modelos
  - una senal estructural global limitada (`R`, luego `RV`)
  - correcciones residuales muy focales

## Familias De Modelo Cerradas Como NO-GO

### Ajustes Incrementales Del Stack Base

- Seeds extra, scans de pesos y variantes cercanas del blend sin senal nueva.
- Regularizacion ciega de `RV`.
- Sample weighting damage-aware.
- Miembros family-aware por stratify compuesto.
- `clean-room baseline rebuild`:
  - `cb_raw`
  - `cb_r`
  - `cb_rv`
  - todos fallan el gate directo contra `v3`

### Especialistas, Challengers Y Overrides Locales

- `specialist masks` adicionales fuera de los que forman `v3`.
- Challengers supervisados sobre `EC / MTM / Fiber`:
  - `classifier`
  - `feature`
  - `gated`
- Reordenamiento de la jerarquia `v3` sin senal nueva.
- `residual hierarchy ablation/compression`:
  - mejor cadena comprimida:
    - `early_all_internet -> fiber_paperless_early -> late_mtm_fiber_paperless`
    - quita `late_mtm_fiber`
    - `delta_vs_v3 = -3.5855e-07`
    - `FAIL`
  - conclusion:
    - la cadena residual no es promocionable en una forma comprimida
    - `late_mtm_fiber` tiene la menor contribucion marginal, pero no hay reemplazo del incumbent
  - ajuste despues de submissions reales del `2026-03-16`:
    - `v6` (3 pasos) empata el mejor publico
    - `v10` (2 pasos) tambien empata el mejor publico
    - las variantes con `early_g` empeoran
    - decision:
      - `v3` sigue siendo el incumbent publico
      - `v6` pasa a ser la working base mas razonable por simplicidad y cercania a `v3`

### Rerankers Y Meta-Modelos

- Ranking reranker global.
- Ranking reranker local.
- `teacher_meta` lineal.
- `teacher_meta` no lineal (`catboost_meta`).
- `uncertainty-band reranker` sobre `v3`:
  - familia objetivo: `Electronic check / Month-to-month / Fiber optic`
  - banda ambigua: `abs(v3 - 0.5) <= 0.20`
  - variantes cerradas:
    - banda pura
    - banda mas estrecha (`0.15`, `0.10`)
    - banda + desacuerdo alto del teacher (`min_teacher_std`)
  - mejor delta observado contra `v3`: `+6.69e-06`
  - no alcanza el gate operativo `1e-05`
- `hard-example stability score` sobre `v3`:
  - fuente: OOF repetidos baratos (`R,V`)
  - score: `stability_pred_std + stability_flip_rate`
  - variantes cerradas:
    - `q80`
    - `q90`
    - `q80 + banda ambigua de v3`
  - mejor delta observado contra `v3`: `+8.28e-07`
  - no alcanza el gate operativo `1e-05`
- `counterfactual teacher sensitivity` contra `v3`:
  - senal: cuanto cae un subconjunto del teacher (`cb`, `r`, `rv`) bajo perturbaciones plausibles del mismo cliente
  - contrafactuales probados:
    - `auto_payment`
    - `paperless_off`
    - `contract_upgrade`
    - `stability_bundle`
  - mejor variante global:
    - `stable_bundle_drop`
    - `alpha = 0.05`
    - `delta_vs_v3 = +5.879856e-06`
  - variantes adicionales:
    - banda `abs(v3-0.5) <= 0.20`: `+2.697875e-06`
    - banda `abs(v3-0.5) <= 0.15`: `+3.279829e-06`
  - mejora la macrofamilia dominante, pero no alcanza el gate global `1e-05`
  - queda `NO-GO` para promocion

### Regularizacion Estructural

- Monotonic constraints.
- Nuevos bloques globales que no sobrevivieron contra `v3`:
  - `F` surface block
  - `T` fit-aware surface block
  - `G` coverage/backoff global

### Semi-Supervisado

- Pseudo-labeling por familia:
  - `hard`
  - `soft`
  - `soft + confidence weighting`

### Intervenciones Data-Centric

- Auditoria de `label noise`:
  - util como diagnostico
  - no hay evidencia para filtrar globalmente
- Mitigacion minima de `near-duplicate conflicts`:
  - `downweight` local por fold
  - `drop` local por fold
  - ambas fallan directamente contra `v3`
- Reglas cerradas:
  - no hacer drops masivos
  - no bajar peso por sospecha global
  - no tocar cohortes grandes completas por auditoria sola

### Familias De Modelo Alternativas

- `linear probe`
- `spline logistic`
- `FFM`
- `MLP embeddings`
- `bi/tri-gram TE + XGBoost`
- `GNN starter + ANN graph`
- `external telco transfer feature`
- `source-aware joint training con Telco original`

## Regla Para No Reabrir Una Familia Cerrada

Una familia cerrada no se reabre salvo que exista al menos una de estas condiciones:

- nueva fuente de senal que no estaba disponible antes
- cambio de validacion que demuestre que el falso negativo era del protocolo y no del modelo
- fix de paridad o leakage que invalide el resultado previo

No basta con:

- cambiar la seed
- mover ligeramente hiperparametros
- hacer otro scan de blend
- probar otra mascara muy parecida

## ROI Restante

La evidencia acumulada apunta a que el ROI restante ya no esta en mas filtros sobre filas sospechosas ni en otra familia de modelo global cercana al stack actual.

Las apuestas que siguen vivas son:

1. fuente de senal materialmente nueva que nazca ya comparada contra `v3`
2. stress tests de generalizacion por familia dominante y por banda ambigua
3. diagnostico de ejemplo duro/inestable solo como soporte, no como challenger

## Filtro Para La Proxima Hipotesis

La siguiente linea solo se abre si cumple al menos una de estas condiciones:

- agrega una fuente de senal que no este ya contenida en `cb/xgb/lgb/r/rv`
- o cambia la geometria del problema de forma demostrable, no solo el orden de un reranker local
- y puede evaluarse contra `v3` desde `smoke` sin depender de un teacher intermedio

La siguiente linea no debe ser:

- otro reranker local sobre la misma macrofamilia dominante
- otro mask dentro de `EC / MTM / Fiber`
- otro challenger global cercano a `R,V`
- otra intervencion data-centric sin senal nueva
- otro score de dureza/estabilidad derivado solo de resampling del mismo stack

## Recomendacion Operativa

Antes de abrir una hipotesis nueva:

1. definir explicita y por adelantado:
   - fuente de senal nueva
   - por que no esta ya absorbida por `v3`
   - como pasaria el gate directo contra `v3`
2. descartar desde el diseno cualquier idea cuyo mejor caso razonable siga estando en el orden de `1e-06`

La hipotesis de `source-aware joint training con Telco original` tambien queda cerrada como `NO-GO`.

Resultado minimo:

- `R,V + dataset_source + filas blastchar`
- `external_weight = 0.25`
- `candidate_oof_auc = 0.9137443511`
- `delta_vs_v3 = -0.0031935925`
- veredicto: `FAIL`

Lectura:

- la fuente externa sigue siendo util como referencia de dominio
- pero esta formulacion minima no agrega una senal competitiva sobre `v3`

La siguiente apuesta recomendada vuelve a ser un filtro, no un modelo concreto:

- solo abrir una hipotesis nueva si explica explicitamente por que su senal no esta ya absorbida por `v3`

## Submission Forensics

- Historial Kaggle auditado:
  - `20` submissions historicas
  - `18` `COMPLETE`
  - `2` `ERROR`
- Mejor submission publico:
  - `playground-series-s6e3-residual-hier-v3.csv`
  - `ref 50828079`
  - `public score 0.91421`
- Familia con supervivencia publica repetida:
  - `residual_hierarchy`
- Familias que tuvieron score publico razonable pero sin repeticion comparable:
  - `teacher_blend_rv`
  - `teacher_blend_r`
  - `pseudo_labeling`
- Lectura operativa:
  - el historial publico refuerza que las mejoras durables vinieron de la familia residual, no de challengers standalone ni de lineas exploratorias aisladas

## Residual Distillation

- La linea `total residual distillation` queda cerrada como `NO-GO` para reemplazar al incumbent.
- Resultado offline:
  - `smoke`: `PASS`
  - `midcap`: `PASS`
  - mejor `delta_vs_v3_oof_auc = +5.1685966e-05`
  - `best_alpha = 4.0`
- Resultado de materializacion:
  - flujo de submit implementado y validado con el mismo `train_csv_path` y la misma referencia base `rvblend`
  - `submission gate`: `PASS`
- Resultado Kaggle:
  - submission `ref 50925374`
  - archivo `playground-series-s6e3-residual_distillation_midcap.csv`
  - `public score = 0.91414`
  - delta vs incumbent `v3`: `-0.00007`
  - traza local: `artifacts/reports/submission_result_residual_distillation_midcap.json`
- Lectura operativa:
  - es la primera linea nueva que pasa `smoke`, `midcap` y `submission gate` directamente contra `v3`
  - aun asi pierde en leaderboard publico
  - conclusion: el stack de validacion actual ya es bastante mejor, pero todavia no separa por completo `offline winner` de `public winner`
  - accion tomada: `submission gate v2` ahora agrega `submission_family_survival_prior`
