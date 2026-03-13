# Model Family Ledger

Fecha: `2026-03-13`

## Incumbent

- Submission operativa vigente: `v3`
- Kaggle `ref`: `50828079`
- `public score`: `0.91421`
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

### Especialistas, Challengers Y Overrides Locales

- `specialist masks` adicionales fuera de los que forman `v3`.
- Challengers supervisados sobre `EC / MTM / Fiber`:
  - `classifier`
  - `feature`
  - `gated`
- Reordenamiento de la jerarquia `v3` sin senal nueva.

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

1. diagnostico de ejemplo duro/inestable con trazabilidad por familia y por banda de confianza
2. fuente de senal materialmente nueva que nazca ya comparada contra `v3`
3. stress tests de generalizacion por familia dominante y por banda ambigua

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

## Recomendacion Operativa

Antes de abrir una hipotesis nueva:

1. definir explicita y por adelantado:
   - fuente de senal nueva
   - por que no esta ya absorbida por `v3`
   - como pasaria el gate directo contra `v3`
2. descartar desde el diseno cualquier idea cuyo mejor caso razonable siga estando en el orden de `1e-06`
