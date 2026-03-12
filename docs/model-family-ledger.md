# Model Family Ledger

Fecha: `2026-03-12`

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

La evidencia acumulada apunta a que el ROI restante ya no esta en mas filtros sobre filas sospechosas ni en otra familia de modelo cercana al stack actual.

Las apuestas que siguen vivas son:

1. uso supervisado de fuente externa alineada al dominio (`telco-customer-churn`) como senal transferida
2. identificacion de ejemplos duros/inestables por fold y por familia, pero solo como diagnostico
3. stress tests de generalizacion por familia dominante

## Proxima Hipotesis

La siguiente linea recomendada es:

- `external telco transfer feature`

Objetivo:

- entrenar un teacher externo sobre el dataset original `telco-customer-churn`
- proyectar esa senal sobre `train/test` de la competencia
- medir si aporta valor frente a `v3` como feature o como miembro complementario
