# Validation Reset Protocol

Fecha: `2026-03-11`

## Objetivo

Definir un protocolo de evaluacion mas realista antes de:
- promover una idea a entrenamiento caro
- autorizar una submission nueva

Este protocolo existe para reducir falsos positivos locales y dejar trazabilidad clara de por que una idea pasa o no pasa.

## Incumbent De Referencia

- Submission vigente: `v3`
- Kaggle `ref`: `50828079`
- `public score`: `0.91421`
- Referencia local minima: comparar siempre contra el incumbent equivalente en `OOF AUC`

## Definiciones Operativas

- `delta local`: `candidate_oof_auc - reference_oof_auc` bajo el mismo split y la misma seed de evaluacion.
- `incumbent comparable`: rerun del incumbent bajo exactamente:
  - mismo split
  - mismas seeds
  - mismo protocolo de features permitidas para la comparacion
- `familia`: por defecto `segment5 = PaymentMethod x Contract x InternetService x PaperlessBilling x tenure_bin`.
- `familia grande`: familia con `train_rows >= 5000` o `test_rows >= 2000`.
- `familia promotable`: familia con `train_rows >= 2000` y `test_rows >= 800`.
- `familia microscopica`: familia con `train_rows < 1000` o `test_rows < 400`.
- `macrofamilia`: por defecto `segment3 = PaymentMethod x Contract x InternetService`.
- `macrofamilia dominante`: `Electronic check / Month-to-month / Fiber optic`.
- `familia objetivo`: la familia principal declarada por la hipotesis.
- `familia hermana`: misma macrofamilia de la hipotesis y mismo `PaperlessBilling`, cambiando solo al bucket de tenure vecino mas cercano en este orden:
  - `0_6 -> 7_12 -> 13_24 -> 25_48 -> 49_plus`
  - si la hipotesis no depende de `tenure_bin`, la familia hermana se define como el mismo grupo con `PaperlessBilling` invertido
  - la familia hermana debe quedar nombrada explicitamente en el experimento
- `top-3 por dano`: top-3 familias grandes ordenadas por `reference_logloss_contribution` en el split evaluado.

## Metricas Definidas

- `OOF AUC`: ROC AUC global sobre todas las filas OOF.
- `OOF AUC on-mask`: ROC AUC calculado solo sobre la familia objetivo o familia auditada.
- `logloss contribution por familia`: `mean_logloss_familia * (rows_familia / rows_eval)`.
- `generalization_gap_logloss_contribution`: `lofo_logloss_contribution - reference_logloss_contribution`.
- `cv_std`: desviacion estandar del AUC por fold.
- `OOF correlation`: correlacion de Pearson entre el vector completo `candidate_pred` OOF y el vector `reference_pred` OOF bajo Split A.
- `standalone delta`: `challenger_oof_auc - reference_oof_auc` bajo Split A, con `alpha = 1.0`.

## Estructura De Evaluacion

### 1. Smoke

Uso:
- ideas nuevas
- bloques nuevos de features
- cambios de gating o weighting

Configuracion minima:
- `2 folds`
- `150-250` iteraciones maximo
- `1 seed`

Gate para pasar:
- mejora local `>= 1e-05` en `OOF AUC`
- en Split B:
  - la macrofamilia dominante no puede caer mas de `2e-04` en `OOF AUC on-mask`
  - ninguna familia grande del top-3 por dano puede empeorar mas de `3%` relativos en `reference_logloss_contribution`
- la mejora no puede depender solo de familias microscopicas

### 2. Midcap

Uso:
- solo ideas que pasaron smoke

Configuracion minima:
- `5 folds`
- `~600` iteraciones
- mismo setup de features que gano smoke

Gate para pasar:
- mejora local `>= 1e-05` contra el incumbent comparable
- `cv_std_candidate <= 1.15 * cv_std_reference`
- en Split B:
  - familia objetivo: `delta OOF AUC on-mask >= -1e-04` contra la referencia on-mask del mismo split
  - macrofamilia dominante: `delta OOF AUC on-mask >= -2e-04` contra la referencia on-mask del mismo split
  - ninguna familia grande del top-3 por dano puede empeorar mas de `3%` relativos en `reference_logloss_contribution` del mismo split

### 3. Submission Candidate

Uso:
- solo ideas que pasaron midcap

Requisitos:
- paridad de train/inferencia verificada
- soporte suficiente tambien en `test`
- mejora local no explicada solo por familias microscopicas

Gate para subir:
- mejora local `>= 1e-05` en Split A
- y familia objetivo `promotable`
- y ninguna familia critica empeorada fuera de threshold; `familias criticas` significa:
  - familia objetivo
  - familia hermana
  - macrofamilia dominante
  - top-3 por dano
  - los thresholds que aplican aqui son los mismos thresholds de Split B, medidos contra la referencia del mismo split
- excepcion 1: `fix de paridad`
  - solo si no cambia el modelo entrenado
  - y corrige un `audit FAIL` o una ruptura train/inferencia demostrable
- excepcion 2: `senal materialmente nueva`
  - solo si la nueva linea tiene `OOF correlation <= 0.995` contra el incumbent comparable
  - o `standalone delta >= 2e-05` bajo Split A
- y `submission_family_survival_prior`
  - si la familia de submission ya tiene supervivencia publica probada, pasa por prior
  - `supervivencia publica probada` significa:
    - al menos `2` submissions historicas en esa familia
    - y `best_public_score >= incumbent_public_score - 1e-05`
  - si no la tiene, solo pasa como excepcion si el `delta local >= 7.5e-05`
  - este check usa `artifacts/reports/submission_forensics_summary.json` como evidencia historica
  - el CLI permite sobrescribir:
    - `--submission-forensics-summary`
    - `--submission-family`
  - si la evidencia historica no esta disponible, no se puede leer o viene incompleta, el check degrada a `WARN`; no debe usarse para autorizar una submission competitiva sin revisar ese vacio

## Tipos De Split

### Split A: CV base

Objetivo:
- medir el comportamiento general

Regla:
- mantener el `StratifiedKFold` actual como baseline de comparacion
- este split define el gate principal de `delta local`

### Split B: Family-aware stress

Objetivo:
- medir sensibilidad a familias dominantes

Regla:
- repetir diagnosticos por familia dominante, al menos:
  - `Electronic check / Month-to-month / Fiber optic`
  - familia secundaria relevante si aparece en la hipotesis

Metricas a mirar:
- `OOF AUC` on-mask
- contribucion a logloss
- soporte en `train` y `test`

Uso en gates:
- Split B no promociona por si solo
- Split B funciona como veto cuando una mejora global esconde dano en familias grandes

### Split C: Leave-one-family-out

Objetivo:
- medir generalizacion fuera de familia

Regla:
- usar `leave-one-family-out` como diagnostico, no como gate principal de promocion

Salida esperada:
- ranking de familias por `generalization_gap_logloss_contribution`

## Stress Tests Obligatorios

Antes de subir una idea:
- revisar familias top por dano actual
- revisar si la mejora depende de familias con soporte bajo en `test`
- revisar si la idea empeora la macrofamilia dominante aunque mejore el global

Lista minima:
1. familia objetivo
2. macrofamilia dominante
3. familia hermana mas cercana
4. mezcla global

Pass minimo:
- ninguna familia grande auditada puede romper los thresholds de Split B
- si la familia objetivo no mejora, la idea no sube salvo que el global gane `>= 2e-05` y no haya dano en familias grandes

## Metricas De Decision

### Primaria

- `OOF AUC`

### Secundarias

- `delta_vs_reference_oof_auc`
- `OOF AUC` on-mask
- `logloss contribution` por familia
- soporte `train/test`
- estabilidad entre folds

### Regla de interpretacion

- una mejora solo on-mask no basta
- una mejora global con dano fuerte en familias grandes tampoco basta
- una mejora menor a `1e-05` no justifica submission
- una mejora sostenida solo en familias microscopicas no justifica promocion

## Checklist Previo A Submission

1. La idea paso smoke
2. La idea paso midcap
3. El delta local supera el gate
4. No depende de una familia de bajo soporte en `test`
5. La inferencia reproduce exactamente el pipeline de train
6. El incumbent contra el que se compara es explicito
7. El artefacto de submission queda trazado con:
   - metrics JSON
   - OOF CSV si existe
   - submission CSV
   - commit SHA o branch origen
8. La familia de submission pasa el prior historico o la excepcion de familia nueva fuerte

Si alguno falla:
- no subir

## Reglas De Descarte

Descartar una linea si ocurre cualquiera:
- `best_alpha = 0.0`
- `delta_vs_reference_oof_auc < 1e-05`
- submission plana o peor:
  - `public_score <= incumbent_public_score`
  - si queda en `[-0.00001, +0.00001]` se considera inconclusa, pero no promociona la linea
- mejora solo contra teacher base y no contra incumbent real
- la idea repite una familia o tecnica ya agotada

## Proxima Aplicacion Recomendada

Usar este protocolo primero en:
1. cualquier hipotesis nueva que toque la macrofamilia `EC / MTM / Fiber`
2. cualquier nueva familia de modelo que pretenda reemplazar o complementar `v3`

## Decision Operativa

- No abrir un nuevo experimento de modelo sin pasar por este protocolo.
- La siguiente fase empieza por evaluacion y criterio de promocion, no por mas CPU.
