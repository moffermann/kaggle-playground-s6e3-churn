# Resumen De Reset De Fase

Fecha: `2026-03-11`

## Incumbent

- Submission operativa vigente: `v3`
- Kaggle `ref`: `50828079`
- `public score`: `0.91421`
- Regla base: no reemplazar `v3` con candidatos cuyo `delta local` en `OOF AUC` contra el incumbent sea `< 1e-05`, salvo fix de paridad o senal materialmente nueva.
- Terminologia corta:
  - `R` y `RV`: ver [README.md](../README.md)
  - `EC / MTM / Fiber`: macrofamilia `Electronic check / Month-to-month / Fiber optic`

## Lo Que Si Funciono

- `CatBoost + XGB + LGB + R + RV` como teacher/blend robusto.
- Jerarquia residual `v3` sobre ese teacher.
- La mejora competitiva real vino de:
  - diversidad moderada de modelo
  - una pequena senal estructural (`R`, luego `RV`)
  - rerankers residuales muy focales

## Lo Que Ya Esta Cerrado Como NO-GO

- Pseudo-labeling por familia:
  - `hard`
  - `soft`
  - `soft + confidence weighting`
- Ranking reranker:
  - global
  - local
- Monotonic constraints.
- Modelos ortogonales baratos:
  - `linear probe`
  - `spline logistic`
  - `FFM`
  - `MLP embeddings`
- `teacher_meta`
- Nuevos `specialist masks` y variantes de superficie sobre familias ya agotadas
- Challengers supervisados directos sobre `EC / MTM / Fiber`:
  - `classifier`
  - `feature`
  - `gated`
- `bi/tri-gram TE + XGBoost`
- `GNN starter + ANN graph`

## Lo Que Si Aprendimos

- La macrofamilia dominante del problema es:
  - `Electronic check / Month-to-month / Fiber optic`
- El cuello de botella no parece ser:
  - falta de otro seed
  - otro blend scan
  - otro mask local parecido
- El cuello parece estar en:
  - generalizacion real a familias/cohortes dominantes
  - evitar falsos positivos locales que no sobreviven al leaderboard

## Ambiguo Pero Util

- `EC / MTM / Fiber` sigue siendo la familia mas danina.
- Hay mejora local en algunas variantes dentro de esa macrofamilia, pero no suficiente ni estable contra `v3`.
- La validacion local sigue siendo util, pero no lo bastante dura como para discriminar bien mejoras pequenas.

## Reglas Operativas

- No subir submissions con `delta local < 1e-05` en `OOF AUC` contra `v3`.
- No promover una idea a `5x600` si no gana su smoke correspondiente.
- No abrir mas masks dentro de familias ya agotadas.
- No repetir scans de pesos o seeds sin una senal nueva y menos correlacionada.
- No abrir mas hipotesis simultaneas; maximo una linea activa a la vez.

## Siguiente Fase Permitida

1. Recomendado: `validation reset`
   - reforzar validacion realista por familia/cohorte
   - medir mejor riesgo `CV -> public/private`
   - usar eso como filtro antes de volver a entrenar modelos nuevos
   - protocolo formal: [validation-reset-protocol.md](./validation-reset-protocol.md)

2. Solo despues del reset:
   - probar una familia de modelo realmente distinta o una representacion nueva
   - no otra variante de CatBoost local sobre la misma macrofamilia

## Decision

- `v3` sigue siendo el incumbent.
- La siguiente apuesta no debe ser otro experimento pequeno sobre `EC / MTM / Fiber`.
- La siguiente apuesta debe empezar por evaluacion y criterio de promocion, no por mas CPU.
