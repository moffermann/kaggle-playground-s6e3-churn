# Resumen De Reset De Fase

Fecha: `2026-03-11`
Ultima actualizacion: `2026-03-13`

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

Lista exhaustiva y mantenida en:
- [model-family-ledger.md](./model-family-ledger.md)

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
- Intervenciones data-centric minimas sobre filas sospechosas:
  - `near-duplicate downweight`
  - `near-duplicate drop`

## Lo Que Si Aprendimos

- La macrofamilia dominante del problema es:
  - `Electronic check / Month-to-month / Fiber optic`
- La auditoria data-centric sirve para orientar investigacion:
  - `label noise` puro parece minoritario
  - la mayor parte de la sospecha se concentra en `near-duplicates / cohortes casi repetidas`
  - pero mitigar eso de forma minima dano el ranking frente a `v3`
- El diagnostico agresivo de dominancia de `v3` agrego una lectura mas precisa:
  - `v3` gana de forma consistente en la macrofamilia dominante para casi todos los challengers
  - la mayor presion aparece en la banda de baja confianza de `v3` (`abs(v3-0.5) <= 0.20`)
  - cuando un challenger se aleja mucho de `v3`, normalmente pierde mas logloss y ranking
  - el residual casi-vivo mejora algo de logloss, pero no logra mejorar ranking global
  - incluso la linea mas alineada con ese hallazgo (`uncertainty-band reranker`) no supero el gate:
    - mejor delta contra `v3`: `+6.69e-06`
    - sigue por debajo del minimo operativo `1e-05`
- El cuello de botella no parece ser:
  - falta de otro seed
  - otro blend scan
  - otro mask local parecido
- El cuello parece estar en:
  - generalizacion real a familias/cohortes dominantes
  - evitar falsos positivos locales que no sobreviven al leaderboard
  - resolver mejor la zona ambigua de `v3`, no reemplazarlo globalmente

## Ambiguo Pero Util

- `EC / MTM / Fiber` sigue siendo la familia mas danina.
- Hay mejora local en algunas variantes dentro de esa macrofamilia, pero no suficiente ni estable contra `v3`.
- La banda ambigua de `v3` si existe como zona de presion real, pero ya no justifica otra linea de reranker local similar.
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
   - probar una fuente de senal realmente distinta o una representacion nueva
   - no otra variante de CatBoost local sobre la misma macrofamilia

## Decision

- `v3` sigue siendo el incumbent.
- El `validation reset` ya quedo implementado y es prerrequisito cumplido.
- La siguiente apuesta no debe ser otro experimento pequeno sobre `EC / MTM / Fiber`.
- La siguiente apuesta debe empezar por evaluacion y criterio de promocion, no por mas CPU.
- La hipotesis de `external telco transfer feature` ya quedo cerrada como `NO-GO`.
- La hipotesis de `uncertainty-band reranker` tambien queda cerrada como `NO-GO`.
- La siguiente hipotesis recomendada ahora no es un modelo especifico, sino un filtro:
  - debe aportar senal materialmente nueva
  - debe explicitar por que `v3` no la absorbe ya
  - debe nacer comparada directamente contra `v3` desde `smoke`
