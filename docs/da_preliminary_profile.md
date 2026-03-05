# DA Preliminary Profile

Fecha: `2026-03-05`
Dataset: `data/raw/train.csv`

## Resumen general

- Rows: `594,194`
- Columns: `21`
- Target distribution:
  - `No`: `460,377` (`77.48%`)
  - `Yes`: `133,817` (`22.52%`)

## Numeric summary (mean/std/min/max/percentiles)

### SeniorCitizen

- mean: `0.1141`
- std: `0.3179`
- min: `0`
- p01: `0`
- p05: `0`
- p25: `0`
- p50: `0`
- p75: `0`
- p95: `1`
- p99: `1`
- max: `1`

### tenure

- mean: `36.5773`
- std: `25.0619`
- min: `1`
- p01: `1`
- p05: `2`
- p25: `14`
- p50: `35`
- p75: `62`
- p95: `72`
- p99: `72`
- max: `72`

### MonthlyCharges

- mean: `65.8662`
- std: `31.0674`
- min: `18.25`
- p01: `19.30`
- p05: `19.65`
- p25: `33.90`
- p50: `74.10`
- p75: `95.25`
- p95: `109.20`
- p99: `115.50`
- max: `118.75`

### TotalCharges

- mean: `2494.3771`
- std: `2353.9167`
- min: `18.80`
- p01: `33.90`
- p05: `78.45`
- p25: `287.95`
- p50: `1433.65`
- p75: `4122.65`
- p95: `7222.75`
- p99: `8240.85`
- max: `8684.80`

## Segmentos sub-representados (feature DA)

Definicion de segmento: `PaymentMethod x Contract x InternetService`.

- Segmentos totales: `36`
- Segmentos con soporte <= `0.50%` del train: `7`
- Mas pequenos:
  - `Mailed check | Two year | Fiber optic`: `897` (`0.151%`), churn `0.0212`
  - `Electronic check | Two year | No`: `907` (`0.153%`), churn `0.0121`
  - `Mailed check | One year | Fiber optic`: `1,007` (`0.169%`), churn `0.0914`

## Segmentos con mayor churn (preliminar)

- `Electronic check | Month-to-month | Fiber optic`: churn `0.6123`, `148,153` filas (`24.933%`)
- `Mailed check | Month-to-month | Fiber optic`: churn `0.4297`, `9,515` filas (`1.601%`)
- `Electronic check | Month-to-month | DSL`: churn `0.3309`, `31,564` filas (`5.312%`)

## Artefactos locales

- `artifacts/reports/da_preliminary_profile.json`
- `artifacts/reports/da_preliminary_profile.md`
