# Market volatility baseline evaluation

This experiment predicts 20-trading-day forward realized volatility, expressed
as an annualized rate. It establishes transparent references before a more
flexible model is considered. The results are development evidence on historical
US large-cap data, not a claim of future trading performance.

## Evaluation contract

- Source snapshot: `us-large-cap-sip-price-v1`, 50 equities, 2016–2025
- Model dataset: `us-large-cap-volatility-v1`
- Train period: 2016–2021
- Validation period: 2022–2023, 24,050 rows
- Test period: 2024–2025, not evaluated during baseline selection
- Primary metric: MAE
- Secondary metrics: RMSE and volatility-specific QLIKE
- Split boundaries use a 20-trading-day purge matching the target horizon.

All rolling and EWMA inputs use observations available by the forecast date.
Ridge preprocessing is fitted only on train rows. Ridge predicts log volatility
and applies `exp` to produce positive forecasts.

## Validation results

| Strategy | MAE | RMSE | QLIKE |
|---|---:|---:|---:|
| Train mean | 0.0869 | 0.1262 | 0.4802 |
| Train median | 0.0962 | 0.1418 | 0.8045 |
| 20-day persistence | 0.0814 | 0.1173 | 0.3510 |
| EWMA (`lambda = 0.94`) | 0.0738 | 0.1064 | **0.2647** |
| Ridge, ten features | 0.0667 | 0.1034 | 0.2747 |
| Ridge, ten features plus EWMA | **0.0667** | **0.1034** | 0.2745 |

Ridge improves MAE over EWMA by about 9.6%, showing that the combined feature
set contains useful signal beyond a single volatility estimate. EWMA retains the
best QLIKE, so the linear model does not dominate every risk-sensitive metric.

Adding EWMA to Ridge changes the metrics by less than 0.05%. The ten-feature
Ridge model therefore remains the linear ML baseline: the negligible gain does
not justify an additional snapshot-derived feature path. EWMA remains a strong
standalone reference and may be tested again with nonlinear models.

## Reproduction

The command below verifies the immutable dataset and source-snapshot checksums,
evaluates only train and validation periods, and writes a versioned report under
the ignored runtime directory:

```powershell
uv run financial-ai-evaluate-market-validation `
  --dataset-version us-large-cap-volatility-v1 `
  --evaluation-version us-large-cap-baselines-v1
```

Reports are written to `data/runtime/ml/market_forecast/evaluations/` and cannot
overwrite an existing version.

## Limitations

- The universe contains current large-cap survivors and is not a
  point-in-time index membership dataset, creating survivorship bias.
- One pooled linear model does not explicitly represent company, sector, or
  market-regime differences.
- Daily data cannot represent intraday volatility dynamics.
- The fixed validation period is development evidence, not an independent
  production benchmark.
- Hyperparameters were not tuned in this baseline experiment.
- Test results remain undisclosed until the feature and model decisions are
  locked.
