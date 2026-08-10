# Market volatility forecasting model card

## Objective

The selected model forecasts 20-trading-day annualized realized volatility for
a pinned universe of 50 US large-cap equities. It is an offline experimental
model for evaluation and portfolio-risk research, not a return forecast,
trading signal, or investment recommendation.

It uses ten backward-looking price/volume features and the chronological split
contract documented in the [data foundation](market_forecast_data_foundation.md).

## Selection protocol

Only the 2016–2021 outer training period was used for model selection. Three
XGBoost configurations were declared before evaluation and compared across
purged expanding-window folds with validation years 2019, 2020, and 2021. Mean
MAE was the selection metric; RMSE, QLIKE, fold stability, and model complexity
were secondary evidence.

The `flexible` candidate (`max_depth=4`, `min_child_weight=5`) ranked first with
mean MAE 0.08333 and mean QLIKE 0.76857. Its median early-stopped fit length was
144 rounds. The final model therefore used the same configuration and exactly
144 rounds on all outer training rows, without further early stopping.

Candidate ranking uses MAE on the original annualized-volatility scale. Early
stopping monitors MAE on the log-transformed target used for fitting. This
scale difference is fixed for V1 and may be compared with an original-scale
custom stopping metric only in a separately evaluated V2.

## Outer-validation results

The frozen candidate was evaluated once on 24,050 observations from 2022–2023:

| Model | MAE | RMSE | QLIKE |
|---|---:|---:|---:|
| EWMA | 0.07380 | 0.10640 | 0.26468 |
| Ridge | 0.06669 | 0.10343 | 0.27465 |
| XGBoost | **0.06491** | **0.09787** | **0.23762** |

Relative to Ridge, XGBoost reduced MAE by 2.7%, RMSE by 5.4%, and QLIKE by
13.5%. The nonlinear model is retained because the improvement extends beyond
the primary metric.

## Diagnostics and limitations

XGBoost performed better in aggregate but did not dominate every slice. It
overestimated realized volatility below 15% and underestimated volatility above
30% by about 10.4 percentage points on average. High-volatility equities such
as NFLX and NVDA were among the largest errors. This regression toward typical
conditions limits the model during abrupt or instrument-specific stress.

Slice membership uses the realized future target and is descriptive only. It
cannot route forecasts at inference time, and it was not used for further
tuning. Sector-specific or mixture models would require point-in-time sector
metadata, sufficient observations per group, and a new nested temporal
evaluation against the pooled model.

The universe has survivorship bias, adjacent targets overlap, and ten years
contain few independent market regimes. Daily price-derived features omit
macroeconomic, volatility-index, news, and point-in-time fundamental context.
Because adjacent targets and instruments share market periods, row-level errors
are correlated and the reported metrics have no confidence intervals. Results
describe observed error differences and do not establish statistical
superiority.

## Final test and decision

After the feature contract, candidate, 144 boosting rounds, metrics, and year
slices were frozen, Ridge and XGBoost were refitted on 95,600 development rows.
The 24,100 observations from 2024–2025 were then evaluated once:

| Model | MAE | RMSE | QLIKE | Bias |
|---|---:|---:|---:|---:|
| EWMA | 0.08399 | 0.12286 | 0.36445 | +0.00305 |
| Ridge | 0.06993 | 0.11184 | 0.35166 | −0.02620 |
| XGBoost | **0.06904** | **0.10764** | **0.31644** | −0.01938 |

XGBoost reduced MAE by 1.3%, RMSE by 3.8%, and QLIKE by 10.0% relative to
Ridge. It also led all three metrics in both predeclared year slices. XGBoost is
therefore retained as the final V1 candidate, while Ridge remains the simpler
linear baseline and EWMA the adaptive statistical reference. No further V1
tuning is permitted after viewing the test.

## Possible V2 experiments

V2 may test broad-market volatility and relative instrument-to-market features,
followed by sector encoding in the pooled model. Separate sector models or a
mixture of experts are justified only if simpler conditioning shows stable
temporal gains.

These experiments introduce unresolved constraints: sector metadata should be
point-in-time correct, small sectors provide few independent companies, static
labels may encourage group memorization, and extra context increases selection
risk. Because 2024–2025 has now been observed, V2 must use nested walk-forward
evaluation or reserve newly available future data; it cannot reuse this period
as an independent final test.

## Reproduction

The versioned snapshot and derived dataset are prerequisites. Their download,
credentials, SIP-access requirement, validation, and build commands are defined
in the [market forecast data foundation](market_forecast_data_foundation.md).

```powershell
uv run financial-ai-select-market-boosting-model `
  --dataset-version us-large-cap-volatility-v1 `
  --selection-version inner-cv-v1

uv run financial-ai-evaluate-market-boosting `
  --dataset-version us-large-cap-volatility-v1 `
  --selection-version inner-cv-v1 `
  --evaluation-version us-large-cap-xgboost-v1

uv run financial-ai-diagnose-market-boosting `
  --dataset-version us-large-cap-volatility-v1 `
  --selection-version inner-cv-v1 `
  --diagnostics-version outer-validation-diagnostics-v1

uv run financial-ai-evaluate-market-final-test `
  --dataset-version us-large-cap-volatility-v1 `
  --validation-version us-large-cap-xgboost-v1 `
  --test-version us-large-cap-final-v1

uv run financial-ai-build-market-forecast-model `
  --dataset-version us-large-cap-volatility-v1
```

Full reports are immutable, checksum-linked runtime artifacts. Compact aggregate
evidence is versioned under
[`data/evaluation/market_forecast`](../../data/evaluation/market_forecast/) so
the selection and evaluation chain can be inspected without redistributing raw
market observations. The final test closes model selection for V1.

The deployable native XGBoost artifact is a post-evaluation refit on all
119,700 labeled rows. Its metadata records the frozen candidate, 144 boosting
rounds, feature order, training coverage, source-dataset checksum, library
versions, and model checksum. Published test metrics remain evidence from the
earlier frozen test fit; they are not recomputed or attributed to this refit.

## Backend inference contract

`GET /v1/market/instruments/{instrument_id}/volatility-forecast` refreshes
cached daily bars through the latest safely completed US session, reconstructs
the ten frozen features, and returns the annualized 20-trading-day volatility
forecast with observation date, model version, source, retrieval time, and
freshness status. A provider failure may fall back only to an existing cache,
which is labeled `stale`; missing models or unusable OHLCV history fail
explicitly.

V1 was trained on Alpaca SIP observations. The default free application feed
is IEX, whose coverage and volume differ. The API therefore reports the
training feed and `feed_match`; IEX forecasts remain a product demonstration,
not evidence that the frozen SIP evaluation transfers unchanged.
