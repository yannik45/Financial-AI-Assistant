# Market forecast data foundation

## Purpose

This branch prepares reproducible daily market observations for later forecasting
experiments. It does not train a model or expose predictions in the product.

The first experiment will compare a price-only baseline across a pinned universe
of liquid US equities. Alpaca SIP daily bars are an acquisition source, not a
training-time dependency. Validated observations will be frozen as immutable
local snapshots with metadata and a checksum before feature engineering begins.

## Raw observation contract

Each row represents one completed trading day for one symbol:

```text
symbol, observed_on, open, high, low, close, adjusted_close, volume
```

The raw layer must not contain rolling indicators, forward returns, targets, or
fundamentals. Those belong to derived versioned datasets so raw observations can
be reused without silently changing their meaning.

## Forecast target

The first supervised target is the annualized realized volatility of the next
20 trading-day returns:

```text
target(t) = standard deviation(log returns from t+1 through t+20) * sqrt(252)
```

The target is a non-negative decimal value, not a percentage. Its forward
window must be constructed independently for each symbol. The final 20 rows per
symbol have an unknown target and are excluded from supervised datasets. This
is a forecasting target, not an estimate of loss probability or return
direction.

## Evaluation periods

The initial evaluation uses fixed chronological periods:

```text
training:   2016-01-01 through 2021-12-31
validation: 2022-01-01 through 2023-12-31
test:       2024-01-01 through 2025-12-31
```

The final 20 observed trading dates before the validation and test boundaries
are purged from the preceding split. This prevents a 20-day forward target in
one period from containing returns observed in the next period. The final test
period must not be used for feature selection, model selection, or threshold
tuning.

## Initial boundaries

- daily US equity observations from 2016 onward;
- historical SIP observations through the end of 2025;
- a pinned symbol universe rather than the provider's changing current catalog;
- immutable snapshots under ignored `data/runtime/ml/market_forecast/` paths;
- no claim that IEX-only observations represent consolidated US market volume;
- no model selection against the final chronological test period;
- SEC fundamentals and point-in-time joins are a later dataset version.

The initial 50-equity universe is pinned in
`data/market/market_forecast_universe_v1.json`. It uses a minimum of three
companies per GICS sector and assigns additional representation to larger
sectors without reproducing market-cap concentration. It remains subject to
survivorship bias because constituents were selected as of 2026-08-01. This
limitation must be retained when interpreting results.

## Delivery sequence

1. Validate and normalize daily OHLCV observations.
2. Export an immutable raw snapshot with metadata and checksum.
3. Define chronological development and final-test boundaries.
4. Build backward-looking features without cross-symbol leakage.
5. Create a forward-return target and remove rows whose future horizon is unknown.
6. Compare expanding and rolling training windows in a later modeling branch.

The validation contract is enforced by
`apps/api/tests/test_market_forecast_daily_bars.py` and implemented by
`financial_ai.ml.market_forecast.daily_bars.validate_daily_bars`.

Validated input files are frozen with:

```powershell
uv run financial-ai-write-market-snapshot `
  --input-csv data/runtime/ml/market_forecast/downloads/us-equities.csv `
  --version us-equities-v1
```

The command writes an immutable CSV and metadata JSON containing provenance,
coverage, schema, and a canonical SHA-256 checksum.

Historical observations can be downloaded and frozen in one command:

```powershell
uv run financial-ai-download-market-snapshot `
  --universe-path data/market/market_forecast_universe_v1.json `
  --date-from 2016-01-01 `
  --date-to 2025-12-31 `
  --version us-large-cap-sip-price-v1
```

The explicit symbol and date arguments keep data acquisition separate from
feature engineering and model evaluation.
