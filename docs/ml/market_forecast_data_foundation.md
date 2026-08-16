# Market forecast data foundation

## Purpose

This foundation defines the reproducible observations, features, targets, and
temporal boundaries used to evaluate and build the market-volatility model.
The backend exposes inference only through a separately built, checksum-verified
deployment artifact.

The experiment uses a pinned universe of liquid US equities. Alpaca SIP daily
bars are an acquisition source, not a training-time dependency. Validated
observations are frozen as immutable local snapshots with metadata and a
checksum before feature engineering.

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

Each symbol's final 20 observations before the validation and test boundaries
are purged from the preceding split. This remains correct when instruments have
different observation calendars and prevents a forward target in one period
from containing returns observed in the next period. The final test period must
not be used for feature selection, model selection, or threshold tuning.

## Initial feature contract

Features are calculated at the end of each completed trading day and use only
the current or earlier observations for the same symbol. The initial feature
set contains one-day log and absolute returns; 5-, 20-, and 60-day realized
volatility; 5- and 20-day momentum; the daily and 20-day average intraday
high-low range; and a 5-to-20-day volume ratio.

Prices and raw volume are not direct model inputs. Returns, relative ranges,
and volume ratios reduce dependence on the nominal price and trading scale of
an instrument. Rows without a complete 60-day history are excluded. Features
must be constructed before chronological rows are filtered into splits so the
first validation and test observations can use information that was genuinely
available before their boundaries.

The feature groups have distinct purposes:

- 5-, 20-, and 60-day realized volatility represent short-, medium-, and
  slower-moving volatility regimes;
- one-day absolute return captures the magnitude of the latest price shock;
- signed returns and momentum retain recent direction without asserting that
  direction alone predicts future returns;
- relative high-low ranges provide an intraday variability signal beyond close
  prices;
- the short-to-medium volume ratio identifies changes in trading activity
  without using nominal volume across differently sized companies.

The initial contract deliberately excludes raw price, raw volume, symbol
identity, sector labels, and a large collection of overlapping technical
indicators. Raw levels are difficult to compare across instruments, symbol
identity can encourage memorization, and adding many variations of the same
rolling signals increases selection and overfitting risk. Fundamentals require
separate point-in-time publication handling and are deferred to a later dataset
version. These exclusions can be reconsidered only through validation-period
experiments defined before the final test is opened.

## Leakage controls

- Raw bars are normalized and sorted before any rolling calculation.
- Returns, rolling windows, and momentum are calculated independently per
  symbol and use no centered windows, negative shifts, or backward filling.
- Features at date `t` use completed observations no later than `t`.
- Targets at date `t` use only returns from `t+1` through `t+20`.
- Twenty observations per symbol are purged before validation and test so target
  windows cannot cross an evaluation boundary.
- Split dates are fixed before model development, and the final test period is
  unavailable for feature or model selection.

Features are calculated across the uninterrupted chronology before split rows
are selected. This allows the first validation and test dates to use legitimate
historical context while preventing either period from influencing earlier
features.

## Derived model dataset

The model-facing table contains only:

```text
symbol, observed_on, ten feature columns, forward volatility target, split
```

Features and targets are independently constructed from the validated raw
snapshot and joined one-to-one by symbol and observation date. The inner join
removes initial rows without 60 days of feature history and final rows without
20 days of future target history. Raw OHLCV values are excluded from the final
column contract so training code cannot accidentally treat nominal price or
future-derived fields as model inputs.

## Initial boundaries

- daily US equity observations from 2016 onward;
- historical SIP observations through the end of 2025;
- a pinned symbol universe rather than the provider's changing current catalog;
- immutable snapshots under ignored `data/runtime/ml/market_forecast/` paths;
- current product quotes may use IEX, but training snapshots use historical SIP
  bars and record the feed in metadata;
- no model selection against the final chronological test period;
- SEC fundamentals and point-in-time joins are a later dataset version.

The initial 50-equity universe is pinned in
`data/market/market_forecast_universe_v1.json`. It uses a minimum of three
companies per GICS sector and assigns additional representation to larger
sectors without reproducing market-cap concentration. It remains subject to
survivorship bias because constituents were selected as of 2026-08-01. This
limitation must be retained when interpreting results.

Additional limitations remain:

- ten calendar years contain a limited number of distinct market regimes even
  though the panel has more than 100,000 rows;
- adjacent 20-day targets overlap and are therefore not independent samples;
- adjusted daily bars do not capture intraday paths, spreads, liquidity costs,
  or trading feasibility;
- the initial feature set has no broad-market, volatility-index, interest-rate,
  macroeconomic, news, or point-in-time fundamental context;
- the universe excludes delisted companies, smaller stocks, and non-US markets;
- predictive performance on this controlled universe does not establish useful
  performance for an arbitrary user-selected instrument;
- annualized volatility describes expected variation, not direction, loss
  probability, or an investment recommendation.

## Delivery sequence

1. Validate and normalize daily OHLCV observations.
2. Export an immutable raw snapshot with metadata and checksum.
3. Define chronological development and final-test boundaries.
4. Build backward-looking features without cross-symbol leakage.
5. Create a forward-volatility target and remove rows whose future horizon is unknown.
6. Compare declared model candidates with purged expanding-window inner folds.
7. Build the frozen deployment artifact only after the final evaluation is closed.

The validation contract is enforced by
`apps/api/tests/test_market_forecast_daily_bars.py` and implemented by
`financial_ai.ml.market_forecast.data.daily_bars.validate_daily_bars`.

Validated input files are frozen with:

```powershell
uv run financial-ai-write-market-snapshot `
  --input-csv data/runtime/ml/market_forecast/downloads/us-equities.csv `
  --version us-equities-v1
```

The command writes an immutable CSV and metadata JSON containing provenance,
coverage, schema, and a canonical SHA-256 checksum.

Historical observations can be downloaded and frozen in one command:

The downloader requires `FINANCIAL_AI_ALPACA_API_KEY` and
`FINANCIAL_AI_ALPACA_SECRET_KEY` in the local `.env`. It explicitly requests
the SIP feed, so the Alpaca account must have access to the requested historical
SIP range.

```powershell
uv run financial-ai-download-market-snapshot `
  --universe-path data/market/market_forecast_universe_v1.json `
  --date-from 2016-01-01 `
  --date-to 2025-12-31 `
  --version us-large-cap-sip-price-v1
```

The explicit symbol and date arguments keep data acquisition separate from
feature engineering and model evaluation.

The derived model table is built and frozen with:

```powershell
uv run financial-ai-build-market-dataset `
  --snapshot-version us-large-cap-sip-price-v1 `
  --dataset-version us-large-cap-volatility-v1
```

The command writes an immutable CSV and metadata JSON under
`data/runtime/ml/market_forecast/datasets/`. Metadata binds the dataset to the
source snapshot checksum, feature and target contracts, split configuration,
row coverage, and its own canonical checksum. Generated market data remains
ignored by Git; later evaluation reports may reference these versions and
checksums without redistributing the underlying observations.

After this dataset exists, build the local deployment artifact with:

```powershell
uv run financial-ai-build-market-forecast-model `
  --dataset-version us-large-cap-volatility-v1
```

The command writes the native XGBoost model and its metadata under
`data/runtime/ml/market_forecast/models/`. Both files remain ignored runtime
artifacts. Model selection, final-test evidence, and the distinction between
the evaluated fit and deployment refit are documented in the
[market volatility model card](market_volatility_boosting.md).
