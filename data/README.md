# Demo data

Security price series are generated at runtime with fixed random seeds. They are
synthetic and must not be interpreted as actual market observations.

`evaluation/transaction_categories/` contains small, versioned development
benchmarks and checksum metadata. Frozen versions are never edited in place;
corrected protocols receive a new version. Generated training datasets, model
artifacts, databases, and full evaluation reports belong under ignored
`runtime/` paths.

`market/ecb_fx.csv` is an unmodified ECB Data API CSV snapshot downloaded on
2026-07-23. It contains daily reference rates for `D.USD+GBP+JPY.EUR.SP00.A`
from 2024-01-01 through 2026-06-30. Source: ECB statistics. Analytics invert the
published foreign-currency-per-EUR quote to obtain EUR per foreign currency.

- API: https://data-api.ecb.europa.eu/service/data/EXR/D.USD+GBP+JPY.EUR.SP00.A
- Reuse policy: https://www.ecb.europa.eu/stats/ecb_statistics/governance_and_quality_framework/html/usage_policy.en.html
