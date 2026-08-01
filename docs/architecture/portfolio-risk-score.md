# Portfolio risk indicators

## Purpose

The dashboard separates measured market risk from portfolio construction. A
single number cannot say simultaneously how volatile a portfolio is, how well
it is diversified, and whether it suits a particular investor or horizon.

The implementation is versioned as `portfolio-risk-score-v2` in
`financial_ai/risk_score.py`. It returns three explainable indicators:

| Indicator | Meaning |
|---|---|
| Market risk | Historical volatility, drawdown, and a small structural fallback |
| Diversification quality | Position, HHI, and sector concentration with limited broad-fund credit |
| Liquidity resilience | Brokerage cash as a share of total equity |

## Market-risk method

```text
market risk = 55% volatility + 35% drawdown + 10% structural exposure
```

Each raw input is mapped to 0–100 by documented piecewise-linear anchors. The
structural component is deliberately small because observed price behavior is
more informative than a coarse asset label. Its current factors are cash 5,
bonds 30, gold or commodities 55, broad equity ETFs 60, individual equities 75,
real estate 75, and crypto 100.

The volatility anchors use the historical UCITS SRRI intervals only as public
reference points. This is not an SRRI or PRIIPs calculation.

| Market-risk score | Label |
|---:|---|
| 0–24.9 | Low |
| 25–49.9 | Moderate |
| 50–74.9 | Elevated |
| 75–100 | High |

## Diversification and fund look-through

The raw concentration risk combines the largest position (50%), HHI (30%), and
largest sector cluster (20%). A broad-market fund is legally one instrument but
economically represents many holdings. Broad-market equity and diversified
fixed-income exposure therefore receive a capped, transparent look-through
credit of at most 75%.

This credit is an approximation. A provider with actual constituent weights
would allow proper issuer, sector, region, and currency look-through. Sector or
thematic ETFs do not receive the broad-market credit.

Diversification and liquidity are quality scores: 75–100 is strong, 50–74.9 is
adequate, and below 50 is weak. They do not change the market-risk number. Cash
can improve immediate liquidity but can still lose purchasing power over time.

## Interpretation boundaries

- A global equity fund can have moderate market risk and strong diversification.
- A single gold instrument can have moderate market risk and weak diversification.
- A long horizon can improve an investor's capacity to wait through losses, but
  it does not erase observed volatility or drawdown.
- Historical risk reconstructs current quantities backwards; it is not actual
  account performance and does not predict future losses.
- Full fund holdings, credit quality, duration, derivatives, and hedges are not
  available.
- Currency denomination is context, not a scored risk component; it is not the
  same as economic currency exposure.
- Thresholds are transparent heuristics and are not calibrated to customer
  outcomes or suitability rules.

The API returns the raw values, normalized component scores, interpretation,
limitations, and methodology version. The UI presents the indicators separately.

## References

- [CESR guidelines on the UCITS synthetic risk and reward indicator](https://www.esma.europa.eu/sites/default/files/library/2015/11/10_673.pdf)
- [UCITS Directive, Article 52 diversification limits](https://eur-lex.europa.eu/legal-content/EN/ALL/?uri=CELEX%3A02009L0065-20230101)
- [FINRA: Concentration risk](https://www.finra.org/investors/insights/concentration-risk)
- [SEC Investor.gov: Asset allocation and diversification](https://www.investor.gov/introduction-investing/getting-started/asset-allocation)
