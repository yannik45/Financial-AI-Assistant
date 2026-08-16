import numpy as np
import pandas as pd
from financial_ai.ml.market_forecast.data.features import (
    FEATURE_COLUMNS,
    build_market_features,
)


def daily_bars(symbol: str, closes: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "symbol": symbol,
            "observed_on": pd.date_range("2023-01-02", periods=len(closes), freq="B"),
            "open": closes * 0.995,
            "high": closes * 1.01,
            "low": closes * 0.99,
            "close": closes,
            "adjusted_close": closes,
            "volume": np.linspace(1_000_000, 2_000_000, len(closes)),
        }
    )


def test_market_features_are_complete_scale_independent_and_symbol_local():
    stable = daily_bars("STABLE", np.full(80, 100.0))
    variable = daily_bars("VARIABLE", 100 * np.exp(np.sin(np.arange(80) / 3) * 0.1))

    result = build_market_features(pd.concat([variable, stable], ignore_index=True))

    assert set(FEATURE_COLUMNS).issubset(result.columns)
    assert not result.loc[:, FEATURE_COLUMNS].isna().any().any()
    assert len(result.loc[result["symbol"] == "STABLE"]) == 20
    assert result.loc[result["symbol"] == "STABLE", "realized_volatility_60d"].eq(0).all()
    assert result.loc[result["symbol"] == "VARIABLE", "realized_volatility_60d"].gt(0).all()


def test_market_features_do_not_use_later_observations():
    original = daily_bars("AAPL", np.linspace(100, 130, 80))
    changed = original.copy(deep=True)
    changed.loc[changed.index[-5:], "adjusted_close"] *= 2
    changed.loc[changed.index[-5:], ["open", "high", "low", "close"]] *= 2

    original_features = build_market_features(original)
    changed_features = build_market_features(changed)
    comparison_date = original.loc[original.index[-6], "observed_on"]

    original_row = original_features.loc[original_features["observed_on"] == comparison_date]
    changed_row = changed_features.loc[changed_features["observed_on"] == comparison_date]
    pd.testing.assert_frame_equal(
        original_row.loc[:, FEATURE_COLUMNS].reset_index(drop=True),
        changed_row.loc[:, FEATURE_COLUMNS].reset_index(drop=True),
    )


def test_market_feature_construction_does_not_modify_source_data():
    source = daily_bars("AAPL", np.linspace(100, 130, 80))
    expected = source.copy(deep=True)

    build_market_features(source)

    pd.testing.assert_frame_equal(source, expected)
