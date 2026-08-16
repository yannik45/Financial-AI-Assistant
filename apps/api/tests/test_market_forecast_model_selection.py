import pandas as pd
import pytest
from financial_ai.ml.market_forecast.modeling.model_selection import build_expanding_training_folds


def model_dataset() -> pd.DataFrame:
    dates = pd.date_range("2017-01-02", "2022-01-10", freq="B")
    rows = []
    for symbol in ("AAPL", "MSFT"):
        symbol_rows = pd.DataFrame(
            {
                "symbol": symbol,
                "observed_on": dates,
                "feature": range(len(dates)),
                "split": "train",
            }
        )
        symbol_rows.loc[symbol_rows["observed_on"].dt.year >= 2022, "split"] = "validation"
        rows.append(symbol_rows)
    return pd.concat(rows, ignore_index=True)


def test_expanding_folds_use_only_outer_train_and_purge_target_overlap():
    result = build_expanding_training_folds(
        model_dataset(),
        validation_years=(2019, 2020, 2021),
        purge_trading_days=3,
    )

    assert tuple(fold.validation_year for fold in result) == (2019, 2020, 2021)
    for fold in result:
        assert set(fold.validation["observed_on"].dt.year) == {fold.validation_year}
        assert fold.train["observed_on"].max() < fold.validation["observed_on"].min()
        validation_start = fold.validation["observed_on"].min()
        source = model_dataset()
        all_prior_dates = (
            source.loc[
                (source["split"] == "train") & (source["observed_on"] < validation_start),
                "observed_on",
            ]
            .drop_duplicates()
            .sort_values()
        )
        assert not fold.train["observed_on"].isin(all_prior_dates.iloc[-3:]).any()
        assert set(fold.train["split"]) == {"train"}


def test_expanding_folds_do_not_modify_dataset():
    source = model_dataset()
    expected = source.copy(deep=True)

    build_expanding_training_folds(source, validation_years=(2020,), purge_trading_days=2)

    pd.testing.assert_frame_equal(source, expected)


@pytest.mark.parametrize(
    "validation_years",
    [(), (2020, 2019), (2020, 2020)],
)
def test_expanding_folds_reject_invalid_validation_years(validation_years):
    with pytest.raises(ValueError, match="Validation years"):
        build_expanding_training_folds(
            model_dataset(),
            validation_years=validation_years,
        )
