from datetime import date

import pandas as pd
import pytest
from financial_ai.ml.market_forecast.data.splits import (
    SPLIT_COLUMN,
    assign_chronological_splits,
    identify_purge_rows,
)
from financial_ai.ml.market_forecast.data.targets import TARGET_COLUMN


def target_frame() -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=12, freq="D")
    return pd.DataFrame(
        [
            {"symbol": symbol, "observed_on": observed_on, TARGET_COLUMN: 0.2}
            for symbol in ("AAPL", "MSFT")
            for observed_on in dates
        ]
    )


def test_chronological_splits_remove_trading_dates_before_each_boundary():
    result = assign_chronological_splits(
        target_frame(),
        validation_start=date(2024, 1, 6),
        test_start=date(2024, 1, 10),
        purge_trading_days=2,
    )

    dates_by_split = {
        split: group["observed_on"].drop_duplicates().dt.date.tolist()
        for split, group in result.groupby(SPLIT_COLUMN, sort=False)
    }
    assert dates_by_split == {
        "train": [date(2024, 1, 1), date(2024, 1, 2), date(2024, 1, 3)],
        "validation": [date(2024, 1, 6), date(2024, 1, 7)],
        "test": [date(2024, 1, 10), date(2024, 1, 11), date(2024, 1, 12)],
    }
    assert len(result) == 16


def test_chronological_splits_do_not_modify_the_source_frame():
    source = target_frame()

    assign_chronological_splits(
        source,
        validation_start=date(2024, 1, 6),
        test_start=date(2024, 1, 10),
        purge_trading_days=2,
    )

    assert SPLIT_COLUMN not in source.columns


def test_purge_uses_each_symbols_observation_calendar():
    frame = pd.DataFrame(
        {
            "symbol": ["AAPL"] * 5 + ["MSFT"] * 3,
            "observed_on": pd.to_datetime(
                [
                    "2024-01-02",
                    "2024-01-03",
                    "2024-01-04",
                    "2024-01-05",
                    "2024-01-08",
                    "2024-01-02",
                    "2024-01-03",
                    "2024-01-05",
                ]
            ),
        }
    )

    purge_mask = identify_purge_rows(
        frame,
        boundary=pd.Timestamp("2024-01-09"),
        purge_trading_days=2,
    )

    purged = frame.loc[purge_mask]
    assert purged.loc[purged["symbol"] == "AAPL", "observed_on"].tolist() == [
        pd.Timestamp("2024-01-05"),
        pd.Timestamp("2024-01-08"),
    ]
    assert purged.loc[purged["symbol"] == "MSFT", "observed_on"].tolist() == [
        pd.Timestamp("2024-01-03"),
        pd.Timestamp("2024-01-05"),
    ]


@pytest.mark.parametrize(
    "validation_start,test_start,purge_days,message",
    [
        (date(2024, 1, 10), date(2024, 1, 6), 2, "validation"),
        (date(2024, 1, 6), date(2024, 1, 10), 0, "purge"),
    ],
)
def test_chronological_splits_reject_invalid_configuration(
    validation_start, test_start, purge_days, message
):
    with pytest.raises(ValueError, match=message):
        assign_chronological_splits(
            target_frame(),
            validation_start=validation_start,
            test_start=test_start,
            purge_trading_days=purge_days,
        )
