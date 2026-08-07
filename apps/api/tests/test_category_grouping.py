import pytest
from financial_ai.ml.transaction_classification.category_grouping import normalize_description_group


@pytest.mark.parametrize(
    ("description", "expected"),
    [
        ("[debit] TARGET #9890", "target"),
        ("[credit] target #1234", "target"),
        ("[debit] PP*SAFEWAY", "safeway"),
        ("[debit] PYPL*SAFEWAY", "safeway"),
        ("[debit] PAYPAL *NETFLIX", "netflix"),
        ("[debit] SQ *ICHIRAN RAMEN #635", "ichiran ramen"),
        ("[debit] 7-ELEVEN #435", "7 eleven"),
        ("[debit] FOOD4LESS #67868", "food4less"),
        ("[debit] DENTAL365 66229", "dental365"),
        ("[debit] 23ANDME", "23andme"),
        ("[debit] 76 1813366", "76"),
        ("[debit] 1-800-CONTACTS 5070", "1 800 contacts"),
    ],
)
def test_normalize_description_group_removes_format_noise(description, expected):
    assert normalize_description_group(description) == expected


def test_normalize_description_group_removes_ach_reference_suffixes():
    descriptions = [
        "[debit] LINCOLN PROPERTY RENT PMT 01446201 PPD ID: 483520035",
        "[debit] LINCOLN PROPERTY RENT PMT 49745749 WEB ID: VOTZ6T0M",
    ]

    assert [normalize_description_group(value) for value in descriptions] == [
        "lincoln property rent pmt",
        "lincoln property rent pmt",
    ]


def test_normalize_description_group_rejects_empty_result():
    with pytest.raises(ValueError, match="Description group must not be empty"):
        normalize_description_group("[debit] #12345")
