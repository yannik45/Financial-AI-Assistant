from financial_ai.demo_bank_feed import generate_demo_bank_feed


def test_demo_bank_feed_is_reproducible_and_seeded() -> None:
    first = generate_demo_bank_feed(seed=42, year=2026, month=8)
    repeated = generate_demo_bank_feed(seed=42, year=2026, month=8)
    different = generate_demo_bank_feed(seed=43, year=2026, month=8)

    assert first == repeated
    assert first != different
    assert len(first) == 15
    assert {item.expected_category for item in first} >= {
        "income",
        "housing",
        "utilities",
    }
    assert all(item.booked_at.year == 2026 and item.booked_at.month == 8 for item in first)


def test_demo_bank_feed_rejects_invalid_generation_parameters() -> None:
    try:
        generate_demo_bank_feed(seed=1, year=2026, month=13)
    except ValueError as exc:
        assert str(exc) == "month must be between 1 and 12"
    else:
        raise AssertionError("Expected invalid month to be rejected")
