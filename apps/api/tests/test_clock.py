from datetime import UTC, datetime

from financial_ai import clock


def test_business_date_uses_configured_timezone(monkeypatch):
    monkeypatch.setattr(clock, "utc_now", lambda: datetime(2026, 7, 31, 22, 30, tzinfo=UTC))

    assert clock.business_today().isoformat() == "2026-08-01"
