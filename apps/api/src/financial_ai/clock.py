from datetime import UTC, date, datetime
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from financial_ai.config import get_settings


def utc_now() -> datetime:
    """Return an aware UTC timestamp for technical event times."""
    return datetime.now(UTC)


def business_today() -> date:
    """Return the calendar date in the configured business timezone."""
    timezone_name = get_settings().app_timezone
    try:
        timezone = ZoneInfo(timezone_name)
    except ZoneInfoNotFoundError as exc:
        raise RuntimeError(f"Unknown FINANCIAL_AI_APP_TIMEZONE: {timezone_name}") from exc
    return utc_now().astimezone(timezone).date()
