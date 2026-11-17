"""Shared validation and configurable reservation-slot policy."""

from datetime import date, datetime, timedelta
import re
from typing import Optional
from zoneinfo import ZoneInfo

from src.core.config import config

PHONE_ERROR = "Enter a valid Indian mobile number: 10 digits starting with 6, 7, 8 or 9, optionally prefixed by +91."
NAME_ERROR = "Enter a name containing letters, spaces, apostrophes, periods or hyphens."
SLOT_ERROR = "Choose a future reservation time on an available operating hours interval."


def validate_customer_name(name: str) -> str:
    import unicodedata

    value = name.strip().rstrip(",;:!?").strip()

    if (
        not value
        or len(value) > 200
        or not any(c.isalpha() for c in value)
        or any(
            not (
                c.isalpha()
                or unicodedata.category(c).startswith("M")
                or c in " '-.’"
            )
            for c in value
        )
    ):
        raise ValueError(NAME_ERROR)

    return value


def validate_indian_mobile(phone: str) -> str:
    """Validate and normalize an Indian mobile number."""
    value = phone.strip()

    # Allow digits and common formatting characters only.
    if not re.fullmatch(r"\+?[0-9().\s-]+", value):
        raise ValueError(PHONE_ERROR)

    # Preserve an optional +91 country code.
    has_country_code = value.startswith("+91")
    number = value[3:] if has_country_code else value

    # Remove formatting characters before validating.
    digits = re.sub(r"\D", "", number)

    if not re.fullmatch(r"[6-9][0-9]{9}", digits):
        raise ValueError(PHONE_ERROR)

    return ("+91" if has_country_code else "") + digits


def parse_reservation_date(value: str) -> date:
    try:
        parsed = datetime.strptime(value, "%Y-%m-%d").date()
        if parsed.isoformat() != value:
            raise ValueError
        return parsed
    except ValueError as exc:
        raise ValueError("Invalid reservation date; use YYYY-MM-DD.") from exc


def restaurant_now() -> datetime:
    """Current time in the restaurant timezone, independent of the host timezone."""
    return datetime.now(ZoneInfo(config.restaurant_timezone))


def restaurant_today() -> date:
    return restaurant_now().date()


def reservation_slots(day: date, *, now: Optional[datetime] = None,
                      duration_minutes: Optional[int] = None) -> list[str]:
    """Return restaurant-local start times; injected clocks must be timezone-aware."""
    zone = ZoneInfo(config.restaurant_timezone)
    current = now if now is not None else restaurant_now()
    if current.tzinfo is None or current.utcoffset() is None:
        raise ValueError("Current time must be timezone-aware.")
    current = current.astimezone(zone)
    horizon = config.reservation_max_advance_days
    if horizon is not None:
        if horizon < 0:
            raise ValueError("RESERVATION_MAX_ADVANCE_DAYS must be non-negative")
        if day > current.date() + timedelta(days=horizon):
            return []
    duration = config.default_reservation_duration_minutes if duration_minutes is None else duration_minutes
    interval = 30
    if duration <= 0:
        raise ValueError("Reservation duration must be positive.")
    try:
        opening = datetime.combine(day, datetime.strptime(config.opening_time, "%H:%M").time(), tzinfo=zone)
        closing = datetime.combine(day, datetime.strptime(config.closing_time, "%H:%M").time(), tzinfo=zone)
    except ValueError as exc:
        raise ValueError("Invalid restaurant hours configuration.") from exc
    if closing <= opening:
        raise ValueError("Closing time must be after opening time.")
    slots = []
    start = opening
    while start < closing and start + timedelta(minutes=duration) <= closing:
        if start > current:
            slots.append(start.strftime("%H:%M"))
        start += timedelta(minutes=interval)
    return slots


def validate_reservation_slot(date_str: str, time_str: str, *, duration_minutes: Optional[int] = None) -> None:
    day = parse_reservation_date(date_str.strip())
    if time_str.strip() not in reservation_slots(day, duration_minutes=duration_minutes):
        raise ValueError(SLOT_ERROR)
