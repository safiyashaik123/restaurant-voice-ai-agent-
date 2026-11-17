"""Focused checks for the shared reservation policy."""
from datetime import date, datetime, timedelta, timezone

import pytest

from src.core.reservation_policy import restaurant_today
from src.core.config import config
from src.core.reservation_policy import reservation_slots, validate_indian_mobile, validate_reservation_slot


@pytest.mark.parametrize(
    ("phone", "expected"),
    [
        ("9876543210", "9876543210"),
        ("+916123456789", "+916123456789"),
        ("987-654-3210", "9876543210"),
        ("987 654 3210", "9876543210"),
        ("(987) 654-3210", "9876543210"),
        ("+91 987-654-3210", "+919876543210"),
    ],
)
def test_valid_indian_mobile(phone, expected):
    assert validate_indian_mobile(phone) == expected


@pytest.mark.parametrize(
    "phone",
    [
        "1234567890",
        "987654321",
        "98765432100",
        "91 9876543210",
        "+9198765432100",
        "98765432AB",
        "++919876543210",
        "+91 1234567890",
    ],
)
def test_invalid_indian_mobile(phone):
    with pytest.raises(ValueError, match="Indian mobile"):
        validate_indian_mobile(phone)


def test_default_slots_and_past_cutoff(monkeypatch):
    monkeypatch.setattr(config, "opening_time", "10:00")
    monkeypatch.setattr(config, "closing_time", "23:00")
    monkeypatch.setattr(config, "default_reservation_duration_minutes", 90)
    now = datetime(2030, 1, 1, 4, 45, tzinfo=timezone.utc)
    assert reservation_slots(now.date(), now=now)[0] == "10:30"
    slots = reservation_slots(now.date() + timedelta(days=1), now=now)
    assert slots[0] == "10:00" and slots[-1] == "21:30"
    assert "22:30" not in slots and "10:15" not in slots
    assert reservation_slots(now.date() - timedelta(days=1), now=now) == []


def test_configured_hours_and_duration(monkeypatch):
    monkeypatch.setattr(config, "opening_time", "12:00")
    monkeypatch.setattr(config, "closing_time", "23:00")
    monkeypatch.setattr(config, "default_reservation_duration_minutes", 30)
    day = restaurant_today() + timedelta(days=1)
    slots = reservation_slots(day)
    assert slots[0] == "12:00" and slots[-1] == "22:30"
    validate_reservation_slot(day.isoformat(), "22:30")
    for time in ("11:30", "12:15", "23:00"):
        with pytest.raises(ValueError):
            validate_reservation_slot(day.isoformat(), time)


def test_slots_interpret_utc_instants_in_kolkata_not_machine_time(monkeypatch):
    from zoneinfo import ZoneInfo
    from src.core import reservation_policy as policy
    from src.services.agent_service import normalize_date_input

    monkeypatch.setattr(config, "opening_time", "10:00")
    monkeypatch.setattr(config, "closing_time", "23:00")
    monkeypatch.setattr(config, "default_reservation_duration_minutes", 90)
    # The same instant is 04:45 UTC and 10:15 in India.
    instant = datetime(2030, 1, 1, 4, 45, tzinfo=timezone.utc)
    monkeypatch.setattr(policy, "restaurant_now", lambda: instant.astimezone(ZoneInfo("Asia/Kolkata")))
    assert policy.restaurant_today() == date(2030, 1, 1)
    assert normalize_date_input("today") == "2030-01-01"
    assert normalize_date_input("tomorrow") == "2030-01-02"
    assert policy.reservation_slots(date(2030, 1, 1))[0] == "10:30"
    with pytest.raises(ValueError):
        policy.validate_reservation_slot("2030-01-01", "10:00")
    policy.validate_reservation_slot("2030-01-01", "10:30")
    with pytest.raises(ValueError):
        policy.validate_reservation_slot("2029-12-31", "21:30")


def test_kolkata_date_rollover_and_aware_clock(monkeypatch):
    from zoneinfo import ZoneInfo
    from src.core import reservation_policy as policy
    from src.services.agent_service import normalize_date_input
    # At 20:00 UTC, Kolkata is already on the following day.
    instant = datetime(2030, 1, 1, 20, 0, tzinfo=timezone.utc)
    monkeypatch.setattr(policy, "restaurant_now", lambda: instant.astimezone(ZoneInfo("Asia/Kolkata")))
    assert normalize_date_input("today") == "2030-01-02"
    assert normalize_date_input("tomorrow") == "2030-01-03"
    assert reservation_slots(date(2030, 1, 1), now=instant) == []
    assert reservation_slots(date(2030, 1, 2), now=instant)[0] == "10:00"
    with pytest.raises(ValueError, match="timezone-aware"):
        reservation_slots(date(2030, 1, 2), now=datetime(2030, 1, 1))
