"""Date normalization and extraction without live Ollama or a production database."""
from datetime import date
import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from src.database.init_db import init_database
from src.services.agent_service import AgentService, Intent, normalize_date_input
from src.services.llm_service import OllamaService


@pytest.fixture(autouse=True)
def fixed_today(monkeypatch):
    monkeypatch.setattr('src.services.agent_service.restaurant_today', lambda: date(2026, 1, 10))


@pytest.fixture
def agent(tmp_path):
    path = tmp_path / 'restaurant.db'
    init_database(path, seed=True)
    return AgentService(db_path=str(path), llm_service=MagicMock(spec=OllamaService))


@pytest.mark.parametrize('value, expected', [
    ('November 15 2026', '2026-11-15'),
    ('November 15, 2026', '2026-11-15'),
    ('Nov 15 2026', '2026-11-15'),
    ('November 15', '2026-11-15'),
    ('2026-11-15', '2026-11-15'),
    ('2026/11/15', '2026-11-15'),
    ('11/15/2026', '2026-11-15'),
    ('15/11/2026', '2026-11-15'),
    ('today', '2026-01-10'),
    ('tonight', '2026-01-10'),
    ('tomorrow', '2026-01-11'),
    ('February 30, 2026', None),
    ('2026-02-29', None),
])
def test_normalize_dates(value, expected):
    assert normalize_date_input(value) == expected


def test_yearless_date_uses_next_occurrence(monkeypatch):
    monkeypatch.setattr('src.services.agent_service.restaurant_today', lambda: date(2026, 12, 1))
    assert normalize_date_input('November 15') == '2027-11-15'
    assert normalize_date_input('November 15 2026') == '2026-11-15'
    assert normalize_date_input('November 15 1900') == '1900-11-15'


@pytest.mark.parametrize('date_text, expected', [
    ('November 15 2026', '2026-11-15'),
    ('November 15, 2026', '2026-11-15'),
    ('Nov 15 2026', '2026-11-15'),
    ('November 15', '2026-11-15'),
    ('2026-11-15', '2026-11-15'),
    ('11/15/2026', '2026-11-15'),
    ('today', '2026-01-10'),
    ('tomorrow', '2026-01-11'),
    ('February 30, 2026', None),
    ('2026-02-29', None),
])
def test_rule_based_date_extraction(agent, date_text, expected):
    intent, entities = agent._extract_rule_based(f'Book a table for 2 on {date_text} at 18:00')
    assert intent == Intent.RESERVATION_REQUEST
    assert entities.requested_reservation_date == expected
    assert entities.number_of_guests == 2 and entities.requested_reservation_time == '18:00'


@pytest.mark.parametrize('date_text', ['November 15 2026', 'November 15, 2026', 'Nov 15 2026'])
def test_llm_date_extraction_normalizes_and_preserves_stated_year(agent, date_text):
    agent.llm.chat.return_value = SimpleNamespace(content=json.dumps({
        'intent': 'reservation_request', 'requested_reservation_date': date_text,
    }))
    intent, entities = agent.extract_intent_and_entities(f'Book a table on {date_text}')
    assert intent == Intent.RESERVATION_REQUEST
    assert entities.requested_reservation_date == '2026-11-15'


def test_llm_cannot_drop_explicit_year_or_turn_invalid_date_valid(agent):
    agent.llm.chat.return_value = SimpleNamespace(content=json.dumps({
        'intent': 'reservation_request', 'requested_reservation_date': '2027-11-15',
    }))
    _, entities = agent.extract_intent_and_entities('Book a table on November 15 2026')
    assert entities.requested_reservation_date == '2026-11-15'
    _, entities = agent.extract_intent_and_entities('Book a table on February 30, 2026')
    assert entities.requested_reservation_date is None
    _, entities = agent.extract_intent_and_entities('Book a table on 2026-02-29')
    assert entities.requested_reservation_date is None


def test_month_date_followed_by_comma_and_guest_count(agent):
    _, entities = agent._extract_rule_based('Book for November 15, 2026, 4 guests at 18:00')
    assert entities.requested_reservation_date == '2026-11-15'
    assert entities.number_of_guests == 4 and entities.requested_reservation_time == '18:00'


def test_llm_missing_date_uses_rule_based_extraction(agent):
    agent.llm.chat.return_value = SimpleNamespace(content='{"intent": "reservation_request"}')
    _, entities = agent.extract_intent_and_entities('Book a table on November 15, 2026')
    assert entities.requested_reservation_date == '2026-11-15'
