"""Reservation references must not be populated from customer phone numbers."""
import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from src.database.init_db import init_database
from src.services.agent_service import AgentService, Intent
from src.services.llm_service import OllamaService


@pytest.fixture
def agent(tmp_path):
    path = tmp_path / 'restaurant.db'
    init_database(path, seed=True)
    return AgentService(db_path=str(path), llm_service=MagicMock(spec=OllamaService))


@pytest.mark.parametrize('message, phone', [
    ('My phone number is 9876543210', '9876543210'),
    ('Phone is +919876543210', '+919876543210'),
    ('9876543210', '9876543210'),
])
def test_rule_based_phone_is_not_a_reservation_reference(agent, message, phone):
    _, entities = agent._extract_rule_based(message)
    assert entities.customer_phone == phone
    assert entities.reservation_identifier is None


@pytest.mark.parametrize('message', ['cancel reservation 42', 'cancel booking number 42', 'cancel #42'])
def test_explicit_reservation_reference_is_preserved(agent, message):
    _, entities = agent._extract_rule_based(message)
    assert entities.reservation_identifier == 42
    assert entities.customer_phone is None


def test_reference_and_phone_in_one_message_are_separate(agent):
    _, entities = agent._extract_rule_based('Cancel reservation 42, phone number is 9876543210')
    assert entities.reservation_identifier == 42
    assert entities.customer_phone == '9876543210'


def test_long_explicit_reference_is_not_extracted_as_phone(agent):
    _, entities = agent._extract_rule_based('Cancel reservation number 9876543210')
    assert entities.reservation_identifier == 9876543210
    assert entities.customer_phone is None


@pytest.mark.parametrize('message, identifier', [
    ('My phone number is 9876543210', 9876543210),
    ('My phone number is 9876543210', 919876543210),
    ('Phone is +919876543210', 9876543210),
    ('Phone is +919876543210', 919876543210),
])
def test_llm_phone_is_not_accepted_as_reference(agent, message, identifier):
    agent.llm.chat.return_value = SimpleNamespace(content=json.dumps({
        'intent': 'unknown', 'reservation_identifier': identifier,
    }))
    _, entities = agent.extract_intent_and_entities(message)
    assert entities.customer_phone is not None
    assert entities.reservation_identifier is None


def test_malformed_hyphenated_phone_after_reservation_is_not_partial_reference(agent):
    _, entities = agent._extract_rule_based('Reservation 98765-43210')
    assert entities.customer_phone == '98765-43210'
    assert entities.reservation_identifier is None


def test_llm_cancellation_still_retains_a_reference(agent):
    agent.llm.chat.return_value = SimpleNamespace(content=json.dumps({
        'intent': Intent.RESERVATION_CANCELLATION, 'reservation_identifier': 42,
    }))
    intent, entities = agent.extract_intent_and_entities('Cancel reservation 42')
    assert intent == Intent.RESERVATION_CANCELLATION
    assert entities.reservation_identifier == 42
    assert entities.customer_phone is None
