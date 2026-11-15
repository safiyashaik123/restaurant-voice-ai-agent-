"""Regression coverage for grounded multi-turn conversation."""
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from src.core.reservation_policy import restaurant_today
from src.database.connection import create_connection
from src.database.init_db import init_database
from src.services.agent_service import AgentService, ExtractedEntities, Intent
from src.services.llm_service import OllamaConnectionError, OllamaService


@pytest.fixture
def agent(tmp_path):
    path = tmp_path / 'restaurant.db'
    init_database(path)
    # Synthetic test values are isolated from seed and restaurant data.
    with create_connection(path) as conn:
        conn.executemany("UPDATE menu_items SET price = ? WHERE name = ?", [
            (299, 'Bruschetta Classica'), (449, 'Caprese Salad'),
            (449, 'Margherita Pizza'), (299, 'Classic Tiramisu'),
        ])
    llm = MagicMock(spec=OllamaService)
    llm.chat.side_effect = OllamaConnectionError('offline')
    return AgentService(db_path=str(path), llm_service=llm)


def test_pending_reservation_ignores_llm_greeting_and_completes(agent):
    session = 'one'
    first = agent.handle_message("I'd like to reserve a table for four people tomorrow evening.", session)
    day = (restaurant_today() + timedelta(days=1)).isoformat()
    assert first.intent == Intent.RESERVATION_REQUEST
    state = agent.get_session(session)
    assert state.slots['number_of_guests'] == 4 and state.slots['requested_reservation_date'] == day
    agent.llm.chat.side_effect = None
    agent.llm.chat.return_value = SimpleNamespace(content='{"intent":"greeting"}')
    agent.llm.chat.reset_mock()
    reply = agent.handle_message('7pm', session)
    assert reply.intent == Intent.RESERVATION_REQUEST and 'name' in reply.text.lower()
    assert state.slots['requested_reservation_time'] == '19:00'
    assert state.slots['number_of_guests'] == 4 and state.slots['requested_reservation_date'] == day
    agent.llm.chat.assert_not_called()
    assert state.awaiting_reservation_slot == 'customer_name'
    agent.handle_message('Alice Smith', session)
    assert state.awaiting_reservation_slot == 'customer_phone'
    booked = agent.handle_message('My phone number is 9876543210', session)
    assert booked.success and booked.data['status'] == 'confirmed'
    assert booked.data['customer_name'] == 'Alice Smith'
    assert booked.data['reservation_date'] == day and booked.data['party_size'] == 4
    assert state.pending_intent is None and not state.slots
    assert state.awaiting_reservation_slot is None
    with create_connection(agent.db_path) as conn:
        assert conn.execute('SELECT COUNT(*) FROM reservations').fetchone()[0] == 1


def test_pending_reservation_still_allows_cancellation(agent):
    agent.handle_message('Book a table for four tomorrow', 'one')
    result = agent.handle_message('Cancel my booking', 'one')
    assert result.intent == Intent.RESERVATION_CANCELLATION
    assert result.needs_clarification


def test_menu_followup_does_not_guess_and_sessions_are_isolated(agent):
    desserts = agent.handle_message('Recommend a dessert.', 'one')
    assert len(desserts.data['items']) == 2
    followup = agent.handle_message('How much does it cost?', 'one')
    assert followup.needs_clarification and 'which dish' in followup.text.lower()
    answer = agent.handle_message('Classic Tiramisu', 'one')
    assert '₹299' in answer.text
    agent.handle_message('How much is Caprese Salad?', 'two')
    assert '₹449' in agent.handle_message('How much does it cost?', 'two').text
    assert '₹299' in agent.handle_message('How much does it cost?', 'one').text


def test_multiple_categories_and_ingredients(agent):
    response = agent.handle_message('What desserts and beverages are available, and what are their prices?')
    assert response.data['categories'] == ['Desserts', 'Beverages']
    items = response.data['items']
    assert len(items) == 4 and len({item['id'] for item in items}) == 4
    assert all(item['name'] in response.text and '₹' in response.text for item in items)
    single = agent.handle_message('What desserts are available?')
    assert single.data['category'] == 'Desserts' and len(single.data['items']) == 2
    ingredients = agent.handle_message('Can you tell me the ingredients used in the Vegan Mediterranean Bowl?')
    assert ingredients.intent == Intent.INGREDIENT_INQUIRY and 'Chickpeas' in ingredients.text


def test_generic_nuts_requests_clarification(agent):
    result = agent.handle_message('I have a nut allergy. Which dishes are safe for me to eat?')
    assert result.needs_clarification and result.is_fallback
    assert 'peanuts' in result.text.lower() and 'tree nuts' in result.text.lower()
    assert 'cross-contact' in result.text.lower()


@pytest.mark.parametrize('phrase', ['the first one', 'that one', 'what about it', 'tell me more about it'])
def test_ambiguous_menu_references_ask_for_a_name(agent, phrase):
    agent.handle_message('Recommend a dessert.', 'one')
    reply = agent.handle_message(phrase, 'one')
    assert reply.needs_clarification
    assert 'which dish' in reply.text.lower()
    assert not reply.entities.menu_item_name
    answer = agent.handle_message('Classic Tiramisu', 'one')
    assert answer.data['name'] == 'Classic Tiramisu'


@pytest.mark.parametrize('phrase', ['that one', 'what about it', 'tell me more about it'])
def test_unambiguous_menu_references_use_last_item(agent, phrase):
    agent.handle_message('How much is Caprese Salad?', 'one')
    reply = agent.handle_message(phrase, 'one')
    assert reply.data['name'] == 'Caprese Salad'
    assert '₹449' in reply.text


def test_ordinal_reference_does_not_guess_even_after_one_dish(agent):
    agent.handle_message('How much is Caprese Salad?', 'one')
    reply = agent.handle_message('the first one', 'one')
    assert reply.needs_clarification and not reply.entities.menu_item_name


def test_multiword_categories_and_aliases(agent):
    conn = create_connection(agent.db_path)
    try:
        conn.execute("INSERT INTO categories (name) VALUES ('Small Plates')")
        category_id = conn.execute("SELECT id FROM categories WHERE name = 'Small Plates'").fetchone()[0]
        conn.execute("INSERT INTO menu_items (category_id, name, price) VALUES (?, 'Test Plate', 100)", (category_id,))
        conn.commit()
    finally:
        conn.close()
    result = agent.handle_message('What SMALL PLATES and dessert are available?')
    assert set(result.data['categories']) == {'Small Plates', 'Desserts'}
    assert {item['name'] for item in result.data['items']} == {'Test Plate', 'Classic Tiramisu', 'Artisan Sorbet Trio'}
    result = agent.handle_message('Do you have small plates and desserts?')
    assert set(result.data['categories']) == {'Small Plates', 'Desserts'}
    result = agent.handle_message('What small plates are available?')
    assert result.data['category'] == 'Small Plates'
    assert [item['name'] for item in result.data['items']] == ['Test Plate']
    result = agent.handle_message('What drinks are available?')
    assert result.data['category'] == 'Beverages'
    result = agent.handle_message('What small platters are available?')
    assert not result.data or result.data.get('category') != 'Small Plates'


def test_plain_name_only_after_explicit_name_prompt(agent):
    state = agent.get_session('one')
    state.pending_intent = Intent.RESERVATION_REQUEST
    state.slots.update(requested_reservation_date='2030-01-01', requested_reservation_time='19:00', number_of_guests=2)
    _, entities = agent._extract_rule_based('Alice Smith', state)
    assert entities.customer_name is None  # Missing is not the same as having been asked.
    state.awaiting_reservation_slot = 'customer_phone'
    _, entities = agent._extract_rule_based('Alice Smith', state)
    assert entities.customer_name is None
    state.awaiting_reservation_slot = 'customer_name'
    _, entities = agent._extract_rule_based('Alice Smith', state)
    assert entities.customer_name == 'Alice Smith'
    state.slots['requested_reservation_time'] = None
    _, entities = agent._extract_rule_based('Alice Smith', state)
    assert entities.customer_name is None
    state.clear()
    assert state.awaiting_reservation_slot is None


def test_ambiguous_ingredient_and_allergen_references_do_not_guess(agent):
    for question, intent in [('What ingredients are in that one?', Intent.INGREDIENT_INQUIRY),
                             ('Does that one contain peanuts?', Intent.ALLERGEN_INQUIRY)]:
        agent.handle_message('Recommend a dessert.', 'one')
        result = agent.handle_message(question, 'one')
        assert result.intent == intent and result.needs_clarification
        assert not result.entities.menu_item_name
        answer = agent.handle_message('Classic Tiramisu', 'one')
        assert answer.intent == intent


def test_overlapping_category_and_drink_alias_do_not_add_extra_category(agent):
    conn = create_connection(agent.db_path)
    try:
        conn.execute("INSERT INTO categories (name) VALUES ('Hot Drinks')")
        category_id = conn.execute("SELECT id FROM categories WHERE name = 'Hot Drinks'").fetchone()[0]
        conn.execute("INSERT INTO menu_items (category_id, name, price) VALUES (?, 'Test Tea', 100)", (category_id,))
        conn.commit()
    finally:
        conn.close()
    response = agent.handle_message('What hot drinks are available?')
    assert response.data['category'] == 'Hot Drinks'
    assert [item['name'] for item in response.data['items']] == ['Test Tea']


def test_name_prompt_rejects_invalid_input_and_cancellation_clears_expected_slot(agent):
    agent.handle_message('Book a table for four tomorrow at 7pm', 'one')
    state = agent.get_session('one')
    assert state.awaiting_reservation_slot == 'customer_name'
    agent.handle_message('123', 'one')
    assert state.slots.get('customer_name') is None
    assert state.awaiting_reservation_slot == 'customer_name'
    result = agent.handle_message('Cancel my booking', 'one')
    assert result.intent == Intent.RESERVATION_CANCELLATION and result.needs_clarification
    assert state.awaiting_reservation_slot is None
    _, entities = agent._extract_rule_based('Safiya', state)
    assert entities.customer_name is None


def test_invalid_stored_name_prompts_for_name_again(agent):
    state = agent.get_session('one')
    conn = create_connection(agent.db_path)
    try:
        result = agent._route_reservation_request(
            conn, '', ExtractedEntities(
                requested_reservation_date=(restaurant_today() + timedelta(days=1)).isoformat(),
                requested_reservation_time='19:00', number_of_guests=2,
                customer_name=' ', customer_phone='9876543210',
            ), state,
        )
    finally:
        conn.close()
    assert result.needs_clarification
    assert state.slots['customer_name'] is None
    assert state.awaiting_reservation_slot == 'customer_name'
    result = agent.handle_message('Safiya', 'one')
    assert result.success and result.data['customer_name'] == 'Safiya'
    assert state.awaiting_reservation_slot is None


def test_confirmed_cancellation_clears_reservation_slot_state(agent):
    agent.handle_message('Book a table for two tomorrow at 7pm', 'booked')
    agent.handle_message('Safiya', 'booked')
    booking = agent.handle_message('9876543210', 'booked')
    reference = booking.data['reservation_id']
    agent.handle_message('Book a table for four tomorrow at 7pm', 'one')
    state = agent.get_session('one')
    assert state.awaiting_reservation_slot == 'customer_name'
    result = agent.handle_message(f'cancel booking {reference}', 'one')
    assert not result.success and 'private access code' in result.text
    assert state.pending_intent == Intent.RESERVATION_REQUEST and state.slots
    with create_connection(agent.db_path) as conn:
        assert conn.execute('SELECT status FROM reservations WHERE id = ?', (reference,)).fetchone()[0] == 'confirmed'


def test_pending_booking_cancellation_waits_for_reference_and_preserves_slots(agent):
    agent.handle_message('Book a table for two tomorrow at 7pm', 'booked')
    agent.handle_message('Safiya', 'booked')
    booking = agent.handle_message('9876543210', 'booked')
    reference = booking.data['reservation_id']

    agent.handle_message('Book a table for four tomorrow at 7pm', 'pending')
    state = agent.get_session('pending')
    original_slots = state.slots.copy()
    prompt = agent.handle_message('Cancel my booking', 'pending')
    assert prompt.intent == Intent.RESERVATION_CANCELLATION
    assert prompt.needs_clarification and 'reference' in prompt.text.lower()
    assert state.awaiting_cancellation_reference
    assert state.pending_intent == Intent.RESERVATION_REQUEST and state.slots == original_slots

    invalid = agent.handle_message('999999', 'pending')
    assert invalid.intent == Intent.RESERVATION_CANCELLATION and not invalid.success
    assert state.awaiting_cancellation_reference
    assert state.pending_intent == Intent.RESERVATION_REQUEST and state.slots == original_slots
    unclear = agent.handle_message('not sure', 'pending')
    assert unclear.intent == Intent.RESERVATION_CANCELLATION and unclear.needs_clarification
    assert state.slots == original_slots

    result = agent.handle_message(str(reference), 'pending')
    assert result.intent == Intent.RESERVATION_CANCELLATION and not result.success
    assert 'private access code' in result.text and result.data is None
    assert state.pending_intent == Intent.RESERVATION_REQUEST and state.slots == original_slots
    conn = create_connection(agent.db_path)
    try:
        assert conn.execute('SELECT status FROM reservations WHERE id = ?', (reference,)).fetchone()[0] == 'confirmed'
    finally:
        conn.close()


@pytest.mark.parametrize('phrase', ["Never mind", "Don't cancel", 'Keep my reservation', 'Continue my booking'])
def test_abandon_cancellation_preserves_booking_and_resumes_reservation(agent, phrase):
    agent.handle_message('Book a table for two tomorrow at 7pm', 'booked')
    agent.handle_message('Safiya', 'booked')
    booked = agent.handle_message('9876543210', 'booked')
    reference = booked.data['reservation_id']

    agent.handle_message('Book a table for four tomorrow at 7pm', 'pending')
    state = agent.get_session('pending')
    original_slots = state.slots.copy()
    agent.handle_message('Cancel my booking', 'pending')
    assert state.awaiting_cancellation_reference
    result = agent.handle_message(phrase, 'pending')
    assert not state.awaiting_cancellation_reference
    assert state.pending_intent == Intent.RESERVATION_REQUEST and state.slots == original_slots
    assert state.awaiting_reservation_slot == 'customer_name'
    assert result.intent == Intent.RESERVATION_REQUEST and 'name' in result.text.lower()
    conn = create_connection(agent.db_path)
    try:
        assert conn.execute('SELECT status FROM reservations WHERE id = ?', (reference,)).fetchone()[0] == 'confirmed'
    finally:
        conn.close()
    following = agent.handle_message('Alice Smith', 'pending')
    assert following.intent == Intent.RESERVATION_REQUEST
    assert state.slots['customer_name'] == 'Alice Smith'
    assert state.awaiting_reservation_slot == 'customer_phone'
    assert state.slots['number_of_guests'] == 4


def test_abandon_cancellation_without_pending_reservation(agent):
    agent.handle_message('Cancel my booking', 'one')
    state = agent.get_session('one')
    assert state.awaiting_cancellation_reference
    result = agent.handle_message('Do not cancel!', 'one')
    assert not state.awaiting_cancellation_reference
    assert state.pending_intent is None and not state.slots
    assert result.intent == Intent.RESERVATION_CANCELLATION


def test_price_comparison_uses_available_database_values_and_context(agent):
    with create_connection(agent.db_path) as conn:
        conn.execute("UPDATE menu_items SET price = 1 WHERE name = 'Caprese Salad'")
        conn.execute("UPDATE menu_items SET price = 0, is_available = 0 WHERE name = 'Classic Tiramisu'")
        conn.commit()
    reply = agent.handle_message('What is the cheapest dish?', 'prices')
    assert reply.data['items'][0]['name'] == 'Caprese Salad'
    assert '₹1' in reply.text
    followup = agent.handle_message('How much does it cost?', 'prices')
    assert followup.data['name'] == 'Caprese Salad'
    highest = agent.handle_message('What is the most expensive beverage?', 'prices')
    assert highest.data['items'] and all(item['category'] == 'Beverages' for item in highest.data['items'])


def test_price_comparison_returns_all_ties(agent):
    with create_connection(agent.db_path) as conn:
        conn.execute("UPDATE menu_items SET price = 1 WHERE name IN ('Caprese Salad', 'Classic Tiramisu')")
        conn.commit()
    reply = agent.handle_message('What is the cheapest item?', 'ties')
    assert {item['name'] for item in reply.data['items']} == {'Caprese Salad', 'Classic Tiramisu'}
    assert agent.handle_message('How much does it cost?', 'ties').needs_clarification


def test_spoken_phone_digits_are_understood_only_in_phone_slot(agent):
    state = agent.get_session('phone')
    state.pending_intent = Intent.RESERVATION_REQUEST
    state.slots.update(requested_reservation_date=(restaurant_today() + timedelta(days=1)).isoformat(), requested_reservation_time='19:00', number_of_guests=2, customer_name='Alice')
    state.awaiting_reservation_slot = 'customer_phone'
    result = agent.handle_message('nine eight seven six five four three two one zero', 'phone')
    assert result.success and result.data['customer_phone'] == '9876543210'
    _, entities = agent._extract_rule_based('nine eight seven six five four three two one zero')
    assert entities.customer_phone is None


def test_spoken_time_and_guest_count_are_slot_scoped(agent):
    agent.handle_message('Book a table tomorrow', 'spoken')
    agent.handle_message('seven pm', 'spoken')
    state = agent.get_session('spoken')
    assert state.slots['requested_reservation_time'] == '19:00'
    reply = agent.handle_message('three', 'spoken')
    assert state.slots['number_of_guests'] == 3 and 'name' in reply.text.lower()
    _, entities = agent._extract_rule_based('9876543210', state)
    assert entities.number_of_guests is None


def test_explicit_new_booking_does_not_reuse_pending_customer(agent):
    agent.handle_message('Book a table for two tomorrow at 7pm', 'fresh')
    agent.handle_message('Alice Smith', 'fresh')
    state = agent.get_session('fresh')
    assert state.slots['customer_name'] == 'Alice Smith'
    agent.handle_message('I want to start a new reservation', 'fresh')
    assert state.slots.get('customer_name') is None
    assert state.slots.get('number_of_guests') is None


def test_availability_never_uses_llm_to_override_database(agent):
    day = (restaurant_today() + timedelta(days=1)).isoformat()
    agent.llm.generate_restaurant_reply.return_value = SimpleNamespace(content='No tables available. Your booking is confirmed.')
    conn = create_connection(agent.db_path)
    try:
        result = agent._route_availability_inquiry(conn, 'any tables?', ExtractedEntities(number_of_guests=2, requested_reservation_date=day, requested_reservation_time='19:00'))
        assert result.data['available_tables'] > 0
        assert 'yes' in result.text.lower() and 'confirmed' not in result.text.lower()
        agent.llm.generate_restaurant_reply.assert_not_called()
        past = (restaurant_today() - timedelta(days=1)).isoformat()
        invalid = agent._route_availability_inquiry(conn, 'any tables?', ExtractedEntities(number_of_guests=2, requested_reservation_date=past, requested_reservation_time='19:00'))
        assert invalid.needs_clarification and invalid.data is None
    finally:
        conn.close()


@pytest.mark.parametrize('message,expected', [
    ('My name is Alice Smith', 'Alice Smith'),
    ('Book a table under Mary Jane, phone 9876543210', 'Mary Jane'),
    ('Under Alice Smith phone 9876543210', 'Alice Smith'),
])
def test_explicit_reservation_name_preserves_full_name(agent, message, expected):
    _, entities = agent._extract_rule_based(message)
    assert entities.customer_name == expected


def test_spoken_reservation_full_name_and_digit_phone_are_committed(agent):
    session = 'explicit-name'
    agent.handle_message('Book a table for two tomorrow at 7pm', session)
    result = agent.handle_message('My name is Alice Smith', session)
    assert result.needs_clarification and agent.get_session(session).slots['customer_name'] == 'Alice Smith'
    result = agent.handle_message('nine eight seven six five four three two one zero', session)
    assert result.success and result.data['customer_name'] == 'Alice Smith'
    assert result.data['customer_phone'] == '9876543210'
    with create_connection(agent.db_path) as conn:
        row = conn.execute('SELECT customer_name, customer_phone FROM reservations WHERE id=?', (result.data['reservation_id'],)).fetchone()
        assert tuple(row) == ('Alice Smith', '9876543210')
