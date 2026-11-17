"""Grounding and conversational context for menu questions."""
from unittest.mock import MagicMock

import pytest

from src.database.connection import create_connection
from src.database.crud import find_safe_items_for_allergens, get_menu_categories, get_menu_items
from src.database.init_db import init_database
from src.services.agent_service import AgentService, Intent, _format_menu_price
from src.services.llm_service import OllamaConnectionError, OllamaService


@pytest.fixture
def agent(tmp_path):
    path = tmp_path / "restaurant.db"
    init_database(path, seed=True)
    # Synthetic test values are isolated from seed and restaurant data.
    with create_connection(path) as conn:
        conn.executemany("UPDATE menu_items SET price = ? WHERE name = ?", [
            (299, 'Bruschetta Classica'), (449, 'Caprese Salad'),
            (449, 'Margherita Pizza'), (299, 'Classic Tiramisu'),
        ])
    llm = MagicMock(spec=OllamaService)
    llm.chat.side_effect = OllamaConnectionError("offline")
    # A model response must not override verified facts.
    llm.generate_restaurant_reply.return_value.content = "Invented ingredient: unicorn dust."
    return AgentService(db_path=str(path), llm_service=llm)


@pytest.mark.parametrize("question", ["What vegan dishes do you have?", "Is Margherita Pizza vegetarian?", "Do you have vegetarian options?"])
def test_dietary_classification_is_not_inferred(agent, question):
    result = agent.handle_message(question)
    assert result.is_fallback and "verified" in result.text.lower() and "staff" in result.text.lower()
    assert "unicorn" not in result.text.lower()
    agent.llm.generate_restaurant_reply.assert_not_called()


def test_ingredients_exactly_match_database_and_empty_is_unknown(agent):
    result = agent.handle_message("What ingredients are in the Bruschetta Classica?")
    assert result.intent == Intent.INGREDIENT_INQUIRY
    with create_connection(agent.db_path) as conn:
        expected = [row[0] for row in conn.execute("SELECT ingredient_name FROM menu_item_ingredients WHERE menu_item_id = (SELECT id FROM menu_items WHERE name = ?) ORDER BY id", ("Bruschetta Classica",))]
        conn.execute("DELETE FROM menu_item_ingredients WHERE menu_item_id = (SELECT id FROM menu_items WHERE name = 'Bruschetta Classica')")
    assert result.data['ingredients'] == expected
    assert all(name in result.text for name in expected)
    assert 'unicorn' not in result.text.lower()
    assert 'do not have ingredient records' in agent.handle_message("What ingredients are in the Bruschetta Classica?").text


def test_allergen_presence_absence_unknown_and_cross_contact(agent):
    present = agent.handle_message("Does Margherita Pizza contain dairy?")
    assert 'contains Dairy' in present.text and 'cross-contact' in present.text
    absent = agent.handle_message("Does Margherita Pizza contain peanuts?")
    assert 'does not list Peanuts' in absent.text and 'cross-contact' in absent.text
    unknown = agent.handle_message("Does Margherita Pizza contain mustard?")
    assert unknown.is_fallback and 'cannot verify' in unknown.text and 'does not' not in unknown.text
    unverified = agent.handle_message("Does Chef's Daily Rotating Special contain dairy?")
    assert unverified.is_fallback and 'not been formally verified' in unverified.text


@pytest.mark.parametrize("question", ["What dishes are mustard-free?", "Is Margherita Pizza mustard free?", "What dishes are nut free?", "What dishes are gluten and mustard free?"])
def test_untracked_ambiguous_or_multiple_allergens_do_not_return_safe_list(agent, question):
    result = agent.handle_message(question)
    assert result.intent == Intent.ALLERGEN_INQUIRY
    assert result.is_fallback and result.needs_clarification
    assert not result.data or 'safe_items' not in result.data


def test_multiple_tracked_allergens_require_clarification(agent):
    result = agent.handle_message('Does the Margherita Pizza contain any dairy or gluten?')
    assert result.intent == Intent.ALLERGEN_INQUIRY
    assert result.is_fallback and result.needs_clarification
    assert 'cannot verify' in result.text.lower()
    assert 'staff' in result.text.lower()


def test_crud_rejects_untracked_allergen(agent):
    with create_connection(agent.db_path) as conn, pytest.raises(ValueError, match="not tracked"):
        find_safe_items_for_allergens(conn, ['Mustard'])


def test_follow_up_context_and_session_isolation(agent):
    agent.handle_message("How much is the Margherita Pizza?", session_id='one')
    ingredients = agent.handle_message("What are its ingredients?", session_id='one')
    assert ingredients.intent == Intent.INGREDIENT_INQUIRY and 'Wheat flour' in ingredients.text
    allergens = agent.handle_message("What about dairy?", session_id='one')
    assert allergens.intent == Intent.ALLERGEN_INQUIRY and 'contains Dairy' in allergens.text
    other = agent.handle_message("What are its ingredients?", session_id='two')
    assert other.needs_clarification and 'Which menu item' in other.text
    answered = agent.handle_message('Bruschetta Classica', session_id='two')
    assert answered.intent == Intent.INGREDIENT_INQUIRY and 'Roma tomatoes' in answered.text


def test_unknown_dish_does_not_reuse_stale_context(agent):
    agent.handle_message("How much is the Margherita Pizza?", session_id='one')
    unknown = agent.handle_message("What ingredients are in the Truffle Pasta?", session_id='one')
    assert unknown.is_fallback and 'could not find' in unknown.text.lower()
    followup = agent.handle_message("What are its ingredients?", session_id='one')
    assert followup.needs_clarification and 'Wheat flour' not in followup.text


def test_category_and_general_menu_are_grounded_without_llm(agent):
    categories = agent.handle_message('What is on the menu?')
    assert 'Desserts' in categories.text
    desserts = agent.handle_message('Do you have desserts?')
    assert desserts.data['category'] == 'Desserts' and desserts.data['items']
    assert all(item['category'] == 'Desserts' for item in desserts.data['items'])
    assert all(item['name'] in desserts.text for item in desserts.data['items'])
    assert 'Margherita Pizza' not in desserts.text
    assert 'Wheat flour' in agent.handle_message("What is in the Margherita Pizza?").text
    assert agent.handle_message('Do you have Margherita Pizza?').data['name'] == 'Margherita Pizza'
    agent.llm.chat.assert_not_called()
    agent.llm.generate_restaurant_reply.assert_not_called()


def test_unknown_dish_with_allergen_is_not_treated_as_general_filter(agent):
    result = agent.handle_message('Does Truffle Pasta contain peanuts?')
    assert result.intent == Intent.ALLERGEN_INQUIRY
    assert result.is_fallback and 'not in our restaurant database' in result.text
    assert not result.data or 'safe_items' not in result.data
    ingredients = agent.handle_message('What ingredients are in Truffle Pasta?')
    assert ingredients.is_fallback and 'could not find' in ingredients.text.lower()


def test_menu_prices_display_rupee_symbol(agent):
    item = agent.handle_message('How much is Caprese Salad?')
    assert '₹449' in item.text and '$' not in item.text
    category = agent.handle_message('What appetizers do you have?')
    assert '₹299' in category.text and '₹449' in category.text
    assert '$' not in category.text
    general = agent.handle_message('What is on the menu?')
    assert '₹' in general.text and '$' not in general.text


@pytest.mark.parametrize(('price', 'display'), [(16, '₹16'), (16.5, '₹16.5'), (16.25, '₹16.25')])
def test_menu_price_display_preserves_database_value(agent, price, display):
    conn = create_connection(agent.db_path)
    try:
        conn.execute("UPDATE menu_items SET price = ? WHERE name = ?", (price, 'Caprese Salad'))
        conn.commit()
    finally:
        conn.close()
    result = agent.handle_message('How much is Caprese Salad?')
    assert result.data['price'] == price and display in result.text and '$' not in result.text
    conn = create_connection(agent.db_path)
    try:
        assert conn.execute("SELECT price FROM menu_items WHERE name = ?", ('Caprese Salad',)).fetchone()[0] == price
    finally:
        conn.close()


def test_explicit_change_of_topic_does_not_reuse_pending_intent(agent):
    agent.handle_message('What ingredients do you use?', session_id='one')
    result = agent.handle_message('How much is Margherita Pizza?', session_id='one')
    assert result.intent == Intent.MENU_INQUIRY and '₹449' in result.text


def test_follow_up_alternate_wording_and_unknown_answer(agent):
    agent.handle_message('How much is Margherita Pizza?', session_id='one')
    result = agent.handle_message('And peanuts?', session_id='one')
    assert result.intent == Intent.ALLERGEN_INQUIRY and 'does not list Peanuts' in result.text
    agent.handle_message('What ingredients do you use?', session_id='two')
    unknown = agent.handle_message('Truffle Pasta', session_id='two')
    assert unknown.intent == Intent.INGREDIENT_INQUIRY and 'could not find' in unknown.text.lower()


def test_similarly_named_unknown_dish_is_not_mistaken_for_known_item(agent):
    result = agent.handle_message('Does Margherita Pizza Deluxe contain dairy?')
    assert result.is_fallback and 'not in our restaurant database' in result.text
    assert 'contains Dairy' not in result.text


def test_multiple_allergens_without_conjunction_are_not_partially_filtered(agent):
    result = agent.handle_message('What dishes are gluten-free dairy-free?')
    assert result.is_fallback and result.needs_clarification
    assert not result.data or 'safe_items' not in result.data


def test_ambiguous_menu_intent_uses_at_most_one_model_call(agent):
    agent.llm.chat.side_effect = None
    agent.llm.chat.return_value.content = '{"intent":"menu_inquiry","menu_item_name":"Margherita Pizza"}'
    response = agent.handle_message('Tell me about the pizza')
    assert response.intent == Intent.MENU_INQUIRY and '₹449' in response.text
    assert agent.llm.chat.call_count == 1
    agent.llm.generate_restaurant_reply.assert_not_called()


def test_unavailable_dish_is_not_claimed_available(agent):
    with create_connection(agent.db_path) as conn:
        conn.execute("UPDATE menu_items SET is_available = 0 WHERE name = 'Margherita Pizza'")
    result = agent.handle_message('How much is Margherita Pizza?')
    assert 'currently unavailable' in result.text
    assert not result.data['is_available']


@pytest.mark.parametrize(
    ("query", "expected_intent"),
    [
        ("List all available dishes.", Intent.MENU_INQUIRY),
        ("What dishes are available today?", Intent.MENU_INQUIRY),
        ("Show me all available menu items.", Intent.MENU_INQUIRY),
        ("List all dishes with their prices.", Intent.MENU_INQUIRY),
        ("Are tables available?", Intent.AVAILABILITY_INQUIRY),
        ("Is a table available for two?", Intent.AVAILABILITY_INQUIRY),
        ("Do you have any free tables?", Intent.AVAILABILITY_INQUIRY),
        ("Is a table available for tonight?", Intent.AVAILABILITY_INQUIRY),
    ],
)
def test_intent_routing_menu_vs_table_availability(agent, query, expected_intent):
    res = agent.handle_message(query)
    assert res.intent == expected_intent


@pytest.mark.parametrize(
    "query",
    [
        "List all available dishes.",
        "List all available dishes with their prices.",
        "Show me the full menu.",
        "Show the complete menu.",
        "List all menu items grouped by category.",
        "Show me all available menu items.",
    ],
)
def test_explicit_full_menu_requests(agent, query):
    res = agent.handle_message(query)
    assert res.intent == Intent.MENU_INQUIRY
    assert not res.is_fallback

    with create_connection(agent.db_path) as conn:
        items = get_menu_items(conn, available_only=True)

    assert items, "Database must have available items for test"
    for item in items:
        assert item["name"] in res.text
        assert _format_menu_price(item["price"]) in res.text

    active_categories = {item["category"] for item in items}
    for category in active_categories:
        assert category in res.text


def test_full_menu_excludes_unavailable_items(agent):
    with create_connection(agent.db_path) as conn:
        conn.execute("UPDATE menu_items SET is_available = 0 WHERE name = 'Margherita Pizza'")
        conn.commit()

    res = agent.handle_message("Show me the full menu.")
    assert res.intent == Intent.MENU_INQUIRY
    assert "Margherita Pizza" not in res.text
    assert "Bruschetta Classica" in res.text


def test_empty_available_menu_returns_friendly_message(agent):
    with create_connection(agent.db_path) as conn:
        conn.execute("UPDATE menu_items SET is_available = 0")
        conn.commit()

    res = agent.handle_message("List all available dishes.")
    assert res.intent == Intent.MENU_INQUIRY
    assert "no available items" in res.text.lower()
    assert res.data and res.data["items"] == []


def test_normal_menu_overview_remains_concise(agent):
    res = agent.handle_message("What is on the menu?")
    assert res.intent == Intent.MENU_INQUIRY
    assert "Available dishes include" in res.text

    with create_connection(agent.db_path) as conn:
        items = get_menu_items(conn, available_only=True)

    # First 4 items should be listed
    for item in items[:4]:
        assert item["name"] in res.text

    # Beyond first 4 items should not all be listed in concise overview
    beyond_four = [item["name"] for item in items[4:]]
    assert any(name not in res.text for name in beyond_four)


@pytest.mark.parametrize("question", [
    "What is your most popular dish?",
    "Which dish sells the most?",
    "What do customers order most?",
    "What is your best-selling item?",
    "What are your most popular dishes?",
])
def test_popularity_questions_do_not_invent_a_winner(agent, question):
    result = agent.handle_message(question)

    assert result.intent == Intent.MENU_INQUIRY
    assert result.is_fallback
    assert "verified sales or popularity data" in result.text.lower()

    with create_connection(agent.db_path) as conn:
        items = get_menu_items(conn, available_only=True)

    assert not any(item["name"] in result.text for item in items)


def test_dessert_recommendation_is_not_treated_as_popularity_question(agent):
    result = agent.handle_message("Recommend a dessert.")

    assert result.intent == Intent.MENU_INQUIRY
    assert result.data is not None
    assert len(result.data["items"]) == 2
    assert all(item["category"] == "Desserts" for item in result.data["items"])


def test_popularity_question_does_not_interrupt_pending_reservation(agent):
    session_id = "popularity_during_booking"

    agent.handle_message("Book a table.", session_id=session_id)
    result = agent.handle_message(
        "What is your most popular dish?",
        session_id=session_id,
    )

    assert result.intent == Intent.RESERVATION_REQUEST
    assert result.needs_clarification
