"""Grounding and conversational context for menu questions."""
from unittest.mock import MagicMock

import pytest

from src.database.connection import create_connection
from src.database.crud import find_safe_items_for_allergens
from src.database.init_db import init_database
from src.services.agent_service import AgentService, Intent
from src.services.llm_service import OllamaConnectionError, OllamaService


@pytest.fixture
def agent(tmp_path):
    path = tmp_path / "restaurant.db"
    init_database(path, seed=True)
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


def test_explicit_change_of_topic_does_not_reuse_pending_intent(agent):
    agent.handle_message('What ingredients do you use?', session_id='one')
    result = agent.handle_message('How much is Margherita Pizza?', session_id='one')
    assert result.intent == Intent.MENU_INQUIRY and '$16.00' in result.text


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
    assert response.intent == Intent.MENU_INQUIRY and '$16.00' in response.text
    assert agent.llm.chat.call_count == 1
    agent.llm.generate_restaurant_reply.assert_not_called()


def test_unavailable_dish_is_not_claimed_available(agent):
    with create_connection(agent.db_path) as conn:
        conn.execute("UPDATE menu_items SET is_available = 0 WHERE name = 'Margherita Pizza'")
    result = agent.handle_message('How much is Margherita Pizza?')
    assert 'currently unavailable' in result.text
    assert not result.data['is_available']
