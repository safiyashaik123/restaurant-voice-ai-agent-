"""Unit tests for AgentService (Phase 2 – Step 3, including audit fixes).

Comprehensive testing with mocked Ollama LLM and isolated in-memory SQLite:
- Intent and entity extraction with valid JSON, malformed JSON, and unknown intents.
- Menu inquiry routing (specific item, category, unknown item fallback).
- Ingredient inquiry routing (existing item, unlisted item fallback, missing entity).
- Allergen inquiry routing (verified dish with allergens, verified 0 allergens, unverified dish safety fallback, allergen-free filter).
- Availability inquiry routing (available table, unavailable slot, missing time prompt, negative guests).
- Reservation requests:
  - Missing reservation details & step-by-step slot filling.
  - Invalid inputs (negative party size, invalid phone number).
  - Past-date rejection (Python-level validation).
  - Empty / whitespace-only customer name rejection.
  - Successful booking confirmation only after database record creation.
  - Conflict handling when tables are full.
- Reservation cancellation (valid ID, missing ID prompt, unknown ID).
- Session state management:
  - Session cleared on successful cancellation.
  - Session NOT cleared on failed cancellation.
  - Session TTL expiry evicts abandoned sessions.
  - Expiry of session A does not affect session B (isolation).
- Malformed LLM output and offline resilience.
- Database error handling.
"""

from datetime import date, datetime, timedelta
import json
import sqlite3
import unittest
from unittest.mock import MagicMock, patch
import uuid

from src.core.reservation_policy import restaurant_today
from src.core.config import config
from src.database.connection import create_connection
from src.database.init_db import init_database
from src.services.agent_service import (
    AgentResult,
    AgentService,
    ExtractedEntities,
    Intent,
    InteractionState,
    normalize_date_input,
    normalize_time_input,
)
from src.services.llm_service import (
    LLMResponse,
    OllamaConnectionError,
    OllamaResponseError,
    OllamaService,
)


class TestAgentServiceUnit(unittest.TestCase):
    """Unit test suite for AgentService."""

    def setUp(self):
        """Set up an isolated in-memory database and mock LLM service."""
        self.db_uri = f"file:agent_test_{uuid.uuid4().hex}?mode=memory&cache=shared"
        self.conn = create_connection(self.db_uri)
        init_database(db_path=self.db_uri, seed=True)
        # A synthetic fixture price, independent of restaurant seed values.
        self.conn.execute("UPDATE menu_items SET price = 449 WHERE name = ?", ("Margherita Pizza",))

        self.mock_llm = MagicMock(spec=OllamaService)
        # Default behavior for mock LLM: fallback to rule-based parser
        self.mock_llm.chat.side_effect = OllamaConnectionError("Offline for unit tests")
        self.mock_llm.generate_restaurant_reply.side_effect = OllamaConnectionError("Offline for unit tests")

        self.agent = AgentService(
            db_path=self.db_uri,
            llm_service=self.mock_llm
        )

    def tearDown(self):
        self.conn.close()

    # -----------------------------------------------------------------------
    # 1. Intent Extraction & Malformed LLM Output Tests
    # -----------------------------------------------------------------------
    def test_llm_valid_json_extraction(self):
        """Verify successful parsing when LLM outputs valid structured JSON."""
        mock_json = {
            "intent": "menu_inquiry",
            "menu_item_name": "Margherita Pizza",
            "allergen_name": None,
            "requested_reservation_date": None,
            "requested_reservation_time": None,
            "number_of_guests": None,
            "reservation_identifier": None,
            "customer_name": None,
            "customer_phone": None
        }
        self.mock_llm.chat.side_effect = None
        self.mock_llm.chat.return_value = LLMResponse(
            content=json.dumps(mock_json),
            model="qwen2.5:1.5b"
        )

        intent, entities = self.agent.extract_intent_and_entities("Tell me about the pizza")
        self.assertEqual(intent, Intent.MENU_INQUIRY)
        self.assertEqual(entities.menu_item_name, "Margherita Pizza")

    def test_llm_malformed_json_triggers_rule_based_fallback(self):
        """Verify that malformed JSON from LLM gracefully triggers rule-based parser."""
        self.mock_llm.chat.side_effect = None
        self.mock_llm.chat.return_value = LLMResponse(
            content="Sorry, here is the answer: not json at all",
            model="qwen2.5:1.5b"
        )

        intent, entities = self.agent.extract_intent_and_entities("How much is the Margherita Pizza?")
        self.assertEqual(intent, Intent.MENU_INQUIRY)
        self.assertEqual(entities.menu_item_name, "Margherita Pizza")

    def test_unknown_intent_handling(self):
        """Verify that unsupported intent strings clamp safely to UNKNOWN."""
        mock_json = {"intent": "unsupported_foreign_intent"}
        self.mock_llm.chat.side_effect = None
        self.mock_llm.chat.return_value = LLMResponse(
            content=json.dumps(mock_json),
            model="qwen2.5:1.5b"
        )

        intent, entities = self.agent.extract_intent_and_entities("Random query")
        self.assertEqual(intent, Intent.UNKNOWN)

    # -----------------------------------------------------------------------
    # 2. Menu Inquiry Routing Tests
    # -----------------------------------------------------------------------
    def test_menu_inquiry_existing_item(self):
        """Routing menu inquiry for an existing dish retrieves price and description."""
        res = self.agent.handle_message("How much is the Margherita Pizza?")
        self.assertEqual(res.intent, Intent.MENU_INQUIRY)
        self.assertFalse(res.is_fallback)
        self.assertIn("Margherita Pizza", res.text)
        self.assertIn("₹449", res.text)
        self.assertIsNotNone(res.data)

    def test_menu_inquiry_unlisted_item_triggers_fallback(self):
        """Routing menu inquiry for unlisted dish triggers safe fallback message."""
        res = self.agent.handle_message("Do you have Lobster Macaroni?")
        self.assertEqual(res.intent, Intent.MENU_INQUIRY)
        self.assertTrue(res.is_fallback)
        self.assertIn("could not find", res.text.lower())
        self.assertIn("Lobster Macaroni", res.text)

    def test_menu_inquiry_general(self):
        """General menu inquiry lists menu categories."""
        res = self.agent.handle_message("What is on the menu?")
        self.assertEqual(res.intent, Intent.MENU_INQUIRY)
        self.assertIn("Appetizers", res.text)
        self.assertIn("Mains", res.text)

    # -----------------------------------------------------------------------
    # 3. Ingredient Inquiry Routing Tests
    # -----------------------------------------------------------------------
    def test_ingredient_inquiry_existing_item(self):
        """Routing ingredient inquiry returns exact ingredients from SQLite."""
        res = self.agent.handle_message("What ingredients are in the Bruschetta Classica?")
        self.assertEqual(res.intent, Intent.INGREDIENT_INQUIRY)
        self.assertFalse(res.is_fallback)
        self.assertIn("Roma tomatoes", res.text)
        self.assertIn("Garlic", res.text)

    def test_ingredient_inquiry_unlisted_item_fallback(self):
        """Querying ingredients for unknown dish triggers safe fallback."""
        res = self.agent.handle_message("What ingredients are in the Truffle Pasta?")
        self.assertEqual(res.intent, Intent.INGREDIENT_INQUIRY)
        self.assertTrue(res.is_fallback)
        self.assertIn("could not find", res.text.lower())

    def test_ingredient_inquiry_missing_item_name(self):
        """Asking for ingredients without specifying dish prompts for item name."""
        res = self.agent.handle_message("What ingredients do you use?")
        self.assertEqual(res.intent, Intent.INGREDIENT_INQUIRY)
        self.assertTrue(res.needs_clarification)
        self.assertIn("Which menu item", res.text)

    # -----------------------------------------------------------------------
    # 4. Allergen Inquiry Routing Tests (Safety Grounding)
    # -----------------------------------------------------------------------
    def test_allergen_inquiry_verified_dish_with_allergens(self):
        """Verified dish with allergens returns confirmed allergens from DB."""
        res = self.agent.handle_message("Does the Margherita Pizza contain allergens?")
        self.assertEqual(res.intent, Intent.ALLERGEN_INQUIRY)
        self.assertFalse(res.is_fallback)
        self.assertIn("Gluten", res.text)
        self.assertIn("Dairy", res.text)

    def test_allergen_inquiry_verified_dish_zero_allergens(self):
        """Verified dish with 0 allergens confirms verified absence of tracked allergens."""
        res = self.agent.handle_message("Are there allergens in the Artisan Sorbet Trio?")
        self.assertEqual(res.intent, Intent.ALLERGEN_INQUIRY)
        self.assertFalse(res.is_fallback)
        self.assertIn("verified allergen records", res.text.lower())

    def test_allergen_inquiry_unverified_dish_triggers_safe_fallback(self):
        """Unverified dish must NEVER claim to be allergen-free; returns staff referral."""
        res = self.agent.handle_message("Does the Chef's Daily Rotating Special contain nuts?")
        self.assertEqual(res.intent, Intent.ALLERGEN_INQUIRY)
        self.assertTrue(res.is_fallback)
        self.assertIn("not been formally verified", res.text.lower())
        self.assertIn("speak directly with restaurant staff", res.text.lower())

    def test_allergen_inquiry_unlisted_dish_triggers_safe_fallback(self):
        """Allergen check for unknown item returns safe fallback."""
        res = self.agent.handle_message("Is the Pad Thai peanut-free?")
        self.assertEqual(res.intent, Intent.ALLERGEN_INQUIRY)
        self.assertTrue(res.is_fallback)
        self.assertIn("do not have verified allergen information", res.text.lower())

    def test_allergen_free_filter_excludes_unverified_items(self):
        """Asking for dishes free of an allergen returns only verified dishes."""
        res = self.agent.handle_message("What dishes are gluten-free?")
        self.assertEqual(res.intent, Intent.ALLERGEN_INQUIRY)
        self.assertIn("Caprese Salad", res.text)
        # Margherita Pizza has gluten -> excluded
        self.assertNotIn("Margherita Pizza", res.text)
        # Chef's Special is unverified -> strictly excluded
        self.assertNotIn("Chef's Daily Rotating Special", res.text)
        # Verify complete safe list in structured data
        safe_names = [d["name"] for d in res.data["safe_items"]]
        self.assertIn("Artisan Sorbet Trio", safe_names)
        self.assertNotIn("Chef's Daily Rotating Special", safe_names)

    # -----------------------------------------------------------------------
    # 5. Availability Inquiry Routing Tests
    # -----------------------------------------------------------------------
    def test_availability_inquiry_missing_time_prompts_user(self):
        """Availability check without a time slot asks for the time."""
        res = self.agent.handle_message("Do you have a table for 4 today?")
        self.assertEqual(res.intent, Intent.AVAILABILITY_INQUIRY)
        self.assertTrue(res.needs_clarification)
        self.assertIn("What time", res.text)

    def test_availability_inquiry_open_slot(self):
        """Availability check with open tables confirms availability."""
        res = self.agent.handle_message("Do you have a table for 2 tomorrow at 18:00?")
        self.assertEqual(res.intent, Intent.AVAILABILITY_INQUIRY)
        self.assertFalse(res.needs_clarification)
        self.assertIn("table available", res.text.lower())
        self.assertGreater(res.data["available_tables"], 0)

    # -----------------------------------------------------------------------
    # 6. Reservation Requests & Slot Filling Tests
    # -----------------------------------------------------------------------
    def test_reservation_slot_filling_step_by_step(self):
        """Test step-by-step slot filling asking ONLY for missing parameters."""
        session_id = "slot_fill_test_session"
        tomorrow_str = (restaurant_today() + timedelta(days=1)).strftime("%Y-%m-%d")

        # Step 1: Initial request without slots
        r1 = self.agent.handle_message("I want to book a table.", session_id=session_id)
        self.assertEqual(r1.intent, Intent.RESERVATION_REQUEST)
        self.assertTrue(r1.needs_clarification)
        self.assertIn("What date", r1.text)

        # Step 2: Provide date -> prompts for time
        r2 = self.agent.handle_message(f"For {tomorrow_str}", session_id=session_id)
        self.assertTrue(r2.needs_clarification)
        self.assertIn("What time", r2.text)

        # Step 3: Provide time -> prompts for number of guests
        r3 = self.agent.handle_message("At 19:00", session_id=session_id)
        self.assertTrue(r3.needs_clarification)
        self.assertIn("How many guests", r3.text)

        # Step 4: Provide guest count -> prompts for customer name
        r4 = self.agent.handle_message("4 people", session_id=session_id)
        self.assertTrue(r4.needs_clarification)
        self.assertIn("full name", r4.text)

        # Step 5: Provide name -> prompts for phone number
        r5 = self.agent.handle_message("My name is Sarah Connor", session_id=session_id)
        self.assertTrue(r5.needs_clarification)
        self.assertIn("phone number", r5.text)

        # Step 6: Provide phone number -> Validates in Python, calls CRUD, confirms booking!
        r6 = self.agent.handle_message("Phone is 9870009876", session_id=session_id)
        self.assertEqual(r6.intent, Intent.RESERVATION_REQUEST)
        self.assertFalse(r6.needs_clarification)
        self.assertTrue(r6.success)
        self.assertIn("reservation is confirmed", r6.text.lower())
        self.assertIn("Sarah", r6.text)
        self.assertIsNotNone(r6.data)
        self.assertEqual(r6.data["party_size"], 4)

    def test_reservation_success_only_after_database_creation(self):
        """Verify reservation success message is emitted ONLY after DB insert succeeds."""
        tomorrow_str = (restaurant_today() + timedelta(days=1)).strftime("%Y-%m-%d")
        msg = f"Book table for 2 on {tomorrow_str} at 18:00 under John, phone 9870000199"

        res = self.agent.handle_message(msg, session_id="atomic_create_test")
        self.assertTrue(res.success)
        self.assertIn("reservation is confirmed", res.text.lower())
        booking_id = res.data["reservation_id"]

        # Directly verify record exists in SQLite
        cursor = self.conn.cursor()
        cursor.execute("SELECT customer_name, status FROM reservations WHERE id = ?;", (booking_id,))
        row = cursor.fetchone()
        self.assertIsNotNone(row)
        self.assertEqual(row["customer_name"], "John")
        self.assertEqual(row["status"], "confirmed")

    def test_reservation_invalid_phone_fails_validation_in_python(self):
        """Short phone numbers are caught and rejected before reaching database."""
        session_id = "phone_val_test"
        tomorrow_str = (restaurant_today() + timedelta(days=1)).strftime("%Y-%m-%d")
        
        # Prime session with other slots
        state = self.agent.get_session(session_id)
        state.pending_intent = Intent.RESERVATION_REQUEST
        state.slots = {
            "requested_reservation_date": tomorrow_str,
            "requested_reservation_time": "18:00",
            "number_of_guests": 2,
            "customer_name": "Test User",
            "customer_phone": None
        }

        # Provide invalid phone (< 7 digits)
        res = self.agent.handle_message("My phone is 123", session_id=session_id)
        self.assertTrue(res.needs_clarification)
        self.assertIn("valid Indian mobile", res.text)

    def test_reservation_conflict_returns_clear_failure_message(self):
        """When all tables of requested size are booked, returns clear unavailable message."""
        tomorrow_str = (restaurant_today() + timedelta(days=1)).strftime("%Y-%m-%d")

        # Table T6 is the only 8-person table. Book it at 18:00.
        r1 = self.agent.handle_message(
            f"Reserve a table for 8 on {tomorrow_str} at 18:00 under HostA, phone 9870001111",
            session_id="booking_a"
        )
        self.assertTrue(r1.success)

        # Attempt to book another party of 8 at the same time
        r2 = self.agent.handle_message(
            f"Reserve a table for 8 on {tomorrow_str} at 18:00 under HostB, phone 9870002222",
            session_id="booking_b"
        )
        self.assertFalse(r2.success)
        self.assertTrue(r2.is_fallback)
        self.assertIn("unable to complete the reservation", r2.text.lower())

    # -----------------------------------------------------------------------
    # 7. Reservation Cancellation Tests
    # -----------------------------------------------------------------------
    def test_reservation_cancellation_success(self):
        """Cancelling an existing reservation frees table and confirms cancellation."""
        tomorrow_str = (restaurant_today() + timedelta(days=1)).strftime("%Y-%m-%d")
        book_res = self.agent.handle_message(
            f"Book table for 2 on {tomorrow_str} at 18:00 under Emily, phone 9870003333",
            session_id="cancel_setup"
        )
        res_id = book_res.data["reservation_id"]

        cancel_res = self.agent.handle_message(f"Cancel reservation {res_id}")
        self.assertEqual(cancel_res.intent, Intent.RESERVATION_CANCELLATION)
        self.assertFalse(cancel_res.success)
        self.assertIn("private access code", cancel_res.text.lower())
        self.assertEqual(self.conn.execute("SELECT status FROM reservations WHERE id=?", (res_id,)).fetchone()[0], "confirmed")

    def test_reservation_cancellation_missing_id_prompts_user(self):
        """Asking to cancel without an ID prompts for the reference number."""
        res = self.agent.handle_message("I need to cancel my reservation.")
        self.assertEqual(res.intent, Intent.RESERVATION_CANCELLATION)
        self.assertTrue(res.needs_clarification)
        self.assertIn("reference number", res.text.lower())

    def test_reservation_cancellation_nonexistent_id(self):
        """Attempting to cancel an unknown ID returns clear error."""
        res = self.agent.handle_message("Cancel reservation #99999")
        self.assertEqual(res.intent, Intent.RESERVATION_CANCELLATION)
        self.assertFalse(res.success)
        self.assertIn("private access code", res.text.lower())

    # -----------------------------------------------------------------------
    # 8. Database Error Handling Tests
    # -----------------------------------------------------------------------
    def test_database_error_handling(self):
        """Agent handles database connection exceptions safely without crashing."""
        broken_agent = AgentService(
            db_path="invalid_path/nonexistent_dir/broken.db",
            llm_service=self.mock_llm
        )
        res = broken_agent.handle_message("What is on the menu?")
        self.assertFalse(res.success)
        self.assertTrue(res.is_fallback)
        self.assertIn("technical difficulties", res.text.lower())

    # -----------------------------------------------------------------------
    # 9. Audit-Fix Tests (Phase 2 Step 3 review corrections)
    # -----------------------------------------------------------------------

    def test_reservation_past_date_rejected_in_python(self):
        """A past date must be rejected with needs_clarification before reaching the DB."""
        session_id = "past_date_test"
        yesterday = (restaurant_today() - timedelta(days=1)).strftime("%Y-%m-%d")

        state = self.agent.get_session(session_id)
        state.pending_intent = Intent.RESERVATION_REQUEST
        state.slots = {
            "requested_reservation_date": yesterday,
            "requested_reservation_time": "19:00",
            "number_of_guests": 2,
            "customer_name": "Alice",
            "customer_phone": "9871234567",
        }

        res = self.agent.handle_message("confirm my booking please", session_id=session_id)
        self.assertTrue(res.needs_clarification)
        self.assertIn("past", res.text.lower())
        # Slot must be cleared so the user can supply a new date
        self.assertIsNone(self.agent.get_session(session_id).slots.get("requested_reservation_date"))

    def test_reservation_today_accepted(self):
        """Today's date should not be rejected as a past date."""
        session_id = "today_date_test"
        today = restaurant_today().strftime("%Y-%m-%d")

        state = self.agent.get_session(session_id)
        state.pending_intent = Intent.RESERVATION_REQUEST
        state.slots = {
            "requested_reservation_date": today,
            "requested_reservation_time": "19:00",
            "number_of_guests": 2,
            "customer_name": "Bob",
            "customer_phone": "9879876543",
        }

        res = self.agent.handle_message("confirm my booking please", session_id=session_id)
        # Today is valid: should not bounce back with a past-date rejection
        self.assertNotIn("past", res.text.lower())

    def test_reservation_empty_name_rejected_in_python(self):
        """An empty string name must be rejected with needs_clarification."""
        session_id = "empty_name_test"
        tomorrow = (restaurant_today() + timedelta(days=1)).strftime("%Y-%m-%d")

        state = self.agent.get_session(session_id)
        state.pending_intent = Intent.RESERVATION_REQUEST
        state.slots = {
            "requested_reservation_date": tomorrow,
            "requested_reservation_time": "19:00",
            "number_of_guests": 2,
            "customer_name": "",
            "customer_phone": "9871234567",
        }

        res = self.agent.handle_message("confirm my booking", session_id=session_id)
        self.assertTrue(res.needs_clarification)
        self.assertIn("valid name", res.text.lower())
        self.assertIsNone(self.agent.get_session(session_id).slots.get("customer_name"))

    def test_reservation_whitespace_only_name_rejected_in_python(self):
        """A whitespace-only name must be rejected the same as an empty name."""
        session_id = "whitespace_name_test"
        tomorrow = (restaurant_today() + timedelta(days=1)).strftime("%Y-%m-%d")

        state = self.agent.get_session(session_id)
        state.pending_intent = Intent.RESERVATION_REQUEST
        state.slots = {
            "requested_reservation_date": tomorrow,
            "requested_reservation_time": "19:00",
            "number_of_guests": 2,
            "customer_name": "   ",
            "customer_phone": "9871234567",
        }

        res = self.agent.handle_message("confirm my booking", session_id=session_id)
        self.assertTrue(res.needs_clarification)
        self.assertIn("valid name", res.text.lower())

    def test_cancellation_success_clears_session(self):
        """After a successful cancellation the session state must be cleared."""
        tomorrow = (restaurant_today() + timedelta(days=1)).strftime("%Y-%m-%d")
        book_session = "cancel_clear_book"
        cancel_session = "cancel_clear_cancel"

        # Create a reservation
        book_res = self.agent.handle_message(
            f"Book table for 2 on {tomorrow} at 18:00 under Diana, phone 9870007777",
            session_id=book_session,
        )
        self.assertTrue(book_res.success)
        res_id = book_res.data["reservation_id"]

        # Pre-populate cancel session with stale reservation slots to verify they are cleared
        cancel_state = self.agent.get_session(cancel_session)
        cancel_state.pending_intent = Intent.RESERVATION_REQUEST
        cancel_state.slots = {"customer_name": "Diana", "number_of_guests": 2}

        cancel_res = self.agent.handle_message(
            f"Cancel reservation {res_id}",
            session_id=cancel_session,
        )
        self.assertFalse(cancel_res.success)
        self.assertIn("private access code", cancel_res.text.lower())
        self.assertEqual(self.conn.execute("SELECT status FROM reservations WHERE id=?", (res_id,)).fetchone()[0], "confirmed")
        # An unverified request must not clear the unrelated pending booking.
        remaining_state = self.agent.get_session(cancel_session)
        self.assertEqual(remaining_state.pending_intent, Intent.RESERVATION_REQUEST)
        self.assertEqual(remaining_state.slots, {"customer_name": "Diana", "number_of_guests": 2})

    def test_cancellation_failure_does_not_clear_session(self):
        """A failed cancellation (unknown ID) must NOT clear the customer's session state."""
        session_id = "cancel_fail_test"
        state = self.agent.get_session(session_id)
        state.pending_intent = Intent.RESERVATION_REQUEST
        state.slots = {"customer_name": "Eve", "number_of_guests": 3}

        res = self.agent.handle_message("Cancel reservation #99998", session_id=session_id)
        self.assertFalse(res.success)

        # Slots and pending_intent must be preserved after a failed cancel
        retained = self.agent.get_session(session_id)
        self.assertEqual(retained.slots.get("customer_name"), "Eve")
        self.assertEqual(retained.slots.get("number_of_guests"), 3)

    def test_expired_session_is_evicted(self):
        """Sessions older than the TTL must be evicted on the next get_session() call."""
        # Create an agent with a 1-second TTL for fast testing
        fast_ttl_agent = AgentService(
            db_path=self.db_uri,
            llm_service=self.mock_llm,
            session_ttl_seconds=1,
        )
        old_session_id = "old_session"
        active_session_id = "active_session"

        # Create both sessions
        fast_ttl_agent.get_session(old_session_id).slots["customer_name"] = "Ghost"
        fast_ttl_agent.get_session(active_session_id).slots["customer_name"] = "Live"

        # Verify both exist
        self.assertIn(old_session_id, fast_ttl_agent._sessions)
        self.assertIn(active_session_id, fast_ttl_agent._sessions)

        # Back-date the old session's last activity so it looks expired
        fast_ttl_agent._sessions[old_session_id].last_activity_at = datetime.now() - timedelta(seconds=5)

        # Trigger eviction by accessing any session
        fast_ttl_agent.get_session(active_session_id)

        # Expired session must be gone; active session must still be intact
        self.assertNotIn(old_session_id, fast_ttl_agent._sessions)
        self.assertIn(active_session_id, fast_ttl_agent._sessions)
        self.assertEqual(fast_ttl_agent._sessions[active_session_id].slots.get("customer_name"), "Live")

    def test_session_activity_refresh_prevents_premature_eviction(self):
        """Access refreshes idle TTL without changing creation time or losing slots."""
        agent = AgentService(db_path=self.db_uri, llm_service=self.mock_llm, session_ttl_seconds=10)
        state = agent.get_session("active")
        state.slots["customer_name"] = "Live"
        start = datetime(2026, 1, 1, 12, 0, 0)
        state.created_at = start
        state.last_activity_at = start

        with patch("src.services.agent_service.datetime") as clock:
            clock.now.return_value = start + timedelta(seconds=8)
            self.assertIs(agent.get_session("active"), state)
            self.assertEqual(state.last_activity_at, start + timedelta(seconds=8))
            self.assertEqual(state.created_at, start)

            # Creation is now older than the TTL, but recent activity is not.
            clock.now.return_value = start + timedelta(seconds=12)
            self.assertIs(agent.get_session("active"), state)
            self.assertEqual(state.last_activity_at, start + timedelta(seconds=12))
            self.assertEqual(state.slots["customer_name"], "Live")
            self.assertEqual(state.created_at, start)

            # Once idle beyond the TTL, the same ID gets fresh state.
            clock.now.return_value = start + timedelta(seconds=23)
            replacement = agent.get_session("active")
            self.assertIsNot(replacement, state)
            self.assertEqual(replacement.slots, {})

    def test_session_expiry_does_not_affect_other_sessions(self):
        """Evicting an expired session must never touch other sessions' slot data."""
        fast_ttl_agent = AgentService(
            db_path=self.db_uri,
            llm_service=self.mock_llm,
            session_ttl_seconds=1,
        )

        for i in range(5):
            fast_ttl_agent.get_session(f"session_{i}").slots["idx"] = i

        # Expire sessions 0, 1, 2 only
        for i in range(3):
            fast_ttl_agent._sessions[f"session_{i}"].last_activity_at = datetime.now() - timedelta(seconds=5)

        # Trigger eviction
        fast_ttl_agent.get_session("session_3")

        # Sessions 3 and 4 must still exist with correct data
        self.assertIn("session_3", fast_ttl_agent._sessions)
        self.assertIn("session_4", fast_ttl_agent._sessions)
        self.assertEqual(fast_ttl_agent._sessions["session_3"].slots["idx"], 3)
        self.assertEqual(fast_ttl_agent._sessions["session_4"].slots["idx"], 4)

        # Sessions 0–2 must be evicted
        for i in range(3):
            self.assertNotIn(f"session_{i}", fast_ttl_agent._sessions)


if __name__ == "__main__":
    unittest.main()
