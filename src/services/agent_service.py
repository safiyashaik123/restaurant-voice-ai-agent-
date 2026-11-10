"""Structured Tool Calling and Grounded Agent Router for Restaurant Voice AI Agent.

Architecture:
1. Intent Classification & Entity Extraction (via local Ollama LLM with validated JSON output, and deterministic fallback).
2. Deterministic Tool Routing: Maps validated intents strictly to existing SQLite CRUD functions.
3. Database Grounding & Safety: SQLite is the source of truth; unverified allergens trigger safe fallbacks; no hallucinations.
4. Reservation Slot Filling: Prompts only for missing parameters across conversational turns.
5. Response Generation: Passes factual database results into generate_restaurant_reply() for natural spoken responses.
6. Error & Failure Handling: Catches invalid JSON, database errors, missing tables, and Ollama connection failures safely.
"""

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
import json
import logging
import re
import sqlite3
from typing import Any, Dict, List, Optional, Tuple

from src.core.config import config
from src.database.connection import create_connection
from src.database.crud import (
    cancel_reservation,
    check_allergens_for_item,
    create_reservation,
    find_available_tables,
    find_safe_items_for_allergens,
    get_ingredients_for_item,
    get_menu_categories,
    get_menu_item_by_name,
    get_menu_items,
    get_reservation_by_id,
)
from src.services.llm_service import (
    OllamaError,
    OllamaService,
    default_llm_service,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Supported Customer Intents & Entities
# ---------------------------------------------------------------------------

class Intent:
    GREETING = "greeting"
    MENU_INQUIRY = "menu_inquiry"
    INGREDIENT_INQUIRY = "ingredient_inquiry"
    ALLERGEN_INQUIRY = "allergen_inquiry"
    AVAILABILITY_INQUIRY = "availability_inquiry"
    RESERVATION_REQUEST = "reservation_request"
    RESERVATION_CANCELLATION = "reservation_cancellation"
    UNKNOWN = "unknown"

    ALL_INTENTS = {
        GREETING,
        MENU_INQUIRY,
        INGREDIENT_INQUIRY,
        ALLERGEN_INQUIRY,
        AVAILABILITY_INQUIRY,
        RESERVATION_REQUEST,
        RESERVATION_CANCELLATION,
        UNKNOWN,
    }


@dataclass
class ExtractedEntities:
    """Structured entities extracted from customer speech or text."""
    menu_item_name: Optional[str] = None
    allergen_name: Optional[str] = None
    requested_reservation_date: Optional[str] = None  # YYYY-MM-DD
    requested_reservation_time: Optional[str] = None  # HH:MM (24-hour)
    number_of_guests: Optional[int] = None
    reservation_identifier: Optional[int] = None
    customer_name: Optional[str] = None
    customer_phone: Optional[str] = None


@dataclass
class AgentResult:
    """Standardized result produced by the Grounded Agent Router."""
    text: str
    intent: str
    entities: ExtractedEntities
    data: Optional[Dict[str, Any]] = None
    needs_clarification: bool = False
    missing_slots: List[str] = field(default_factory=list)
    is_fallback: bool = False
    success: bool = True


@dataclass
class InteractionState:
    """In-memory state preserving slot values for the current interaction session."""
    session_id: str
    pending_intent: Optional[str] = None
    slots: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    last_activity_at: datetime = field(default_factory=datetime.now)

    def update_slots(self, entities: ExtractedEntities) -> None:
        """Accumulate new entity values into slot state."""
        if entities.number_of_guests is not None:
            self.slots["number_of_guests"] = entities.number_of_guests
        if entities.requested_reservation_date is not None:
            self.slots["requested_reservation_date"] = entities.requested_reservation_date
        if entities.requested_reservation_time is not None:
            self.slots["requested_reservation_time"] = entities.requested_reservation_time
        if entities.customer_name is not None:
            self.slots["customer_name"] = entities.customer_name
        if entities.customer_phone is not None:
            self.slots["customer_phone"] = entities.customer_phone

    def get_missing_reservation_slots(self) -> List[str]:
        """Return list of mandatory reservation fields not yet provided."""
        required = [
            "requested_reservation_date",
            "requested_reservation_time",
            "number_of_guests",
            "customer_name",
            "customer_phone",
        ]
        return [slot for slot in required if self.slots.get(slot) is None]

    def clear(self) -> None:
        """Reset state upon booking confirmation or cancellation."""
        self.slots.clear()
        self.pending_intent = None


# ---------------------------------------------------------------------------
# Date and Time Normalization Helpers
# ---------------------------------------------------------------------------

def normalize_date_input(date_str: Optional[str]) -> Optional[str]:
    """Normalize user date strings ('today', 'tomorrow', 'YYYY-MM-DD') to YYYY-MM-DD."""
    if not date_str:
        return None
    raw = str(date_str).strip().lower()
    today = date.today()

    if raw in ("today", "tonight", "this evening"):
        return today.strftime("%Y-%m-%d")
    if raw == "tomorrow":
        return (today + timedelta(days=1)).strftime("%Y-%m-%d")

    # Match YYYY-MM-DD
    match_iso = re.search(r"\b(\d{4}-\d{2}-\d{2})\b", raw)
    if match_iso:
        try:
            parsed = datetime.strptime(match_iso.group(1), "%Y-%m-%d").date()
            return parsed.strftime("%Y-%m-%d")
        except ValueError:
            return None

    # Match common human date formats
    for fmt in ("%Y/%m/%d", "%m/%d/%Y", "%d/%m/%Y", "%B %d", "%b %d"):
        try:
            parsed = datetime.strptime(raw, fmt)
            if parsed.year == 1900:
                parsed = parsed.replace(year=today.year)
                if parsed.date() < today:
                    parsed = parsed.replace(year=today.year + 1)
            return parsed.strftime("%Y-%m-%d")
        except ValueError:
            continue

    return None


def normalize_time_input(time_str: Optional[str]) -> Optional[str]:
    """Normalize user time expressions ('7 pm', '19:00', '7:30pm') to HH:MM (24-hour)."""
    if not time_str:
        return None
    raw = str(time_str).strip().lower()

    match = re.search(r"\b(\d{1,2})(?::(\d{2}))?\s*(am|pm)?\b", raw)
    if not match:
        return None

    hour = int(match.group(1))
    minute = int(match.group(2)) if match.group(2) else 0
    meridiem = match.group(3)

    if meridiem == "pm" and hour < 12:
        hour += 12
    elif meridiem == "am" and hour == 12:
        hour = 0

    if 0 <= hour <= 23 and 0 <= minute <= 59:
        return f"{hour:02d}:{minute:02d}"
    return None


# ---------------------------------------------------------------------------
# LLM Intent & Entity Extraction System Prompt
# ---------------------------------------------------------------------------

INTENT_EXTRACTION_PROMPT = """You are an intent and entity extraction parser for a restaurant voice assistant.
Analyze the user message and output ONLY a valid JSON object matching this exact schema:

{
  "intent": "greeting | menu_inquiry | ingredient_inquiry | allergen_inquiry | availability_inquiry | reservation_request | reservation_cancellation | unknown",
  "menu_item_name": "string or null",
  "allergen_name": "string or null",
  "requested_reservation_date": "string or null",
  "requested_reservation_time": "string or null",
  "number_of_guests": number or null,
  "reservation_identifier": number or null,
  "customer_name": "string or null",
  "customer_phone": "string or null"
}

RULES:
- Return ONLY JSON. Do not write markdown blocks, explanations, or code fences.
- Intent must be one of the 8 allowed values. If uncertain, use "unknown".
- If an entity is not explicitly mentioned, use null.
- "number_of_guests" must be an integer or null.
- "reservation_identifier" must be an integer (e.g. reservation #3 -> 3) or null.
"""


# ---------------------------------------------------------------------------
# Grounded Agent Service
# ---------------------------------------------------------------------------

class AgentService:
    """Modular agent service coordinating extraction, database routing, and voice synthesis."""

    # Abandoned sessions older than this are evicted on the next access.
    SESSION_TTL_SECONDS: int = 1800  # 30 minutes

    def __init__(
        self,
        db_path: Optional[str] = None,
        llm_service: Optional[OllamaService] = None,
        session_ttl_seconds: Optional[int] = None,
    ) -> None:
        self.db_path = db_path or str(config.db_path)
        self.llm = llm_service or default_llm_service
        self._sessions: Dict[str, InteractionState] = {}
        self._session_ttl = timedelta(
            seconds=session_ttl_seconds if session_ttl_seconds is not None else self.SESSION_TTL_SECONDS
        )

    def _evict_expired_sessions(self) -> None:
        """Remove sessions that have been idle longer than SESSION_TTL_SECONDS.

        Called lazily on each get_session() call.  Operates only on the
        current AgentService instance; other instances are unaffected.
        """
        cutoff = datetime.now() - self._session_ttl
        expired = [sid for sid, state in self._sessions.items() if state.last_activity_at < cutoff]
        for sid in expired:
            logger.debug("Evicting expired session '%s' (idle since %s)", sid, self._sessions[sid].last_activity_at)
            del self._sessions[sid]

    def get_session(self, session_id: str) -> InteractionState:
        """Get or initialize in-memory interaction state for a session.

        Expired sessions are evicted on each call, including the requested
        session if it has been idle too long. Each session_id remains isolated.
        """
        self._evict_expired_sessions()
        if session_id not in self._sessions:
            self._sessions[session_id] = InteractionState(session_id=session_id)
        else:
            self._sessions[session_id].last_activity_at = datetime.now()
        return self._sessions[session_id]

    def clear_session(self, session_id: str) -> None:
        """Remove a specific session when an interaction concludes."""
        if session_id in self._sessions:
            del self._sessions[session_id]

    # -----------------------------------------------------------------------
    # Step 2: Intent Classification and Entity Extraction
    # -----------------------------------------------------------------------

    def extract_intent_and_entities(
        self,
        user_message: str,
        session: Optional[InteractionState] = None
    ) -> Tuple[str, ExtractedEntities]:
        """Extract structured intent and entities via local LLM with deterministic fallback."""
        try:
            # 1. Ask local Ollama to parse structured JSON
            llm_resp = self.llm.chat(
                messages=[{"role": "user", "content": user_message.strip()}],
                system_prompt=INTENT_EXTRACTION_PROMPT,
                temperature=0.0,
                timeout=10.0
            )
            raw_json = llm_resp.content.strip()

            # Clean any accidental markdown code fences
            if raw_json.startswith("```"):
                raw_json = re.sub(r"^```(?:json)?\s*", "", raw_json)
                raw_json = re.sub(r"\s*```$", "", raw_json)

            data = json.loads(raw_json)

            if not isinstance(data, dict):
                raise ValueError("LLM returned non-dictionary JSON")

            # Validate and clamp intent
            intent = data.get("intent", Intent.UNKNOWN)
            if intent not in Intent.ALL_INTENTS:
                intent = Intent.UNKNOWN

            # Parse integer guest count
            guests = data.get("number_of_guests")
            try:
                guests = int(guests) if guests is not None else None
            except (ValueError, TypeError):
                guests = None

            # Parse integer reservation identifier
            res_id = data.get("reservation_identifier")
            try:
                res_id = int(res_id) if res_id is not None else None
            except (ValueError, TypeError):
                res_id = None

            entities = ExtractedEntities(
                menu_item_name=data.get("menu_item_name") or None,
                allergen_name=data.get("allergen_name") or None,
                requested_reservation_date=normalize_date_input(data.get("requested_reservation_date")),
                requested_reservation_time=normalize_time_input(data.get("requested_reservation_time")),
                number_of_guests=guests,
                reservation_identifier=res_id,
                customer_name=data.get("customer_name") or None,
                customer_phone=data.get("customer_phone") or None,
            )

            # Post-check menu_item_name against database if LLM didn't extract it
            if not entities.menu_item_name:
                try:
                    conn = create_connection(self.db_path)
                    items = get_menu_items(conn, available_only=False)
                    for item in items:
                        if item["name"].lower() in user_message.lower():
                            entities.menu_item_name = item["name"]
                            break
                    conn.close()
                except Exception:
                    pass

            # Availability vs Reservation distinction:
            # If user asks "do you have a table" without explicit booking words, clamp to availability_inquiry
            msg_lower = user_message.lower()
            if intent == Intent.RESERVATION_REQUEST and not any(w in msg_lower for w in ("book", "reserve", "reservation", "booking")):
                if any(w in msg_lower for w in ("do you have a table", "have a table", "table available", "any tables")):
                    intent = Intent.AVAILABILITY_INQUIRY

            # Preserve reservation context across multi-turn slot filling
            if session and session.pending_intent == Intent.RESERVATION_REQUEST and intent in (Intent.UNKNOWN, Intent.GREETING):
                if any([
                    entities.number_of_guests,
                    entities.requested_reservation_date,
                    entities.requested_reservation_time,
                    entities.customer_name,
                    entities.customer_phone,
                ]):
                    intent = Intent.RESERVATION_REQUEST

            return intent, entities

        except Exception as e:
            logger.info(f"Ollama JSON extraction fallback triggered ({e}). Falling back to rule-based parser.")
            return self._extract_rule_based(user_message, session)

    def _extract_rule_based(
        self,
        user_message: str,
        session: Optional[InteractionState] = None
    ) -> Tuple[str, ExtractedEntities]:
        """Deterministic keyword/regex parser used when LLM is unavailable or returns invalid JSON."""
        msg = user_message.lower().strip()
        entities = ExtractedEntities()

        # 1. Extract reservation date
        date_match = re.search(r"\b(\d{4}-\d{2}-\d{2})\b", msg)
        if "today" in msg or "tonight" in msg:
            entities.requested_reservation_date = date.today().strftime("%Y-%m-%d")
        elif "tomorrow" in msg:
            entities.requested_reservation_date = (date.today() + timedelta(days=1)).strftime("%Y-%m-%d")
        elif date_match:
            entities.requested_reservation_date = date_match.group(1)

        # 2. Extract customer phone number (mask out ISO date first to avoid matching YYYY-MM-DD)
        text_without_dates = re.sub(r"\b\d{4}-\d{2}-\d{2}\b", "", user_message)
        phone_explicit = re.search(r"\b(?:phone(?:\s*number)?|number|call me at)\s*(?:is|:)?\s*([+\d -]+)\b", text_without_dates, re.I)
        if phone_explicit:
            entities.customer_phone = phone_explicit.group(1).strip()
        else:
            phone_match = re.search(r"\b(\+?\d[\d -]{6,}\d)\b", text_without_dates)
            if phone_match:
                digits = re.sub(r"\D", "", phone_match.group(1))
                if len(digits) >= 7:
                    entities.customer_phone = phone_match.group(1).strip()

        # 3. Extract number of guests
        party_match = re.search(r"\b(?:table for|party of)\s+(\d{1,2})\b", msg)
        if not party_match:
            party_match = re.search(r"\b(\d{1,2})\s*(?:people|guests|persons|pax)\b", msg)
        if not party_match:
            # "for 4", avoiding years (e.g., 2026) or dates or times
            party_match = re.search(r"\bfor\s+(\d{1,2})\b(?!\d|[-/:]|\s*(?:am|pm))", msg)
        if party_match:
            entities.number_of_guests = int(party_match.group(1))

        # 4. Extract reservation time (avoid matching isolated single digits unless prefixed by 'at')
        time_match = re.search(r"\b(\d{1,2}:\d{2}\s*(?:am|pm)?)\b", msg)
        if not time_match:
            time_match = re.search(r"\b(\d{1,2}\s*(?:am|pm))\b", msg)
        if not time_match:
            time_match = re.search(r"\bat\s+(\d{1,2})\b(?!\d|[-/])", msg)
        if time_match:
            entities.requested_reservation_time = normalize_time_input(time_match.group(1))

        # 5. Extract reservation ID (e.g. "cancel reservation 3", "cancel #4")
        id_match = re.search(r"\b(?:reservation\s*(?:#|number|id)?|booking\s*(?:#|number|id)?|#)\s*(\d+)\b", msg)
        if id_match:
            entities.reservation_identifier = int(id_match.group(1))

        # 6. Extract customer name
        name_match = re.search(r"\b(?:my name is|under the name|name'?s|under)\s+([A-Za-z]+)\b", msg)
        if name_match:
            entities.customer_name = name_match.group(1).capitalize()

        # 7. Extract allergen names
        known_allergens = ["gluten", "dairy", "eggs", "fish", "shellfish", "tree nuts", "peanuts", "peanut", "soy", "sesame", "nuts"]
        for allergen in known_allergens:
            if allergen in msg:
                norm = "Peanuts" if allergen in ("peanut", "peanuts") else ("Tree Nuts" if allergen == "nuts" else allergen.capitalize())
                entities.allergen_name = norm
                break

        # 8. Match menu items from database or extract candidate name
        try:
            conn = create_connection(self.db_path)
            items = get_menu_items(conn, available_only=False)
            for item in items:
                if item["name"].lower() in msg:
                    entities.menu_item_name = item["name"]
                    break
            conn.close()
        except Exception:
            pass

        if not entities.menu_item_name:
            # Pattern-based dish candidate extraction for unlisted dishes
            dish_patterns = [
                r"\b(?:do you have|do you serve|can i get|can i order|is there any)\s+([a-zA-Z\s']+?)(?:\s*(?:on the menu|\?|$))",
                r"\b(?:what is in the|what's in the|ingredients in the)\s+([a-zA-Z\s']+?)(?:\s*(?:made of|\?|$))",
                r"\b(?:does the|in the|about the|is the)\s+([a-zA-Z\s']+?)\s*(?:contain|have|made of|recipe|-free|\?|$)",
            ]
            for pattern in dish_patterns:
                dm = re.search(pattern, user_message, re.IGNORECASE)
                if dm:
                    candidate = dm.group(1).strip()
                    candidate = re.sub(r"^(?:the|a|an)\s+", "", candidate, flags=re.IGNORECASE).strip()
                    if len(candidate) > 2 and candidate.lower() not in ("menu", "table", "reservation"):
                        entities.menu_item_name = candidate.title()
                        break

        # 9. Determine Intent
        is_cancellation = bool(re.search(r"\b(?:cancel|cancelling|cancellation)\s*(?:my|the|a)?\s*(?:reservation|booking|table|#|\d)", msg)) or msg.startswith("cancel")
        if is_cancellation and not any(w in msg for w in ("book", "reserve", "make a reservation")):
            return Intent.RESERVATION_CANCELLATION, entities

        if session and session.pending_intent == Intent.RESERVATION_REQUEST:
            # If user is in an active reservation slot-filling session, keep collecting
            return Intent.RESERVATION_REQUEST, entities

        if any(w in msg for w in ("reserve", "book", "reservation", "booking")):
            return Intent.RESERVATION_REQUEST, entities

        if any(w in msg for w in ("allergy", "allergens", "allergic", "safe to eat", "-free", "dairy-free", "gluten-free", "peanut-free")):
            return Intent.ALLERGEN_INQUIRY, entities

        if entities.allergen_name and any(w in msg for w in ("contain", "have", "with", "in")):
            return Intent.ALLERGEN_INQUIRY, entities

        if any(w in msg for w in ("ingredient", "ingredients", "what is in", "what's in", "recipe")):
            return Intent.INGREDIENT_INQUIRY, entities

        if any(w in msg for w in ("available", "availability", "free table", "open table", "any tables", "have a table", "table for")):
            return Intent.AVAILABILITY_INQUIRY, entities

        if any(w in msg for w in ("menu", "dish", "dishes", "appetizer", "appetizers", "main", "mains", "dessert", "desserts", "beverage", "beverages", "drink", "drinks", "price", "how much")) or entities.menu_item_name is not None:
            return Intent.MENU_INQUIRY, entities

        if any(w in msg for w in ("hello", "hi", "hey", "good morning", "good evening", "hours", "open", "close")):
            return Intent.GREETING, entities

        return Intent.UNKNOWN, entities

    # -----------------------------------------------------------------------
    # Step 3: Deterministic Tool Router & Orchestration
    # -----------------------------------------------------------------------

    def handle_message(
        self,
        user_message: str,
        session_id: str = "default_session"
    ) -> AgentResult:
        """Route customer message deterministically to SQLite CRUD operations and synthesize reply."""
        session = self.get_session(session_id)
        intent, entities = self.extract_intent_and_entities(user_message, session)

        conn = create_connection(self.db_path)
        try:
            if intent == Intent.GREETING:
                return self._route_greeting(user_message, entities)

            elif intent == Intent.MENU_INQUIRY:
                return self._route_menu_inquiry(conn, user_message, entities)

            elif intent == Intent.INGREDIENT_INQUIRY:
                return self._route_ingredient_inquiry(conn, user_message, entities)

            elif intent == Intent.ALLERGEN_INQUIRY:
                return self._route_allergen_inquiry(conn, user_message, entities)

            elif intent == Intent.AVAILABILITY_INQUIRY:
                return self._route_availability_inquiry(conn, user_message, entities)

            elif intent == Intent.RESERVATION_REQUEST:
                return self._route_reservation_request(conn, user_message, entities, session)

            elif intent == Intent.RESERVATION_CANCELLATION:
                return self._route_reservation_cancellation(conn, user_message, entities, session)

            else:
                return self._route_unknown(user_message, entities)

        except Exception as e:
            logger.error(f"Error handling message: {e}", exc_info=True)
            text = (
                "I am currently experiencing technical difficulties accessing restaurant records. "
                "Please speak directly with our team so we can assist you."
            )
            return AgentResult(
                text=text,
                intent=intent,
                entities=entities,
                is_fallback=True,
                success=False
            )
        finally:
            conn.close()

    # -----------------------------------------------------------------------
    # Intent Routers & Handlers
    # -----------------------------------------------------------------------

    def _route_greeting(
        self,
        user_message: str,
        entities: ExtractedEntities
    ) -> AgentResult:
        """Handle greetings and provide restaurant hours."""
        facts = f"Restaurant is open daily from {config.opening_time} to {config.closing_time}."
        default_text = (
            f"Hello! Welcome to our restaurant. We are open daily from {config.opening_time} to {config.closing_time}. "
            "How can I help you today with our menu, allergens, or a table reservation?"
        )
        text = self._synthesize_voice_response(user_message, facts, default_text)
        return AgentResult(text=text, intent=Intent.GREETING, entities=entities)

    def _route_menu_inquiry(
        self,
        conn: sqlite3.Connection,
        user_message: str,
        entities: ExtractedEntities
    ) -> AgentResult:
        """Query menu items, categories, or prices from database."""
        # 1. Specific dish lookup via get_menu_item_by_name
        if entities.menu_item_name:
            item = get_menu_item_by_name(conn, entities.menu_item_name)
            if not item:
                text = (
                    f"I could not find '{entities.menu_item_name}' on our verified menu. "
                    "Our staff would be happy to check if we have any daily specials available."
                )
                return AgentResult(text=text, intent=Intent.MENU_INQUIRY, entities=entities, is_fallback=True)

            facts = (
                f"Dish: {item['name']}\n"
                f"Category: {item['category']}\n"
                f"Price: ${item['price']:.2f}\n"
                f"Description: {item['description']}\n"
                f"Available: {'Yes' if item['is_available'] else 'No'}"
            )
            default_text = (
                f"Our {item['name']} is {item['description']} "
                f"It is available for ${item['price']:.2f}."
            )
            text = self._synthesize_voice_response(user_message, facts, default_text)
            return AgentResult(text=text, intent=Intent.MENU_INQUIRY, entities=entities, data=item)

        # 2. General menu listing via get_menu_items
        items = get_menu_items(conn, available_only=True)
        categories = get_menu_categories(conn)
        sample_dishes = [f"{i['name']} (${i['price']:.2f})" for i in items[:4]]

        facts = (
            f"Categories: {', '.join(categories)}\n"
            f"Sample Dishes: {', '.join(sample_dishes)}"
        )
        default_text = (
            f"Our menu features {', '.join(categories)}. "
            f"Popular dishes include {', '.join(sample_dishes)}. What would you like to hear more about?"
        )
        text = self._synthesize_voice_response(user_message, facts, default_text)
        return AgentResult(text=text, intent=Intent.MENU_INQUIRY, entities=entities, data={"items": items})

    def _route_ingredient_inquiry(
        self,
        conn: sqlite3.Connection,
        user_message: str,
        entities: ExtractedEntities
    ) -> AgentResult:
        """Look up verified dish ingredients directly from SQLite."""
        if not entities.menu_item_name:
            text = "Which menu item would you like to know the ingredients for?"
            return AgentResult(text=text, intent=Intent.INGREDIENT_INQUIRY, entities=entities, needs_clarification=True)

        ingredients = get_ingredients_for_item(conn, entities.menu_item_name)
        if ingredients is None:
            text = (
                f"I could not find '{entities.menu_item_name}' in our database. "
                "Please speak directly with our kitchen staff for ingredient details."
            )
            return AgentResult(text=text, intent=Intent.INGREDIENT_INQUIRY, entities=entities, is_fallback=True)

        facts = f"Item: {entities.menu_item_name}\nIngredients: {', '.join(ingredients)}"
        default_text = f"The {entities.menu_item_name} is made with: {', '.join(ingredients)}."
        text = self._synthesize_voice_response(user_message, facts, default_text)
        return AgentResult(text=text, intent=Intent.INGREDIENT_INQUIRY, entities=entities, data={"ingredients": ingredients})

    def _route_allergen_inquiry(
        self,
        conn: sqlite3.Connection,
        user_message: str,
        entities: ExtractedEntities
    ) -> AgentResult:
        """Verify allergens with strict database grounding and safe fallbacks."""
        # Case A: Allergen inquiry for a specific dish
        if entities.menu_item_name:
            item = get_menu_item_by_name(conn, entities.menu_item_name)
            if item is None:
                text = (
                    f"I do not have verified allergen information for '{entities.menu_item_name}' "
                    "because it is not in our restaurant database. Please speak directly with restaurant staff."
                )
                return AgentResult(text=text, intent=Intent.ALLERGEN_INQUIRY, entities=entities, is_fallback=True)

            # Strict Safety Invariant: unverified items must never be assumed allergen-free
            if item.get("allergen_status") == "unverified":
                text = (
                    f"Allergen information for '{item['name']}' has not been formally verified by our kitchen. "
                    "For your safety, please speak directly with restaurant staff before ordering."
                )
                return AgentResult(text=text, intent=Intent.ALLERGEN_INQUIRY, entities=entities, data=item, is_fallback=True)

            # Verified dish
            allergens = item.get("allergens", [])
            facts = (
                f"Dish: {item['name']}\n"
                f"Allergen Status: verified\n"
                f"Verified Allergens: {', '.join(allergens) if allergens else 'None (free of tracked major allergens)'}"
            )
            if allergens:
                text = f"The {item['name']} contains the following verified allergens: {', '.join(allergens)}."
            else:
                text = f"The {item['name']} has verified allergen records and does not contain tracked major allergens."

            return AgentResult(text=text, intent=Intent.ALLERGEN_INQUIRY, entities=entities, data=item)

        # Case B: Inquiring about safe dishes excluding an allergen
        if entities.allergen_name:
            safe_items = find_safe_items_for_allergens(conn, [entities.allergen_name])
            if not safe_items:
                text = (
                    f"I could not find verified dishes free of {entities.allergen_name}. "
                    "Please speak directly with our chef about customized preparations."
                )
                return AgentResult(text=text, intent=Intent.ALLERGEN_INQUIRY, entities=entities)

            dish_names = [d["name"] for d in safe_items[:4]]
            text = (
                f"For guests avoiding {entities.allergen_name}, our verified safe options include: "
                f"{', '.join(dish_names)}. Would you like details on any of these?"
            )
            return AgentResult(text=text, intent=Intent.ALLERGEN_INQUIRY, entities=entities, data={"safe_items": safe_items})

        text = (
            "We take food allergies very seriously. Please let me know which dish or allergen "
            "you are inquiring about, and I will check our verified records."
        )
        return AgentResult(text=text, intent=Intent.ALLERGEN_INQUIRY, entities=entities)

    def _route_availability_inquiry(
        self,
        conn: sqlite3.Connection,
        user_message: str,
        entities: ExtractedEntities
    ) -> AgentResult:
        """Check table availability via find_available_tables without creating bookings."""
        party_size = entities.number_of_guests or 2
        res_date = entities.requested_reservation_date or date.today().strftime("%Y-%m-%d")
        res_time = entities.requested_reservation_time

        if not res_time:
            text = (
                f"We would be delighted to host your party of {party_size}. "
                "What time were you thinking of joining us?"
            )
            return AgentResult(text=text, intent=Intent.AVAILABILITY_INQUIRY, entities=entities, needs_clarification=True)

        # Python-level input validation
        if party_size <= 0:
            text = "Please specify a party size of at least 1 person."
            return AgentResult(text=text, intent=Intent.AVAILABILITY_INQUIRY, entities=entities, needs_clarification=True)

        tables = find_available_tables(
            conn,
            party_size=party_size,
            reservation_date=res_date,
            reservation_time=res_time
        )

        if tables:
            facts = f"Party Size: {party_size}\nDate: {res_date}\nTime: {res_time}\nAvailable Tables: {len(tables)}"
            default_text = (
                f"Yes, we have a table available for {party_size} guests on {res_date} at {res_time}. "
                "Would you like me to book it for you?"
            )
            text = self._synthesize_voice_response(user_message, facts, default_text)
            return AgentResult(text=text, intent=Intent.AVAILABILITY_INQUIRY, entities=entities, data={"available_tables": len(tables)})
        else:
            facts = f"Party Size: {party_size}\nDate: {res_date}\nTime: {res_time}\nAvailable Tables: 0"
            default_text = (
                f"I am sorry, but we do not have an open table for {party_size} guests on {res_date} at {res_time}. "
                "Would you like to check a different time?"
            )
            text = self._synthesize_voice_response(user_message, facts, default_text)
            return AgentResult(text=text, intent=Intent.AVAILABILITY_INQUIRY, entities=entities, data={"available_tables": 0})

    def _route_reservation_request(
        self,
        conn: sqlite3.Connection,
        user_message: str,
        entities: ExtractedEntities,
        session: InteractionState
    ) -> AgentResult:
        """Handle table booking with slot-filling and deterministic database insertion."""
        session.pending_intent = Intent.RESERVATION_REQUEST
        session.update_slots(entities)
        missing = session.get_missing_reservation_slots()

        # Step 5: Ask ONLY for the missing information
        if "requested_reservation_date" in missing:
            text = "Certainly! What date would you like to reserve for?"
            return AgentResult(text=text, intent=Intent.RESERVATION_REQUEST, entities=entities, needs_clarification=True, missing_slots=missing)

        if "requested_reservation_time" in missing:
            text = f"What time would you like to dine with us on {session.slots['requested_reservation_date']}?"
            return AgentResult(text=text, intent=Intent.RESERVATION_REQUEST, entities=entities, needs_clarification=True, missing_slots=missing)

        if "number_of_guests" in missing:
            text = "How many guests will be joining us in your party?"
            return AgentResult(text=text, intent=Intent.RESERVATION_REQUEST, entities=entities, needs_clarification=True, missing_slots=missing)

        if "customer_name" in missing:
            text = "May I please have the full name for the reservation?"
            return AgentResult(text=text, intent=Intent.RESERVATION_REQUEST, entities=entities, needs_clarification=True, missing_slots=missing)

        if "customer_phone" in missing:
            text = f"Thank you, {session.slots['customer_name']}. What phone number can we reach you at to confirm?"
            return AgentResult(text=text, intent=Intent.RESERVATION_REQUEST, entities=entities, needs_clarification=True, missing_slots=missing)

        # All slots collected -> Validate inputs in Python before calling CRUD
        party_size = session.slots["number_of_guests"]
        res_date = session.slots["requested_reservation_date"]
        res_time = session.slots["requested_reservation_time"]
        name = session.slots["customer_name"].strip()
        phone = session.slots["customer_phone"].strip()

        if party_size <= 0:
            text = "Party size must be at least 1 person. How many guests will be dining?"
            session.slots["number_of_guests"] = None
            return AgentResult(text=text, intent=Intent.RESERVATION_REQUEST, entities=entities, needs_clarification=True)

        if not name:
            text = "Please provide a valid name for the reservation."
            session.slots["customer_name"] = None
            return AgentResult(text=text, intent=Intent.RESERVATION_REQUEST, entities=entities, needs_clarification=True)

        try:
            parsed_date = datetime.strptime(res_date, "%Y-%m-%d").date()
        except ValueError:
            text = "The reservation date appears to be invalid. What date would you like to book?"
            session.slots["requested_reservation_date"] = None
            return AgentResult(text=text, intent=Intent.RESERVATION_REQUEST, entities=entities, needs_clarification=True)

        if parsed_date < date.today():
            text = (
                f"'{res_date}' is in the past. Please provide a future date for your reservation."
            )
            session.slots["requested_reservation_date"] = None
            return AgentResult(text=text, intent=Intent.RESERVATION_REQUEST, entities=entities, needs_clarification=True)

        if len(re.sub(r"[^\d]", "", phone)) < 7:
            text = "Please provide a valid phone number with at least 7 digits."
            session.slots["customer_phone"] = None
            return AgentResult(text=text, intent=Intent.RESERVATION_REQUEST, entities=entities, needs_clarification=True)

        # Execute create_reservation
        try:
            booking = create_reservation(
                conn,
                customer_name=name,
                customer_phone=phone,
                party_size=party_size,
                reservation_date=res_date,
                reservation_time=res_time
            )
            session.clear()

            facts = (
                f"Reservation Confirmed: Yes\n"
                f"Booking ID: {booking['reservation_id']}\n"
                f"Table: {booking['table_number']}\n"
                f"Party Size: {booking['party_size']}\n"
                f"Date: {booking['reservation_date']}\n"
                f"Time: {booking['reservation_time']}\n"
                f"Guest Name: {booking['customer_name']}"
            )
            default_text = (
                f"Your reservation is confirmed! Booking reference number {booking['reservation_id']} "
                f"for {booking['party_size']} guests on {booking['reservation_date']} at {booking['reservation_time']} "
                f"under the name {booking['customer_name']}."
            )
            text = self._synthesize_voice_response(user_message, facts, default_text)
            return AgentResult(text=text, intent=Intent.RESERVATION_REQUEST, entities=entities, data=booking, success=True)

        except ValueError as e:
            # Table unavailable or outside hours
            error_msg = str(e)
            text = f"I was unable to complete the reservation: {error_msg} Would you like to try a different time or date?"
            return AgentResult(text=text, intent=Intent.RESERVATION_REQUEST, entities=entities, is_fallback=True, success=False)

    def _route_reservation_cancellation(
        self,
        conn: sqlite3.Connection,
        user_message: str,
        entities: ExtractedEntities,
        session: InteractionState
    ) -> AgentResult:
        """Handle reservation cancellations using existing cancel_reservation CRUD."""
        res_id = entities.reservation_identifier

        if not res_id:
            text = "To cancel your booking, could you please provide your reservation reference number?"
            return AgentResult(text=text, intent=Intent.RESERVATION_CANCELLATION, entities=entities, needs_clarification=True)

        existing = get_reservation_by_id(conn, res_id)
        if not existing:
            text = f"I could not find an active reservation with reference number {res_id}. Please check the number and try again."
            return AgentResult(text=text, intent=Intent.RESERVATION_CANCELLATION, entities=entities, is_fallback=True, success=False)

        cancelled = cancel_reservation(conn, res_id)
        if cancelled:
            facts = f"Reservation {res_id} status: cancelled"
            default_text = f"Reservation number {res_id} has been successfully cancelled. We hope to see you another time!"
            text = self._synthesize_voice_response(user_message, facts, default_text)
            session.clear()  # Fix: clear session state after confirmed cancellation
            return AgentResult(text=text, intent=Intent.RESERVATION_CANCELLATION, entities=entities, data={"reservation_id": res_id}, success=True)
        else:
            text = f"Reservation number {res_id} could not be cancelled. It may have already been cancelled."
            return AgentResult(text=text, intent=Intent.RESERVATION_CANCELLATION, entities=entities, is_fallback=True, success=False)

    def _route_unknown(
        self,
        user_message: str,
        entities: ExtractedEntities
    ) -> AgentResult:
        """Handle unrecognized or out-of-domain input."""
        text = (
            "I am here to assist you with our menu, verified allergens, table availability, and reservations. "
            "How can I help you today?"
        )
        return AgentResult(text=text, intent=Intent.UNKNOWN, entities=entities, is_fallback=True)

    # -----------------------------------------------------------------------
    # Step 6: Factual Response Synthesis
    # -----------------------------------------------------------------------

    def _synthesize_voice_response(
        self,
        user_message: str,
        factual_context: str,
        default_fallback_text: str
    ) -> str:
        """Generate conversational voice reply using local LLM grounded on facts, with deterministic fallback."""
        try:
            resp = self.llm.generate_restaurant_reply(
                user_message=user_message,
                context_facts=factual_context,
                timeout=12.0
            )
            if resp and resp.content and resp.content.strip():
                return resp.content.strip()
        except OllamaError as e:
            logger.info(f"Ollama unavailable during voice synthesis ({e}). Using deterministic factual template.")
        except Exception as e:
            logger.warning(f"Unexpected error during synthesis ({e}). Using deterministic template.")

        return default_fallback_text


# Singleton instance
default_agent_service = AgentService()
