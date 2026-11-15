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
from datetime import datetime, timedelta
import json
import logging
import re
import sqlite3
from time import perf_counter
from typing import Any, Dict, List, Optional, Tuple

from src.core.config import config
from src.core.reservation_policy import restaurant_today, validate_customer_name, validate_indian_mobile, validate_reservation_slot
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


# Allergen aliases are deliberately limited to names with an unambiguous mapping.
ALLERGEN_PATTERNS = [
    (r"\btree nuts?\b", "Tree Nuts"), (r"\bpeanuts?\b", "Peanuts"),
    (r"\bshellfish\b", "Shellfish"), (r"\bfish\b", "Fish"),
    (r"\bgluten\b", "Gluten"), (r"\b(?:dairy|milk)\b", "Dairy"),
    (r"\beggs?\b", "Eggs"), (r"\bsoy\b", "Soy"), (r"\bsesame\b", "Sesame"),
]


def _format_menu_price(price: float) -> str:
    """Format a menu price in Indian rupees without changing its value."""
    return "₹" + f"{price:.2f}".rstrip("0").rstrip(".")


def _ambiguous_allergen_request(message: str) -> bool:
    text = re.sub(r"^\s*(?:and|or)\b", "", message, flags=re.I)
    matched = {name for pattern, name in ALLERGEN_PATTERNS if re.search(pattern, text, re.I)}
    free_terms = re.findall(r"\b([a-z]+)[ -]free\b", text, re.I)
    return len(matched) > 1 or len(free_terms) > 1 or bool(re.search(r"\b(?:and|or)\b|[,/]", text, re.I))


CATEGORY_ALIASES = {"appetizer": "appetizers", "main": "mains", "dessert": "desserts",
                    "beverage": "beverages", "drink": "beverages", "drinks": "beverages"}


def _requested_menu_categories(message: str, categories: List[str]) -> List[str]:
    """Match configured category phrases and common aliases without partial matches."""
    aliases = CATEGORY_ALIASES
    matches = []
    for category in categories:
        phrases = [category] + [alias for alias, target in aliases.items() if target == category.lower()]
        for phrase in phrases:
            pattern = r"(?<!\w)" + re.escape(phrase).replace(r"\ ", r"\s+") + r"(?!\w)"
            matches.extend((match.start(), match.end(), category) for match in re.finditer(pattern, message, re.I))
    # A configured phrase such as "Hot Drinks" takes precedence over the
    # overlapping "drinks" alias, but independently mentioned categories remain.
    selected = []
    for start, end, category in sorted(matches, key=lambda match: -(match[1] - match[0])):
        if not any(start < other_end and end > other_start for other_start, other_end, _ in selected):
            selected.append((start, end, category))
    return [category for category in categories if any(match[2] == category for match in selected)]


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
    last_menu_item: Optional[str] = None
    awaiting_menu_intent: Optional[str] = None
    has_menu_context: bool = False
    menu_context_count: int = 0
    awaiting_reservation_slot: Optional[str] = None
    awaiting_cancellation_reference: bool = False

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
        self.last_menu_item = None
        self.awaiting_menu_intent = None
        self.has_menu_context = False
        self.menu_context_count = 0
        self.awaiting_reservation_slot = None
        self.awaiting_cancellation_reference = False


# ---------------------------------------------------------------------------
# Date and Time Normalization Helpers
# ---------------------------------------------------------------------------

MONTH_DATE_PATTERN = (
    r"\b(?:january|jan|february|feb|march|mar|april|apr|may|june|jun|"
    r"july|jul|august|aug|september|sep|october|oct|november|nov|december|dec)"
    r"\s+\d{1,2}\b(?:\s*,?\s*\d{4}\b|(?!\s*,?\s*\d))"
)
EXPLICIT_DATE_PATTERN = rf"\b\d{{4}}-\d{{2}}-\d{{2}}\b|\b\d{{1,4}}/\d{{1,2}}/\d{{1,4}}\b|{MONTH_DATE_PATTERN}"


def _identifier_is_phone(identifier: Optional[int], phone: Optional[str]) -> bool:
    """Do not expose a captured phone number as a reservation reference."""
    if identifier is None or not phone:
        return False
    digits = re.sub(r"\D", "", phone)
    return (str(identifier) == digits
            or (phone.startswith("+91") and str(identifier) == digits[2:])
            or (len(digits) == 10 and str(identifier) == "91" + digits))


def normalize_date_input(date_str: Optional[str]) -> Optional[str]:
    """Normalize user date strings ('today', 'tomorrow', 'YYYY-MM-DD') to YYYY-MM-DD."""
    if not date_str:
        return None
    raw = str(date_str).strip().lower()
    today = restaurant_today()

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

    # Explicit years must be preserved; yearless dates use the next occurrence.
    for fmt in ("%Y/%m/%d", "%m/%d/%Y", "%d/%m/%Y",
                "%B %d %Y", "%B %d, %Y", "%b %d %Y", "%b %d, %Y",
                "%B %d", "%b %d"):
        try:
            parsed = datetime.strptime(raw, fmt)
            if fmt in ("%B %d", "%b %d"):
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
                customer_phone=None,
            )

            # Preserve explicitly stated dates even if the model omits or alters the year.
            # Never trust a model to normalize a phone number: extract its raw text.
            _, raw_entities = self._extract_rule_based(user_message, session)
            if re.search(EXPLICIT_DATE_PATTERN, user_message, re.I):
                entities.requested_reservation_date = raw_entities.requested_reservation_date
            elif not entities.requested_reservation_date:
                entities.requested_reservation_date = raw_entities.requested_reservation_date
            entities.customer_phone = raw_entities.customer_phone
            # The model may mistake a phone number for a booking reference.
            if intent == Intent.RESERVATION_REQUEST or (session and session.pending_intent == Intent.RESERVATION_REQUEST):
                entities.reservation_identifier = raw_entities.reservation_identifier
            elif _identifier_is_phone(entities.reservation_identifier, entities.customer_phone):
                entities.reservation_identifier = raw_entities.reservation_identifier

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

        # 1. Extract and validate reservation dates in natural-language messages.
        date_match = re.search(EXPLICIT_DATE_PATTERN, msg)
        relative = re.search(r"\b(?:today|tonight|tomorrow)\b", msg)
        if date_match:
            entities.requested_reservation_date = normalize_date_input(date_match.group())
        elif relative:
            entities.requested_reservation_date = normalize_date_input(relative.group())

        # 2. Distinguish explicit booking references from customer phone numbers.
        text_without_dates = re.sub(r"\b\d{4}-\d{2}-\d{2}\b", "", user_message)
        id_match = re.search(
            r"(?<!\w)(?:(?:reservation|booking)\s*(?:(?:reference\s*)?(?:number|id)|reference|#)?\s*(?:is\s*)?|#)\s*(\d+)\b(?![-./]\d)",
            text_without_dates, re.I,
        )
        phone_explicit = re.search(
            r"\b(?:phone(?:\s*number)?|(?<!reservation )(?<!booking )(?<!reference )number|call me at)\s*(?:is|:)?\s*([^\s,;]+(?:[ -]+[0-9][^\s,;]*)*)",
            text_without_dates, re.I,
        )
        if phone_explicit:
            entities.customer_phone = phone_explicit.group(1).strip()
        else:
            # A bare booking reference, even a long one, is not a phone number.
            text_without_reference = (text_without_dates[:id_match.start(1)] + ' ' + text_without_dates[id_match.end(1):]) if id_match else text_without_dates
            phone_match = re.search(r"(?<![\w+()./-])([+]?[0-9][0-9+(). -]{6,}[0-9])(?![\w+()./-])", text_without_reference)
            if phone_match:
                entities.customer_phone = phone_match.group(1).strip()

        # Interpret spoken individual digits only while explicitly collecting a phone.
        # Do not guess at ambiguous words or silently correct an invalid number.
        if (session and session.pending_intent == Intent.RESERVATION_REQUEST
                and session.awaiting_reservation_slot == 'customer_phone'):
            spoken = re.sub(r'^\s*(?:my\s+)?(?:phone(?:\s+number)?|number)\s*(?:is|:)?\s*', '', msg)
            words = re.findall(r"[a-z]+|[0-9]+|\+", spoken)
            digit_words = {'zero': '0', 'oh': '0', 'one': '1', 'two': '2', 'three': '3', 'four': '4', 'five': '5', 'six': '6', 'seven': '7', 'eight': '8', 'nine': '9'}
            if (re.fullmatch(r'[a-z0-9+\s,.-]+', spoken) and words
                    and all(word in digit_words or (word.isascii() and word.isdigit()) or word == '+' for word in words)):
                candidate = ''.join(digit_words.get(word, word) for word in words)
                try:
                    entities.customer_phone = validate_indian_mobile(candidate)
                except ValueError:
                    pass

        # 3. Extract number of guests
        party_match = re.search(r"\b(?:table for|party of)\s+(\d{1,2})\b", msg)
        if not party_match:
            party_match = re.search(r"\b(\d{1,2})\s*(?:people|guests|persons|pax)\b", msg)
        if not party_match:
            # "for 4", avoiding years (e.g., 2026) or dates or times
            party_match = re.search(r"\bfor\s+(\d{1,2})\b(?!\d|[-/:]|\s*(?:am|pm))", msg)
        if party_match:
            entities.number_of_guests = int(party_match.group(1))
        else:
            word_party = re.search(r"\b(?:for|party of)\s+(one|two|three|four|five|six|seven|eight|nine|ten)\b", msg)
            if word_party:
                entities.number_of_guests = {'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
                                             'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10}[word_party.group(1)]

        if (session and session.pending_intent == Intent.RESERVATION_REQUEST
                and session.awaiting_reservation_slot == 'number_of_guests'
                and not entities.customer_phone and not entities.number_of_guests):
            answer = re.fullmatch(r"\s*(?:for\s+)?(\d{1,3}|one|two|three|four|five|six|seven|eight|nine|ten)(?:\s+(?:people|guests|persons))?[.!?]?\s*", msg)
            if answer:
                word = answer.group(1)
                numbers = {'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5, 'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10}
                entities.number_of_guests = int(word) if word.isdigit() else numbers[word]

        # 4. Extract reservation time (avoid matching isolated single digits unless prefixed by 'at')
        time_match = re.search(r"\b(\d{1,2}:\d{2}\s*(?:am|pm)?)\b", msg)
        if not time_match:
            time_match = re.search(r"\b(\d{1,2}\s*(?:am|pm))\b", msg)
        if not time_match:
            time_match = re.search(r"\bat\s+(\d{1,2})\b(?!\d|[-/])", msg)
        if time_match:
            entities.requested_reservation_time = normalize_time_input(time_match.group(1))

        if (not entities.requested_reservation_time and session
                and session.pending_intent == Intent.RESERVATION_REQUEST
                and session.awaiting_reservation_slot == 'requested_reservation_time'):
            spoken_time = re.fullmatch(r"\s*(?:at\s+)?(one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\s*(a\.?m\.?|p\.?m\.?)\s*[!?]?\s*", msg)
            if spoken_time:
                hours = {'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5, 'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10, 'eleven': 11, 'twelve': 12}
                entities.requested_reservation_time = normalize_time_input(str(hours[spoken_time.group(1)]) + spoken_time.group(2).replace('.', ''))

        # 5. Retain an explicit reference only when it is not the captured phone.
        if id_match:
            identifier = int(id_match.group(1))
            if not _identifier_is_phone(identifier, entities.customer_phone):
                entities.reservation_identifier = identifier

        # 6. Extract customer name
        name_match = re.search(
            r"\b(?:my name is|under the name|name'?s|under)\s+"
            r"([A-Za-z]+(?:[ '-][A-Za-z]+)*?)(?=\s+(?:phone|number|for|on|at)\b|[,;.!?]|$)",
            user_message, re.I,
        )
        if name_match:
            entities.customer_name = name_match.group(1).strip()
        elif (session and session.pending_intent == Intent.RESERVATION_REQUEST
              and session.awaiting_reservation_slot == 'customer_name'
              and session.get_missing_reservation_slots()[:1] == ['customer_name']
              and re.fullmatch(r"[A-Za-z]+(?:[ '-][A-Za-z]+)*", user_message.strip())
              and msg not in {'hello', 'hi', 'hey', 'yes', 'no', 'cancel'}):
            entities.customer_name = user_message.strip()

        # 7. Recognize tracked allergen aliases without substring collisions (fish/shellfish).
        for pattern, allergen in ALLERGEN_PATTERNS:
            if re.search(pattern, msg):
                entities.allergen_name = allergen
                break
        if not entities.allergen_name and re.search(r"\bnuts?\b", msg):
            entities.allergen_name = "nuts"
        if not entities.allergen_name:
            unknown = re.search(r"\b(?:contains?|has|allergic to|allergy to)\s+([a-z]+)(?:-free)?\b|\b([a-z]+)[ -]free\b", msg)
            if not unknown:
                unknown = re.search(r"\bdoes\b.*?\bhave\s+([a-z]+)\b|\b(?:free (?:of|from)|without|avoid(?:ing)?)\s+([a-z]+)\b", msg)
            if unknown:
                candidate = unknown.group(1) or (unknown.group(2) if unknown.lastindex and unknown.lastindex > 1 else None)
                if candidate not in {"allergens", "allergen", "ingredients", "ingredient", "a", "any", "table"}:
                    entities.allergen_name = candidate

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

        # Do not silently turn an unlisted longer dish name into a known substring.
        if entities.menu_item_name:
            clause = re.search(r"\b(?:does|is)\s+(?:the\s+)?([a-zA-Z][a-zA-Z\s']+?)\s+(?:contain|have|made of|made with)\b", user_message, re.I)
            if not clause:
                clause = re.search(r"\bingredients?(?:\s+are)?\s+(?:in|of|for)\s+(?:the\s+)?([a-zA-Z][a-zA-Z\s']+?)(?:\s*\?|$)", user_message, re.I)
            if clause:
                candidate = re.sub(r"\s+please$", "", clause.group(1).strip(), flags=re.I)
                known = entities.menu_item_name.lower()
                if candidate.lower().startswith(known + " "):
                    entities.menu_item_name = candidate.title()

        if not entities.menu_item_name:
            # Pattern-based dish candidate extraction for unlisted dishes
            dish_patterns = [
                r"\b(?:does|is)\s+(?:the\s+)?([a-zA-Z][a-zA-Z\s']+?)\s+(?:contain|have|made of|made with)\b",
                r"\b(?:ingredients?(?:\s+are)?\s+(?:in|of|for)|what(?: is|'s) in)\s+(?:the\s+)?([a-zA-Z][a-zA-Z\s']+?)(?:\s*\?|$)",
                r"\b(?:do you have|do you serve|can i get|can i order|is there any)\s+([a-zA-Z\s']+?)(?:\s*(?:on the menu|\?|$))",
                r"\b(?:what is in the|what's in the|ingredients in the)\s+([a-zA-Z\s']+?)(?:\s*(?:made of|\?|$))",
                r"\b(?:does the|in the|about the|is the)\s+([a-zA-Z\s']+?)\s*(?:contain|have|made of|recipe|-free|\?|$)",
            ]
            for pattern in dish_patterns:
                dm = re.search(pattern, user_message, re.IGNORECASE)
                if dm:
                    candidate = dm.group(1).strip()
                    candidate = re.sub(r"^(?:the|a|an)\s+", "", candidate, flags=re.IGNORECASE).strip()
                    if len(candidate) > 2 and candidate.lower() not in ("menu", "table", "reservation", "it", "this", "that", "there"):
                        entities.menu_item_name = candidate.title()
                        break

        # 9. Determine Intent
        is_cancellation = bool(re.search(r"\b(?:cancel|cancelling|cancellation)\s*(?:my|the|a)?\s*(?:reservation|booking|table|#|\d)", msg)) or msg.startswith("cancel")
        if is_cancellation and not re.search(r"\b(?:book|reserve|make a reservation)\b", msg):
            return Intent.RESERVATION_CANCELLATION, entities

        if session and session.pending_intent == Intent.RESERVATION_REQUEST:
            # If user is in an active reservation slot-filling session, keep collecting
            return Intent.RESERVATION_REQUEST, entities

        if any(w in msg for w in ("reserve", "book", "reservation", "booking")):
            return Intent.RESERVATION_REQUEST, entities

        if any(w in msg for w in ("allergy", "allergen", "allergic", "safe to eat", "-free", "dairy-free", "gluten-free", "peanut-free", "free of", "free from", "without", "avoiding")) or re.search(r"\b[a-z]+ free\b", msg):
            return Intent.ALLERGEN_INQUIRY, entities

        if entities.allergen_name and any(w in msg for w in ("contain", "have", "with", " in ", "has")):
            return Intent.ALLERGEN_INQUIRY, entities

        if any(w in msg for w in ("ingredient", "ingredients", "what is in", "what's in", "recipe", "made of", "made with")):
            return Intent.INGREDIENT_INQUIRY, entities

        if (re.search(r"\b(?:appetizers?|mains?|desserts?|beverages?|drinks?)\b", msg)
                and not re.search(r"\b(?:table|tables|reservation|booking)\b", msg)):
            return Intent.MENU_INQUIRY, entities

        if any(w in msg for w in ("available", "availability", "free table", "open table", "any tables", "have a table", "table for")):
            return Intent.AVAILABILITY_INQUIRY, entities

        if any(w in msg for w in ("menu", "dish", "dishes", "appetizer", "appetizers", "main", "mains", "dessert", "desserts", "beverage", "beverages", "drink", "drinks", "price", "how much", "vegetarian", "vegan")) or entities.menu_item_name is not None:
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
        # An explicit request for a new booking must not inherit a previous
        # incomplete booking's customer or party details.
        if (session.pending_intent == Intent.RESERVATION_REQUEST
                and re.search(r'\b(?:book|reserve|make|start)\b.*\b(?:another|new)\s+(?:reservation|booking|table)\b', user_message, re.I)
                and not re.search(r'\bcancel\b', user_message, re.I)):
            session.clear()
        if session.awaiting_cancellation_reference and re.fullmatch(
            r"\s*(?:never\s*mind|don['’]t\s+cancel|do\s+not\s+cancel|"
            r"keep\s+my\s+reservation|continue\s+my\s+booking)\s*[.!?]*\s*",
            user_message, re.I,
        ):
            session.awaiting_cancellation_reference = False
            if session.pending_intent == Intent.RESERVATION_REQUEST:
                missing = session.get_missing_reservation_slots()
                session.awaiting_reservation_slot = missing[0] if missing else None
                prompts = {
                    "requested_reservation_date": "What date would you like to reserve for?",
                    "requested_reservation_time": "What time would you like to dine?",
                    "number_of_guests": "How many guests will be joining you?",
                    "customer_name": "May I have the full name for the reservation?",
                    "customer_phone": "What phone number can we reach you at?",
                }
                text = "I won't cancel your booking. " + (prompts[missing[0]] if missing else "Your reservation details are still available.")
                return AgentResult(text=text, intent=Intent.RESERVATION_REQUEST,
                                   entities=ExtractedEntities(), needs_clarification=bool(missing), missing_slots=missing)
            return AgentResult(text="I won't cancel your booking.", intent=Intent.RESERVATION_CANCELLATION,
                               entities=ExtractedEntities())
        classification_started = perf_counter()
        # Explicit menu questions can be answered from SQLite without an LLM round trip.
        # Keep the model for ambiguous requests and reservation slot filling.
        rule_intent, rule_entities = self._extract_rule_based(user_message, session)
        configured_category_query = False
        if (session.pending_intent != Intent.RESERVATION_REQUEST
                and rule_intent in (Intent.UNKNOWN, Intent.AVAILABILITY_INQUIRY, Intent.MENU_INQUIRY)
                and not re.search(r"\b(?:table|tables|reservation|booking)\b", user_message, re.I)):
            try:
                category_conn = create_connection(self.db_path)
                try:
                    categories = get_menu_categories(category_conn)
                    configured_category_query = bool(_requested_menu_categories(user_message, categories))
                    if configured_category_query and rule_entities.menu_item_name:
                        terms = re.split(r"\s*(?:,|\band\b|\bor\b)\s*", rule_entities.menu_item_name.lower())
                        category_terms = {name.lower() for name in categories} | set(CATEGORY_ALIASES)
                        if all(term.strip() in category_terms for term in terms):
                            rule_entities.menu_item_name = None
                finally:
                    category_conn.close()
            except sqlite3.Error:
                pass
            if configured_category_query:
                rule_intent = Intent.MENU_INQUIRY
        ordinal_reference = bool(re.search(r"\b(?:first|second|third|last|\d+(?:st|nd|rd|th))\s+(?:one|dish|item)\b", user_message, re.I))
        reference = bool(re.search(r"\b(?:it|its|that one|this one|those|same dish)\b", user_message, re.I)) or ordinal_reference
        menu_followup = bool(reference and (session.has_menu_context or session.awaiting_menu_intent
                             or re.search(r"\b(?:what about it|tell me more about it)\b", user_message, re.I)
                             or ordinal_reference))
        allergen_followup = bool(session.last_menu_item and rule_entities.allergen_name
                                 and re.match(r"\s*(?:and|what about|how about)\b", user_message, re.I))
        if session.pending_intent != Intent.RESERVATION_REQUEST and allergen_followup:
            rule_intent = Intent.ALLERGEN_INQUIRY
        # A short answer to "which item?" can itself be an unlisted dish.
        if (session.awaiting_menu_intent and not menu_followup and not rule_entities.menu_item_name
                and rule_intent == Intent.UNKNOWN and re.fullmatch(r"[A-Za-z][A-Za-z '-]{1,80}[?.!]?", user_message.strip())):
            rule_entities.menu_item_name = user_message.strip().rstrip("?.!")
            rule_intent = session.awaiting_menu_intent
        price_comparison = bool(re.search(r"\b(?:cheapest|lowest[ -](?:priced|price|cost)|least expensive|most expensive|highest[ -](?:priced|price|cost))\b", user_message, re.I))
        menu_cues = re.search(
            r"\b(?:menu|dish(?:es)?|appetizers?|mains?|desserts?|beverages?|drinks?|"
            r"ingredients?|recipe|allergens?|allerg(?:y|ic)|vegetarian|vegan|"
            r"contain(?:s)?|made (?:of|with)|what is in|what's in|do you (?:have|serve)|how much|price|without|avoiding|free of|free from)\b|\b[a-z]+[ -]free\b", user_message, re.I)
        if session.awaiting_cancellation_reference:
            # A bare reference is not a phone number or a reservation slot.
            bare_reference = re.fullmatch(r"\s*#?\s*([0-9]{1,18})\s*[?.!]?\s*", user_message)
            reference_id = int(bare_reference.group(1)) if bare_reference else rule_entities.reservation_identifier
            intent = Intent.RESERVATION_CANCELLATION
            entities = ExtractedEntities(reservation_identifier=reference_id)
        elif session.pending_intent == Intent.RESERVATION_REQUEST:
            # Explicit cancellation must remain available during slot filling.
            intent, entities = rule_intent, rule_entities
        elif menu_followup:
            intent = rule_intent if rule_intent in (Intent.INGREDIENT_INQUIRY, Intent.ALLERGEN_INQUIRY) else Intent.MENU_INQUIRY
            entities = rule_entities
        elif (
            (rule_intent == Intent.ALLERGEN_INQUIRY and allergen_followup)
            or configured_category_query
            or price_comparison
            or (menu_cues and rule_intent in (Intent.MENU_INQUIRY, Intent.INGREDIENT_INQUIRY, Intent.ALLERGEN_INQUIRY))
            or (session.awaiting_menu_intent and rule_entities.menu_item_name)
        ):
            intent, entities = rule_intent, rule_entities
            if price_comparison:
                intent = Intent.MENU_INQUIRY
        else:
            intent, entities = self.extract_intent_and_entities(user_message, session)

        logger.debug("Intent classification and extraction: %.1f ms", (perf_counter() - classification_started) * 1000)
        database_started = perf_counter()
        conn = create_connection(self.db_path)
        try:
            menu_intents = (Intent.MENU_INQUIRY, Intent.INGREDIENT_INQUIRY, Intent.ALLERGEN_INQUIRY)
            if price_comparison and session.pending_intent != Intent.RESERVATION_REQUEST and not session.awaiting_cancellation_reference:
                categories = _requested_menu_categories(user_message, get_menu_categories(conn))
                if len(categories) > 1:
                    return AgentResult(text="Which one category should I compare?", intent=Intent.MENU_INQUIRY, entities=entities, needs_clarification=True)
                items = get_menu_items(conn, category_name=categories[0] if categories else None, available_only=True)
                if not items:
                    return AgentResult(text="There are no available items to compare.", intent=Intent.MENU_INQUIRY, entities=entities)
                highest = bool(re.search(r'\b(?:most expensive|highest)\b', user_message, re.I))
                price = (max if highest else min)(item['price'] for item in items)
                matches = [item for item in items if item['price'] == price]
                session.last_menu_item = matches[0]['name'] if len(matches) == 1 else None
                session.has_menu_context = True
                session.menu_context_count = len(matches)
                session.awaiting_menu_intent = None
                names = ', '.join(item['name'] for item in matches)
                label = 'highest-priced' if highest else 'cheapest'
                return AgentResult(text=f"The {label} {'items are' if len(matches) > 1 else 'item is'} {names} at {_format_menu_price(price)}.", intent=Intent.MENU_INQUIRY, entities=entities, data={'items': matches})
            if menu_followup and session.pending_intent != Intent.RESERVATION_REQUEST:
                if entities.menu_item_name and not get_menu_item_by_name(conn, entities.menu_item_name):
                    entities.menu_item_name = None
                if ordinal_reference or (not entities.menu_item_name and not session.last_menu_item):
                    session.awaiting_menu_intent = intent
                    return AgentResult(text="Which dish do you mean? Please tell me its name.",
                                       intent=intent, entities=entities, needs_clarification=True)
                if not entities.menu_item_name:
                    entities.menu_item_name = session.last_menu_item
            if session.awaiting_menu_intent and entities.menu_item_name and intent == Intent.MENU_INQUIRY:
                answer = re.fullmatch(r"(?:the\s+|what about\s+)?" + re.escape(entities.menu_item_name) + r"(?:\s+please)?[?.!]*", user_message.strip(), re.I)
                if answer:
                    intent = session.awaiting_menu_intent
            elif (intent in menu_intents
                  and not entities.menu_item_name and session.last_menu_item
                  and (allergen_followup or re.search(r"\b(?:it|its|that|this|those|same dish|what about)\b", user_message, re.I))):
                entities.menu_item_name = session.last_menu_item

            if intent in menu_intents:
                if entities.menu_item_name:
                    item = get_menu_item_by_name(conn, entities.menu_item_name)
                    session.last_menu_item = item["name"] if item else None
                    session.has_menu_context = bool(item)
                    session.menu_context_count = 1 if item else 0
                    session.awaiting_menu_intent = None
                elif (intent == Intent.INGREDIENT_INQUIRY or (intent == Intent.ALLERGEN_INQUIRY and not entities.allergen_name)
                      or (intent == Intent.MENU_INQUIRY and re.search(r"\b(?:it|its|that|this)\b", user_message, re.I))):
                    session.awaiting_menu_intent = intent
                    if intent == Intent.MENU_INQUIRY:
                        return AgentResult(text="Which dish would you like the price for?", intent=intent, entities=entities, needs_clarification=True)
                else:
                    session.awaiting_menu_intent = None
                if (intent == Intent.MENU_INQUIRY and re.search(r"\b(?:vegetarian|vegan)\b", user_message, re.I)
                        and (not entities.menu_item_name or re.search(r"\bis\s+(?:the\s+)?[^?]+\s+(?:vegetarian|vegan)\b", user_message, re.I))):
                    label = "vegan" if re.search(r"\bvegan\b", user_message, re.I) else "vegetarian"
                    text = f"Our database does not have verified {label} classifications. Please ask restaurant staff before ordering."
                    return AgentResult(text=text, intent=Intent.MENU_INQUIRY, entities=entities, is_fallback=True)
            else:
                session.awaiting_menu_intent = None
            if intent == Intent.GREETING:
                return self._route_greeting(user_message, entities)

            elif intent == Intent.MENU_INQUIRY:
                result = self._route_menu_inquiry(conn, user_message, entities)
                if not entities.menu_item_name and result.data and isinstance(result.data.get('items'), list):
                    items = result.data['items']
                    session.last_menu_item = items[0]['name'] if len(items) == 1 else None
                    session.has_menu_context = bool(items)
                    session.menu_context_count = len(items)
                return result

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
            logger.debug("Agent database routing: %.1f ms", (perf_counter() - database_started) * 1000)

    # -----------------------------------------------------------------------
    # Intent Routers & Handlers
    # -----------------------------------------------------------------------

    def _route_greeting(
        self,
        user_message: str,
        entities: ExtractedEntities
    ) -> AgentResult:
        """Handle greetings and provide restaurant hours."""
        text = (
            f"Hello! Welcome to our restaurant. We are open daily from {config.opening_time} to {config.closing_time}. "
            "How can I help you today with our menu, allergens, or a table reservation?"
        )
        return AgentResult(text=text, intent=Intent.GREETING, entities=entities)

    def _route_menu_inquiry(
        self,
        conn: sqlite3.Connection,
        user_message: str,
        entities: ExtractedEntities
    ) -> AgentResult:
        """Query menu items, categories, or prices from database."""
        # Category names are not unknown dishes, even in "do you have desserts?".
        if entities.menu_item_name:
            category_words = {name.lower() for name in get_menu_categories(conn)} | {"appetizer", "main", "dessert", "beverage", "drink", "drinks"}
            if entities.menu_item_name.lower() in category_words:
                entities.menu_item_name = None

        # 1. Specific dish lookup via get_menu_item_by_name
        if entities.menu_item_name:
            item = get_menu_item_by_name(conn, entities.menu_item_name)
            if not item:
                text = (
                    f"I could not find '{entities.menu_item_name}' on our verified menu. "
                    "Our staff would be happy to check if we have any daily specials available."
                )
                return AgentResult(text=text, intent=Intent.MENU_INQUIRY, entities=entities, is_fallback=True)

            availability = "available" if item["is_available"] else "currently unavailable"
            description = f" {item['description']}" if item["description"] else ""
            text = f"Our {item['name']} is {availability} at {_format_menu_price(item['price'])}.{description}"
            return AgentResult(text=text, intent=Intent.MENU_INQUIRY, entities=entities, data=item)

        categories = get_menu_categories(conn)
        # Match full configured names as well as common singular aliases.
        requested = _requested_menu_categories(user_message, categories)
        if requested:
            items = [item for category in requested for item in get_menu_items(conn, category_name=category, available_only=True)]
            names = ", ".join(f"{item['name']} ({_format_menu_price(item['price'])})" for item in items)
            label = ' and '.join(requested)
            text = f"Our {label} include: {names}." if names else f"There are no currently available items in {label}."
            data = {"items": items, "category": requested[0]} if len(requested) == 1 else {"items": items, "categories": requested}
            return AgentResult(text=text, intent=Intent.MENU_INQUIRY, entities=entities, data=data)
        items = get_menu_items(conn, available_only=True)

        sample_dishes = [f"{item['name']} ({_format_menu_price(item['price'])})" for item in items[:4]]
        text = f"Our menu features {', '.join(categories)}."
        if sample_dishes:
            text += f" Available dishes include {', '.join(sample_dishes)}."
        text += " What would you like to hear more about?"
        return AgentResult(text=text, intent=Intent.MENU_INQUIRY, entities=entities, data={"categories": categories, "items": items})

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

        if not ingredients:
            text = f"I do not have ingredient records for {entities.menu_item_name}. Please ask restaurant staff."
            return AgentResult(text=text, intent=Intent.INGREDIENT_INQUIRY, entities=entities, is_fallback=True, data={"ingredients": []})
        text = f"The recorded ingredients for {entities.menu_item_name} are: {', '.join(ingredients)}."
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

            allergens = item.get("allergens", [])
            if entities.allergen_name:
                tracked = {row["name"].lower(): row["name"] for row in conn.execute("SELECT name FROM allergens")}
                requested = tracked.get(entities.allergen_name.strip().lower())
                if not requested or _ambiguous_allergen_request(user_message):
                    text = "I cannot verify that allergen request from our records. Please specify one tracked allergen and check with restaurant staff."
                    return AgentResult(text=text, intent=Intent.ALLERGEN_INQUIRY, entities=entities, data=item, is_fallback=True, needs_clarification=True)
                present = requested.lower() in {name.lower() for name in allergens}
                status = f"contains {requested}" if present else f"does not list {requested}"
                text = f"Verified allergen records say {item['name']} {status}. Please check with restaurant staff about cross-contact before ordering."
            elif allergens:
                text = f"The {item['name']} contains the following verified allergens: {', '.join(allergens)}. Please check with staff about cross-contact."
            else:
                text = f"The {item['name']} has verified allergen records and does not contain tracked major allergens. Please check with staff about cross-contact."
            return AgentResult(text=text, intent=Intent.ALLERGEN_INQUIRY, entities=entities, data=item)

        # Case B: Inquiring about safe dishes excluding an allergen
        if entities.allergen_name:
            if entities.allergen_name.lower() == 'nuts':
                return AgentResult(text="Do you mean peanuts or tree nuts? Please check with restaurant staff about allergies and cross-contact.",
                                   intent=Intent.ALLERGEN_INQUIRY, entities=entities, is_fallback=True, needs_clarification=True)
            tracked = {row["name"].lower() for row in conn.execute("SELECT name FROM allergens")}
            if entities.allergen_name.strip().lower() not in tracked or _ambiguous_allergen_request(user_message):
                text = "I cannot verify that allergen request from our records. Please specify one tracked allergen and check with restaurant staff."
                return AgentResult(text=text, intent=Intent.ALLERGEN_INQUIRY, entities=entities, is_fallback=True, needs_clarification=True)
            safe_items = find_safe_items_for_allergens(conn, [entities.allergen_name])
            if not safe_items:
                text = (
                    f"I could not find verified dishes free of {entities.allergen_name}. "
                    "Please speak directly with our chef about customized preparations."
                )
                return AgentResult(text=text, intent=Intent.ALLERGEN_INQUIRY, entities=entities)

            dish_names = [d["name"] for d in safe_items[:4]]
            text = (
                f"For guests avoiding {entities.allergen_name}, verified records do not list it in: "
                f"{', '.join(dish_names)}. Please check with staff about cross-contact before ordering."
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
        party_size = entities.number_of_guests if entities.number_of_guests is not None else 2
        res_date = entities.requested_reservation_date or restaurant_today().strftime("%Y-%m-%d")
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

        try:
            validate_reservation_slot(res_date, res_time)
        except ValueError as exc:
            return AgentResult(text=str(exc), intent=Intent.AVAILABILITY_INQUIRY,
                               entities=entities, needs_clarification=True)

        tables = find_available_tables(
            conn, party_size=party_size, reservation_date=res_date,
            reservation_time=res_time,
        )
        # Do not let a language model contradict the database result.
        if tables:
            text = (f"Yes, we have a table available for {party_size} guests on {res_date} at {res_time}. "
                    "Would you like me to book it for you?")
        else:
            text = (f"I am sorry, but we do not have an open table for {party_size} guests on {res_date} at {res_time}. "
                    "Would you like to check a different time?")
        return AgentResult(text=text, intent=Intent.AVAILABILITY_INQUIRY, entities=entities,
                           data={"available_tables": len(tables)})

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
        invalid_name = False
        if session.slots.get('customer_name') is not None:
            try:
                session.slots['customer_name'] = validate_customer_name(session.slots['customer_name'])
            except ValueError:
                session.slots['customer_name'] = None
                invalid_name = True
        missing = session.get_missing_reservation_slots()
        session.awaiting_reservation_slot = missing[0] if missing else None

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
            session.awaiting_reservation_slot = "customer_name"
            text = "Please provide a valid name for the reservation." if invalid_name else "May I please have the full name for the reservation?"
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
            session.awaiting_reservation_slot = "customer_name"
            return AgentResult(text=text, intent=Intent.RESERVATION_REQUEST, entities=entities, needs_clarification=True)

        try:
            parsed_date = datetime.strptime(res_date, "%Y-%m-%d").date()
        except ValueError:
            text = "The reservation date appears to be invalid. What date would you like to book?"
            session.slots["requested_reservation_date"] = None
            return AgentResult(text=text, intent=Intent.RESERVATION_REQUEST, entities=entities, needs_clarification=True)

        if parsed_date < restaurant_today():
            text = (
                f"'{res_date}' is in the past. Please provide a future date for your reservation."
            )
            session.slots["requested_reservation_date"] = None
            return AgentResult(text=text, intent=Intent.RESERVATION_REQUEST, entities=entities, needs_clarification=True)

        try:
            validate_indian_mobile(phone)
        except ValueError as exc:
            session.slots["customer_phone"] = None
            return AgentResult(text=str(exc), intent=Intent.RESERVATION_REQUEST, entities=entities, needs_clarification=True)

        try:
            validate_reservation_slot(res_date, res_time)
        except ValueError as exc:
            session.slots["requested_reservation_time"] = None
            return AgentResult(text=str(exc), intent=Intent.RESERVATION_REQUEST, entities=entities, needs_clarification=True)

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

            # Confirmation must reflect the committed booking, not an unconstrained LLM reply.
            guests = "guest" if booking['party_size'] == 1 else "guests"
            text = (
                f"Your reservation is confirmed! Booking reference number {booking['reservation_id']} "
                f"for {booking['party_size']} {guests} on {booking['reservation_date']} at {booking['reservation_time']} "
                f"under the name {booking['customer_name']}. Your table is {booking['table_number']}. "
                f"Save your private access code {booking['access_code']} to view or cancel online."
            )
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
        session.awaiting_reservation_slot = None
        session.awaiting_cancellation_reference = True
        res_id = entities.reservation_identifier

        if not res_id:
            text = "To cancel your booking, could you please provide your reservation reference number?"
            return AgentResult(text=text, intent=Intent.RESERVATION_CANCELLATION, entities=entities, needs_clarification=True)

        # A sequential reference is not proof of ownership. Do not disclose
        # existence or mutate data through chat/voice without the access code.
        return AgentResult(
            text="For your privacy, use your booking reference and private access code on the Reservations page to review and confirm cancellation, or contact the restaurant.",
            intent=Intent.RESERVATION_CANCELLATION, entities=entities,
            needs_clarification=True, success=False,
        )

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
