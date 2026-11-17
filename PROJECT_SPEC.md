# PROJECT_SPEC.md — Restaurant Voice AI Agent

## 1. Project Overview & Scope

The **Restaurant Voice AI Agent** is an interactive, voice-driven restaurant assistant designed strictly around core restaurant operations:

1. **Customer Voice Interface:** A modern React + TypeScript web interface capturing customer speech via microphone, displaying live transcripts and status, and playing synthesized voice responses.

2. **Speech Understanding (STT):** Convert customer speech into text using local `faster-whisper`.

3. **Menu Inquiries:** Query and present dishes, descriptions, and prices from the local SQLite database.

4. **Ingredient & Allergen Verification:** Provide verified ingredient and allergen facts retrieved directly from the database.

5. **Table Availability:** Check table capacity and existing bookings.

6. **Reservation Management:** Validate and store customer bookings deterministically in SQLite, with lookup and cancellation via private access codes.

7. **Speech Generation (TTS):** Convert assistant responses into natural spoken audio using local Piper TTS.

**Out-of-scope features (e.g., payment gateways, external POS integrations, SMS notifications, complex web analytics) will not be introduced.**

---

## 2. Technology Stack & Constraints

### 2.1 Core Technologies

- **Frontend Framework:** React 18+ with TypeScript, Vite, and CSS

- **Frontend Tooling:** Node.js v24.11.1 (pre-installed on system, no reinstallation required)

- **Backend Language:** Python 3.10+

- **API Framework:** FastAPI (with Uvicorn)

- **Local Large Language Model (LLM):** Ollama running `qwen2.5:1.5b` (verified present on local disk, no re-download required)

- **Database:** SQLite 3 (relational schema with foreign keys and WAL mode)

- **Speech-to-Text (STT):** `faster-whisper` (CTranslate2-backed Whisper on CPU, `tiny.en` default or `base.en` opt-in, with `int8` quantization)

- **Text-to-Speech (TTS):** Piper TTS (local ONNX voice model `en_US-lessac-medium` running on CPU)

- **LLM Client:** Python standard library `urllib.request` communicating with the Ollama `/api/chat` endpoint; no third-party SDK or cloud API keys required

### 2.2 Strict Constraints & Operating Principles

1. **100% Free & Local:** Zero paid APIs, zero subscriptions, zero cloud LLM services, and no API keys required.

2. **Windows Environment:** Developed and verified on a Windows host with 8 GB system RAM and CPU-based inference (Intel integrated graphics, no CUDA).

3. **Database-Grounded Responses & Safe Fallbacks:**

   - The LLM is never relied upon to "remember" or invent recipe ingredients or allergen flags.

   - All ingredient and allergen answers are retrieved directly from SQLite records.

   - If a menu item or allergen record is absent or ambiguous, the system triggers a deterministic, safe fallback message (e.g., **"I do not have verified allergen information for that item. Please speak with our restaurant staff before ordering."**).

   - Dishes with `allergen_status = 'unverified'` are never claimed to be allergen-free and are excluded from safe recommendations.

4. **Deterministic Business Logic for Bookings:**

   - The LLM does not write to the database or compute calendar availability.

   - The LLM acts purely as an entity extractor to collect booking intent (party size, date, time slot, customer name, phone).

   - Backend Python logic deterministically checks table capacity, operating hours, and booking overlaps before confirming any reservation.

5. **Configurable Reservation Duration:**

   - The conflict detection window is fully configurable via `DEFAULT_RESERVATION_DURATION_MINUTES` (default 90 minutes) rather than hardcoded, allowing the restaurant to adjust seating duration.

6. **System Resource Observations:**

   - Preliminary resource observations have been made on the 8 GB RAM / Intel integrated graphics Windows host during development and manual testing. These are not systematic benchmarks.

7. **Empirical Benchmarking (Pending):**

   - Comprehensive full-stack peak memory, STT latency, LLM inference latency, and TTS synthesis latency benchmarks have not yet been measured. No performance figures are claimed in this document.

---

## 3. System Architecture & Components

```

+-----------------------------------------------------------------------+

|                    Frontend Layer (React + TS + Vite)                 |

|  - Customer Voice Interface (Microphone Capture / Audio Playback)     |

|  - Real-time Status & Conversation Transcript Display                 |

|  - Menu browsing, reservation form, lookup & cancellation             |

|  - Direct REST / Audio upload communication with FastAPI backend      |

+-----------------------------------------------------------------------+

                                   │ Audio / JSON

                                   ▼

+-----------------------------------------------------------------------+

|                          FastAPI Backend                              |

|  +--------------------+  +--------------------+  +-----------------+  |

|  |     STT Engine     |  |  Agent Controller  |  |   TTS Engine    |  |

|  |  (faster-whisper)  |  |  - Intent Extractor|  |   (Piper TTS)   |  |

|  |  (int8 CPU engine) |  |  - Grounded Router |  | (ONNX medium)   |  |

|  +--------------------+  +--------------------+  +-----------------+  |

|                                  │                                    |

|                                  ▼                                    |

|                   +-------------------------------+                   |

|                   |   Local LLM (Ollama Client)   |                   |

|                   |   qwen2.5:1.5b (local)        |                   |

|                   +-------------------------------+                   |

|                                  │                                    |

|                                  ▼                                    |

|                   +-------------------------------+                   |

|                   |  Database & Business Logic    |                   |

|                   |  - Menu & Allergen Service    |                   |

|                   |  - Configurable Booking Logic |                   |

|                   |  - SQLite (WAL mode)          |                   |

|                   +-------------------------------+                   |

+-----------------------------------------------------------------------+

```

---

## 4. Database Schema (SQLite)

The database uses `PRAGMA foreign_keys = ON` and WAL journal mode. All tables are created with `CREATE TABLE IF NOT EXISTS`. Performance indexes are defined on frequently queried columns.

### 4.1 `categories`

- `id` (INTEGER, PRIMARY KEY AUTOINCREMENT)

- `name` (TEXT, UNIQUE, NOT NULL, COLLATE NOCASE)

### 4.2 `menu_items`

- `id` (INTEGER, PRIMARY KEY AUTOINCREMENT)

- `category_id` (INTEGER, NOT NULL, FOREIGN KEY → `categories.id` ON DELETE RESTRICT)

- `name` (TEXT, UNIQUE, NOT NULL, COLLATE NOCASE)

- `description` (TEXT)

- `price` (REAL, NOT NULL, CHECK (price >= 0))

- `is_available` (INTEGER, NOT NULL, DEFAULT 1, CHECK (is_available IN (0, 1)))

- `allergen_status` (TEXT, NOT NULL, DEFAULT 'unverified', CHECK (allergen_status IN ('verified', 'unverified')))

  - `'verified'`: kitchen/supplier allergen audit completed; authoritative allergen data present.

  - `'unverified'`: allergen profile pending review; must **never** be assumed allergen-free.

### 4.3 `allergens`

- `id` (INTEGER, PRIMARY KEY AUTOINCREMENT)

- `name` (TEXT, UNIQUE, NOT NULL, COLLATE NOCASE)

### 4.4 `menu_item_ingredients`

- `id` (INTEGER, PRIMARY KEY AUTOINCREMENT)

- `menu_item_id` (INTEGER, NOT NULL, FOREIGN KEY → `menu_items.id` ON DELETE CASCADE)

- `ingredient_name` (TEXT, NOT NULL, COLLATE NOCASE)

### 4.5 `menu_item_allergens`

- `menu_item_id` (INTEGER, NOT NULL, FOREIGN KEY → `menu_items.id` ON DELETE CASCADE)

- `allergen_id` (INTEGER, NOT NULL, FOREIGN KEY → `allergens.id` ON DELETE CASCADE)

- PRIMARY KEY (`menu_item_id`, `allergen_id`)

### 4.6 `tables`

- `id` (INTEGER, PRIMARY KEY AUTOINCREMENT)

- `table_number` (TEXT, UNIQUE, NOT NULL)

- `capacity` (INTEGER, NOT NULL, CHECK (capacity > 0))

- `is_active` (INTEGER, NOT NULL, DEFAULT 1, CHECK (is_active IN (0, 1)))

### 4.7 `reservations`

- `id` (INTEGER, PRIMARY KEY AUTOINCREMENT)

- `table_id` (INTEGER, NOT NULL, FOREIGN KEY → `tables.id` ON DELETE RESTRICT)

- `customer_name` (TEXT, NOT NULL)

- `customer_phone` (TEXT, NOT NULL)

- `party_size` (INTEGER, NOT NULL, CHECK (party_size > 0))

- `reservation_date` (TEXT, NOT NULL) — format: `YYYY-MM-DD`

- `reservation_time` (TEXT, NOT NULL) — format: `HH:MM` (24-hour)

- `duration_minutes` (INTEGER, NOT NULL, DEFAULT 90, CHECK (duration_minutes > 0)) — stored per booking

- `status` (TEXT, NOT NULL, DEFAULT 'confirmed', CHECK (status IN ('confirmed', 'cancelled')))

- `created_at` (TEXT, NOT NULL, DEFAULT (datetime('now', 'localtime')))

### 4.8 `reservation_access`

- `reservation_id` (INTEGER, PRIMARY KEY, FOREIGN KEY → `reservations.id` ON DELETE CASCADE)

- `secret_hash` (TEXT, NOT NULL) — SHA-256 digest of the one-time access code

This table is created lazily on first booking and allows secure reservation lookup and cancellation without exposing the access code. Older reservations created before this table was introduced cannot be retrieved online.

---

## 5. Functional Requirements

### 5.1 Customer Frontend Interface (React + TypeScript)

Implemented as a single-page application with four main components: `Assistant`, `Menu`, `Reservations`, and `Sidebar`.

- Responsive, intuitive customer interface styled with CSS.

- **Sidebar navigation:** switching between Assistant, Menu, and Reservations pages.

- **Status indicator:** live visual states — idle, listening, processing, speaking.

- **Assistant page:** chat transcript showing user queries and grounded agent responses; microphone recording control; immediate playback of Piper TTS synthesized responses as WAV audio; text chat fallback via keyboard.

- **Menu page:** browsable menu grid with category filtering; displays dish name, category, description, price (in Indian rupees ₹), availability status, and allergen verification status per item.

- **Reservations page:** date picker that fetches available slots dynamically from `GET /api/reservations/slots`; booking form collecting guest name, Indian mobile number, party size, date, and time slot; booking confirmation display including allocated table number and private access code; reservation lookup form (reference number + access code); reservation cancellation with explicit confirm step.

- Per-tab session UUID is generated with `crypto.randomUUID()` and reused across chat and voice turns within that tab.

### 5.2 Speech Understanding (STT)

- Ingest audio input from the frontend client via multipart upload.

- Inspect audio with PyAV before decoding; reject files with no audio stream.

- Decode audio once with faster-whisper's decoder to 16 kHz mono float PCM.

- Reject files with no decodable samples (HTTP 400) or digitally all-zero audio (HTTP 422) before loading the model.

- Transcribe with `faster-whisper`, explicitly requesting English transcription (`language="en"`, `task="transcribe"`).

- Configuration: `WHISPER_MODEL_SIZE` (default `tiny.en`; `base.en` opt-in), `WHISPER_DEVICE` (default `cpu`), `WHISPER_COMPUTE_TYPE` (default `int8`), `WHISPER_BEAM_SIZE` (default 5), `WHISPER_VAD_FILTER` (default false).

- The model is loaded lazily on first valid request and reused for subsequent requests; startup and health checks do not load or download it.

- Only locally available model files are used; the service never downloads a model at runtime.

- Empty or unintelligible transcriptions return HTTP 422 with an appropriate error.

### 5.3 Menu, Ingredient & Allergen Inquiries

- Query menu items by name, category, or full menu listing from SQLite.

- Provide factual ingredient breakdowns strictly from `menu_item_ingredients`.

- Allergen responses distinguish three states:

  1. **Allergens present** (`allergen_status = 'verified'` and one or more allergens are mapped): confirms which allergens are recorded for the dish.

  2. **No allergens recorded** (`allergen_status = 'verified'` and the allergen list is empty): states that no allergens at all are recorded in the verified database for that dish and advises confirming ingredients and cross-contact risk with staff. This does not prove the dish is free from any specific allergen — only that none are recorded.

  3. **Unverified item** (`allergen_status = 'unverified'`): explicitly states allergen information has not been formally verified by the kitchen and advises speaking with staff.

- `find_safe_items_for_allergens()` exclusively returns available items with `allergen_status = 'verified'` that have no mapping for the excluded allergen; unverified items are strictly excluded.

- `find_items_containing_allergen()` returns available verified items that contain a specified allergen; unverified items are excluded.

- Items absent from the database return `None`, triggering a safe fallback rather than a guessed response.

- The database contains no verified vegetarian or vegan classification; the assistant honestly directs guests to staff for dietary classification questions.

- Popularity or sales data is not stored; the assistant honestly acknowledges this rather than guessing.

- Cross-contact risk cannot be verified from the database; guests are advised to confirm with staff before ordering.

### 5.4 Reservation Management

**Validation and policy:**

- Operating hours default to `10:00`–`23:00` (Asia/Kolkata IST), configurable via `RESTAURANT_OPENING_TIME` and `RESTAURANT_CLOSING_TIME`.

- Reservation start times are spaced in 30-minute intervals from opening time.

- Each slot's end time must fall at or before closing time (default 90-minute duration means last start is `21:30`).

- All date and time interpretation uses Asia/Kolkata (IST), independent of server or browser timezone.

- Slots must be in the future; past slots cannot be booked.

- An optional `RESERVATION_MAX_ADVANCE_DAYS` limits how far ahead a booking can be made.

- Indian mobile numbers are validated and normalized by `validate_indian_mobile()`. The validator strips leading and trailing whitespace, then checks that the input contains only digits and common formatting characters (spaces, hyphens, parentheses, periods, and an optional leading `+`). It detects and preserves an optional `+91` country code prefix, strips all remaining formatting characters from the subscriber number, and validates that the resulting 10 digits match the pattern `[6-9][0-9]{9}`. Other prefixes, letters, or symbols are rejected.

- Customer names must contain at least one letter and may include letters, spaces, apostrophes, hyphens, and periods; maximum 200 characters.

**Deterministic table allocation:**

- Tables are sorted by `capacity ASC, table_number ASC` (best-fit allocation).

- Only active tables (`is_active = 1`) with capacity >= party size are candidates.

- Overlap detection compares requested slot against confirmed bookings using each booking's stored `duration_minutes`.

- Exact end-to-start boundary is available for the next booking (no forced buffer between bookings).

- Duplicate booking prevention: a matching phone + name + date + time + party size combination is rejected.

**Transaction integrity:**

- `create_reservation()` uses `BEGIN IMMEDIATE` when no transaction is active; callers that own a transaction must commit after confirmation.

- Failed bookings are rolled back atomically.

**Access codes and lookup:**

- Each new booking generates a cryptographically random access code (`secrets.token_urlsafe(24)`); only its SHA-256 digest is stored.

- Reservation lookup and cancellation both require the reservation ID and matching access code (constant-time comparison via `hmac.compare_digest`).

- Incorrect code and nonexistent reference return the same response to prevent enumeration.

- Secure cancellation is performed through the **Reservations page** workflow using the booking's reference number and private access code. The voice and chat agent acknowledges cancellation intent but does not perform secure cancellation by reference alone; customers must use the Reservations page to complete a cancellation.

**Multi-turn slot filling (voice/chat path):**

- The agent collects missing reservation parameters across conversational turns: date, time, party size, customer name, phone.

- Required slots: `requested_reservation_date`, `requested_reservation_time`, `number_of_guests`, `customer_name`, `customer_phone`.

- Booking is only attempted when all five slots are populated and validated.

- In-memory session state stores slot values across turns and expires after 30 minutes of inactivity.

### 5.5 Speech Generation (TTS)

- Synthesize assistant responses to WAV audio using Piper TTS on CPU.

- Voice model: `en_US-lessac-medium` (ONNX format, locally stored in `models/piper/`).

- Model directory configured via `PIPER_MODEL_DIR`; voice name via `PIPER_MODEL_NAME`.

- The voice is loaded lazily on first synthesis request and reused; model files are never downloaded at runtime.

- Output is written atomically via a temporary file, replaced in place only on success.

- Empty or whitespace-only text is rejected before synthesis.

- Both synthesized audio (WAV) and text are returned to the frontend.

---

## 6. API Endpoints

The FastAPI application exposes the following endpoints under the `/api` prefix. Configurable CORS origins default to `http://localhost:5173,http://127.0.0.1:5173`. Run one API process worker; session state is in-process memory and disappears on restart.

### `GET /api/health`

Returns `{"status": "ok", "database": "ok"}`. Returns HTTP 503 if the database file or required tables are missing. Does not load STT or TTS models.

### `POST /api/chat`

**Request:** JSON `{"session_id": "\<UUID>", "message": "\<text>"}` (message 1–4000 characters, non-blank).

**Response:** JSON with `session_id`, `text` (agent reply), `intent`, `entities` (all entity fields), `data`, `success`, `is_fallback`, `needs_clarification`, `missing_slots`.

### `POST /api/voice`

**Request:** Multipart form with `session_id` (UUID) and `file` (audio upload). Supported content types include `audio/wav`, `audio/mpeg`, `audio/mp4`, `audio/ogg`, `audio/webm`, `audio/flac`, `application/octet-stream`.

**Response:** WAV audio body. Response headers `X-Transcript`, `X-Response-Text`, and `X-Session-ID` are exposed to allowed browser origins; transcript and response text are UTF-8 percent-encoded. `Cache-Control: no-store` is set.

Error codes: 400 (empty/missing file), 413 (upload too large; limit configurable via `API_MAX_UPLOAD_BYTES`, default 10 MiB), 415 (unsupported audio type), 422 (invalid or unintelligible audio), 503 (STT or TTS unavailable).

### `GET /api/menu`

Optional query parameter: `category` (max 100 characters). Returns `{"categories": [...], "items": [...]}`. Items include `id`, `category`, `name`, `description`, `price`, `is_available`, `allergen_status`. The categories list is always complete; the items list is filtered to available items in the requested category if provided.

### `POST /api/reservations`

**Request:** JSON with `customer_name`, `customer_phone`, `party_size`, `reservation_date` (YYYY-MM-DD), `reservation_time` (HH:MM).

**Response (HTTP 201):** JSON with `access_code`, `reservation_id`, `table_id`, `table_number`, `table_capacity`, `customer_name`, `customer_phone`, `party_size`, `reservation_date`, `reservation_time`, `duration_minutes`, `status`, `created_at`.

Error codes: 409 (no table available, or duplicate reservation), 422 (invalid phone, invalid time slot, or other validation failure).

### `GET /api/reservations/slots`

**Query parameters:** `date` (YYYY-MM-DD, required), `party_size` (integer >= 1, optional). Returns `{"date": "...", "slots": ["HH:MM", ...]}` listing permissible start times in the restaurant's IST timezone. If `party_size` is provided, slots are filtered by table availability. Past dates return an empty list. Invalid dates return HTTP 422. The backend re-validates on booking; an elapsed slot cannot be confirmed.

### `POST /api/reservations/lookup`

**Request:** JSON with `reservation_id` (integer > 0) and `access_code` (1–128 characters).

**Response:** Reservation details if the access code matches: `reservation_id`, `reservation_date`, `reservation_time`, `party_size`, `status`, `table_number`.

Error: HTTP 404 if reference or access code is incorrect. Per-process rate limiting: 20 attempts per IP per 60-second window.

### `POST /api/reservations/cancel`

**Request:** JSON with `reservation_id`, `access_code`, and `confirm: true`.

**Response:** Same as lookup on success. The reservation row is retained with `status = 'cancelled'`.

Error: HTTP 422 if `confirm` is not `true`; HTTP 404 if reference or access code is incorrect.

---

## 7. Agent Architecture & Grounding

### 7.1 Supported Customer Intents

The agent supports nine intents:

| Intent | Description |

|---|---|

| `greeting` | General greetings and operating hour questions |

| `menu_inquiry` | Menu listings, category browsing, full menu, prices |

| `ingredient_inquiry` | Ingredient breakdown for a specific dish |

| `allergen_inquiry` | Allergen check for a dish or dishes free from / containing an allergen |

| `availability_inquiry` | Table availability questions (not reservation requests) |

| `reservation_request` | New booking, with multi-turn slot filling |

| `reservation_cancellation` | Cancel an existing booking |

| `reservation_lookup` | View an existing reservation by reference and access code |

| `unknown` | Unclear or out-of-scope input |

### 7.2 Intent Extraction Pipeline

1. **LLM extraction (primary):** The local Ollama `qwen2.5:1.5b` model receives the user message and a strict JSON schema prompt (`INTENT_EXTRACTION_PROMPT`). It returns a JSON object containing `intent` and all entity fields (`menu_item_name`, `allergen_name`, `requested_reservation_date`, `requested_reservation_time`, `number_of_guests`, `reservation_identifier`, `customer_name`, `customer_phone`).

2. **Post-extraction corrections applied to LLM output:**

   - If the LLM classifies a message as `reservation_request` but the message is a table availability question (detected by `_is_table_availability_question()`), the intent is corrected to `availability_inquiry`.

   - If the LLM classifies a message as `availability_inquiry` but the message contains menu/dish keywords or a recognized menu item name without table/seating context, the intent is corrected to `menu_inquiry`.

   - Explicit dates in the user message override LLM date extraction (using `_extract_rule_based`).

   - Phone numbers are always extracted by rule, never trusted from LLM output.

3. **Rule-based fallback:** Used when Ollama is unavailable or returns invalid JSON. Applies deterministic regex patterns for cancellations, reservations, allergen keywords, ingredient keywords, menu keywords, table availability, and greetings. Extracts entities: dates (ISO, relative, named month formats), times (12h and 24h), party size (digits and English words), phone numbers, booking references, customer names, and allergen names.

4. **Session-aware extraction:** During active slot-filling, the rule-based parser applies context-specific patterns (e.g., bare integer answers for party size when that is the awaiting slot; spoken digit words for phone numbers).

5. **Date and time normalization:** `normalize_date_input()` converts `"today"`, `"tomorrow"`, `"tonight"`, `"next Monday"`, named month formats, and `YYYY-MM-DD` to ISO format. `normalize_time_input()` converts 12h and 24h time expressions to `HH:MM`.

### 7.3 Multi-Turn Reservation Slot Filling

Required slots: `requested_reservation_date`, `requested_reservation_time`, `number_of_guests`, `customer_name`, `customer_phone`.

The agent prompts for one missing slot at a time (`awaiting_reservation_slot` tracks which slot is pending). Each user reply is merged into session slot state via `update_slots()`. Booking is attempted only when all five slots are present and pass validation. Session context is preserved across conversational turns and expires after 30 minutes of idle inactivity.

### 7.4 Reservation Lookup (Multi-Turn)

The agent prompts for the reservation reference number, then for the private access code. It calls `verified_reservation()` which performs a constant-time credential comparison. The result is presented as plain text suitable for voice delivery.

### 7.5 Session Management

- In-memory `InteractionState` per `session_id`, stored in a process-local dictionary.

- `last_activity_at` is updated on every session access; sessions idle for more than 30 minutes are evicted lazily on the next access.

- Session state stores: pending intent, slot values, last menu item context, awaiting flags for slot-filling turns, cancellation and lookup reference state.

- Sessions are isolated per session ID; clearing one session does not affect others.

---

## 8. Food Safety Grounding Invariants

1. **No invented data:** The LLM never generates ingredient or allergen facts. All such data comes exclusively from SQLite.

2. **Unverified items:** Any dish with `allergen_status = 'unverified'` triggers an explicit fallback advising guests to speak with staff, regardless of what allergens may or may not be mapped.

3. **Missing items:** Any item not present in the database triggers a safe fallback; the system never guesses.

4. **Allergen absence:** When a verified item has no allergens recorded at all in the database, the response states "no allergens are recorded in the verified allergen database" and advises confirming ingredients and cross-contact risk with staff. The absence of recorded allergens does not prove the dish is free from any specific allergen; it is never described as "allergen-free".

5. **Untracked allergens:** Allergens not in the `allergens` table raise a `ValueError`; the system does not speculate.

6. **Vegetarian/vegan classification:** Not present in the schema; the assistant cannot and does not classify dishes by dietary category.

7. **Popularity data:** No sales or order-frequency data is stored; popularity questions are honestly answered with this acknowledgment.

---

## 9. Non-Functional & Reliability Requirements

1. **Data Integrity:** ACID-compliant transactions with SQLite Write-Ahead Logging (`PRAGMA journal_mode=WAL`) and foreign key enforcement (`PRAGMA foreign_keys = ON`). Reservations use `BEGIN IMMEDIATE` write transactions to prevent booking conflicts under concurrent access.

2. **Deterministic Fallbacks:** Hardcoded safe fallback responses are triggered for system failures, unverified or missing ingredients, Ollama disconnection, STT failures, and TTS failures.

3. **Input Validation:** Pydantic request schemas use `StrictStr`, `StrictInt`, and `StrictBool` on key input fields where type coercion must be prevented (e.g., `ReservationRequest`, `ReservationAccessRequest`, `ReservationCancelRequest`, and the `message` field of `ChatRequest`). Response and entity models use standard Pydantic types. Regex validation is applied to phone numbers and customer names. Operating hour and slot validation is applied to all reservation times, both via the agent path and the direct API path.

4. **Error Isolation:** Internal stack traces and file paths are not exposed in error responses. Service and unexpected failures return HTTP 503 or 500 respectively.

5. **LLM Communication:** Zero-dependency `urllib.request` client. Structured error hierarchy: `OllamaConnectionError`, `OllamaTimeoutError`, `OllamaResponseError`. Strict response validation: response must be a JSON object with a non-empty `message.content` string.

6. **LLM Output Contract:** Low temperature (0.2 for grounded replies, 0.0 for intent extraction), 1–3 sentence responses, no markdown, no emojis, Indian rupee symbol for prices.

7. **Memory & Performance:** Piper and faster-whisper models are loaded lazily and reused; they are not reloaded per request. Preliminary resource observations have been made during development and manual testing, but comprehensive full-stack peak memory and systematic STT, LLM inference, and TTS latency benchmarks are **pending** and have not been measured; no performance figures are claimed.

8. **Rate Limiting:** A per-process, per-IP throttle on reservation lookup and cancel endpoints (20 attempts per 60-second window). This is not a substitute for a network-level rate limiter in a production deployment.

9. **CORS:** Configurable trusted origins via `API_ALLOWED_ORIGINS`; wildcards are not supported.

10. **Single Worker:** Session state is in-process memory. Run one Uvicorn worker process; sessions do not persist across restarts.
