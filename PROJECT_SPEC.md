# PROJECT_SPEC.md — Restaurant Voice AI Agent

## 1. Project Overview & Scope
The **Restaurant Voice AI Agent** is an interactive, voice-driven restaurant assistant designed strictly around core restaurant operations:
1. **Customer Voice Interface:** A modern React + TypeScript web interface capturing customer speech via microphone, displaying live transcripts and status, and playing synthesized voice responses.
2. **Speech Understanding (STT):** Convert customer speech into text using local `faster-whisper`.
3. **Menu Inquiries:** Query and present dishes, descriptions, and prices from the local SQLite database.
4. **Ingredient & Allergen Verification:** Provide verified ingredient and allergen facts retrieved directly from the database.
5. **Table Availability:** Check table capacity and existing bookings.
6. **Reservation Management:** Validate and store customer bookings deterministically in SQLite.
7. **Speech Generation (TTS):** Convert assistant responses into natural spoken audio using local Piper TTS.

*Out-of-scope features (e.g., payment gateways, external POS integrations, SMS notifications, complex web analytics) will not be introduced.*

---

## 2. Technology Stack & Constraints

### 2.1 Core Technologies
- **Frontend Framework:** React 18+ with TypeScript, Vite, and CSS
- **Frontend Tooling:** Node.js v24.11.1 (pre-installed on system, no reinstallation required)
- **Backend Language:** Python 3.10+
- **API Framework:** FastAPI (with Uvicorn)
- **Local Large Language Model (LLM):** Ollama running `qwen2.5:1.5b` (verified present on local disk, no re-download required)
- **Database:** SQLite 3 (relational schema with foreign keys and WAL mode)
- **Speech-to-Text (STT):** `faster-whisper` (CTranslate2-backed Whisper on CPU, `tiny.en` or `base.en` with `int8` quantization)
- **Text-to-Speech (TTS):** Piper TTS (local ONNX voice model running on CPU)

### 2.2 Strict Constraints & Operating Principles
1. **100% Free & Local:** Zero paid APIs, zero subscriptions, zero cloud LLM services, and no API keys required.
2. **Database-Grounded Responses & Safe Fallbacks:**
   - The LLM is never relied upon to "remember" or invent recipe ingredients or allergen flags.
   - All ingredient and allergen answers are retrieved directly from SQLite records.
   - If a menu item or allergen record is absent or ambiguous, the system must trigger a deterministic, safe fallback message (e.g., *"I do not have verified allergen information for that item. Please speak with our restaurant staff before ordering."*).
3. **Deterministic Business Logic for Bookings:**
   - The LLM does not write to the database or compute calendar availability.
   - The LLM acts purely as an entity extractor to collect booking intent (party size, date, time slot, customer name, phone).
   - Backend Python logic deterministically checks table capacity, operating hours, and booking overlaps before confirming any reservation.
4. **Configurable Reservation Duration:**
   - The conflict detection window is fully configurable (via configuration setting `DEFAULT_RESERVATION_DURATION_MINUTES`) rather than assumed or hardcoded, allowing the restaurant to adjust seating duration.
5. **System Resource Monitoring:**
   - Memory and CPU usage will be actively profiled and monitored on the 8 GB RAM / Intel integrated graphics host to ensure stability alongside the Windows OS.
6. **Empirical Benchmarking:**
   - Latency figures across STT, LLM inference, and TTS synthesis are target benchmarks to measure empirically during development, not guaranteed minimums.

---

## 3. System Architecture & Components

```
+-----------------------------------------------------------------------+
|                    Frontend Layer (React + TS + Vite)                 |
|  - Customer Voice Interface (Microphone Capture / Audio Playback)     |
|  - Real-time Status & Conversation Transcript Display                 |
|  - Direct REST / Audio upload communication with FastAPI backend      |
+-----------------------------------------------------------------------+
                                   │ Audio / JSON
                                   ▼
+-----------------------------------------------------------------------+
|                          FastAPI Backend                              |
|  +--------------------+  +--------------------+  +-----------------+  |
|  |     STT Engine     |  |  Agent Controller  |  |   TTS Engine    |  |
|  |  (faster-whisper)  |  |  - Intent Extractor|  |   (Piper TTS)   |  |
|  |  (int8 CPU engine) |  |  - Grounded Context|  | (ONNX medium)   |  |
|  +--------------------+  +--------------------+  +-----------------+  |
|                                  │                                    |
|                                  ▼                                    |
|                   +-------------------------------+                   |
|                   |   Local LLM (Ollama Client)   |                   |
|                   |   qwen2.5:1.5b (local)        |                   |
|                   +-------------------------------+                   |
|                                  │                                    |
|                                  ▼                                    |
|                   +-------------------------------+                   |
|                   |  Database & Business Logic    |                   |
|                   |  - Menu & Allergen Service    |                   |
|                   |  - Configurable Booking Logic |                   |
|                   |  - SQLite (WAL mode)          |                   |
|                   +-------------------------------+                   |
+-----------------------------------------------------------------------+
```

---

## 4. Database Schema (SQLite)

### 4.1 `categories`
- `id` (INTEGER, PRIMARY KEY AUTOINCREMENT)
- `name` (TEXT, UNIQUE, NOT NULL COLLATE NOCASE)

### 4.2 `menu_items`
- `id` (INTEGER, PRIMARY KEY AUTOINCREMENT)
- `category_id` (INTEGER, FOREIGN KEY -> categories.id)
- `name` (TEXT, UNIQUE, NOT NULL COLLATE NOCASE)
- `description` (TEXT)
- `price` (REAL, NOT NULL CHECK (price >= 0))
- `is_available` (INTEGER, NOT NULL DEFAULT 1 CHECK (is_available IN (0, 1)))
- `allergen_status` (TEXT, NOT NULL DEFAULT 'unverified' CHECK (allergen_status IN ('verified', 'unverified')))


### 4.3 `allergens`
- `id` (INTEGER, PRIMARY KEY AUTOINCREMENT)
- `name` (TEXT, UNIQUE, NOT NULL COLLATE NOCASE)

### 4.4 `menu_item_ingredients`
- `id` (INTEGER, PRIMARY KEY AUTOINCREMENT)
- `menu_item_id` (INTEGER, FOREIGN KEY -> menu_items.id)
- `ingredient_name` (TEXT, NOT NULL COLLATE NOCASE)

### 4.5 `menu_item_allergens`
- `menu_item_id` (INTEGER, FOREIGN KEY -> menu_items.id)
- `allergen_id` (INTEGER, FOREIGN KEY -> allergens.id)
- PRIMARY KEY (`menu_item_id`, `allergen_id`)

### 4.6 `tables`
- `id` (INTEGER, PRIMARY KEY AUTOINCREMENT)
- `table_number` (TEXT, UNIQUE, NOT NULL)
- `capacity` (INTEGER, NOT NULL)
- `is_active` (BOOLEAN, DEFAULT 1)

### 4.7 `reservations`
- `id` (INTEGER, PRIMARY KEY AUTOINCREMENT)
- `table_id` (INTEGER, FOREIGN KEY -> tables.id)
- `customer_name` (TEXT, NOT NULL)
- `customer_phone` (TEXT, NOT NULL)
- `party_size` (INTEGER, NOT NULL)
- `reservation_date` (TEXT, NOT NULL) -- YYYY-MM-DD
- `reservation_time` (TEXT, NOT NULL) -- HH:MM (24h)
- `status` (TEXT, DEFAULT 'confirmed') -- 'confirmed', 'cancelled'
- `created_at` (TIMESTAMP, DEFAULT CURRENT_TIMESTAMP)

---

## 5. Functional Requirements

### 5.1 Customer Frontend Interface (React + TypeScript)
- Responsive, intuitive customer interface styled with clean CSS.
- Audio recording control with live visual status (idle, recording, transcribing, thinking, speaking).
- Audio player for immediate playback of the Piper TTS synthesized response.
- Chat transcript showing user queries and grounded agent responses.

### 5.2 Speech Understanding (STT)
- Ingest audio input from the frontend client.
- Transcribe speech to text using `faster-whisper` on CPU (`int8` quantization).
- Gracefully handle inaudible, silent, or noisy audio with appropriate user clarification prompts.

### 5.3 Menu, Ingredient & Allergen Inquiries
- Query menu items by name, category, or dietary preference.
- Provide factual ingredient breakdowns and verified allergens strictly from SQLite.
- If an item or allergen record is unverified or missing, immediately return a safe disclaimer fallback.

### 5.4 Reservation Management
- Check table capacity and availability against operating hours.
- Perform conflict detection using the configurable duration window (`DEFAULT_RESERVATION_DURATION_MINUTES`).
- Validate customer inputs (name, phone format, party size, valid future date and time).
- Deterministically insert confirmed bookings into SQLite.

### 5.5 Speech Generation (TTS)
- Synthesize assistant responses to WAV audio via Piper TTS on CPU.
- Return both synthesized audio and text to the frontend client.

---

## 6. Non-Functional & Reliability Requirements
1. **Memory Profiling:** Active monitoring of RAM usage during inference on the 8 GB Windows laptop.
2. **Deterministic Fallbacks:** Fallback responses hardcoded for system failures, unverified ingredients, or Ollama disconnection.
3. **Data Integrity:** ACID-compliant transactions with SQLite Write-Ahead Logging (`PRAGMA journal_mode=WAL`) and foreign keys enforced.
4. **Input Validation:** Strict Pydantic and regex validation for all reservation entities before database execution.
