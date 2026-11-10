# PROGRESS.md — Restaurant Voice AI Agent

## Project Status Overview
- **Status:** Phase 2, Step 2 Completed (Local LLM Service Verified)
- **Last Updated:** 2026-11-09

---

## Roadmap & Phases

- [x] **Phase 0: Project Setup & Specification**
  - [x] Workspace inspection (verified clean workspace).
  - [x] Node.js and npm pre-installation verified (`v24.11.1` / `11.21.2`; no reinstallation needed).
  - [x] Ollama installation & `qwen2.5:1.5b` model verification on local disk (found at `C:\Users\Safiya Shaik\.ollama\models\manifests\registry.ollama.ai\library\qwen2.5\1.5b`, size 986 MB; no re-download required).
  - [x] Frontend technology stack defined: **React + TypeScript + Vite + CSS**.
  - [x] Backend stack confirmed: **Python + FastAPI + SQLite + faster-whisper + Piper TTS**.
  - [x] Scope alignment: strictly core voice assistant capabilities (Menu, Allergens/Ingredients, Availability, Reservations, STT, LLM Tool Calling, SQLite, TTS).
  - [x] Refinement of safety rules: grounded database responses, deterministic backend validation, and safe fallbacks.
  - [x] Configurable reservation duration parameter (`DEFAULT_RESERVATION_DURATION_MINUTES`) replacing hardcoded assumptions.
  - [x] RAM footprint strategy revised to active memory profiling on the 8 GB Windows host.
  - [x] Latency figures framed as empirical benchmarks to measure during testing.
  - [x] Updated `PROJECT_SPEC.md` and `PROGRESS.md`.

- [x] **Phase 1: Database Layer & Scaffolding (Verified)**
  - [x] Created clean project modular directory structure (`src/core`, `src/database`, `src/services`, `src/api`, `data`, `models`, `tests`).
  - [x] Implemented SQLite schema with foreign key constraints, WAL mode, and indexes (`src/database/schema.sql`).
  - [x] Implemented thread-safe connection manager with automatic transaction commit/rollback (`src/database/connection.py`).
  - [x] Created realistic seed dataset (`data/seed_data.json`) covering 4 categories, 9 major allergens, 12 dishes (with full ingredient lists and allergen mappings), and 6 tables with varying capacities (2, 4, 6, 8 pax).
  - [x] Implemented explicit `allergen_status` (`'verified'` vs `'unverified'`) ensuring unverified items are never assumed allergen-free.
  - [x] Implemented deterministic CRUD services (`src/database/crud.py`) for menu queries, allergen verification, and configurable reservation conflict validation.
  - [x] Executed full database unit test suite: **21 / 21 unit tests passed** in `tests/test_database.py`.
  - [x] Persistent database operational at `data/restaurant.db`.

- [x] **Phase 2 – Step 1 & Step 2: Local LLM Service (Verified)**
  - [x] Verified local Ollama daemon and `qwen2.5:1.5b` model inference.
  - [x] Implemented modular LLM service (`src/services/llm_service.py`):
    - Connects to local Ollama API via `/api/chat`.
    - 100% standard library implementation (`urllib.request`) with zero cloud dependencies or external API keys.
    - Configurable timeouts (`ollama_timeout_seconds`, default 30.0s).
    - Structured error hierarchy (`OllamaConnectionError`, `OllamaTimeoutError`, `OllamaResponseError`).
    - Standardized `LLMResponse` data transfer object with token counts and duration metrics.
    - **Strict Response Validation:** Enforces that response payload is a JSON object, `message` is a dictionary, and `message.content` is a non-empty string; raises `OllamaResponseError` on missing, malformed, or empty content.
  - [x] Implemented voice-optimized restaurant system prompt (`DEFAULT_RESTAURANT_SYSTEM_PROMPT`):
    - Concise spoken communication (1-3 sentences, natural TTS delivery).
    - Prohibits markdown formatting, bullet points, headers, and emojis.
    - Strict safety invariant: never invents dishes, ingredients, or allergens.
    - Requires database context grounding and explicit referral to staff for unverified facts.
  - [x] Added grounded reply method (`generate_restaurant_reply()`):
    - Injects authoritative SQLite facts directly into system prompt.
  - [x] Added and executed focused unit tests (`tests/test_llm_service.py`):
    - **15 / 15 unit tests passed** (including missing message, invalid message structure, top-level array, missing content, and empty/whitespace content; 0.011s).
  - [x] Added and executed live integration tests (`tests/test_llm_integration.py`):
    - **4 / 4 integration tests passed** against local Ollama running `qwen2.5:1.5b`.
    - Measured latency: ~3.25s - 4.01s for conversational turn on CPU (~23 tokens/sec).
  - [x] Re-ran Phase 1 database test suite: **21 / 21 tests passed** (0.479s), verifying database layer remains intact.

- [x] **Phase 2 – Step 3: Structured Tool Calling & Grounded Agent Router (Verified)**
  - [x] Implemented modular Grounded Agent service ([`src/services/agent_service.py`](file:///c:/Users/Safiya%20Shaik/Downloads/restaurant-voice-agent/src/services/agent_service.py)):
    - Supported 8 standardized customer intents: `greeting`, `menu_inquiry`, `ingredient_inquiry`, `allergen_inquiry`, `availability_inquiry`, `reservation_request`, `reservation_cancellation`, `unknown`.
    - Extracted entities: `menu_item_name`, `allergen_name`, `requested_reservation_date`, `requested_reservation_time`, `number_of_guests`, `reservation_identifier`, `customer_name`, `customer_phone`.
    - Local LLM structured extraction via validated JSON schema (`INTENT_EXTRACTION_PROMPT`) with deterministic regex/keyword fallback (`_extract_rule_based`).
    - Deterministic routing to Phase 1 SQLite CRUD operations (`get_menu_items`, `get_menu_item_by_name`, `get_ingredients_for_item`, `check_allergens_for_item`, `find_safe_items_for_allergens`, `find_available_tables`, `create_reservation`, `cancel_reservation`).
    - Food Safety Grounding: Unverified dishes and unlisted items strictly trigger safe fallbacks; authoritative database allergen facts are delivered directly to prevent LLM hallucination.
    - Multi-turn reservation slot filling: In-memory session state tracks pending booking parameters and prompts only for missing slots before Python-level validation and atomic DB creation.
  - [x] Implemented and executed unit test suite ([`tests/test_agent_service.py`](file:///c:/Users/Safiya%20Shaik/Downloads/restaurant-voice-agent/tests/test_agent_service.py)):
    - **24 / 24 unit tests passed** (0.840s; covers intent extraction, routing, slot filling, boundary validation, cancellation, and error handling).
  - [x] Implemented and executed live integration test suite ([`tests/test_agent_integration.py`](file:///c:/Users/Safiya%20Shaik/Downloads/restaurant-voice-agent/tests/test_agent_integration.py)):
    - **4 / 4 integration tests passed** against live local Ollama `qwen2.5:1.5b` and SQLite `data/restaurant.db`.
  - [x] Re-ran full project regression test suite:
    - **60 / 60 offline unit tests passed** (21 DB + 15 LLM + 24 Agent; 1.237s).
    - **8 / 8 live Ollama integration tests passed** (4 LLM + 4 Agent).
    - **Grand Total: 68 / 68 tests passing across the project (100%)**.

- [ ] **Phase 3: Speech-to-Text (STT) via faster-whisper**
  - [ ] Configure `faster-whisper` (`base.en` / `tiny.en` on CPU with `int8` quantization).
  - [ ] Implement `stt_service.py` to ingest audio and return text transcriptions.
  - [ ] Benchmark transcription latency and memory usage on CPU.

- [ ] **Phase 4: Text-to-Speech (TTS) via Piper**
  - [ ] Set up local Piper ONNX voice model.
  - [ ] Implement `tts_service.py` for WAV speech synthesis.
  - [ ] Benchmark speech synthesis latency and verify audio output.

- [ ] **Phase 5: FastAPI Backend Orchestration**
  - [ ] Build FastAPI server with REST and audio endpoints (`/api/voice`, `/api/chat`, `/api/menu`, `/api/reservations`).
  - [ ] Wire end-to-end pipeline: Audio -> faster-whisper -> Grounded Agent -> SQLite -> Piper TTS -> Audio response.

- [ ] **Phase 6: React + TypeScript Frontend Development**
  - [ ] Initialize Vite + React + TypeScript + CSS in `frontend/` using existing Node.js environment.
  - [ ] Implement microphone capture, live status indicator (idle, listening, processing, speaking), transcript view, and audio response player.
  - [ ] Connect frontend to FastAPI backend endpoints.

- [ ] **Phase 7: Quality Assurance, Benchmarking & Failure Handling**
  - [ ] Validate edge cases: allergen inquiry for unlisted items (safe fallback test), overbooking prevention, invalid date/time formats.
  - [ ] Profile memory and CPU utilization on the 8 GB machine during full pipeline execution.
  - [ ] Verify graceful handling when Ollama is unreachable or audio is corrupt.

---

## Architectural & Technical Decisions Log

| Decision ID | Topic | Decision | Justification |
|-------------|-------|----------|---------------|
| **DEC-001** | STT Model Selection | `faster-whisper` with `base.en` / `tiny.en` (int8, CPU) | Provides robust transcription accuracy while keeping memory footprint low on an 8 GB system. |
| **DEC-002** | TTS Engine | Piper TTS (ONNX voice model on CPU) | Fast local speech synthesis without cloud dependencies or paid APIs. |
| **DEC-003** | Local LLM Verification | Use existing local `qwen2.5:1.5b` (Q4_K_M) via Ollama | Verified 986 MB model blob is already cached on disk; prevents redundant downloads and runs locally on CPU. |
| **DEC-004** | Grounded Safety Architecture | DB-Grounded Retrieval with Deterministic Fallbacks | LLM does not invent recipe ingredients. All safety answers are grounded strictly in SQLite records; unlisted items trigger a safe fallback message. |
| **DEC-005** | Reservation Duration Logic | Configurable booking conflict window (`DEFAULT_RESERVATION_DURATION_MINUTES`) | Avoids hardcoded 90-minute assumptions; allows the restaurant's seating policy to be configured in settings. |
| **DEC-006** | Performance & Memory Policy | Empirical benchmarking and active memory profiling | Avoids absolute guarantees; monitors memory footprint on the 8 GB host to ensure system stability. |
| **DEC-007** | Frontend Framework | React + TypeScript + Vite + CSS | Modern, lightweight, type-safe client with fast development cycle; leverages existing Node.js v24.11.1 installation without extra runtime installations. |
| **DEC-008** | Explicit Allergen Verification Status | `allergen_status` column in `menu_items` (`'verified'` vs `'unverified'`) | Prevents unverified dishes from being treated as allergen-free. `find_safe_items_for_allergens()` strictly excludes unverified dishes. |
| **DEC-009** | Best-Fit Table Allocation | Tables sorted by `capacity ASC, table_number ASC` | Assigns the smallest table that can accommodate the party size, preserving larger tables (e.g. 6-8 pax) for larger groups. |
| **DEC-010** | Database Test Isolation | Unique in-memory URI per test method (`file:memdb_{uuid}?mode=memory&cache=shared`) | Guarantees complete isolation across unit tests, preventing shared state contamination in concurrent or sequential test runs. |
| **DEC-011** | Zero-Dependency LLM Client | Standard Python `urllib.request` communicating with Ollama `/api/chat` | Eliminates external SDK dependencies and API keys; handles timeouts, network failures, and latency tracking cleanly. |
| **DEC-012** | Voice-First LLM Output Contract | Low temperature (0.2), 1-3 sentences max, no markdown, no emojis | Ensures generated text sounds natural, clear, and unfragmented when fed to Piper Text-to-Speech in subsequent phases. |
| **DEC-013** | Direct Grounded Allergen Delivery | Deliver verified SQLite allergen facts directly to caller without LLM re-synthesis | Prevents local LLMs from hallucinating negations (e.g. inverting "contains gluten" to "gluten-free") during natural language generation for critical food safety inquiries. |
| **DEC-014** | In-Memory Session Slot Filling | `InteractionState` class scoped per interaction session | Preserves pending booking parameters (date, time, party size, name, phone) across conversational turns without requiring external database state. |

