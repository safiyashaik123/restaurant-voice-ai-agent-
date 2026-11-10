# PROGRESS.md — Restaurant Voice AI Agent

## Project Status Overview

- **Status:** Phases 0–5 implemented; Phase 6 (React frontend) not started; Phase 7 (QA and benchmarking) pending.
- **Last Updated:** 2026-11-10

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
  - [x] Implemented modular Grounded Agent service (`src/services/agent_service.py`):
    - Supported 8 standardized customer intents: `greeting`, `menu_inquiry`, `ingredient_inquiry`, `allergen_inquiry`, `availability_inquiry`, `reservation_request`, `reservation_cancellation`, `unknown`.
    - Extracted entities: `menu_item_name`, `allergen_name`, `requested_reservation_date`, `requested_reservation_time`, `number_of_guests`, `reservation_identifier`, `customer_name`, `customer_phone`.
    - Local LLM structured extraction via validated JSON schema (`INTENT_EXTRACTION_PROMPT`) with deterministic regex/keyword fallback (`_extract_rule_based`).
    - Deterministic routing to Phase 1 SQLite CRUD operations (`get_menu_items`, `get_menu_item_by_name`, `get_ingredients_for_item`, `check_allergens_for_item`, `find_safe_items_for_allergens`, `find_available_tables`, `create_reservation`, `cancel_reservation`).
    - Food Safety Grounding: Unverified dishes and unlisted items strictly trigger safe fallbacks; authoritative database allergen facts are delivered directly to prevent LLM hallucination.
    - Multi-turn reservation slot filling: In-memory session state tracks pending booking parameters and prompts only for missing slots before Python-level validation and atomic DB creation.
  - [x] Implemented and executed unit test suite (`tests/test_agent_service.py`):
    - **33 / 33 unit tests passed**, covering intent extraction, routing, reservation slot filling, validation, cancellation, session isolation, idle TTL expiration, and error handling.
  - [x] Implemented and executed live integration test suite (`tests/test_agent_integration.py`):
    - **4 / 4 integration tests passed** against live local Ollama `qwen2.5:1.5b` and SQLite `data/restaurant.db`.
  - [x] Implemented idle-session TTL:
    - Added `last_activity_at` to `InteractionState`.
    - Sessions expire based on inactivity rather than creation time.
    - Existing sessions refresh their activity timestamp on access.
    - Added tests for activity refresh, idle expiration, and session isolation.
  - [x] Executed complete project regression test suite:
    - Agent integration: **4 / 4 passed**.
    - Agent service: **33 / 33 passed**.
    - Database: **21 / 21 passed**.
    - Ollama integration: **4 / 4 passed**.
    - LLM service: **15 / 15 passed**.
    - **Grand Total: 77 / 77 tests passed (100%).**

- [x] **Phase 3: Speech-to-Text (STT) via faster-whisper (Complete)**
  - [x] Configure `faster-whisper` (`tiny.en` by default on CPU with `int8` quantization); verify the locally downloaded model loads.
  - [x] Implement `stt_service.py` to ingest audio and return text transcriptions, with lazy model reuse and error handling.
  - [x] Run mocked STT unit tests: **16 passed, 2 subtests passed**.

- [x] **Phase 4: Text-to-Speech (TTS) via Piper (Complete)**
  - [x] Implement `tts_service.py` for local WAV speech synthesis using `piper-tts>=1.8.0`; the default voice is `en_US-lessac-medium`.
  - [x] Configure `PIPER_MODEL_DIR` (default `models/piper`) and `PIPER_MODEL_NAME`; load the model lazily and reuse it without downloading it.
  - [x] Add mocked tests for voice loading, WAV output, configuration and failure handling: **11 passed, 13 subtests passed**.
  - [x] Run regression tests on Python 3.11: **96 passed, 8 skipped, 15 subtests passed**; live Ollama tests skipped because Ollama was unavailable.
  - [x] Download and verify the local `en_US-lessac-medium` Piper ONNX voice model.
  - [x] Generate a real WAV on Windows using Python 3.10; manually verify playback is audible, clear, natural and not cut off.
  - [x] Run local TTS tests: **11 / 11 passed**.
  - [x] Run the complete local test suite: **104 / 104 passed in 46.09 seconds**.

- [x] **Phase 5: FastAPI Backend Orchestration (Implemented; mocked verification)**
  - [x] Build FastAPI endpoints (`/api/health`, `/api/voice`, `/api/chat`, `/api/menu`, `/api/reservations`) with schemas, configurable CORS, and bounded audio uploads.
  - [x] Wire Audio -> faster-whisper -> Grounded Agent -> SQLite -> Piper TTS -> WAV response using existing services; clean up temporary files.
  - [x] Document startup, configuration, endpoint examples and deployment limitations in `API.md`.
  - [x] Run API unit tests: **18 passed**; full suite on Python 3.11: **114 passed, 8 skipped, 15 subtests passed**. The skipped integration tests require live Ollama.
  - [ ] Verify the complete pipeline with live local models and target Windows environment during Phase 7.

- [ ] **Phase 6: React + TypeScript Frontend Development (Not started)**
  - [ ] Initialize Vite + React + TypeScript + CSS in `frontend/` using existing Node.js environment.
  - [ ] Implement microphone capture, live status indicator (idle, listening, processing, speaking), transcript view, and audio response player.
  - [ ] Connect frontend to FastAPI backend endpoints.

- [ ] **Phase 7: Quality Assurance, Benchmarking & Failure Handling (Pending)**
  - [ ] Benchmark transcription latency and memory usage on CPU.
  - [ ] Benchmark speech synthesis latency and memory usage on the target machine.
  - [ ] Validate edge cases: allergen inquiry for unlisted items (safe fallback test), overbooking prevention, invalid date/time formats.
  - [ ] Profile memory and CPU utilization on the 8 GB machine during full pipeline execution.
  - [ ] Verify graceful handling when Ollama is unreachable or audio is corrupt.

---

## Architectural & Technical Decisions Log

| Decision ID | Topic                                 | Decision                                                                            | Justification                                                                                                                                                               |
| ----------- | ------------------------------------- | ----------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **DEC-001** | STT Model Selection                   | `faster-whisper` with `base.en` / `tiny.en` (int8, CPU)                             | Provides robust transcription accuracy while keeping memory footprint low on an 8 GB system.                                                                                |
| **DEC-002** | TTS Engine                            | Piper TTS (ONNX voice model on CPU)                                                 | Fast local speech synthesis without cloud dependencies or paid APIs.                                                                                                        |
| **DEC-003** | Local LLM Verification                | Use existing local `qwen2.5:1.5b` (Q4_K_M) via Ollama                               | Verified 986 MB model blob is already cached on disk; prevents redundant downloads and runs locally on CPU.                                                                 |
| **DEC-004** | Grounded Safety Architecture          | DB-Grounded Retrieval with Deterministic Fallbacks                                  | LLM does not invent recipe ingredients. All safety answers are grounded strictly in SQLite records; unlisted items trigger a safe fallback message.                         |
| **DEC-005** | Reservation Duration Logic            | Configurable booking conflict window (`DEFAULT_RESERVATION_DURATION_MINUTES`)       | Avoids hardcoded 90-minute assumptions; allows the restaurant's seating policy to be configured in settings.                                                                |
| **DEC-006** | Performance & Memory Policy           | Empirical benchmarking and active memory profiling                                  | Avoids absolute guarantees; monitors memory footprint on the 8 GB host to ensure system stability.                                                                          |
| **DEC-007** | Frontend Framework                    | React + TypeScript + Vite + CSS                                                     | Modern, lightweight, type-safe client with fast development cycle; leverages existing Node.js v24.11.1 installation without extra runtime installations.                    |
| **DEC-008** | Explicit Allergen Verification Status | `allergen_status` column in `menu_items` (`'verified'` vs `'unverified'`)           | Prevents unverified dishes from being treated as allergen-free. `find_safe_items_for_allergens()` strictly excludes unverified dishes.                                      |
| **DEC-009** | Best-Fit Table Allocation             | Tables sorted by `capacity ASC, table_number ASC`                                   | Assigns the smallest table that can accommodate the party size, preserving larger tables (e.g. 6-8 pax) for larger groups.                                                  |
| **DEC-010** | Database Test Isolation               | Unique in-memory URI per test method (`file:memdb_{uuid}?mode=memory&cache=shared`) | Guarantees complete isolation across unit tests, preventing shared state contamination in concurrent or sequential test runs.                                               |
| **DEC-011** | Zero-Dependency LLM Client            | Standard Python `urllib.request` communicating with Ollama `/api/chat`              | Eliminates external SDK dependencies and API keys; handles timeouts, network failures, and latency tracking cleanly.                                                        |
| **DEC-012** | Voice-First LLM Output Contract       | Low temperature (0.2), 1-3 sentences max, no markdown, no emojis                    | Ensures generated text sounds natural, clear, and unfragmented when fed to Piper Text-to-Speech in subsequent phases.                                                       |
| **DEC-013** | Direct Grounded Allergen Delivery     | Deliver verified SQLite allergen facts directly to caller without LLM re-synthesis  | Prevents local LLMs from hallucinating negations (e.g. inverting "contains gluten" to "gluten-free") during natural language generation for critical food safety inquiries. |
| **DEC-014** | In-Memory Session Slot Filling        | `InteractionState` class scoped per interaction session                             | Preserves pending booking parameters (date, time, party size, name, phone) across conversational turns without requiring external database state.                           |
| **DEC-015** | Session Idle TTL                      | `last_activity_at` with a configurable 30-minute idle timeout                       | Expires inactive sessions while preserving active customer conversations and their pending reservation details.                                                             |
