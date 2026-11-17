# Restaurant Voice AI Agent

A local, voice-enabled restaurant assistant that understands customer requests, answers menu and allergen questions, checks table availability, and manages reservations through natural conversation.

The application combines a React frontend, FastAPI backend, local speech recognition, a local language model, text-to-speech, and SQLite database operations.

## Features

- **Voice interaction:** Capture customer speech through the browser microphone and respond with synthesized audio.
- **Speech-to-text:** Transcribe speech locally using Faster-Whisper.
- **Local AI:** Understand customer requests using Ollama and Qwen 2.5.
- **Menu inquiries:** Retrieve available dishes, categories, descriptions, and prices from SQLite.
- **Ingredient information:** Retrieve recorded ingredients directly from the restaurant database.
- **Allergen safety:** Use verified database records for allergen answers. Unverified information triggers a safe fallback rather than an AI-generated guess.
- **Table availability:** Check table capacity and existing reservations.
- **Reservation management:** Collect missing booking details, validate them, and create reservations deterministically.
- **Reservation lookup and cancellation:** Use a reservation reference and private access code through the Reservations interface.
- **Voice responses:** Generate spoken responses locally using Piper TTS.

## Architecture

```text
Customer
   |
   v
React + TypeScript Frontend
   |
   | Text / Audio
   v
FastAPI Backend
   |
   v
Agent Service
   |
   +---- LLM Service (Ollama)
   |
   +---- STT Service (Faster-Whisper)
   |
   +---- TTS Service (Piper)
   |
   +---- Core Reservation Policies
   |
   v
SQLite Database
   |
   +---- Menu and Categories
   +---- Ingredients and Allergens
   +---- Tables
   +---- Reservations
   +---- Reservation Access
```

## Technology Stack

| Component | Technology |
|---|---|
| Frontend | React, TypeScript, Vite |
| Backend | Python, FastAPI |
| Local LLM | Ollama, Qwen 2.5 1.5B |
| Speech recognition | Faster-Whisper |
| Speech generation | Piper TTS |
| Database | SQLite |
| Data validation | Pydantic |
| API communication | REST |
| Version control | Git and GitHub |

## Voice Agent Workflow

1. The customer speaks through the browser microphone.
2. Faster-Whisper converts the audio into text.
3. The agent identifies the customer's intent and relevant information.
4. The agent retrieves factual information or performs the required database operation.
5. Python business rules validate reservation and availability operations.
6. The response is prepared using grounded information.
7. Piper converts the response into audio.
8. The frontend displays the response and plays the audio.

## Food Safety and Grounding

Food safety is handled through deterministic database lookups.

- Ingredient information is retrieved from SQLite.
- Allergen information is checked against verified menu records.
- Items marked as unverified are never assumed to be allergen-free.
- Missing or ambiguous allergen information triggers a safe response directing customers to restaurant staff.
- Cross-contact warnings are included where appropriate.

The LLM is not treated as the source of truth for food safety.

## Reservation Management

The reservation workflow collects:

- Customer name
- Phone number
- Reservation date
- Reservation time
- Number of guests

The backend validates the information, checks restaurant operating hours, verifies availability, and creates the reservation in SQLite.

The default reservation duration is 90 minutes and can be configured.

Customers can use their reservation reference and private access code to review or cancel a booking through the Reservations interface.

## Example Conversation

**Customer:** Can I reserve a table for four people at 7:30 PM?

**Agent:** Certainly. What date would you like to reserve?

The agent continues collecting missing booking details, validates availability, and confirms the booking after successful database insertion.

**Customer:** Does the pesto contain nuts?

**Agent:** The agent checks the verified allergen records. If the information has not been formally verified, it directs the customer to restaurant staff before ordering.

## Error Handling and Reliability

The application includes safe handling for:

- Unavailable Ollama service
- Speech recognition and synthesis failures
- Invalid or unintelligible audio
- Missing database records
- Unverified allergen information
- Invalid reservation details
- Unavailable tables and conflicting bookings
- Invalid reservation credentials

When a required service or database operation fails, the agent uses fallback responses rather than inventing successful results.

## Local Setup

### Prerequisites

- Python 3.11
- Node.js and npm
- Ollama
- Git

The required speech recognition and speech generation models must also be available locally.

### 1. Clone the repository

After publishing the repository, replace the placeholder URL with its GitHub URL:

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd restaurant-voice-agent
```

### 2. Set up the Python environment

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 3. Configure Ollama

Install Ollama and download the configured model:

```bash
ollama pull qwen2.5:1.5b
```

Ensure Ollama is running locally.

### 4. Configure speech models

Ensure the configured Faster-Whisper model and Piper ONNX model, together with its JSON configuration file, are available locally.

Default settings:

- Whisper model: `tiny.en`
- Whisper compute type: `int8`
- Whisper device: `cpu`
- Piper model: `en_US-lessac-medium`

### 5. Start the backend

From the project root, activate the virtual environment and run:

```bash
python -m uvicorn src.api.main:app --reload
```

### 6. Start the frontend

Open another terminal:

```powershell
cd frontend
npm install
npm run dev
```

Open the local frontend address displayed by Vite.

> **Note:** These are the intended local development commands. Confirm the model file locations and any project-specific environment settings before publishing.

## Configuration

The application supports environment-variable overrides.

| Variable | Default |
|---|---|
| `OLLAMA_BASE_URL` | `http://localhost:11434` |
| `OLLAMA_MODEL` | `qwen2.5:1.5b` |
| `WHISPER_MODEL_SIZE` | `tiny.en` |
| `WHISPER_COMPUTE_TYPE` | `int8` |
| `WHISPER_DEVICE` | `cpu` |
| `PIPER_MODEL_NAME` | `en_US-lessac-medium` |
| `RESERVATION_DURATION_MINUTES` | `90` |
| `RESTAURANT_OPENING_TIME` | `10:00` |
| `RESTAURANT_CLOSING_TIME` | `23:00` |
| `API_MAX_UPLOAD_BYTES` | `10485760` |

Additional configuration options are available in `src/core/config.py`.

## API Overview

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/api/health` | Check API and database health |
| POST | `/api/chat` | Process text conversations |
| POST | `/api/voice` | Process audio and return a spoken response |
| GET | `/api/menu` | Retrieve menu information |
| GET | `/api/reservations/slots` | Retrieve available reservation slots |
| POST | `/api/reservations` | Create a reservation |
| POST | `/api/reservations/lookup` | Retrieve a reservation |
| POST | `/api/reservations/cancel` | Cancel a reservation |

## Testing and Verification

The project has undergone:

- Backend and frontend regression testing
- API and database verification
- Voice interaction and audio playback checks
- Menu, ingredient, and allergen verification
- Reservation creation, lookup, and cancellation checks
- Error-handling verification
- End-to-end manual testing
- Performance benchmarking
- Responsive frontend checks

## Project Constraints

- Local-first architecture
- No external cloud AI APIs required
- No paid LLM API keys required
- SQLite-based restaurant data
- Deterministic reservation operations
- Database-grounded food safety responses

## Future Scope

Possible extensions include telephony integration, restaurant POS integration, multilingual speech support, and additional restaurant operational workflows.

## License

This project is licensed under the MIT License.

You are free to use, copy, modify, merge, publish, distribute, sublicense, and sell copies of this software, subject to the conditions of the MIT License.

See the LICENSE file for the complete license terms.

Copyright © 2026 Safiya Shaik.
