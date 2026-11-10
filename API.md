# Restaurant Voice Agent API

The API uses the existing local SQLite, Ollama, faster-whisper and Piper services. No audio models are loaded by the health endpoint. Run one server worker: agent sessions are held in memory, and the API's session/booking lock applies only within one process. Sessions disappear on process restart; there is no authentication yet. Do not expose this API to untrusted networks.

## Setup and start

Install dependencies from `requirements.txt` in your virtual environment. Configure Ollama and make the locally cached Whisper model and Piper model files available as described by the existing project configuration. Initialize the database if needed:

```bash
python -m src.database.init_db
uvicorn src.api.main:app --host 127.0.0.1 --port 8000
```

On the target machine this serves `http://127.0.0.1:8000`; interactive Swagger documentation is at `http://127.0.0.1:8000/docs`. `init_db` is idempotent; do not delete an existing production database. The API does not initialize it automatically. `GET /api/health` reports 503 if the database file or required tables are missing.

Set `API_ALLOWED_ORIGINS` to a comma-separated list of trusted browser origins; the development default is `http://localhost:5173,http://127.0.0.1:5173`. Wildcards are not supported. Configure `API_MAX_UPLOAD_BYTES` (positive integer; default 10485760, or 10 MiB). Limit requests at the reverse proxy too: FastAPI parses multipart before the endpoint enforces the per-file limit. Existing `WHISPER_*`, `PIPER_MODEL_DIR`, and `PIPER_MODEL_NAME` settings control the models. Do not commit voice models, audio uploads, or generated WAV files.

The future frontend can use relative `/api/...` URLs with a Vite proxy to the backend, or call the backend origin if it is explicitly allowed by CORS. Generate a fresh UUID for each customer (for example, `crypto.randomUUID()`); reuse it for that customer's chat and voice turns. Do not share it between customers. The UUID is an isolation key, **not** an authentication credential.

## Endpoints

### `GET /api/health`

```bash
curl http://127.0.0.1:8000/api/health
```

Example: `{"status":"ok","database":"ok"}`. It checks SQLite without loading Whisper or Piper.

### `POST /api/chat`

```bash
curl -X POST http://127.0.0.1:8000/api/chat \
  -H 'Content-Type: application/json' \
  -d '{"session_id":"a3c4bfde-5476-4a71-996c-79585754add2","message":"Hello"}'
```

Example response (text varies with the agent):

```json
{"session_id":"a3c4bfde-5476-4a71-996c-79585754add2","text":"Hello! How can I help?","intent":"greeting","entities":{"menu_item_name":null,"allergen_name":null,"requested_reservation_date":null,"requested_reservation_time":null,"number_of_guests":null,"reservation_identifier":null,"customer_name":null,"customer_phone":null},"data":null,"success":true,"is_fallback":false,"needs_clarification":false,"missing_slots":[]}
```

### `POST /api/voice`

Multipart fields: `session_id` (UUID) and `file` (audio file). The content type must be a supported audio type or `application/octet-stream`; actual decoding is validated by STT.

```bash
curl -X POST http://127.0.0.1:8000/api/voice \
  -F 'session_id=a3c4bfde-5476-4a71-996c-79585754add2' \
  -F 'file=@recording.wav;type=audio/wav' -D voice-headers.txt -o response.wav
```

The response body is WAV. `X-Session-ID`, `X-Transcript`, and `X-Response-Text` are exposed to allowed browser origins; transcript and text headers are UTF-8 percent-encoded and should be decoded with `decodeURIComponent`. The response has `Cache-Control: no-store`. These headers can contain customer speech; handle them as sensitive data. Uploads and output files are temporary and removed after processing. Invalid/empty audio returns 400 or 422, unsupported media 415, oversized files 413, and unavailable STT/TTS 503. Missing fields return 422.

### `GET /api/menu`

```bash
curl 'http://127.0.0.1:8000/api/menu?category=Appetizers'
```

Response: `{"categories":["Appetizers", "..."],"items":[{"id":1,"category":"Appetizers","name":"...","description":"...","price":9.0,"is_available":true,"allergen_status":"verified"}]}`. The optional category filter applies to available items; the categories list remains complete.

### `POST /api/reservations`

```bash
curl -X POST http://127.0.0.1:8000/api/reservations \
  -H 'Content-Type: application/json' \
  -d '{"customer_name":"Alice","customer_phone":"555-1234","party_size":2,"reservation_date":"2030-05-12","reservation_time":"18:00"}'
```

A 201 response includes `reservation_id`, table details, customer and booking details, `status`, and `created_at`, only after commit. Invalid details return 422; no matching table returns 409. This direct endpoint intentionally uses existing CRUD validation. Unlike the agent reservation path, CRUD currently does **not** reject past dates or count phone digits the same way; unifying those rules requires a separate business-logic change. Existing agent booking also uses autocommit; API calls are serialized only inside this process. Do not run multiple API workers or other concurrent booking writers until database-wide booking atomicity is addressed.

Errors do not include internal paths or stack traces. Service and unexpected failures may return 503 or 500 respectively.
