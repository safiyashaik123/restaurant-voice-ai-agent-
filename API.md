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
  -d '{"customer_name":"Alice","customer_phone":"9876543210","party_size":2,"reservation_date":"2030-05-12","reservation_time":"18:00"}'
```

A 201 response includes `reservation_id`, table details, customer and booking details, `status`, and `created_at`, only after commit. Invalid details return 422; no matching table returns 409. The direct and agent booking paths share phone and time validation. Reservation allocation uses an immediate SQLite transaction; the API also serializes its in-process agent and booking calls. Run one API worker because session state remains process-local.

Errors do not include internal paths or stack traces. Service and unexpected failures may return 503 or 500 respectively.

### Reservation policy and slots

`GET /api/reservations/slots?date=YYYY-MM-DD` returns `{ "date": "YYYY-MM-DD", "slots": ["10:00", ...] }` for permissible start times, not table availability. Invalid dates return 422; past dates return an empty list. Dates and times are interpreted in the restaurant's Asia/Kolkata timezone (IST), regardless of the server or browser timezone. The form displays the returned 24-hour times unchanged as restaurant-local times. The frontend fetches these slots when the date changes; the server validates again on booking, so an elapsed slot cannot be booked.

Indian mobile numbers must be exactly ten ASCII digits starting with 6–9, optionally immediately preceded by `+91` (no spaces, punctuation or other prefixes). Invalid phone or time details return 422 with an explanation. Operating hours default to 10:00–23:00 and are configured with `RESTAURANT_OPENING_TIME` and `RESTAURANT_CLOSING_TIME` in 24-hour `HH:MM` format. Starts are spaced 30 minutes from opening and must be in the future and fit entirely before closing. `RESERVATION_DURATION_MINUTES` defaults to 90, so with the default hours the last start is 21:30, not 22:30; a 30-minute duration permits 22:30. Overnight hours are not supported. Install the requirements including `tzdata` for IANA timezone support on Windows.

### Grounded menu conversations

Menu, category, ingredient, and allergen replies use SQLite records directly rather than model-generated factual prose. Explicit menu queries use deterministic recognition; ambiguous messages still use the configured Ollama model for intent extraction. Follow-up dish context is isolated by in-memory session and expires with the existing session TTL. Unknown dishes and missing ingredient records are not guessed.

The existing schema has no verified vegetarian or vegan classification. The assistant therefore cannot recommend or classify dishes as vegetarian or vegan and directs guests to staff; it does not infer suitability from names, descriptions, or ingredients. For allergens, a verified item's mapped allergens indicate recorded presence or absence only for allergens tracked by the database. Unverified dishes, untracked or ambiguous allergens, and compound allergen requests require clarification or staff assistance. Cross-contact cannot be verified by this database, so guests are advised to check with staff before ordering.

## English speech recognition and Windows A/B benchmark

The service uses faster-whisper locally with `language="en"` and `task="transcribe"`. Defaults are `WHISPER_MODEL_SIZE=tiny.en`, `WHISPER_DEVICE=cpu`, `WHISPER_COMPUTE_TYPE=int8`, `WHISPER_BEAM_SIZE=5` (positive integer), and `WHISPER_VAD_FILTER=false` (`true`/`false`, also `1`/`0`, `yes`/`no`, `on`/`off`). Beam 5 and VAD off match faster-whisper defaults; VAD is optional because it can omit short or quiet speech. `WHISPER_MODEL_SIZE=base.en` opts into the larger English model. Existing local model-directory configuration remains supported. Set environment variables before launching the process; restart the API to change models. Each STT service instance lazily loads and reuses one model; startup, health checks, and empty or silent inputs do not load or download one.

Audio is inspected with PyAV, then decoded once with faster-whisper's decoder to 16 kHz mono float audio; there is no gain normalization, denoising or other aggressive preprocessing. Files with no decodable samples are invalid; digitally all-zero audio is treated as no speech before loading a model. Other quiet/noisy silence relies on VAD and the model's no-speech behavior and may still be misrecognized. A nonempty transcript is not a guarantee of accuracy. The existing `/api/voice` multipart request, WAV response and Piper integration are unchanged.

### Prepare on Windows (PowerShell, Python 3.10, 8 GB RAM)

1. In the repository, activate your existing virtual environment and install the project requirements (`python -m pip install -r requirements.txt`). The following **explicit** commands need internet on your PC and download the converted models into the Hugging Face cache; they are not run on application startup:

   ```powershell
   python -c "from faster_whisper.utils import download_model; print(download_model('tiny.en'))"
   python -c "from faster_whisper.utils import download_model; print(download_model('base.en'))"
   ```

   Ensure the returned local model directories contain `tokenizer.json`; incomplete caches fail with `STTModelError` rather than triggering an automatic download. Keep downloaded models outside Git. A configured local converted-model directory may also be supplied as `WHISPER_MODEL_SIZE`.
2. For a direct API trial, set `$env:WHISPER_MODEL_SIZE = 'base.en'`, then start Uvicorn as above. The first non-silent voice request loads the model; subsequent requests reuse it. Set the variable back to `tiny.en` and restart to revert. Optional example: `$env:WHISPER_BEAM_SIZE = '5'; $env:WHISPER_VAD_FILTER = 'false'`. Base may consume more memory and run slower; on an 8 GB PC close unnecessary applications and monitor Task Manager, especially when Ollama and Piper are also running. Actual memory and latency must be measured on that machine.
3. Make several short recordings of the **same English phrases**, including “Tell me about Caprese Salad,” with your normal microphone, accent, background noise, and speaking distance. Include varied menu names and longer/shorter utterances. Obtain consent for recordings, keep them private and outside the repository, and use the same files and manually checked verbatim references for both models. The benchmark processes local files only; it does not record or download audio/models.
4. In a private folder, create a UTF-8 JSON Lines manifest named `recordings.jsonl`, with one object per line. Paths are relative to the manifest (or absolute); point them at your actual audio files. Example:

   ```jsonl
   {"audio":"caprese.wav","text":"Tell me about Caprese Salad"}
   {"audio":"booking.wav","text":"Book a table for two tomorrow at seven"}
   ```

5. In the activated environment, from the repository root, run the models in **separate processes**, with the same manifest and settings. Replace the path below with your manifest path:

   ```powershell
   $env:WHISPER_DEVICE = 'cpu'
   $env:WHISPER_COMPUTE_TYPE = 'int8'
   $env:WHISPER_BEAM_SIZE = '5'
   $env:WHISPER_VAD_FILTER = 'false'
   python -m scripts.benchmark_stt "$env:USERPROFILE\stt-benchmark\recordings.jsonl" --model tiny.en
   python -m scripts.benchmark_stt "$env:USERPROFILE\stt-benchmark\recordings.jsonl" --model base.en
   ```

   Add `--vad` to both benchmark commands to enable VAD; without the flag the benchmark disables VAD regardless of `WHISPER_VAD_FILTER`. This does not change the API's STT configuration.

   To test a vocabulary prompt, compare the same model and recordings with and without the flag below, keeping the other settings identical:

   ```powershell
   python -m scripts.benchmark_stt "$env:USERPROFILE\stt-benchmark\recordings.jsonl" --model base.en --initial-prompt "Caprese Salad, ingredients, vegetarian, reservation, menu"
   ```

   The prompt applies to the warm-up and each measured transcription, not to production API calls. Check individual transcripts for prompted words that were not actually spoken as well as comparing WER and time; an accuracy improvement is not guaranteed.

   Compare per-recording transcripts, word error rates and wall times, plus aggregate WER and mean warm time. WER is `(substitutions + deletions + insertions) / reference words`. Raw WER uses whitespace-tokenized text with case and punctuation retained; normalized WER uses Unicode-normalized, case-folded text with punctuation removed before whitespace tokenization. Each aggregate WER is total errors divided by total reference words for that tokenization. WER can exceed 1. The separately reported first transcription includes model loading and inference, but may benefit from the OS disk cache; it is excluded from WER and warm timings. A warm-up transcribes the first recording, then every recording is measured once. A failure aborts the run rather than counting it as a successful result. Repeat runs under comparable system load and inspect errors as well as averages. Output contains transcript text and filenames: handle it as sensitive. No benchmark results are claimed here.

### Reservation access

New bookings return a one-time `access_code` alongside the reference. Customers
must save it privately: only its SHA-256 digest is stored. `POST
/api/reservations/lookup` accepts JSON `{ "reservation_id": 42, "access_code":
"..." }` and returns only that booking's reference, date, time, party size and
status. `POST /api/reservations/cancel` accepts the same fields plus
`"confirm": true`; it retains the row with cancelled status. An incorrect code
and a nonexistent reference return the same 404 response. Bookings made before
access codes were introduced cannot be recovered online; contact staff. Chat and
voice do not cancel using references alone: use the Reservations page.

`RESERVATION_MAX_ADVANCE_DAYS` optionally sets the number of restaurant-local
calendar days ahead accepted; unset means no maximum. `party_size` may be passed
to `/api/reservations/slots` to filter slots by table availability. Availability
is rechecked when booking. The per-process lookup throttle is not a substitute
for a shared, proxy-aware rate limiter and HTTPS in a network deployment. The
access code is a bearer secret: anyone who obtains it and the reference can act
on the booking. Do not log, share, or store it on a shared device.
