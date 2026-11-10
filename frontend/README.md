# Savour frontend

A React + TypeScript + Vite interface for the local Restaurant Voice AI Agent.

## Run locally

1. Install Node.js (20.19+ or 22.12+), then install dependencies:
   ```sh
   cd frontend
   npm ci
   ```
2. In another terminal, initialize and start the FastAPI backend as described in `../API.md`:
   ```sh
   python -m src.database.init_db
   uvicorn src.api.main:app --host 127.0.0.1 --port 8000
   ```
   Run those commands from the project root and use your configured local Ollama, Whisper and Piper models for real voice interaction.
3. Start Vite from `frontend/`:
   ```sh
   npm run dev
   ```
   Open the address printed by Vite (normally `http://localhost:5173`).

By default the frontend calls relative `/api` URLs, and Vite proxies them to `http://127.0.0.1:8000` on the **server**. Set `VITE_API_PROXY_TARGET` in a local `.env` file to change the dev proxy target, or set `VITE_API_BASE_URL` to an absolute backend origin if not using the proxy. For a direct cross-origin backend, configure its `API_ALLOWED_ORIGINS` to include the frontend origin. A production deployment must route `/api` to FastAPI or build with an appropriate `VITE_API_BASE_URL`; Vite's proxy applies only to the development server.

The backend keeps conversations in memory. The frontend stores a UUID in per-tab `sessionStorage` so it survives reloads in that tab but is not shared as a global customer session. It is not authentication. Recording requires microphone permission and a secure context (HTTPS or localhost) and browser `MediaRecorder` support. Recordings are sent to `/api/voice`; responses are played as WAV and can be replayed with the audio controls. The backend returns only available menu items. Prices follow the dollar-denominated examples in the existing backend documentation; check the restaurant's actual currency before deployment.

The reservation form sends requests to the existing CRUD endpoint. As noted in `../API.md`, its date and phone validation differs from the conversational agent; confirmation is shown only after the API returns success. The assistant's allergen answers should be confirmed with restaurant staff.

## Checks

```sh
npm test
npm run build
```

The tests mock HTTP and do not require local models. Real microphone, playback and end-to-end model testing must be performed in a supported browser with the backend and models running.
