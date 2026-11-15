# Savour frontend

A React + TypeScript + Vite interface for the local Restaurant Voice AI Agent.

## Run locally

1. Install Node.js (20.19+ or 22.12+), then install dependencies:
   ```sh
   cd frontend
   npm ci
   ```
2. In another terminal, start the FastAPI backend from the project root as described in `../API.md`:
   ```sh
   uvicorn src.api.main:app --host 127.0.0.1 --port 8000
   ```
   Use an existing configured database or initialize a new test database only when appropriate. Do not reset an existing database. Configure local Ollama, Whisper and Piper models for real voice interaction.
3. Start Vite from `frontend/`:
   ```sh
   npm run dev
   ```
   Open the address printed by Vite (normally `http://localhost:5173`).

By default the frontend calls relative `/api` URLs, and Vite proxies them to `http://127.0.0.1:8000` on the **server**. Set `VITE_API_PROXY_TARGET` in a local `.env` file to change the dev proxy target, or set `VITE_API_BASE_URL` to an absolute backend origin if not using the proxy. For a direct cross-origin backend, configure its `API_ALLOWED_ORIGINS` to include the frontend origin. A production deployment must route `/api` to FastAPI or build with an appropriate `VITE_API_BASE_URL`; Vite's proxy applies only to the development server.

The backend keeps conversations in memory. The frontend stores a UUID in per-tab `sessionStorage` so it survives reloads in that tab but is not shared as a global customer session. It is not authentication. Recording requires microphone permission and a secure context (HTTPS or localhost) and browser `MediaRecorder` support. Recordings are sent to `/api/voice`; responses are played as WAV and can be replayed with the audio controls. The backend returns only available menu items. Prices are displayed with the Indian rupee symbol using the numeric values returned by the database; verify the restaurant’s actual prices before deployment.

The reservation form uses the API and shared backend reservation policy; confirmation is shown only after the booking succeeds. Save the private access code shown with the reference to look up or explicitly cancel a new booking. Older bookings without an access code require staff assistance. The assistant’s allergen answers should be confirmed with restaurant staff.

## Checks

```sh
npm test
npm run build
```

The tests mock HTTP and do not require local models. Real microphone, playback and end-to-end model testing must be performed in a supported browser with the backend and models running.
