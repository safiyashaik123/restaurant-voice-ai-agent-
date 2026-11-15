"""HTTP endpoints for restaurant operations."""

from typing import Annotated, Optional
from collections import defaultdict, deque
from threading import Lock
from time import monotonic
from urllib.parse import quote
from uuid import UUID

from fastapi import APIRouter, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import Response

from src.core.reservation_policy import PHONE_ERROR, SLOT_ERROR, parse_reservation_date, reservation_slots
from src.api.orchestration import EmptyUpload, UploadTooLarge
from src.api.schemas import ChatRequest, ChatResponse, HealthResponse, MenuResponse, ReservationRequest, ReservationResponse, ReservationAccessRequest, ReservationCancelRequest
from src.services.stt_service import EmptyTranscriptionError, InvalidAudioError, STTError
from src.services.tts_service import TTSError

router = APIRouter(prefix="/api")
# Local per-process throttle. A deployment exposed to the Internet also needs
# a shared, proxy-aware rate limiter and HTTPS at its trusted edge.
_attempts = defaultdict(deque)
_attempt_lock = Lock()


def limit_access(request: Request):
    now = monotonic()
    host = request.client.host if request.client else 'unknown'
    with _attempt_lock:
        for key in list(_attempts):
            while _attempts[key] and _attempts[key][0] <= now - 60:
                _attempts[key].popleft()
            if not _attempts[key]:
                del _attempts[key]
        if len(_attempts[host]) >= 20:
            raise HTTPException(status_code=429, detail="Too many attempts. Please try again later.")
        _attempts[host].append(now)



@router.get("/health", response_model=HealthResponse)
def health(request: Request):
    if not request.app.state.backend.database_healthy():
        raise HTTPException(status_code=503, detail="Database unavailable")
    return {"status": "ok", "database": "ok"}


@router.post("/chat", response_model=ChatResponse)
def chat(payload: ChatRequest, request: Request):
    return request.app.state.backend.chat(payload.message, str(payload.session_id))


@router.get("/menu", response_model=MenuResponse)
def menu(request: Request, category: Optional[str] = Query(default=None, max_length=100)):
    return request.app.state.backend.menu(category)


@router.get("/reservations/slots")
def slots(request: Request, date: str = Query(...), party_size: Optional[int] = Query(default=None, ge=1)):
    try:
        day = parse_reservation_date(date)
        return {"date": date, "slots": request.app.state.backend.available_slots(day, party_size) if party_size else reservation_slots(day)}
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.post("/reservations", response_model=ReservationResponse, status_code=201)
def reserve(payload: ReservationRequest, request: Request, response: Response):
    response.headers["Cache-Control"] = "no-store"
    try:
        return request.app.state.backend.reserve(payload.model_dump())
    except ValueError as exc:
        if str(exc).startswith("No tables available"):
            raise HTTPException(status_code=409, detail="No table is available for that request") from exc
        if str(exc).startswith("A reservation for this phone"):
            raise HTTPException(status_code=409, detail="A matching reservation already exists. Please contact the restaurant if you need help.") from exc
        detail = str(exc) if str(exc) in (PHONE_ERROR, SLOT_ERROR) else "Invalid reservation details"
        raise HTTPException(status_code=422, detail=detail) from exc


@router.post("/reservations/lookup")
def lookup(payload: ReservationAccessRequest, request: Request, response: Response):
    response.headers["Cache-Control"] = "no-store"
    limit_access(request)
    result = request.app.state.backend.lookup_reservation(payload.reservation_id, payload.access_code)
    if result is None:
        raise HTTPException(status_code=404, detail="Reference or access code is incorrect; contact the restaurant for older bookings.")
    return result


@router.post("/reservations/cancel")
def cancel(payload: ReservationCancelRequest, request: Request, response: Response):
    response.headers["Cache-Control"] = "no-store"
    limit_access(request)
    if payload.confirm is not True:
        raise HTTPException(status_code=422, detail="Cancellation must be confirmed.")
    result = request.app.state.backend.lookup_reservation(payload.reservation_id, payload.access_code, cancel=True)
    if result is None:
        raise HTTPException(status_code=404, detail="Reference or access code is incorrect; contact the restaurant for older bookings.")
    return result


@router.post("/voice", response_class=Response, responses={200: {"content": {"audio/wav": {"schema": {"type": "string", "format": "binary"}}}}})
def voice(request: Request, session_id: Annotated[UUID, Form()], file: Annotated[UploadFile, File()]):
    try:
        if not file.filename:
            raise HTTPException(status_code=400, detail="An audio file is required")
        allowed = {"audio/wav", "audio/x-wav", "audio/wave", "audio/mpeg", "audio/mp4", "audio/ogg", "audio/webm", "audio/flac", "audio/x-flac", "application/octet-stream"}
        if (file.content_type or "").lower() not in allowed:
            raise HTTPException(status_code=415, detail="Unsupported audio content type")
        audio, transcript, response_text = request.app.state.backend.voice(file.file, str(session_id))
        return Response(content=audio, media_type="audio/wav", headers={
            "X-Transcript": quote(transcript, safe=""),
            "X-Response-Text": quote(response_text, safe=""),
            "X-Session-ID": str(session_id),
            "Cache-Control": "no-store",
        })
    except EmptyUpload as exc:
        raise HTTPException(status_code=400, detail="Audio file is empty") from exc
    except UploadTooLarge as exc:
        raise HTTPException(status_code=413, detail="Audio file exceeds upload limit") from exc
    except (InvalidAudioError, EmptyTranscriptionError) as exc:
        raise HTTPException(status_code=422, detail="Audio could not be understood") from exc
    except STTError as exc:
        raise HTTPException(status_code=503, detail="Speech recognition unavailable") from exc
    except TTSError as exc:
        raise HTTPException(status_code=503, detail="Speech synthesis unavailable") from exc
    finally:
        file.file.close()
