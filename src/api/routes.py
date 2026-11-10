"""HTTP endpoints for restaurant operations."""

from typing import Annotated, Optional
from urllib.parse import quote
from uuid import UUID

from fastapi import APIRouter, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import Response

from src.api.orchestration import EmptyUpload, UploadTooLarge
from src.api.schemas import ChatRequest, ChatResponse, HealthResponse, MenuResponse, ReservationRequest, ReservationResponse
from src.services.stt_service import EmptyTranscriptionError, InvalidAudioError, STTError
from src.services.tts_service import TTSError

router = APIRouter(prefix="/api")


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


@router.post("/reservations", response_model=ReservationResponse, status_code=201)
def reserve(payload: ReservationRequest, request: Request):
    try:
        return request.app.state.backend.reserve(payload.model_dump())
    except ValueError as exc:
        if str(exc).startswith("No tables available"):
            raise HTTPException(status_code=409, detail="No table is available for that request") from exc
        raise HTTPException(status_code=422, detail="Invalid reservation details") from exc


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
