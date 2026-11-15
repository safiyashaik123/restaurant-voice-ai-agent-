"""Orchestrate existing services for the HTTP layer without changing business logic."""

from dataclasses import asdict
from pathlib import Path
from threading import RLock
from typing import BinaryIO, Optional
import sqlite3
import tempfile
import re
import logging
from time import perf_counter

logger = logging.getLogger(__name__)

from src.database.connection import create_connection, get_db
from src.database.crud import create_reservation, get_menu_categories, get_menu_items, verified_reservation, cancel_reservation, find_available_tables
from src.services.agent_service import AgentService
from src.services.stt_service import STTService
from src.services.tts_service import TTSService


class UploadTooLarge(Exception):
    """The uploaded audio exceeds the configured limit."""


class EmptyUpload(Exception):
    """The uploaded audio contains no bytes."""


class APIBackend:
    """Long-lived service instances and short-lived database/file operations."""

    def __init__(self, db_path: Path, max_upload_bytes: int,
                 agent: Optional[AgentService] = None,
                 stt: Optional[STTService] = None,
                 tts: Optional[TTSService] = None) -> None:
        self.db_path = Path(db_path)
        self.max_upload_bytes = max_upload_bytes
        self.agent = agent if agent is not None else AgentService(db_path=str(db_path))
        self.stt = stt if stt is not None else STTService()
        self.tts = tts if tts is not None else TTSService()
        # AgentService's session dictionary and its booking path are not thread-safe.
        # This protects calls within this process, not across multiple workers.
        self._agent_and_booking_lock = RLock()

    def database_healthy(self) -> bool:
        if not self.db_path.is_file():
            return False
        conn = None
        try:
            conn = create_connection(self.db_path)
            rows = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
            return {"categories", "menu_items", "tables", "reservations"}.issubset({row[0] for row in rows})
        except (sqlite3.Error, OSError):
            return False
        finally:
            if conn is not None:
                conn.close()

    def chat(self, message: str, session_id: str) -> dict:
        with self._agent_and_booking_lock:
            result = self.agent.handle_message(message, session_id=session_id)
        return {"session_id": session_id, **asdict(result)}

    def menu(self, category: Optional[str]) -> dict:
        with get_db(self.db_path) as conn:
            return {"categories": get_menu_categories(conn), "items": get_menu_items(conn, category_name=category)}

    def _menu_vocabulary_prompt(self) -> Optional[str]:
        """Build a bounded prompt from available live menu items, if accessible."""
        if not self.db_path.is_file():
            return None
        try:
            with get_db(self.db_path) as conn:
                items = get_menu_items(conn, available_only=True)
        except (sqlite3.Error, OSError):
            return None
        prefix = "Restaurant menu items: "
        names = []
        length = len(prefix)
        for item in items:
            name = " ".join(item["name"].split())
            extra = len(name) + (2 if names else 0)
            if name and length + extra <= 400:
                names.append(name)
                length += extra
        return prefix + ", ".join(names) if names else None

    def reserve(self, values: dict) -> dict:
        # Exit the transaction context (and commit) before returning confirmation.
        with self._agent_and_booking_lock:
            with get_db(self.db_path, immediate=True) as conn:
                reservation = create_reservation(conn, **values)
        return reservation

    def available_slots(self, day, party_size: int) -> list[str]:
        from src.core.reservation_policy import reservation_slots
        with get_db(self.db_path) as conn:
            return [slot for slot in reservation_slots(day) if find_available_tables(conn, party_size, day.isoformat(), slot)]

    def lookup_reservation(self, reference: int, code: str, *, cancel: bool = False):
        with get_db(self.db_path, immediate=cancel) as conn:
            booking = verified_reservation(conn, reference, code)
            if booking and cancel and booking['status'] == 'confirmed':
                cancel_reservation(conn, reference)
                booking['status'] = 'cancelled'
            return booking

    def voice(self, upload: BinaryIO, session_id: str) -> tuple[bytes, str, str]:
        with tempfile.TemporaryDirectory(prefix="restaurant-voice-") as directory:
            input_path = Path(directory) / "input.audio"
            output_path = Path(directory) / "output.wav"
            size = 0
            with input_path.open("wb") as target:
                while True:
                    chunk = upload.read(min(65536, self.max_upload_bytes + 1 - size))
                    if not chunk:
                        break
                    size += len(chunk)
                    if size > self.max_upload_bytes:
                        raise UploadTooLarge()
                    target.write(chunk)
            if not size:
                raise EmptyUpload()
            prompt = self._menu_vocabulary_prompt()
            started = perf_counter()
            transcript = self.stt.transcribe(input_path, initial_prompt=prompt)
            logger.debug('Speech transcription: %.1f ms', (perf_counter() - started) * 1000)
            started = perf_counter()
            result = self.chat(transcript, session_id)
            logger.debug('Voice chat turn: %.1f ms', (perf_counter() - started) * 1000)
            started = perf_counter()
            spoken_text = re.sub(r"₹\s*(\d+(?:\.\d+)?)", r"\1 rupees", result["text"])
            self.tts.synthesize(spoken_text, output_path)
            logger.debug('Speech synthesis: %.1f ms', (perf_counter() - started) * 1000)
            return output_path.read_bytes(), transcript, result["text"]
