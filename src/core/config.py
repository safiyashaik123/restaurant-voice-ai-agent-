"""Configuration module for the Restaurant Voice AI Agent.

Provides centralized configuration with environment variable overrides.
Designed to be dependency-free so it can be loaded in any Python environment.
"""

from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import ClassVar


BASE_DIR = Path(__file__).resolve().parent.parent.parent


@dataclass
class AppConfig:
    project_name: str = "Restaurant Voice AI Agent"
    base_dir: Path = BASE_DIR
    data_dir: Path = field(default_factory=lambda: BASE_DIR / "data")
    models_dir: Path = field(default_factory=lambda: BASE_DIR / "models")
    
    # Database settings
    db_path: Path = field(default_factory=lambda: BASE_DIR / "data" / "restaurant.db")
    
    # Reservation configuration (configurable per restaurant policy)
    restaurant_timezone: ClassVar[str] = "Asia/Kolkata"
    default_reservation_duration_minutes: int = int(
        os.environ.get("RESERVATION_DURATION_MINUTES", "90")
    )
    opening_time: str = os.environ.get("RESTAURANT_OPENING_TIME", "10:00")
    closing_time: str = os.environ.get("RESTAURANT_CLOSING_TIME", "23:00")
    
    # Local LLM settings (Ollama)
    ollama_base_url: str = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
    ollama_model: str = os.environ.get("OLLAMA_MODEL", "qwen2.5:1.5b")
    ollama_timeout_seconds: float = float(os.environ.get("OLLAMA_TIMEOUT_SECONDS", "30.0"))
    
    # STT & TTS settings
    whisper_model_size: str = os.environ.get("WHISPER_MODEL_SIZE", "tiny.en")
    whisper_compute_type: str = os.environ.get("WHISPER_COMPUTE_TYPE", "int8")
    piper_model_name: str = os.environ.get("PIPER_MODEL_NAME", "en_US-lessac-medium")
    
    # Safety fallbacks
    allergen_fallback_message: str = (
        "I do not have verified allergen information for that item. "
        "For your safety, please speak directly with our restaurant staff before ordering."
    )
    general_error_message: str = (
        "I am currently experiencing technical difficulties. "
        "Please let a team member know so we can assist you right away."
    )

    whisper_device: str = os.environ.get("WHISPER_DEVICE", "cpu")

    @property
    def reservation_max_advance_days(self) -> int | None:
        value = os.environ.get("RESERVATION_MAX_ADVANCE_DAYS", "").strip()
        if not value:
            return None
        days = int(value)
        if days < 0:
            raise ValueError("RESERVATION_MAX_ADVANCE_DAYS must be non-negative")
        return days

    @property
    def whisper_beam_size(self) -> int:
        """Beam width for English transcription (faster-whisper default: 5)."""
        try:
            value = int(os.environ.get("WHISPER_BEAM_SIZE", "5"))
        except ValueError as exc:
            raise ValueError("WHISPER_BEAM_SIZE must be a positive integer") from exc
        if value < 1:
            raise ValueError("WHISPER_BEAM_SIZE must be a positive integer")
        return value

    @property
    def whisper_vad_filter(self) -> bool:
        """Optional voice activity filtering; disabled by default for short speech."""
        value = os.environ.get("WHISPER_VAD_FILTER", "false").strip().lower()
        if value in ("true", "1", "yes", "on"):
            return True
        if value in ("false", "0", "no", "off"):
            return False
        raise ValueError("WHISPER_VAD_FILTER must be true or false")

    @property
    def api_allowed_origins(self) -> list[str]:
        """Explicit browser origins allowed to access the API."""
        return [origin.strip() for origin in os.environ.get(
            "API_ALLOWED_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173"
        ).split(",") if origin.strip() and origin.strip() != "*"]

    @property
    def api_max_upload_bytes(self) -> int:
        """Maximum audio upload size, defaulting to 10 MiB."""
        value = int(os.environ.get("API_MAX_UPLOAD_BYTES", str(10 * 1024 * 1024)))
        if value <= 0:
            raise ValueError("API_MAX_UPLOAD_BYTES must be positive")
        return value

    @property
    def piper_model_dir(self) -> Path:
        """Directory containing local Piper ONNX voices and their JSON configs."""
        return Path(os.environ.get("PIPER_MODEL_DIR", str(self.models_dir / "piper")))


# Singleton instance
config = AppConfig()
