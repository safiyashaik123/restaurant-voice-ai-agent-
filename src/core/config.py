"""Configuration module for the Restaurant Voice AI Agent.

Provides centralized configuration with environment variable overrides.
Designed to be dependency-free so it can be loaded in any Python environment.
"""

from dataclasses import dataclass, field
import os
from pathlib import Path


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
    default_reservation_duration_minutes: int = int(
        os.environ.get("RESERVATION_DURATION_MINUTES", "90")
    )
    opening_time: str = os.environ.get("RESTAURANT_OPENING_TIME", "11:00")
    closing_time: str = os.environ.get("RESTAURANT_CLOSING_TIME", "22:00")
    
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


# Singleton instance
config = AppConfig()
