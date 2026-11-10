"""Local file-based text-to-speech using Piper."""

import json
import os
from pathlib import Path
import re
import tempfile
from threading import Lock
from typing import Optional, Union
import wave

from piper import PiperVoice
from piper.config import PiperConfig

from src.core.config import config


class TTSError(Exception):
    """Base exception for text-to-speech service errors."""


class EmptyTextError(TTSError):
    """The supplied text is empty or contains only whitespace."""


class TTSModelNotFoundError(TTSError):
    """A local voice model or its configuration file is missing."""


class TTSConfigError(TTSError):
    """The voice name or voice configuration is invalid."""


class TTSModelError(TTSError):
    """The local voice model could not be loaded."""


class TTSSynthesisError(TTSError):
    """Audio synthesis or WAV file output failed."""


class TTSService:
    """Synthesize WAV files with a lazily loaded, reusable local Piper voice.

    The configured model directory must contain <voice>.onnx and
    <voice>.onnx.json. Model files are never downloaded by this service.
    """

    def __init__(
        self,
        model_dir: Optional[Union[str, os.PathLike]] = None,
        voice_name: Optional[str] = None,
    ) -> None:
        try:
            self.model_dir = Path(model_dir) if model_dir is not None else config.piper_model_dir
        except (TypeError, ValueError) as exc:
            raise TTSConfigError(f"Invalid Piper model directory: {model_dir!r}") from exc
        self.voice_name = voice_name if voice_name is not None else config.piper_model_name
        self._voice: Optional[PiperVoice] = None
        self._voice_lock = Lock()

    def _get_voice(self) -> PiperVoice:
        with self._voice_lock:
            if self._voice is not None:
                return self._voice
            if not isinstance(self.voice_name, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", self.voice_name):
                raise TTSConfigError("Voice name must contain only letters, numbers, underscores, and hyphens")

            model_path = self.model_dir / f"{self.voice_name}.onnx"
            config_path = self.model_dir / f"{self.voice_name}.onnx.json"
            for path in (model_path, config_path):
                if not path.is_file():
                    raise TTSModelNotFoundError(f"Local Piper voice file not found: {path}")

            try:
                with config_path.open("r", encoding="utf-8") as file:
                    voice_config = json.load(file)
                if not isinstance(voice_config, dict):
                    raise ValueError("Voice configuration must be a JSON object")
                parsed = PiperConfig.from_dict(voice_config)
                if (not isinstance(parsed.sample_rate, int) or isinstance(parsed.sample_rate, bool)
                        or parsed.sample_rate <= 0):
                    raise ValueError("audio.sample_rate must be a positive integer")
                if not isinstance(parsed.phoneme_id_map, dict):
                    raise ValueError("phoneme_id_map must be an object")
            except (OSError, UnicodeError, ValueError, TypeError, KeyError, AttributeError) as exc:
                raise TTSConfigError(f"Invalid Piper voice configuration {config_path}: {exc}") from exc

            try:
                self._voice = PiperVoice.load(model_path, config_path=config_path, use_cuda=False)
            except Exception as exc:
                raise TTSModelError(f"Could not load local Piper voice {model_path}: {exc}") from exc
            return self._voice

    def synthesize(self, text: str, output_path: Union[str, os.PathLike]) -> Path:
        """Synthesize text to a WAV file and return its path.

        The output directory must exist. A temporary file in that directory is
        replaced into place only on success, preserving an existing output if
        synthesis fails.
        """
        if not isinstance(text, str) or not text.strip():
            raise EmptyTextError("Text to synthesize must not be empty")
        try:
            path = Path(output_path)
        except (TypeError, ValueError) as exc:
            raise TTSSynthesisError(f"Invalid WAV output path: {output_path!r}") from exc
        if path.suffix.lower() != ".wav":
            raise TTSSynthesisError("Output path must have a .wav extension")

        voice = self._get_voice()
        temp_path = None
        try:
            with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".wav", delete=False) as temp_file:
                temp_path = Path(temp_file.name)
            wav_file = wave.open(str(temp_path), "wb")
            try:
                voice.synthesize_wav(text.strip(), wav_file)
                if wav_file.getnframes() == 0:
                    raise ValueError("Piper produced no audio frames")
            except Exception:
                try:
                    wav_file.close()
                except Exception:
                    pass  # Do not mask the original synthesis failure.
                raise
            else:
                wav_file.close()
            os.replace(temp_path, path)
        except Exception as exc:
            raise TTSSynthesisError(f"Could not synthesize WAV file {path}: {exc}") from exc
        finally:
            if temp_path is not None:
                try:
                    temp_path.unlink(missing_ok=True)
                except OSError:
                    pass  # Preserve the original synthesis error if cleanup fails.
        return path
