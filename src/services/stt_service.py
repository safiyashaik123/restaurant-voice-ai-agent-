"""Local, file-based speech transcription using faster-whisper."""

from pathlib import Path
from threading import Lock
from typing import Optional, Union
import os

import av
import numpy as np
from av.error import FFmpegError
from faster_whisper import WhisperModel
from faster_whisper.audio import decode_audio
from faster_whisper.utils import download_model

from src.core.config import config


class STTError(Exception):
    """Base exception for speech-to-text service errors."""


class AudioFileNotFoundError(STTError):
    """The requested audio file does not exist."""


class InvalidAudioError(STTError):
    """The audio input is not a decodable audio file."""


class EmptyTranscriptionError(STTError):
    """No recognizable text was produced from the audio."""


class STTModelError(STTError):
    """The locally cached speech recognition model could not be loaded."""


class TranscriptionError(STTError):
    """The speech recognition operation failed."""


class STTService:
    """Transcribe audio files locally, reusing one model per service instance.

    The model is loaded on the first valid transcription request. Only locally
    available model files are used; this service never downloads a model.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        compute_type: Optional[str] = None,
        beam_size: Optional[int] = None,
        vad_filter: Optional[bool] = None,
    ) -> None:
        self.model_name = model_name if model_name is not None else config.whisper_model_size
        self.device = device if device is not None else config.whisper_device
        self.compute_type = compute_type if compute_type is not None else config.whisper_compute_type
        self.beam_size = beam_size if beam_size is not None else config.whisper_beam_size
        self.vad_filter = vad_filter if vad_filter is not None else config.whisper_vad_filter
        if isinstance(self.beam_size, bool) or not isinstance(self.beam_size, int) or self.beam_size < 1:
            raise ValueError("beam_size must be a positive integer")
        if not isinstance(self.vad_filter, bool):
            raise ValueError("vad_filter must be a boolean")
        self._model: Optional[WhisperModel] = None
        self._model_lock = Lock()

    def _get_model(self) -> WhisperModel:
        """Initialize the model once, even if first requests arrive concurrently."""
        with self._model_lock:
            if self._model is None:
                try:
                    model_path = self.model_name
                    if not os.path.isdir(model_path):
                        model_path = download_model(model_path, local_files_only=True)
                    # faster-whisper otherwise falls back to downloading a tokenizer.
                    if not (Path(model_path) / "tokenizer.json").is_file():
                        raise FileNotFoundError("Local model is missing tokenizer.json")
                    self._model = WhisperModel(
                        model_path,
                        device=self.device,
                        compute_type=self.compute_type,
                        local_files_only=True,
                    )
                except Exception as exc:
                    raise STTModelError(
                        f"Could not load local Whisper model {self.model_name!r}: {exc}"
                    ) from exc
            return self._model

    def transcribe(self, audio_path: Union[str, os.PathLike]) -> str:
        """Return recognized text from an audio file, or raise an STTError."""
        path = Path(audio_path)
        if not path.exists():
            raise AudioFileNotFoundError(f"Audio file does not exist: {path}")
        if not path.is_file():
            raise InvalidAudioError(f"Audio path is not a file: {path}")

        try:
            with av.open(str(path), mode="r", metadata_errors="ignore") as container:
                if not container.streams.audio:
                    raise InvalidAudioError(f"Media file contains no audio stream: {path}")
        except InvalidAudioError:
            raise
        except (FFmpegError, EOFError) as exc:
            raise InvalidAudioError(f"Could not decode audio file {path}: {exc}") from exc
        except Exception as exc:
            raise TranscriptionError(f"Could not inspect audio file {path}: {exc}") from exc

        # Use faster-whisper's decoder once: it downmixes channels and resamples
        # to the model's expected 16 kHz mono PCM without extra processing.
        try:
            audio = decode_audio(str(path), sampling_rate=16000)
        except (FFmpegError, EOFError, ValueError) as exc:
            raise InvalidAudioError(f"Could not decode audio file {path}: {exc}") from exc
        except Exception as exc:
            raise TranscriptionError(f"Could not decode audio file {path}: {exc}") from exc
        if audio.size == 0:
            raise InvalidAudioError(f"Audio file contains no decodable samples: {path}")
        if not np.any(audio):
            raise EmptyTranscriptionError(f"No speech was recognized in silent audio file: {path}")

        model = self._get_model()
        try:
            segments, _ = model.transcribe(audio, language="en", task="transcribe",
                                           beam_size=self.beam_size, vad_filter=self.vad_filter)
            # faster-whisper returns a lazy iterator: errors may occur here too.
            text = " ".join(segment.text.strip() for segment in segments if segment.text.strip())
        except (FFmpegError, EOFError) as exc:
            raise InvalidAudioError(f"Could not decode audio file {path}: {exc}") from exc
        except Exception as exc:
            raise TranscriptionError(f"Could not transcribe audio file {path}: {exc}") from exc

        if not text:
            raise EmptyTranscriptionError(f"No speech was recognized in audio file: {path}")
        return text
