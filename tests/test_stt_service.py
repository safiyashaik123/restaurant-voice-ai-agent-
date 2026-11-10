"""Unit tests for local STT without loading or downloading a Whisper model."""

from dataclasses import fields
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from av.error import InvalidDataError

from src.core.config import AppConfig, config
from src.services.stt_service import (
    AudioFileNotFoundError,
    EmptyTranscriptionError,
    InvalidAudioError,
    STTModelError,
    STTService,
    TranscriptionError,
)


class TestSTTService(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.audio_path = Path(self.temp_dir.name) / "audio.wav"
        self.audio_path.write_bytes(b"mock audio")
        self.model_dir = Path(self.temp_dir.name) / "model"
        self.model_dir.mkdir()
        (self.model_dir / "tokenizer.json").write_text("{}", encoding="utf-8")
        model_lookup = patch("src.services.stt_service.download_model", return_value=str(self.model_dir))
        self.mock_model_lookup = model_lookup.start()
        self.addCleanup(model_lookup.stop)
        audio_open = patch("src.services.stt_service.av.open")
        self.mock_audio_open = audio_open.start()
        self.mock_audio_open.return_value.__enter__.return_value.streams.audio = [object()]
        self.addCleanup(audio_open.stop)
        audio_decode = patch("src.services.stt_service.decode_audio", return_value=np.array([0.1], dtype=np.float32))
        self.mock_audio_decode = audio_decode.start()
        self.addCleanup(audio_decode.stop)

    def test_config_preserves_existing_positional_field_order(self):
        self.assertEqual(
            [item.name for item in fields(AppConfig)],
            ["project_name", "base_dir", "data_dir", "models_dir", "db_path",
             "default_reservation_duration_minutes", "opening_time", "closing_time",
             "ollama_base_url", "ollama_model", "ollama_timeout_seconds",
             "whisper_model_size", "whisper_compute_type", "piper_model_name",
             "allergen_fallback_message", "general_error_message", "whisper_device"],
        )

    @patch("src.services.stt_service.WhisperModel")
    def test_defaults_and_model_reuse(self, model_class):
        with patch.object(config, "whisper_model_size", "tiny.en"), \
             patch.object(config, "whisper_device", "cpu"), \
             patch.object(config, "whisper_compute_type", "int8"):
            service = STTService()
        model = model_class.return_value
        model.transcribe.side_effect = [
            (iter([SimpleNamespace(text=" Hello "), SimpleNamespace(text=" world! ")]), None),
            (iter([SimpleNamespace(text=" Again ")]), None),
        ]
        model_class.assert_not_called()
        self.assertEqual(service.transcribe(self.audio_path), "Hello world!")
        self.assertEqual(service.transcribe(str(self.audio_path)), "Again")
        self.mock_model_lookup.assert_called_once_with("tiny.en", local_files_only=True)
        model_class.assert_called_once_with(str(self.model_dir), device="cpu", compute_type="int8", local_files_only=True)
        self.assertEqual(model.transcribe.call_count, 2)
        args, kwargs = model.transcribe.call_args
        np.testing.assert_array_equal(args[0], np.array([0.1], dtype=np.float32))
        self.assertEqual(kwargs, {"language": "en", "task": "transcribe", "beam_size": 5, "vad_filter": False})
        self.assertEqual(self.mock_audio_decode.call_count, 2)
        self.mock_audio_decode.assert_called_with(str(self.audio_path), sampling_rate=16000)

    @patch("src.services.stt_service.WhisperModel")
    def test_explicit_configuration(self, model_class):
        model_class.return_value.transcribe.return_value = (iter([SimpleNamespace(text="hello")]), None)
        service = STTService(model_name="custom-model", device="cpu", compute_type="float32")
        self.assertEqual(service.transcribe(self.audio_path), "hello")
        self.mock_model_lookup.assert_called_once_with("custom-model", local_files_only=True)
        model_class.assert_called_once_with(str(self.model_dir), device="cpu", compute_type="float32", local_files_only=True)

    @patch("src.services.stt_service.WhisperModel")
    def test_missing_file_does_not_load_model(self, model_class):
        with self.assertRaisesRegex(AudioFileNotFoundError, "does not exist"):
            STTService().transcribe(self.audio_path.parent / "missing.wav")
        model_class.assert_not_called()

    @patch("src.services.stt_service.WhisperModel")
    def test_directory_is_invalid_audio(self, model_class):
        with self.assertRaisesRegex(InvalidAudioError, "not a file"):
            STTService().transcribe(self.audio_path.parent)
        model_class.assert_not_called()

    @patch("src.services.stt_service.WhisperModel")
    def test_media_without_audio_stream_is_invalid(self, model_class):
        self.mock_audio_open.return_value.__enter__.return_value.streams.audio = []
        with self.assertRaisesRegex(InvalidAudioError, "no audio stream"):
            STTService().transcribe(self.audio_path)
        self.mock_audio_open.assert_called_once_with(str(self.audio_path), mode="r", metadata_errors="ignore")
        model_class.assert_not_called()

    @patch("src.services.stt_service.WhisperModel")
    def test_inspection_decode_error_is_invalid_audio(self, model_class):
        self.mock_audio_open.side_effect = InvalidDataError(22, "invalid audio")
        with self.assertRaises(InvalidAudioError):
            STTService().transcribe(self.audio_path)
        model_class.assert_not_called()

    @patch("src.services.stt_service.WhisperModel")
    def test_unrelated_index_error_remains_transcription_error(self, model_class):
        model_class.return_value.transcribe.side_effect = IndexError("inference indexing bug")
        with self.assertRaisesRegex(TranscriptionError, "inference indexing bug") as ctx:
            STTService().transcribe(self.audio_path)
        self.assertIsInstance(ctx.exception.__cause__, IndexError)

    @patch("src.services.stt_service.WhisperModel")
    def test_model_load_failure_is_clear_and_chained(self, model_class):
        model_class.side_effect = RuntimeError("model not cached")
        with self.assertRaisesRegex(STTModelError, "model not cached") as ctx:
            STTService().transcribe(self.audio_path)
        self.assertIsInstance(ctx.exception.__cause__, RuntimeError)

    @patch("src.services.stt_service.WhisperModel")
    def test_missing_local_tokenizer_does_not_fall_back_to_download(self, model_class):
        (self.model_dir / "tokenizer.json").unlink()
        with self.assertRaisesRegex(STTModelError, "tokenizer.json"):
            STTService().transcribe(self.audio_path)
        model_class.assert_not_called()

    @patch("src.services.stt_service.WhisperModel")
    def test_local_model_directory_does_not_use_cache_lookup(self, model_class):
        model_class.return_value.transcribe.return_value = (iter([SimpleNamespace(text="hello")]), None)
        self.assertEqual(STTService(model_name=str(self.model_dir)).transcribe(self.audio_path), "hello")
        self.mock_model_lookup.assert_not_called()

    @patch("src.services.stt_service.WhisperModel")
    def test_invalid_audio_is_clear_and_chained(self, model_class):
        model_class.return_value.transcribe.side_effect = InvalidDataError(22, "invalid audio")
        with self.assertRaisesRegex(InvalidAudioError, "decode audio") as ctx:
            STTService().transcribe(self.audio_path)
        self.assertIsInstance(ctx.exception.__cause__, InvalidDataError)

    @patch("src.services.stt_service.WhisperModel")
    def test_empty_or_whitespace_transcription(self, model_class):
        model = model_class.return_value
        for segments in ([], [SimpleNamespace(text="  "), SimpleNamespace(text="\n")]):
            with self.subTest(segments=segments):
                model.transcribe.return_value = (iter(segments), None)
                with self.assertRaisesRegex(EmptyTranscriptionError, "No speech"):
                    STTService().transcribe(self.audio_path)

    @patch("src.services.stt_service.WhisperModel")
    def test_transcription_failure_is_clear_and_chained(self, model_class):
        model_class.return_value.transcribe.side_effect = RuntimeError("inference failed")
        with self.assertRaisesRegex(TranscriptionError, "inference failed") as ctx:
            STTService().transcribe(self.audio_path)
        self.assertIsInstance(ctx.exception.__cause__, RuntimeError)

    @patch("src.services.stt_service.WhisperModel")
    def test_lazy_segment_iteration_failure_is_wrapped(self, model_class):
        def failing_segments():
            yield SimpleNamespace(text="partial")
            raise RuntimeError("inference failed during iteration")

        model_class.return_value.transcribe.return_value = (failing_segments(), None)
        with self.assertRaisesRegex(TranscriptionError, "during iteration"):
            STTService().transcribe(self.audio_path)

    @patch("src.services.stt_service.WhisperModel")
    def test_lazy_audio_decoding_failure_is_wrapped(self, model_class):
        def failing_segments():
            yield SimpleNamespace(text="partial")
            raise InvalidDataError(22, "invalid audio")

        model_class.return_value.transcribe.return_value = (failing_segments(), None)
        with self.assertRaises(InvalidAudioError):
            STTService().transcribe(self.audio_path)


if __name__ == "__main__":
    unittest.main()
