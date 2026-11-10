"""Focused Piper TTS unit tests; no voice model or network is required."""

import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import wave

from src.core.config import AppConfig, config
from src.services.tts_service import (
    EmptyTextError,
    TTSConfigError,
    TTSModelError,
    TTSModelNotFoundError,
    TTSService,
    TTSSynthesisError,
)


class TestTTSService(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.directory = Path(self.temp_dir.name)
        self.model_path = self.directory / "en_US-lessac-medium.onnx"
        self.config_path = self.directory / "en_US-lessac-medium.onnx.json"
        self.output_path = self.directory / "output.wav"
        self.model_path.write_bytes(b"mock model")
        self.valid_config = {
            "num_symbols": 10,
            "num_speakers": 1,
            "audio": {"sample_rate": 22050},
            "espeak": {"voice": "en-us"},
            "phoneme_id_map": {"a": [1]},
        }
        self.config_path.write_text(json.dumps(self.valid_config), encoding="utf-8")

    @staticmethod
    def write_mock_audio(text, wav_file):
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(22050)
        wav_file.writeframes(b"\x00\x00\x01\x00")

    @patch("src.services.tts_service.PiperVoice.load")
    def test_synthesizes_valid_wav_and_reuses_model(self, load):
        load.return_value.synthesize_wav.side_effect = self.write_mock_audio
        service = TTSService(model_dir=self.directory)
        load.assert_not_called()
        self.assertEqual(service.synthesize(" Hello there ", self.output_path), self.output_path)
        self.assertEqual(service.synthesize("Again", str(self.output_path)), self.output_path)
        load.assert_called_once_with(self.model_path, config_path=self.config_path, use_cuda=False)
        self.assertEqual(load.return_value.synthesize_wav.call_count, 2)
        self.assertEqual(load.return_value.synthesize_wav.call_args.args[0], "Again")
        with wave.open(str(self.output_path), "rb") as wav_file:
            self.assertEqual((wav_file.getnchannels(), wav_file.getsampwidth(), wav_file.getframerate()), (1, 2, 22050))
            self.assertEqual(wav_file.getnframes(), 2)

    def test_config_defaults_and_environment_override(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(AppConfig().piper_model_dir, config.models_dir / "piper")
        with patch.dict(os.environ, {"PIPER_MODEL_DIR": str(self.directory)}):
            with patch.object(config, "piper_model_name", "en_US-lessac-medium"):
                service = TTSService()
            self.assertEqual(service.model_dir, self.directory)
            self.assertEqual(service.voice_name, "en_US-lessac-medium")

    @patch("src.services.tts_service.PiperVoice.load")
    def test_empty_text_rejected_before_loading(self, load):
        for value in ("", " \n ", None):
            with self.subTest(value=value):
                with self.assertRaises(EmptyTextError):
                    TTSService(model_dir=self.directory).synthesize(value, self.output_path)
        load.assert_not_called()
        self.assertFalse(self.output_path.exists())

    @patch("src.services.tts_service.PiperVoice.load")
    def test_missing_model_and_config_files(self, load):
        for path in (self.model_path, self.config_path):
            with self.subTest(path=path):
                contents = path.read_bytes()
                path.unlink()
                with self.assertRaisesRegex(TTSModelNotFoundError, path.name):
                    TTSService(model_dir=self.directory).synthesize("Hello", self.output_path)
                path.write_bytes(contents)
        load.assert_not_called()

    @patch("src.services.tts_service.PiperVoice.load")
    def test_invalid_voice_names(self, load):
        for name in ("", "../voice", "voice.onnx", "a\\b"):
            with self.subTest(name=name):
                with self.assertRaises(TTSConfigError):
                    TTSService(model_dir=self.directory, voice_name=name).synthesize("Hello", self.output_path)
        load.assert_not_called()

    def test_invalid_model_directory(self):
        with self.assertRaises(TTSConfigError):
            TTSService(model_dir=42)

    @patch("src.services.tts_service.PiperVoice.load")
    def test_invalid_voice_configuration(self, load):
        for content in ("not json", "[]", "{}", json.dumps({**self.valid_config, "audio": {"sample_rate": -1}})):
            with self.subTest(content=content):
                self.config_path.write_text(content, encoding="utf-8")
                with self.assertRaises(TTSConfigError):
                    TTSService(model_dir=self.directory).synthesize("Hello", self.output_path)
        load.assert_not_called()

    @patch("src.services.tts_service.PiperVoice.load")
    def test_model_load_failure(self, load):
        load.side_effect = RuntimeError("bad onnx")
        with self.assertRaisesRegex(TTSModelError, "bad onnx") as ctx:
            TTSService(model_dir=self.directory).synthesize("Hello", self.output_path)
        self.assertIsInstance(ctx.exception.__cause__, RuntimeError)

    @patch("src.services.tts_service.PiperVoice.load")
    def test_synthesis_failure_preserves_existing_output_and_cleans_temp(self, load):
        self.output_path.write_bytes(b"existing")
        load.return_value.synthesize_wav.side_effect = RuntimeError("synthesis failed")
        with self.assertRaisesRegex(TTSSynthesisError, "synthesis failed") as ctx:
            TTSService(model_dir=self.directory).synthesize("Hello", self.output_path)
        self.assertIsInstance(ctx.exception.__cause__, RuntimeError)
        self.assertEqual(self.output_path.read_bytes(), b"existing")
        self.assertEqual(list(self.directory.glob("*.wav")), [self.output_path])

    @patch("src.services.tts_service.PiperVoice.load")
    def test_no_audio_frames_is_a_synthesis_failure(self, load):
        def no_audio(text, wav_file):
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(22050)
        load.return_value.synthesize_wav.side_effect = no_audio
        with self.assertRaisesRegex(TTSSynthesisError, "no audio frames"):
            TTSService(model_dir=self.directory).synthesize("Hello", self.output_path)
        self.assertFalse(self.output_path.exists())

    @patch("src.services.tts_service.PiperVoice.load")
    def test_output_path_errors(self, load):
        service = TTSService(model_dir=self.directory)
        with self.assertRaisesRegex(TTSSynthesisError, "wav extension"):
            service.synthesize("Hello", self.directory / "output.mp3")
        with self.assertRaises(TTSSynthesisError):
            service.synthesize("Hello", None)
        load.assert_not_called()
        with self.assertRaises(TTSSynthesisError):
            service.synthesize("Hello", self.directory / "missing" / "output.wav")


if __name__ == "__main__":
    unittest.main()
