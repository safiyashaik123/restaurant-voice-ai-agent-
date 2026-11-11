"""STT configuration and real audio decoding without model downloads."""
import math
import struct
import wave
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from av.error import InvalidDataError

from src.core.config import AppConfig
from src.services.stt_service import EmptyTranscriptionError, InvalidAudioError, STTService


def write_wav(path, rate=48000, channels=2, silent=False, frames=4800):
    with wave.open(str(path), 'wb') as wav:
        wav.setnchannels(channels)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        for index in range(frames):
            value = 0 if silent else int(10000 * math.sin(2 * math.pi * 440 * index / rate))
            wav.writeframesraw(struct.pack('<' + 'h' * channels, *([value] * channels)))


def test_settings_defaults_environment_and_validation(monkeypatch):
    monkeypatch.delenv('WHISPER_BEAM_SIZE', raising=False)
    monkeypatch.delenv('WHISPER_VAD_FILTER', raising=False)
    assert AppConfig().whisper_beam_size == 5 and AppConfig().whisper_vad_filter is False
    monkeypatch.setenv('WHISPER_BEAM_SIZE', '3')
    monkeypatch.setenv('WHISPER_VAD_FILTER', 'true')
    service = STTService(model_name='base.en')
    assert service.model_name == 'base.en' and service.beam_size == 3 and service.vad_filter is True
    assert service._model is None
    monkeypatch.setenv('WHISPER_BEAM_SIZE', '0')
    with pytest.raises(ValueError, match='positive'):
        STTService()
    monkeypatch.setenv('WHISPER_BEAM_SIZE', 'wide')
    with pytest.raises(ValueError, match='WHISPER_BEAM_SIZE'):
        STTService()
    monkeypatch.setenv('WHISPER_BEAM_SIZE', '5')
    monkeypatch.setenv('WHISPER_VAD_FILTER', 'sometimes')
    with pytest.raises(ValueError, match='true or false'):
        STTService()
    with pytest.raises(ValueError, match='beam_size'):
        STTService(beam_size=0, vad_filter=False)
    with pytest.raises(ValueError, match='vad_filter'):
        STTService(beam_size=5, vad_filter='true')


@pytest.mark.parametrize('rate,channels,frames', [(48000, 2, 4800), (8000, 1, 800)])
def test_real_audio_is_decoded_once_to_16khz_mono(tmp_path, rate, channels, frames):
    from types import SimpleNamespace
    path = tmp_path / 'test.wav'
    write_wav(path, rate=rate, channels=channels, frames=frames)
    model = MagicMock()
    model.transcribe.return_value = (iter([SimpleNamespace(text='Tell me about Caprese Salad')]), None)
    service = STTService(model_name='base.en', beam_size=3, vad_filter=True)
    with patch.object(service, '_get_model', return_value=model) as get_model:
        assert service.transcribe(path) == 'Tell me about Caprese Salad'
    get_model.assert_called_once()
    audio = model.transcribe.call_args.args[0]
    assert audio.dtype == np.float32 and audio.ndim == 1
    assert len(audio) == 1600 and 0.1 < float(np.max(np.abs(audio))) < 0.5
    assert model.transcribe.call_args.kwargs == {'language': 'en', 'task': 'transcribe', 'beam_size': 3, 'vad_filter': True, 'initial_prompt': None}


def test_silent_and_zero_sample_audio_do_not_load_model(tmp_path):
    silent, empty = tmp_path / 'silent.wav', tmp_path / 'empty.wav'
    write_wav(silent, silent=True)
    write_wav(empty, frames=0)
    service = STTService()
    with patch.object(service, '_get_model') as model:
        with pytest.raises(EmptyTranscriptionError, match='No speech'):
            service.transcribe(silent)
        with pytest.raises(InvalidAudioError, match='no decodable samples'):
            service.transcribe(empty)
    model.assert_not_called()


def test_decoder_failure_does_not_load_model(tmp_path):
    path = tmp_path / 'audio.wav'
    write_wav(path)
    service = STTService()
    with patch('src.services.stt_service.decode_audio', side_effect=InvalidDataError(22, 'bad audio')), patch.object(service, '_get_model') as model:
        with pytest.raises(InvalidAudioError, match='decode audio'):
            service.transcribe(path)
    model.assert_not_called()


def test_base_model_loads_locally_only_and_reuses_one_instance(tmp_path):
    from types import SimpleNamespace
    path = tmp_path / 'sound.wav'
    write_wav(path)
    model_dir = tmp_path / 'model'
    model_dir.mkdir()
    (model_dir / 'tokenizer.json').write_text('{}', encoding='utf-8')
    with patch('src.services.stt_service.download_model', return_value=str(model_dir)) as lookup, patch('src.services.stt_service.WhisperModel') as cls:
        cls.return_value.transcribe.side_effect = [(iter([SimpleNamespace(text='first')]), None), (iter([SimpleNamespace(text='second')]), None)]
        service = STTService(model_name='base.en')
        lookup.assert_not_called()
        cls.assert_not_called()
        assert service.transcribe(path) == 'first'
        assert service.transcribe(path) == 'second'
        lookup.assert_called_once_with('base.en', local_files_only=True)
        cls.assert_called_once_with(str(model_dir), device=service.device, compute_type=service.compute_type, local_files_only=True)


def test_concurrent_first_use_loads_only_one_model(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event
    model_dir = tmp_path / 'model'
    model_dir.mkdir()
    (model_dir / 'tokenizer.json').write_text('{}', encoding='utf-8')
    started, release = Event(), Event()

    def load_model(*args, **kwargs):
        started.set()
        assert release.wait(timeout=5)
        return object()

    with patch('src.services.stt_service.WhisperModel', side_effect=load_model) as cls:
        service = STTService(model_name=str(model_dir))
        with ThreadPoolExecutor(max_workers=2) as pool:
            first = pool.submit(service._get_model)
            assert started.wait(timeout=5)
            second = pool.submit(service._get_model)
            release.set()
            assert first.result(timeout=5) is second.result(timeout=5)
        cls.assert_called_once()
