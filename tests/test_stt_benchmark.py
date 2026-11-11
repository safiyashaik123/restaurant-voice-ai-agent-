"""Benchmark metrics and manifest behavior without speech models."""
import json
from unittest.mock import patch

import pytest

from scripts.benchmark_stt import load_manifest, main, word_errors, words


def test_word_error_rate_counts_insertions_deletions_and_substitutions():
    reference = words('Tell me about Caprese Salad.')
    assert words('tell ME about caprese salad!') == reference
    assert word_errors(reference, words('tell me about caesar salad')) == 1
    assert word_errors(reference, words('tell me caprese salad please')) == 2
    assert word_errors(reference, words('tell me about caprese salad please')) == 1


def test_manifest_validation_and_relative_audio_paths(tmp_path):
    path = tmp_path / 'recordings.jsonl'
    path.write_text(json.dumps({'audio': 'clip.wav', 'text': 'Hello there'}) + '\n', encoding='utf-8')
    assert load_manifest(path) == [(tmp_path / 'clip.wav', 'Hello there')]
    path.write_text('{"audio":"clip.wav","text":"!!!"}\n', encoding='utf-8')
    with pytest.raises(ValueError, match='nonempty'):
        load_manifest(path)


@pytest.mark.parametrize('vad_args, expected_vad', [([], False), (['--vad'], True)])
def test_benchmark_warmup_aggregate_and_model_selection(tmp_path, capsys, monkeypatch, vad_args, expected_vad):
    # The CLI flag, not the production environment setting, controls this run.
    monkeypatch.setenv('WHISPER_VAD_FILTER', 'true')
    path = tmp_path / 'recordings.jsonl'
    path.write_text('\n'.join(json.dumps(item) for item in [
        {'audio': 'a.wav', 'text': 'tell me about caprese salad'},
        {'audio': 'b.wav', 'text': 'hello world'},
    ]), encoding='utf-8')
    with patch('scripts.benchmark_stt.STTService') as cls, patch('sys.argv', ['benchmark_stt', str(path), '--model', 'base.en', *vad_args]):
        service = cls.return_value
        service.vad_filter = expected_vad
        service.transcribe.side_effect = ['warm up', 'tell me about caesar salad', 'hello world']
        assert main() == 0
    cls.assert_called_once_with(model_name='base.en', vad_filter=expected_vad)
    assert service.transcribe.call_count == 3
    output = capsys.readouterr().out
    assert 'Cold first transcription' in output and 'Aggregate WER: 0.143 (1/7)' in output
