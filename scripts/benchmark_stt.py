"""Compare local Whisper models on a JSONL manifest, one model per process."""
import argparse
import json
from pathlib import Path
import re
import sys
import time
import unicodedata

from src.services.stt_service import STTError, STTService


def words(text: str) -> list[str]:
    """Case-fold and remove punctuation before whitespace-tokenized WER."""
    text = unicodedata.normalize("NFKC", text).casefold()
    return re.sub(r"[^\w\s]", " ", text).split()


def word_errors(reference: list[str], hypothesis: list[str]) -> int:
    """Levenshtein distance in words (substitutions, insertions, deletions)."""
    row = list(range(len(hypothesis) + 1))
    for index, expected in enumerate(reference, start=1):
        next_row = [index]
        for j, actual in enumerate(hypothesis, start=1):
            next_row.append(min(next_row[-1] + 1, row[j] + 1, row[j - 1] + (expected != actual)))
        row = next_row
    return row[-1]


def load_manifest(path: Path) -> list[tuple[Path, str]]:
    entries = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), start=1):
        if not line.strip():
            continue
        entry = json.loads(line)
        if not isinstance(entry, dict) or not isinstance(entry.get("audio"), str) or not isinstance(entry.get("text"), str):
            raise ValueError(f"Line {line_number}: expected string 'audio' and 'text' fields")
        if not entry["audio"].strip() or not words(entry["text"]):
            raise ValueError(f"Line {line_number}: audio and reference text must be nonempty")
        audio = Path(entry["audio"])
        entries.append((audio if audio.is_absolute() else path.parent / audio, entry["text"]))
    if not entries:
        raise ValueError("Manifest contains no recordings")
    return entries


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path, help="UTF-8 JSONL with audio and text fields")
    parser.add_argument("--model", required=True, choices=("tiny.en", "base.en"))
    parser.add_argument("--vad", action="store_true", help="Enable voice activity detection")
    parser.add_argument("--initial-prompt", help="Optional vocabulary prompt for transcription")
    args = parser.parse_args()
    try:
        entries = load_manifest(args.manifest)
        service = STTService(model_name=args.model, vad_filter=args.vad)
        print(f"Model: {args.model}; device: {service.device}; compute: {service.compute_type}; "
              f"beam: {service.beam_size}; VAD: {service.vad_filter}; language: en")
        # Record cold load plus first inference separately; the actual comparison is warm.
        start = time.perf_counter()
        service.transcribe(entries[0][0], initial_prompt=args.initial_prompt)
        print(f"Cold first transcription: {time.perf_counter() - start:.3f} s (excluded from WER and warm timing)")
        total_errors = total_words = 0
        raw_errors = raw_words = 0
        total_seconds = 0.0
        for path, reference in entries:
            start = time.perf_counter()
            hypothesis = service.transcribe(path, initial_prompt=args.initial_prompt)
            elapsed = time.perf_counter() - start
            expected_words = words(reference)
            errors = word_errors(expected_words, words(hypothesis))
            raw_reference = reference.split()
            raw_error = word_errors(raw_reference, hypothesis.split())
            raw_errors += raw_error
            raw_words += len(raw_reference)
            total_errors += errors
            total_words += len(expected_words)
            total_seconds += elapsed
            print(f"\n{path}: {elapsed:.3f} s, Raw WER {raw_error / len(raw_reference):.3f} ({raw_error}/{len(raw_reference)}), "
                  f"Normalized WER {errors / len(expected_words):.3f} ({errors}/{len(expected_words)})")
            print(f"  Reference:  {reference}\n  Transcript: {hypothesis}")
        print(f"\nAggregate Raw WER: {raw_errors / raw_words:.3f} ({raw_errors}/{raw_words}); "
              f"Aggregate Normalized WER: {total_errors / total_words:.3f} ({total_errors}/{total_words}); "
              f"warm total: {total_seconds:.3f} s; warm mean: {total_seconds / len(entries):.3f} s")
        return 0
    except (STTError, OSError, ValueError) as exc:
        print(f"Benchmark failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
