#!/usr/bin/env python3
"""Generate spelling bee audio files from a CSV using the ElevenLabs API.

CSV format (two columns only, no header required):
    filename,text

Examples:
    Word_abacus,abacus
    Definition_abacus,A counting frame used for arithmetic.
    Sentence_abacus,She used an abacus to solve the problem.

This script:
1. Reads the CSV.
2. Sends each text value to ElevenLabs TTS.
3. Saves the resulting MP3 as <output_dir>/<filename>.mp3
4. Generates/refreshes manifest.json and manifest.js for the local study page.

Environment variable:
    ELEVENLABS_API_KEY

Usage:
    python generate_audio.py words.csv --voice-id VOICE_ID
    python generate_audio.py words.csv --voice-id VOICE_ID --output-dir ./audio
    python generate_audio.py --manifest-only --output-dir ./audio
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path
from typing import List, Tuple

import requests

API_URL_TEMPLATE = "https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
DEFAULT_MODEL_ID = "eleven_multilingual_v2"
DEFAULT_OUTPUT_FORMAT = "mp3_44100_128"
DEFAULT_LANGUAGE_CODE = "en"
DEFAULT_SPEED = 0.85
DEFAULT_STABILITY = 0.80
DEFAULT_SIMILARITY_BOOST = 1.00
DEFAULT_STYLE = 0.00
DEFAULT_USE_SPEAKER_BOOST = False
VALID_PREFIXES = ("Word_", "Definition_", "Sentence_")


def sanitize_stem(value: str) -> str:
    value = value.strip()
    value = re.sub(r'[\\/:*?"<>|]+', "_", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value


def read_rows(csv_path: Path) -> List[Tuple[str, str]]:
    rows: List[Tuple[str, str]] = []

    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        for i, row in enumerate(reader, start=1):
            if not row or all(not cell.strip() for cell in row):
                continue
            if len(row) < 2:
                raise ValueError(f"Row {i} must have at least 2 columns.")

            filename = sanitize_stem(row[0])
            text = row[1].strip()

            if not filename:
                raise ValueError(f"Row {i} has an empty filename.")
            if not text:
                raise ValueError(f"Row {i} has empty text.")

            rows.append((filename, text))

    if not rows:
        raise ValueError("CSV contains no usable rows.")

    return rows


def synthesize_to_mp3(
    api_key: str,
    voice_id: str,
    text: str,
    destination: Path,
    *,
    model_id: str,
    output_format: str,
    language_code: str | None,
    speed: float,
    stability: float,
    similarity_boost: float,
    style: float,
    use_speaker_boost: bool,
    timeout: int = 120,
) -> None:
    url = API_URL_TEMPLATE.format(voice_id=voice_id)
    params = {"output_format": output_format}
    payload = {
        "text": text,
        "model_id": model_id,
        "voice_settings": {
            "speed": speed,
            "stability": stability,
            "similarity_boost": similarity_boost,
            "style": style,
            "use_speaker_boost": use_speaker_boost,
        },
    }
    if language_code:
        payload["language_code"] = language_code
    headers = {
        "xi-api-key": api_key,
        "Content-Type": "application/json",
        "Accept": "audio/mpeg",
    }

    response = requests.post(
        url,
        params=params,
        headers=headers,
        json=payload,
        timeout=timeout,
    )

    if response.status_code != 200:
        try:
            details = response.json()
        except Exception:
            details = response.text
        raise RuntimeError(f"ElevenLabs error {response.status_code}: {details}")

    destination.write_bytes(response.content)


def derive_word_key(stem: str) -> tuple[str | None, str | None]:
    for prefix in VALID_PREFIXES:
        if stem.startswith(prefix):
            return prefix[:-1].lower(), stem[len(prefix):]
    return None, None


def build_manifest_list(audio_dir: Path) -> list[dict]:
    grouped: dict[str, dict] = {}

    for mp3_file in sorted(audio_dir.glob("*.mp3")):
        kind, word = derive_word_key(mp3_file.stem)
        if not kind or not word:
            continue

        item = grouped.setdefault(
            word,
            {
                "word": word,
                "wordAudio": None,
                "definitionAudio": None,
                "sentenceAudio": None,
            },
        )

        rel = f"./audio/{mp3_file.name}"
        if kind == "word":
            item["wordAudio"] = rel
        elif kind == "definition":
            item["definitionAudio"] = rel
        elif kind == "sentence":
            item["sentenceAudio"] = rel

    manifest = list(grouped.values())
    manifest.sort(key=lambda x: x["word"].lower())
    return manifest


def write_manifests(audio_dir: Path) -> tuple[Path, Path]:
    manifest = build_manifest_list(audio_dir)

    manifest_json_path = audio_dir / "manifest.json"
    manifest_js_path = audio_dir / "manifest.js"

    manifest_json_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    manifest_js_path.write_text(
        "window.AUDIO_MANIFEST = "
        + json.dumps(manifest, indent=2, ensure_ascii=False)
        + ";\n",
        encoding="utf-8",
    )

    return manifest_json_path, manifest_js_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate spelling bee MP3s from CSV using ElevenLabs."
    )
    parser.add_argument(
        "csv_path",
        type=Path,
        nargs="?",
        help="Path to the 2-column CSV file.",
    )
    parser.add_argument(
        "--voice-id",
        help="ElevenLabs voice ID to use.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./audio"),
        help="Output folder for MP3 files.",
    )
    parser.add_argument(
        "--model-id",
        default=DEFAULT_MODEL_ID,
        help=f"ElevenLabs model_id. Default: {DEFAULT_MODEL_ID}",
    )
    parser.add_argument(
        "--output-format",
        default=DEFAULT_OUTPUT_FORMAT,
        help=f"ElevenLabs output_format. Default: {DEFAULT_OUTPUT_FORMAT}",
    )
    parser.add_argument(
        "--language-code",
        default=DEFAULT_LANGUAGE_CODE,
        help=f"Language override for ElevenLabs. Default: {DEFAULT_LANGUAGE_CODE}",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=DEFAULT_SPEED,
        help=f"Voice speed. Default: {DEFAULT_SPEED}",
    )
    parser.add_argument(
        "--stability",
        type=float,
        default=DEFAULT_STABILITY,
        help=f"Voice stability. Default: {DEFAULT_STABILITY}",
    )
    parser.add_argument(
        "--similarity-boost",
        type=float,
        default=DEFAULT_SIMILARITY_BOOST,
        help=f"Voice similarity boost. Default: {DEFAULT_SIMILARITY_BOOST}",
    )
    parser.add_argument(
        "--style",
        type=float,
        default=DEFAULT_STYLE,
        help=f"Voice style exaggeration. Default: {DEFAULT_STYLE}",
    )
    parser.add_argument(
        "--speaker-boost",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_USE_SPEAKER_BOOST,
        help=f"Enable ElevenLabs speaker boost. Default: {DEFAULT_USE_SPEAKER_BOOST}",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip files that already exist in the output folder.",
    )
    parser.add_argument(
        "--manifest-only",
        action="store_true",
        help="Do not call ElevenLabs. Only regenerate manifest files from existing MP3 files.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.manifest_only:
        manifest_json_path, manifest_js_path = write_manifests(output_dir)
        print(f"Generated {manifest_json_path}")
        print(f"Generated {manifest_js_path}")
        return 0

    if args.csv_path is None:
        print("csv_path is required unless --manifest-only is used.", file=sys.stderr)
        return 1

    if not args.voice_id:
        print("--voice-id is required unless --manifest-only is used.", file=sys.stderr)
        return 1

    api_key = os.getenv("ELEVENLABS_API_KEY")
    if not api_key:
        print("Missing ELEVENLABS_API_KEY environment variable.", file=sys.stderr)
        return 1

    try:
        rows = read_rows(args.csv_path)
    except Exception as exc:
        print(f"CSV error: {exc}", file=sys.stderr)
        return 1

    total = len(rows)
    failures = 0

    for index, (filename, text) in enumerate(rows, start=1):
        destination = output_dir / f"{filename}.mp3"

        if args.skip_existing and destination.exists():
            print(f"[{index}/{total}] Skipping existing file: {destination.name}")
            continue

        print(f"[{index}/{total}] Generating: {destination.name}")
        try:
            synthesize_to_mp3(
                api_key=api_key,
                voice_id=args.voice_id,
                text=text,
                destination=destination,
                model_id=args.model_id,
                output_format=args.output_format,
                language_code=args.language_code,
                speed=args.speed,
                stability=args.stability,
                similarity_boost=args.similarity_boost,
                style=args.style,
                use_speaker_boost=args.speaker_boost,
            )
        except Exception as exc:
            failures += 1
            print(f"Failed for {filename}: {exc}", file=sys.stderr)

    manifest_json_path, manifest_js_path = write_manifests(output_dir)
    print(f"Generated {manifest_json_path}")
    print(f"Generated {manifest_js_path}")

    if failures:
        print(f"Completed with {failures} failure(s).", file=sys.stderr)
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
