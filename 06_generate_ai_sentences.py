"""Generate a CSV of AI-drafted sentiment examples using local Ollama.

Windows VS Code example:
  ollama pull llama3.2:1b
  python 06_generate_ai_sentences.py --per-class 50

Only the standard library is required. Review generated examples and labels
before training. Do not evaluate on this synthetic dataset alone.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = ROOT / "data" / "ai_sentences.csv"
TOPICS = [
    "movies and TV", "online shopping", "restaurants and food", "customer service",
    "mobile apps", "learning and classes", "public transport", "travel",
    "games", "workplaces", "books", "sports", "phones", "hotels", "music",
    "health and fitness apps", "social media", "delivery services", "concerts",
    "home appliances", "school assignments", "websites", "cafes", "software updates",
]


def read_existing(path: Path) -> list[tuple[str, int]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = csv.DictReader(f)
        if not {"text", "label"}.issubset(rows.fieldnames or []):
            raise ValueError(f"Existing file {path} must have columns text,label")
        result = []
        for row in rows:
            text = (row.get("text") or "").strip()
            label = (row.get("label") or "").strip()
            if text and label in {"0", "1"}:
                result.append((text, int(label)))
        return result


def save_csv(path: Path, rows: list[tuple[str, int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    with temp.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["text", "label"])
        writer.writerows(rows)
    temp.replace(path)


def request_sentences(model: str, label: int, number: int, topics: list[str]) -> list[str]:
    sentiment = "positive" if label == 1 else "negative"
    instruction = (
        f"Create {number} UNIQUE natural-sounding English sentences expressing a clearly {sentiment} opinion. "
        f"Use varied topics and wording, including {', '.join(topics)}. "
        "Include some simple and some subtle sentences. Do not repeat phrases, templates, or sentence starters. "
        "Keep each sentence between 5 and 25 words. Do not put the label in the text. "
        'Respond with a JSON object ONLY, like {"sentences": ["First example", "Second example"]}. '
        "Return exactly the requested number of strings."
    )
    payload = {
        "model": model,
        "prompt": instruction,
        "stream": False,
        "format": "json",
        "options": {"temperature": 0.85},
    }
    req = Request(
        "http://127.0.0.1:11434/api/generate",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(req, timeout=240) as resp:
            response = json.load(resp)
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:500]
        raise RuntimeError(f"Ollama HTTP error {exc.code}: {detail}") from exc
    except URLError as exc:
        raise RuntimeError(
            "Could not connect to Ollama at localhost:11434. Start Ollama and run "
            f"'ollama pull {model}' in PowerShell. Error: {exc.reason}"
        ) from exc
    body = response.get("response", "")
    try:
        result = json.loads(body)
    except (ValueError, TypeError) as exc:
        raise RuntimeError(f"Ollama did not return valid JSON: {str(body)[:250]}") from exc
    if not isinstance(result, dict) or not isinstance(result.get("sentences"), list):
        raise RuntimeError("Ollama's JSON must contain a 'sentences' list")
    return [t.strip() for t in result["sentences"] if isinstance(t, str)]


def main():
    parser = argparse.ArgumentParser(description="Generate AI sentiment training data using local Ollama")
    parser.add_argument("--model", default="llama3.2:1b", help="Name from 'ollama list'")
    parser.add_argument("--per-class", type=int, default=50, help="Goal for each label")
    parser.add_argument("--batch-size", type=int, default=10, help="Examples requested per Ollama call")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.per_class < 5 or not 1 <= args.batch_size <= 25:
        parser.error("--per-class must be at least 5; --batch-size must be between 1 and 25")
    random.seed(args.seed)
    destination = args.output.resolve()
    rows = read_existing(destination)
    seen = {" ".join(text.lower().split()) for text, _ in rows}
    counts = {0: sum(label == 0 for _, label in rows), 1: sum(label == 1 for _, label in rows)}
    print(f"Output file: {destination}")
    print(f"Existing: positive={counts[1]}, negative={counts[0]}")
    for label in (1, 0):
        attempts = 0
        max_attempts = max(8, ((args.per_class + args.batch_size - 1) // args.batch_size) * 5)
        while counts[label] < args.per_class and attempts < max_attempts:
            attempts += 1
            need = min(args.batch_size, args.per_class - counts[label])
            topics = random.sample(TOPICS, k=4)
            try:
                sentences = request_sentences(args.model, label, need, topics)
            except (RuntimeError, TimeoutError, ValueError) as exc:
                print(f"Generation problem: {exc}", file=sys.stderr)
                if attempts >= 3 and not rows:
                    sys.exit(1)
                continue
            added = 0
            for sentence in sentences:
                sentence = " ".join(sentence.split()).strip(' "')
                normalized = sentence.casefold()
                if not (5 <= len(sentence.split()) <= 30) or normalized in seen:
                    continue
                rows.append((sentence, label))
                seen.add(normalized)
                counts[label] += 1
                added += 1
                if counts[label] >= args.per_class:
                    break
            save_csv(destination, rows)
            print(f"{'Positive' if label == 1 else 'Negative'}: {counts[label]}/{args.per_class} (+{added} new)")
        if counts[label] < args.per_class:
            print(f"Warning: Only got {counts[label]} examples for class {label}. Review model output.")
    random.shuffle(rows)
    save_csv(destination, rows)
    print(f"Saved {len(rows)} examples to {destination}")
    print("IMPORTANT: Open the CSV, remove incorrect/duplicate examples, and verify labels before training.")
    print("Evaluate accuracy on untouched REAL labeled sentences, not only AI-generated text.")


if __name__ == "__main__":
    main()
