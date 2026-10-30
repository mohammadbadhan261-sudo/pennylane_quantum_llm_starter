"""Download balanced IMDb sentiment examples without the `datasets` package.

Usage:
    python 04_download_imdb_no_datasets.py --per-class 100
    python 02_train_hybrid_model.py --dataset data/imdb_sentences.csv --epochs 10

Requires only the Python standard library. Uses the documented
Hugging Face Dataset Viewer `/rows` API (maximum 100 rows per request).
"""

import argparse
import csv
import json
import random
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

DATASET = "stanfordnlp/imdb"
CONFIG = "plain_text"
SPLIT = "train"
API = "https://datasets-server.huggingface.co/rows"
ROWS_IN_SPLIT = 25000
PAGE_SIZE = 100


def fetch_rows(offset: int):
    query = urlencode({
        "dataset": DATASET,
        "config": CONFIG,
        "split": SPLIT,
        "offset": offset,
        "length": PAGE_SIZE,
    })
    req = Request(f"{API}?{query}", headers={"User-Agent": "PennyLane-educational-IMDb-downloader/1.0"})
    for attempt in range(3):
        try:
            with urlopen(req, timeout=40) as response:
                body = json.load(response)
            return body["rows"]
        except (HTTPError, URLError, TimeoutError, ValueError, KeyError) as exc:
            if attempt == 2:
                raise RuntimeError(
                    "IMDb download failed. Check your internet connection, "
                    "and whether the Hugging Face Dataset Viewer is available."
                ) from exc
            time.sleep(2 * (attempt + 1))
    return []


def main():
    parser = argparse.ArgumentParser(description="Download balanced IMDb examples as CSV")
    parser.add_argument("--per-class", type=int, default=100)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "data" / "imdb_sentences.csv")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if not 2 <= args.per_class <= 2000:
        parser.error("--per-class must be between 2 and 2000")

    # The IMDb training split contains 25,000 reviews. Randomized pages allow
    # sampling across the split without assuming its sentiment label order.
    rng = random.Random(args.seed)
    offsets = list(range(0, ROWS_IN_SPLIT, PAGE_SIZE))
    rng.shuffle(offsets)
    counts = {0: 0, 1: 0}
    rows = []
    print(f"Downloading {args.per_class} reviews for each sentiment class...")
    for offset in offsets:
        page = fetch_rows(offset)
        rng.shuffle(page)
        for item in page:
            record = item.get("row", {})
            label = record.get("label")
            content = record.get("text")
            if label in counts and isinstance(content, str) and content.strip() and counts[label] < args.per_class:
                rows.append((content, label))
                counts[label] += 1
        print(f"  Positive: {counts[1]}/{args.per_class}; negative: {counts[0]}/{args.per_class}")
        if all(n == args.per_class for n in counts.values()):
            break

    if any(n < args.per_class for n in counts.values()):
        raise RuntimeError(f"Could not collect enough examples: {counts}. Try a smaller --per-class value.")

    rng.shuffle(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["text", "label"])
        writer.writerows(rows)
    print(f"Saved {len(rows)} balanced examples to: {args.output.resolve()}")
    print("Now run: python 02_train_hybrid_model.py --dataset data/imdb_sentences.csv --epochs 10")


if __name__ == "__main__":
    main()
