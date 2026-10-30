"""Optional: download a balanced IMDb movie-review dataset from Hugging Face.

First install: python -m pip install datasets
Then run:     python 04_download_imdb.py --per-class 100
Then train:   python 02_train_hybrid_model.py --dataset data/imdb_sentences.csv
"""

import argparse
import csv
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-class", type=int, default=100, help="Number of positive and negative reviews each")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "data" / "imdb_sentences.csv")
    args = parser.parse_args()
    if not 2 <= args.per_class <= 12500:
        parser.error("--per-class must be from 2 to 12500")

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit("Install Hugging Face datasets first: python -m pip install datasets") from exc
    print("Downloading IMDb training split from Hugging Face (internet required)...")
    dataset = load_dataset("stanfordnlp/imdb", split="train")
    dataset = dataset.shuffle(seed=42)
    counts = {0: 0, 1: 0}
    rows = []
    for record in dataset:
        label = int(record["label"])
        if label in counts and counts[label] < args.per_class:
            rows.append((record["text"], label))
            counts[label] += 1
        if all(n >= args.per_class for n in counts.values()):
            break
    if any(n < args.per_class for n in counts.values()):
        raise RuntimeError(f"Not enough examples in source data: {counts}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["text", "label"])
        writer.writerows(rows)
    print(f"Saved {len(rows)} reviews to {args.output.resolve()}")
    print(f"Next: python 02_train_hybrid_model.py --dataset {args.output} --epochs 10")
    print("Note: Dataset contains movie reviews. The GPT-2 encoder truncates each text to 64 tokens in hybrid_model.py.")


if __name__ == "__main__":
    main()
