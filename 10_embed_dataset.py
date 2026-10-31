"""Create a reusable, local AI-embedded dataset from text CSV.

Input CSV: `text` is required; `label` (0 negative / 1 positive) is optional.
Embeddings: frozen DistilGPT2 masked-mean last-hidden-state vectors (max 64 tokens).

Examples:
 python 10_embed_dataset.py --input data/commoncrawl_raw.csv --output data/commoncrawl_raw_embeddings.pt
 python 10_embed_dataset.py --input data/train_mixed.csv --output data/train_mixed_embeddings.pt

Only use .pt files you created or trust. Git ignores saved caches.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import torch
from hybrid_model import LANGUAGE_MODEL_ID, load_language_model, encode_texts

ROOT = Path(__file__).resolve().parent
POOLING = "distilgpt2_masked_mean_last_hidden_max64_v1"


def digest_rows(texts: list[str], labels: list[int]) -> str:
    canonical = json.dumps(list(zip(texts, labels)), ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def read_input(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Dataset missing: {path}")
    texts, labels, urls, seen = [], [], [], {}
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or "text" not in reader.fieldnames:
            raise ValueError("Input CSV must have a 'text' column; label is optional")
        for n, row in enumerate(reader, 2):
            content = (row.get("text") or "").strip()
            if not content:
                continue
            raw_label = (row.get("label") or "").strip()
            if raw_label not in ("0", "1", ""):
                raise ValueError(f"Line {n}: label must be 0, 1, or empty")
            value = int(raw_label) if raw_label else -1
            key = " ".join(content.casefold().split())
            if key in seen:
                if seen[key] != value:
                    raise ValueError(f"Line {n}: conflicting labels for duplicate text")
                continue
            seen[key] = value
            texts.append(content)
            labels.append(value)
            urls.append((row.get("source_url") or "").strip())
    if not texts:
        raise ValueError("No usable rows found in CSV")
    return texts, labels, urls


def main():
    p = argparse.ArgumentParser(description="Embed a real Common Crawl or combined CSV using frozen DistilGPT2")
    p.add_argument("--input", type=Path, required=True, help="CSV with text and optional label columns")
    p.add_argument("--output", type=Path, default=ROOT / "data" / "text_embeddings.pt")
    p.add_argument("--batch-size", type=int, default=8)
    a = p.parse_args()
    if a.batch_size < 1:
        p.error("batch-size must be positive")
    texts, labels, urls = read_input(a.input)
    print(f"Loaded {len(texts)} distinct texts (negative={labels.count(0)}, positive={labels.count(1)}, unlabeled={labels.count(-1)})")
    print("Loading pretrained DistilGPT2 to build AI embeddings...")
    tokenizer, encoder = load_language_model(LANGUAGE_MODEL_ID)
    vectors = encode_texts(texts, tokenizer, encoder, batch_size=a.batch_size).cpu().contiguous()
    if vectors.ndim != 2 or vectors.shape[0] != len(texts) or not torch.isfinite(vectors).all():
        raise ValueError("Encoder produced an invalid embedding tensor")
    a.output.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "format_version": 1,
        "embeddings": vectors,
        "labels": torch.tensor(labels, dtype=torch.long),
        "texts": texts,
        "source_urls": urls,
        "encoder_name": LANGUAGE_MODEL_ID,
        "pooling": POOLING,
        "source_csv": str(a.input.resolve()),
        "row_sha256": digest_rows(texts, labels),
    }
    torch.save(record, a.output)
    print(f"Saved AI embeddings: {a.output.resolve()}")
    print(f"Embedding shape: {tuple(vectors.shape)}")
    if -1 in labels:
        print("IMPORTANT: This file contains unlabeled text. You may search its embeddings, but sentiment training requires reviewed 0/1 labels.")
    else:
        print("All records are labeled; you can train using --embeddings and --dataset.")


if __name__ == "__main__":
    main()
