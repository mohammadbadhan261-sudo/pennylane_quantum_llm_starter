"""Encode a CSV with OpenAI embeddings and save a reusable training/search cache."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import torch
from openai_text_api import EMBED_MODEL, embed_texts

ROOT = Path(__file__).resolve().parent


def fingerprint(texts, labels):
    payload = json.dumps(list(zip(texts, labels)), ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def read_csv(path):
    unique = {}
    with path.open("r", newline="", encoding="utf-8-sig") as fp:
        reader = csv.DictReader(fp)
        if "text" not in (reader.fieldnames or []):
            raise ValueError("CSV requires a column named text (and optional label)")
        for row in reader:
            text = (row.get("text") or "").strip()
            if not text:
                continue
            value = (row.get("label") or "").strip()
            if value not in ("0", "1", ""):
                raise ValueError("Labels must be 0, 1 or blank")
            label = int(value) if value else -1
            key = " ".join(text.casefold().split())
            if key in unique and unique[key][1] != label:
                raise ValueError(f"Conflicting labels for: {text[:50]}")
            unique.setdefault(key, (text, label))
    if not unique:
        raise ValueError("No usable texts in CSV")
    return [r[0] for r in unique.values()], [r[1] for r in unique.values()]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, default=ROOT / "data" / "sentences.csv")
    p.add_argument("--output", type=Path, default=ROOT / "data" / "openai_embeddings.pt")
    p.add_argument("--model", default=EMBED_MODEL)
    p.add_argument("--batch-size", type=int, default=16)
    args = p.parse_args()
    texts, labels = read_csv(args.input)
    print(f"Input rows: {len(texts)}, labeled: {len([x for x in labels if x>=0])}, unlabeled: {labels.count(-1)}")
    print("WARNING: input text will be sent to the OpenAI API; check its rights/privacy first.")
    embeddings = embed_texts(texts, model=args.model, batch_size=args.batch_size)
    args.output.parent.mkdir(exist_ok=True, parents=True)
    torch.save({
        "format_version": 2,
        "model_name": args.model,
        "embedding_dim": embeddings.shape[1],
        "embeddings": embeddings,
        "texts": texts,
        "labels": torch.tensor(labels, dtype=torch.long),
        "sha256": fingerprint(texts, labels),
        "source_csv": str(args.input),
    }, args.output)
    print(f"Saved {tuple(embeddings.shape)} embeddings to {args.output}")
    if -1 in labels:
        print("UNLABELED cache: suitable for similarity search, NOT supervised sentiment training.")

if __name__ == "__main__":
    main()
