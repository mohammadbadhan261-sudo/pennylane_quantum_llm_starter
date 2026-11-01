"""Create a reusable Ollama embedding cache from labeled or unlabeled CSV text."""
import argparse
import csv
from pathlib import Path
import torch
from local_ai import EMBED_MODEL, embed_texts

ROOT = Path(__file__).resolve().parent


def read_csv(path: Path):
    if not path.is_file():
        raise FileNotFoundError(f"Missing input dataset: {path}")
    rows, seen = [], set()
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        names = set(reader.fieldnames or [])
        if "text" not in names:
            raise ValueError("CSV must contain a 'text' column")
        labeled = "label" in names
        for row in reader:
            sentence = (row.get("text") or "").strip()
            key = " ".join(sentence.casefold().split())
            if not sentence or key in seen:
                continue
            seen.add(key)
            label = (row.get("label") or "").strip() if labeled else ""
            if label not in ("", "0", "1"):
                raise ValueError(f"Label must be 0 or 1; got {label!r} for {sentence[:60]!r}")
            rows.append((sentence, int(label) if label else None))
    if not rows:
        raise ValueError("No valid text rows found")
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, default=ROOT / "data" / "sentences.csv")
    p.add_argument("--output", type=Path, default=ROOT / "data" / "local_embeddings.pt")
    p.add_argument("--model", default=EMBED_MODEL, help="Ollama embedding model name")
    p.add_argument("--batch-size", type=int, default=8)
    args = p.parse_args()
    if args.batch_size < 1:
        p.error("batch-size must be positive")
    rows = read_csv(args.input)
    features = embed_texts([x for x,_ in rows], model=args.model, batch_size=args.batch_size)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "embeddings": features,
        "labels": [label for _, label in rows],
        "texts": [x for x,_ in rows],
        "embedding_model": args.model,
        "source_file": str(args.input),
    }
    torch.save(data, args.output)
    counts = {label: data["labels"].count(label) for label in (0,1,None)}
    print(f"Saved {tuple(features.shape)} embeddings to {args.output}")
    print(f"Labels: negative={counts[0]}, positive={counts[1]}, unlabeled={counts[None]}")
    if counts[None]:
        print("Unlabeled rows can be used for retrieval, but not supervised sentiment training.")


if __name__ == "__main__":
    main()
