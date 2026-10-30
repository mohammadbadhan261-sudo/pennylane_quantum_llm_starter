"""Train the hybrid sentiment head on a labeled CSV.

Examples:
  python 02_train_hybrid_model.py --dataset data/imdb_sentences.csv --epochs 20 --lr 0.003
  python 02_train_hybrid_model.py --dataset data/sentences.csv --epochs 20
"""
import argparse
import csv
import random
from pathlib import Path

import torch
from torch import nn

from hybrid_model import LANGUAGE_MODEL_ID, QuantumTextHead, encode_texts, load_language_model

ROOT = Path(__file__).resolve().parent
CHECKPOINT = ROOT / "models" / "quantum_head.pt"


def load_examples(path):
    if not path.exists():
        raise FileNotFoundError(f"Could not find {path}")
    seen = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not {"text", "label"}.issubset(reader.fieldnames or []):
            raise ValueError("CSV must have text,label columns")
        for row in reader:
            text = (row.get("text") or "").strip()
            label = (row.get("label") or "").strip()
            if not text:
                continue
            if label not in ("0", "1"):
                raise ValueError(f"Label for {text[:40]!r} must be 0 or 1")
            key = " ".join(text.casefold().split())
            if key in seen and seen[key][1] != int(label):
                raise ValueError(f"Conflicting labels for duplicate text: {text[:40]!r}")
            seen.setdefault(key, (text, int(label)))
    examples = list(seen.values())
    counts = [sum(y == label for _, y in examples) for label in (0, 1)]
    if min(counts) < 5:
        raise ValueError(f"Need at least 5 unique texts per class; got negative={counts[0]}, positive={counts[1]}")
    return examples, counts


def stratified_split(examples, test_fraction, seed):
    rng = random.Random(seed)
    per_label = [[i for i, (_, y) in enumerate(examples) if y == label] for label in (0, 1)]
    train, test = [], []
    for ids in per_label:
        rng.shuffle(ids)
        ntest = min(len(ids) - 1, max(1, round(len(ids)*test_fraction)))
        train.extend(ids[ntest:])
        test.extend(ids[:ntest])
    rng.shuffle(train)
    rng.shuffle(test)
    return train, test


def evaluate(head, x, labels, batch_size):
    head.eval()
    pred, positive_scores = [], []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            scores = torch.softmax(head(x[start:start+batch_size]), dim=1)
            positive_scores.extend(scores[:, 1].cpu().tolist())
            pred.extend(scores.argmax(dim=1).cpu().tolist())
    actual = labels.cpu().tolist()
    correct = sum(p == y for p, y in zip(pred, actual))
    tp = sum(p == y == 1 for p, y in zip(pred, actual))
    fp = sum(p == 1 and y == 0 for p, y in zip(pred, actual))
    fn = sum(p == 0 and y == 1 for p, y in zip(pred, actual))
    precision = tp / (tp+fp) if tp+fp else 0.0
    recall = tp / (tp+fn) if tp+fn else 0.0
    f1 = 2*precision*recall/(precision+recall) if precision+recall else 0.0
    return {
        "accuracy": correct/len(actual), "f1_positive": f1,
        "predicted_positive": sum(pred), "predicted_negative": len(pred) - sum(pred),
        "positive_score_min": min(positive_scores), "positive_score_max": max(positive_scores),
        "actual_positive": sum(actual), "actual_negative": len(actual)-sum(actual),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=ROOT / "data" / "sentences.csv")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.003)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.lr <= 0 or not 0 < args.test_fraction < 0.5:
        parser.error("epochs/batch-size/lr must be positive; test-fraction must be between 0 and 0.5")
    path = args.dataset.resolve()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    examples, counts = load_examples(path)
    train_ids, test_ids = stratified_split(examples, args.test_fraction, args.seed)
    print(f"Dataset: {path}")
    print(f"Unique rows: {len(examples)} | Negative: {counts[0]} | Positive: {counts[1]}")
    print(f"Training: {len(train_ids)} | Held-out test: {len(test_ids)}")
    print("Loading frozen DistilGPT2 encoder...")
    tokenizer, encoder = load_language_model()
    embeddings = encode_texts([t for t, _ in examples], tokenizer, encoder)
    labels = torch.tensor([y for _, y in examples], dtype=torch.long)
    x_train, y_train = embeddings[train_ids], labels[train_ids]
    x_test, y_test = embeddings[test_ids], labels[test_ids]
    print(f"Feature shape: {tuple(embeddings.shape)}")
    head = QuantumTextHead(embeddings.shape[1])
    optimizer = torch.optim.Adam(head.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(1, args.epochs + 1):
        head.train()
        order = torch.randperm(len(x_train))
        total_loss = 0.0
        for indices in order.split(args.batch_size):
            optimizer.zero_grad()
            loss = loss_fn(head(x_train[indices]), y_train[indices])
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(indices)
        if epoch == 1 or epoch % 5 == 0 or epoch == args.epochs:
            print(f"Epoch {epoch:02d}/{args.epochs} | train loss: {total_loss/len(x_train):.4f}")

    metrics = evaluate(head, x_test, y_test, args.batch_size)
    print(f"Held-out accuracy: {metrics['accuracy']:.1%} | positive F1: {metrics['f1_positive']:.3f}")
    print(f"Test actual: positive={metrics['actual_positive']}, negative={metrics['actual_negative']}")
    print(f"Test predicted: positive={metrics['predicted_positive']}, negative={metrics['predicted_negative']}")
    print(f"Positive score range: {metrics['positive_score_min']:.1%} to {metrics['positive_score_max']:.1%}")
    if metrics['predicted_positive'] == 0 or metrics['predicted_negative'] == 0:
        print("WARNING: All held-out examples received the same predicted class. Model may have collapsed.")

    CHECKPOINT.parent.mkdir(exist_ok=True, parents=True)
    torch.save({
        "state_dict": head.state_dict(), "embedding_dim": embeddings.shape[1],
        "model_name": LANGUAGE_MODEL_ID, "train_examples": len(train_ids),
        "test_examples": len(test_ids), "dataset_path": str(path),
        "test_metrics": metrics, "architecture": "layernorm-quantum-classical-skip-v2",
    }, CHECKPOINT)
    print(f"Saved checkpoint: {CHECKPOINT}")
    print("Restart Streamlit fully: Ctrl+C, then python -m streamlit run 03_app.py")
    print("This hybrid includes a classical skip path, so results do NOT establish quantum advantage.")


if __name__ == "__main__":
    main()
