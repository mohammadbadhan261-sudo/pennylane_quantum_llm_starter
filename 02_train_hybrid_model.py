"""Train a PennyLane quantum sentiment head from any labeled CSV file.

Examples:
    python 02_train_hybrid_model.py
    python 02_train_hybrid_model.py --dataset data/imdb_sentences.csv --epochs 10
    python 02_train_hybrid_model.py --dataset data/sentences.csv --epochs 20 --batch-size 8
"""

import argparse
import csv
import random
from pathlib import Path

import torch
from torch import nn

from hybrid_model import LANGUAGE_MODEL_ID, QuantumTextHead, encode_texts, load_language_model

ROOT = Path(__file__).resolve().parent


def load_examples(csv_path: Path):
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found: {csv_path}")
    examples = []
    seen = {}
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or not {"text", "label"}.issubset(reader.fieldnames):
            raise ValueError("CSV requires exactly named columns: text,label")
        for line_no, row in enumerate(reader, start=2):
            sentence = (row.get("text") or "").strip()
            label_text = (row.get("label") or "").strip()
            if not sentence:
                continue
            if label_text not in ("0", "1"):
                raise ValueError(f"Line {line_no}: label must be 0 or 1; got {label_text!r}")
            label = int(label_text)
            key = " ".join(sentence.casefold().split())
            if key in seen:
                if seen[key] != label:
                    raise ValueError(f"Line {line_no}: same text has conflicting labels")
                continue  # Skip duplicate sentences, preventing train/test leakage.
            seen[key] = label
            examples.append((sentence, label))

    per_label = {0: sum(y == 0 for _, y in examples), 1: sum(y == 1 for _, y in examples)}
    if min(per_label.values()) < 2:
        raise ValueError(f"You need at least 2 distinct sentences for each label (have {per_label}).")
    return examples, per_label


def split_by_class(examples, test_fraction, seed):
    rng = random.Random(seed)
    indices = {0: [], 1: []}
    for i, (_, label) in enumerate(examples):
        indices[label].append(i)

    train_indices = []
    test_indices = []
    for label in (0, 1):
        ids = indices[label]
        rng.shuffle(ids)
        n_test = min(len(ids) - 1, max(1, round(len(ids) * test_fraction)))
        test_indices += ids[:n_test]
        train_indices += ids[n_test:]
    rng.shuffle(train_indices)
    rng.shuffle(test_indices)
    return train_indices, test_indices


def main():
    parser = argparse.ArgumentParser(description="Train a hybrid quantum sentiment classifier")
    parser.add_argument("--dataset", type=Path, default=ROOT / "data" / "sentences.csv", help="CSV with text,label columns")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--lr", type=float, default=0.03)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.lr <= 0 or not 0 < args.test_fraction < 0.5:
        parser.error("Requires --epochs >= 1, --batch-size >= 1, --lr > 0 and 0 < --test-fraction < 0.5")

    torch.manual_seed(args.seed)
    examples, class_counts = load_examples(args.dataset)
    train_ids, test_ids = split_by_class(examples, args.test_fraction, args.seed)
    print(f"Dataset: {args.dataset.resolve()}")
    print(f"Unique sentences: {len(examples)} (negative={class_counts[0]}, positive={class_counts[1]})")
    print(f"Training sentences: {len(train_ids)} | Test sentences: {len(test_ids)}")

    tokenizer, language_model = load_language_model()
    print("Extracting frozen GPT-2 text features (this may take a while)...")
    embeddings = encode_texts([text for text, _ in examples], tokenizer, language_model)
    labels = torch.tensor([y for _, y in examples], dtype=torch.long)
    print("Embedding shape:", tuple(embeddings.shape))
    x_train, y_train = embeddings[train_ids], labels[train_ids]
    x_test, y_test = embeddings[test_ids], labels[test_ids]

    model = QuantumTextHead(embedding_dim=embeddings.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()
    print("Training classical projection + quantum circuit + classifier (GPT-2 remains frozen)...")
    for epoch in range(1, args.epochs + 1):
        model.train()
        perm = torch.randperm(len(x_train))
        epoch_loss = 0.0
        for inds in perm.split(args.batch_size):
            optimizer.zero_grad()
            output = model(x_train[inds])
            loss = loss_fn(output, y_train[inds])
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(inds)
        if epoch == 1 or epoch % 5 == 0 or epoch == args.epochs:
            print(f"Epoch {epoch:02d}/{args.epochs} | training loss: {epoch_loss / len(x_train):.4f}")

    model.eval()
    correct = 0
    with torch.no_grad():
        # Evaluate in batches to avoid a single large quantum-simulator call.
        for start in range(0, len(x_test), args.batch_size):
            end = start + args.batch_size
            predicted = model(x_test[start:end]).argmax(dim=1)
            correct += int((predicted == y_test[start:end]).sum())
    accuracy = correct / len(x_test)
    print(f"Held-out test accuracy: {accuracy:.1%} ({correct}/{len(x_test)})")

    out = ROOT / "models" / "quantum_head.pt"
    out.parent.mkdir(exist_ok=True, parents=True)
    torch.save({
        "state_dict": model.state_dict(),
        "embedding_dim": embeddings.shape[1],
        "model_name": LANGUAGE_MODEL_ID,
        "train_examples": len(train_ids),
        "test_examples": len(test_ids),
        "dataset_path": str(args.dataset),
    }, out)
    print(f"Saved trained quantum head: {out}")
    print("Run the app: python -m streamlit run 03_app.py")
    print("Note: This trains the quantum/classical head, not GPT-2 weights. Test accuracy is not a quantum-advantage claim.")


if __name__ == "__main__":
    main()
