"""Train a free, local PennyLane sentiment head using saved Ollama embeddings."""
import argparse
import random
from pathlib import Path
import torch
from torch import nn
from local_quantum_model import LocalQuantumHead

ROOT = Path(__file__).resolve().parent


def split_indices(labels: list[int], fraction: float, seed: int):
    rnd = random.Random(seed)
    by_class = {0: [], 1: []}
    for idx, label in enumerate(labels):
        by_class[label].append(idx)
    train, test = [], []
    for key, indices in by_class.items():
        if len(indices) < 2:
            raise ValueError(f"Need at least two different sentences for class {key}")
        rnd.shuffle(indices)
        n_test = max(1, min(len(indices) - 1, round(len(indices) * fraction)))
        test.extend(indices[:n_test])
        train.extend(indices[n_test:])
    rnd.shuffle(train)
    rnd.shuffle(test)
    return train, test


def metrics(target: torch.Tensor, prediction: torch.Tensor):
    correct = int((target == prediction).sum())
    total = len(target)
    tp = int(((target == 1) & (prediction == 1)).sum())
    fp = int(((target == 0) & (prediction == 1)).sum())
    fn = int(((target == 1) & (prediction == 0)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2*precision*recall / (precision + recall) if precision+recall else 0.0
    return {"accuracy": correct / total, "positive_f1": f1, "test_count": total,
            "predicted_positive": int((prediction == 1).sum()), "predicted_negative": int((prediction == 0).sum())}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--embeddings", type=Path, default=ROOT / "data" / "local_embeddings.pt")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=0.003)
    p.add_argument("--test-fraction", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.lr <= 0 or not 0 < args.test_fraction < 0.5:
        p.error("Invalid epochs, batch size, learning rate, or test fraction")
    torch.manual_seed(args.seed)
    data = torch.load(args.embeddings, map_location="cpu", weights_only=True)
    labels = data["labels"]
    if any(label not in (0, 1) for label in labels):
        raise ValueError("Unlabeled embeddings cannot train sentiment model. Label and review them first.")
    embeddings = data["embeddings"].float()
    if embeddings.ndim != 2 or len(labels) != len(embeddings):
        raise ValueError("Embedding cache and label lengths do not match")
    train, test = split_indices(labels, args.test_fraction, args.seed)
    y = torch.tensor(labels, dtype=torch.long)
    model = LocalQuantumHead(embedding_dim=embeddings.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()
    print(f"Embeddings {tuple(embeddings.shape)} | model {data['embedding_model']}")
    print(f"Train {len(train)} | test {len(test)} | positive {sum(y==1)} | negative {sum(y==0)}")
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = torch.randperm(len(train)).tolist()
        total_loss = 0.0
        for part in [order[i:i+args.batch_size] for i in range(0,len(order), args.batch_size)]:
            indices = [train[i] for i in part]
            optimizer.zero_grad()
            logits = model(embeddings[indices])
            loss = loss_fn(logits, y[indices])
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * len(indices)
        if epoch == 1 or epoch % 5 == 0 or epoch == args.epochs:
            print(f"Epoch {epoch:02d}/{args.epochs} | loss {total_loss/len(train):.4f}")
    model.eval()
    with torch.no_grad():
        scores = model(embeddings[test])
        predictions = scores.argmax(dim=1)
    result = metrics(y[test], predictions)
    print("Held-out results (illustrative dataset may be too small for reliable accuracy):")
    print(f"Accuracy={result['accuracy']:.3f} Positive F1={result['positive_f1']:.3f}")
    print(f"Predicted positive={result['predicted_positive']}, negative={result['predicted_negative']}")
    out = ROOT / "models" / "local_quantum_head.pt"
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "embedding_dim": embeddings.shape[1],
                "embedding_model": data["embedding_model"], "train_examples":len(train),
                "test_examples":len(test), "metrics":result}, out)
    print("Saved model:", out)


if __name__ == "__main__":
    main()
