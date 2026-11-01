"""Train only the PennyLane/PyTorch head on cached OpenAI embeddings."""
import argparse
import random
from pathlib import Path
import torch
from torch import nn
from openai_quantum_head import QuantumTextHead
from openai_text_api import EMBED_MODEL
from importlib import import_module

fp = import_module("12_prepare_openai_embeddings").fingerprint
ROOT = Path(__file__).resolve().parent


def stratified_indices(labels, test_fraction, seed):
    rng = random.Random(seed)
    tr, te = [], []
    for label in (0, 1):
        indices = [i for i, y in enumerate(labels.tolist()) if y == label]
        rng.shuffle(indices)
        if len(indices) < 5:
            raise ValueError(f"At least 5 reviewed records needed per class; label={label} has {len(indices)}")
        n = min(len(indices)-1, max(1, round(len(indices)*test_fraction)))
        te.extend(indices[:n]); tr.extend(indices[n:])
    rng.shuffle(tr); rng.shuffle(te)
    return tr, te


def metrics(model, features, labels, batch_size):
    model.eval()
    predictions=[]
    with torch.no_grad():
        for chunk in features.split(batch_size):
            predictions.extend(model(chunk).argmax(dim=1).tolist())
    ys = labels.tolist()
    tp=sum(p==y==1 for p,y in zip(predictions,ys))
    fp=sum(p==1 and y==0 for p,y in zip(predictions,ys))
    fn=sum(p==0 and y==1 for p,y in zip(predictions,ys))
    precision=tp/(tp+fp) if tp+fp else 0
    recall=tp/(tp+fn) if tp+fn else 0
    f1=2*precision*recall/(precision+recall) if precision+recall else 0
    return {"accuracy": sum(p==y for p,y in zip(predictions,ys))/len(ys),
            "positive_f1": f1,
            "pred_positive": sum(predictions),
            "pred_negative": len(predictions)-sum(predictions)}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--embeddings", type=Path, default=ROOT/"data"/"openai_embeddings.pt")
    parser.add_argument("--output", type=Path, default=ROOT/"models"/"openai_quantum_head.pt")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.003)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    args=parser.parse_args()
    if args.epochs<1 or args.batch_size<1 or args.lr<=0 or not 0<args.test_fraction<0.5:
        parser.error("Invalid training parameters")
    random.seed(args.seed); torch.manual_seed(args.seed)
    # Load only files you created or explicitly trust.
    data=torch.load(args.embeddings, map_location="cpu",weights_only=True)
    if data.get("format_version")!=2 or data.get("model_name")!=EMBED_MODEL:
        raise ValueError(f"Expected cache from {EMBED_MODEL}; re-embed using 12_prepare_openai_embeddings.py")
    x,y,texts=data["embeddings"],data["labels"],data["texts"]
    if x.ndim!=2 or x.shape[0]!=len(y) or len(y)!=len(texts) or x.shape[1]!=data["embedding_dim"]:
        raise ValueError("Invalid feature/label cache shapes")
    if not torch.isfinite(x).all():
        raise ValueError("Invalid numeric features")
    if set(y.tolist())-set((0,1)):
        raise ValueError("Unlabeled examples detected. Review sentiment labels before training.")
    if fp(texts,y.tolist())!=data["sha256"]:
        raise ValueError("Cache fingerprint mismatch")
    train_ids,test_ids=stratified_indices(y,args.test_fraction,args.seed)
    print(f"Embeddings {tuple(x.shape)} from {data['model_name']} | train={len(train_ids)} | test={len(test_ids)}")
    head=QuantumTextHead(x.shape[1]); optimizer=torch.optim.Adam(head.parameters(),lr=args.lr)
    criterion=nn.CrossEntropyLoss()
    for epoch in range(1,args.epochs+1):
        head.train(); order=torch.randperm(len(train_ids)); total=0
        for ids in order.split(args.batch_size):
            batch_indices=[train_ids[j] for j in ids.tolist()]
            optimizer.zero_grad()
            loss=criterion(head(x[batch_indices]),y[batch_indices]); loss.backward(); optimizer.step()
            total+=loss.item()*len(batch_indices)
        if epoch==1 or epoch%5==0 or epoch==args.epochs:
            print(f"Epoch {epoch:02d}/{args.epochs} | loss={total/len(train_ids):.4f}")
    result=metrics(head,x[test_ids],y[test_ids],args.batch_size)
    print(f"Held-out accuracy: {result['accuracy']:.1%} | positive F1={result['positive_f1']:.3f}")
    print(f"Predicted positive={result['pred_positive']} negative={result['pred_negative']}")
    if result['pred_positive']==0 or result['pred_negative']==0:
        print("WARNING: Predictions collapsed into one class")
    args.output.parent.mkdir(exist_ok=True,parents=True)
    torch.save({"state_dict":head.state_dict(),"embedding_dim":x.shape[1],
                "embedding_model": data["model_name"],
                "train_examples":len(train_ids),"test_examples":len(test_ids),
                "test_metrics":result,"architecture":"openai-q4-classical-skip-v1"},args.output)
    print(f"Saved {args.output} (NOT the original GPT-2 checkpoint)")

if __name__=="__main__":main()
