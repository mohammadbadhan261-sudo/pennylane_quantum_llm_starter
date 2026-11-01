"""Combine an existing labeled CSV with manually reviewed Common Crawl labels."""
import argparse
import csv
from pathlib import Path

ROOT=Path(__file__).resolve().parent


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--base",type=Path,default=ROOT/"data"/"sentences.csv")
    p.add_argument("--review",type=Path,default=ROOT/"data"/"commoncrawl_labeled.csv")
    p.add_argument("--output",type=Path,default=ROOT/"data"/"train_mixed.csv")
    args=p.parse_args()
    found=set(); rows=[]
    for path,require_review in ((args.base,False),(args.review,True)):
        if not path.is_file():
            raise FileNotFoundError(path)
        with path.open(encoding="utf-8-sig",newline="") as file:
            reader=csv.DictReader(file)
            if not {"text","label"}.issubset(set(reader.fieldnames or [])):
                raise ValueError(f"Missing text/label in {path}")
            for row in reader:
                if require_review and row.get("reviewed", "").strip().lower() not in ("yes","true","1"):
                    continue
                text=(row.get("text") or "").strip()
                label=(row.get("label") or "").strip()
                key=" ".join(text.casefold().split())
                if text and label in ("0","1") and key not in found:
                    rows.append({"text":text,"label":label})
                    found.add(key)
    if not rows:
        raise ValueError("No labeled examples to merge")
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open("w",newline="",encoding="utf-8") as file:
        writer=csv.DictWriter(file,fieldnames=["text","label"])
        writer.writeheader(); writer.writerows(rows)
    print(f"Wrote {len(rows)} deduplicated, labeled examples to {args.output}")


if __name__=="__main__":
    main()
