"""Suggest labels for Common Crawl text using GPT API. Never mark as reviewed."""
import argparse
import csv
from pathlib import Path
from openai_text_api import GPT_MODEL,classify_with_gpt

ROOT=Path(__file__).resolve().parent

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--input",type=Path,default=ROOT/"data"/"commoncrawl_raw.csv")
    p.add_argument("--output",type=Path,default=ROOT/"data"/"commoncrawl_labeled.csv")
    p.add_argument("--model",default=GPT_MODEL)
    p.add_argument("--max-rows",type=int,default=30)
    args=p.parse_args()
    if args.max_rows<1:p.error("--max-rows must be >= 1")
    with args.input.open(encoding="utf-8-sig",newline="") as f:
        source=csv.DictReader(f)
        if "text" not in (source.fieldnames or []):
            raise ValueError("CSV requires 'text' column")
        rows=[]
        for row in source:
            text=(row.get("text") or "").strip()
            if not text:continue
            guess=classify_with_gpt(text,model=args.model).upper().strip(" .!\n\t")
            if guess=="POSITIVE":label="1"
            elif guess=="NEGATIVE":label="0"
            else:label=""  # skip neutral or malformed answer
            rows.append({"text":text,"label":label,"reviewed":"no",
                         "source_url":row.get("source_url", ""),"ai_model":args.model,
                         "ai_response":guess[:200]})
            print(f"Suggested {len(rows)}/{args.max_rows}: {guess}")
            if len(rows)>=args.max_rows:break
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open("w",encoding="utf-8",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=["text","label","reviewed","source_url","ai_model","ai_response"])
        writer.writeheader();writer.writerows(rows)
    print(f"Saved {len(rows)} suggestions to {args.output} (ALL reviewed=no).")
    print("Check labels manually, then set reviewed=yes for accepted positive/negative examples.")

if __name__=="__main__": main()
