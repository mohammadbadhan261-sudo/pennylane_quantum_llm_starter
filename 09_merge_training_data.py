"""Merge reviewed Common Crawl sentiment excerpts with an existing labeled CSV.

Outputs data/train_mixed.csv (text,label) for the existing training script.
Any Common Crawl row must be verified by a human (reviewed=yes) by default.
"""
from __future__ import annotations
import argparse,csv,random
from pathlib import Path
from collections import Counter

ROOT=Path(__file__).resolve().parent

def read_csv(path):
    if not path.exists():raise SystemExit(f'Cannot find {path}')
    with path.open('r',encoding='utf-8-sig',newline='') as f:
        rows=list(csv.DictReader(f))
    if not rows or not {'text','label'}.issubset(rows[0].keys()):
        raise SystemExit(f'Expected text,label columns in {path}')
    return rows

def merge(base,extra,allow_ai=False,seed=42):
    merged={};counts=Counter()
    for source,rows in [('base',base),('commoncrawl',extra)]:
        for row in rows:
            sentence=' '.join((row.get('text') or '').split())
            label=str(row.get('label') or '').strip()
            if not sentence or label not in ('0','1'): continue
            if source=='commoncrawl' and not allow_ai and str(row.get('reviewed') or '').strip().lower() not in ('yes','true','1'):
                counts['unreviewed_skipped']+=1;continue
            key=sentence.casefold()
            if key in merged:
                counts['duplicates_skipped']+=1;continue
            merged[key]={'text':sentence,'label':label,'source':source,
                         'source_url':row.get('source_url','') if source=='commoncrawl' else ''}
            counts[source]+=1
    values=list(merged.values());random.Random(seed).shuffle(values)
    return values,counts

def main():
    p=argparse.ArgumentParser(description='Join labeled data, safely excluding unreviewed Common Crawl examples')
    p.add_argument('--base',type=Path,default=ROOT/'data'/'sentences.csv')
    p.add_argument('--commoncrawl',type=Path,default=ROOT/'data'/'commoncrawl_labeled.csv')
    p.add_argument('--output',type=Path,default=ROOT/'data'/'train_mixed.csv')
    p.add_argument('--allow-ai-labels',action='store_true',help='EXPERIMENTAL: allow unreviewed pseudo-labels, not suitable for measured accuracy claims')
    a=p.parse_args()
    combined,counts=merge(read_csv(a.base),read_csv(a.commoncrawl),a.allow_ai_labels)
    if not combined: raise SystemExit('No training examples found')
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['text','label']);w.writeheader();w.writerows({'text':r['text'],'label':r['label']} for r in combined)
    prov=a.output.with_name(a.output.stem+'_provenance.csv')
    with prov.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['text','label','source','source_url']);w.writeheader();w.writerows(combined)
    classcounts=Counter(r['label'] for r in combined)
    print(f'Base added: {counts["base"]}, Common Crawl added: {counts["commoncrawl"]}, unreviewed skipped: {counts["unreviewed_skipped"]}')
    print(f'Class counts: negative={classcounts["0"]}, positive={classcounts["1"]}')
    print(f'Training CSV: {a.output.resolve()}')
    print(f'Private provenance CSV: {prov.resolve()}')
    print('Evaluate on a separate, untouched, manually labeled test set. Do not treat AI labels as verified truth.')

if __name__=='__main__':main()
