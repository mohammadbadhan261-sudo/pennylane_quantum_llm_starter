"""Sample actual Common Crawl WET plaintext without downloading a full archive.

Source: https://data.commoncrawl.org/crawl-data/CC-MAIN-2026-43/wet.paths.gz
Needs: python -m pip install warcio
Run: python 07_fetch_commoncrawl.py --target 50 --max-wet-files 3

No labels are assigned here. Keep provenance and review copyright/privacy before use.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import random
import re
import sys
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parent
HOST = "https://data.commoncrawl.org/"
UA = "PennyLane-education-research-sampler/1.0"
EMAIL = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I)
PHONE = re.compile(r"(?:\+?\d[\d\s().-]{8,}\d)")
LINK = re.compile(r"https?://|www\.", re.I)
SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")

class BoundedReader:
    """End the HTTP stream once a safe approximate compressed-byte limit is reached."""
    def __init__(self, reader, byte_limit):
        self.reader = reader
        self.remaining = byte_limit
    def read(self, amount=-1):
        if self.remaining <= 0:
            return b""
        amount = min(self.remaining, 65536 if amount is None or amount < 0 else amount)
        chunk = self.reader.read(amount)
        self.remaining -= len(chunk)
        return chunk
    def close(self):
        self.reader.close()


def select_sentences(content, max_per_page=2):
    """Keep short, readable English-like sentences and reject obvious personal data."""
    result = []
    for line in SPLIT.split(content):
        s = " ".join(line.split()).strip(" \"'\t")
        words = s.split()
        if not 8 <= len(words) <= 65 or not 45 <= len(s) <= 450:
            continue
        if EMAIL.search(s) or PHONE.search(s) or LINK.search(s):
            continue
        if '<' in s or '>' in s or s.count('|') >= 2:
            continue
        if sum(c.isascii() and c.isalpha() for c in s) / max(1,len(s)) < 0.62:
            continue
        if not any(c in s for c in '.!?'):
            continue
        result.append(s)
        if len(result) == max_per_page:
            break
    return result


def get_manifest(crawl):
    url = f"{HOST}crawl-data/{crawl}/wet.paths.gz"
    with urlopen(Request(url, headers={"User-Agent": UA}), timeout=40) as response:
        raw = response.read(4_000_000)
    return [line.strip() for line in gzip.decompress(raw).decode('utf-8').splitlines() if line.strip()]


def fetch_some(path, output_rows, target, seen, byte_limit, url_contains):
    try:
        from warcio.archiveiterator import ArchiveIterator
    except ImportError as exc:
        raise SystemExit("Install required parser: python -m pip install warcio") from exc
    url = HOST + path
    print(f"  Streaming WET file: {path.split('/')[-1]} (up to {byte_limit//(1024*1024)} MiB)",flush=True)
    with urlopen(Request(url, headers={"User-Agent": UA}), timeout=75) as resp:
        limited = BoundedReader(resp,byte_limit)
        try:
            for record in ArchiveIterator(limited):
                if len(output_rows) >= target:
                    break
                if record.rec_type != 'conversion':
                    continue
                origin = record.rec_headers.get_header('WARC-Target-URI') or ''
                if not origin.startswith(('http://','https://')):
                    continue
                if url_contains and url_contains.casefold() not in origin.casefold():
                    continue
                try:
                    text = record.content_stream().read(80000).decode('utf-8',errors='replace')
                except Exception:
                    continue
                for sentence in select_sentences(text):
                    key = ' '.join(sentence.casefold().split())
                    if key in seen:
                        continue
                    seen.add(key)
                    output_rows.append({'text': sentence, 'source_url': origin, 'wet_path': path})
                    if len(output_rows)>=target:
                        break
        except (OSError, EOFError, ValueError) as exc:
            # An incomplete record at the configured byte limit is expected.
            print(f"  Reached a WET stream boundary or read limit ({type(exc).__name__}).")


def main():
    p=argparse.ArgumentParser(description='Sample actual Common Crawl WET plaintext; output is unlabeled')
    p.add_argument('--crawl',default='CC-MAIN-2026-43')
    p.add_argument('--target',type=int,default=50)
    p.add_argument('--max-wet-files',type=int,default=3)
    p.add_argument('--max-mb-per-file',type=int,default=30)
    p.add_argument('--url-contains',default='',help='Optional URL filter, e.g. reviews (may return no rows)')
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--output',type=Path,default=ROOT/'data'/'commoncrawl_raw.csv')
    a=p.parse_args()
    if a.target<1 or a.max_wet_files<1 or not 1<=a.max_mb_per_file<=500:
        p.error('target, max-wet-files must be positive and max-mb-per-file between 1 and 500')
    try:
        paths=get_manifest(a.crawl)
    except (HTTPError, URLError, OSError) as exc:
        raise SystemExit(f'Unable to load Common Crawl WET manifest: {exc}. Check internet connection.') from exc
    if not paths:
        raise SystemExit('WET manifest was empty')
    rng=random.Random(a.seed)
    choices=rng.sample(paths,k=min(a.max_wet_files,len(paths)))
    rows=[];seen=set()
    for path in choices:
        try:
            fetch_some(path,rows,a.target,seen,a.max_mb_per_file*1024*1024,a.url_contains)
        except (HTTPError, URLError, TimeoutError, ConnectionError) as exc:
            print(f'  Network error: {exc}; continuing to next WET file',file=sys.stderr)
        print(f'  Collected {len(rows)}/{a.target} sentences')
        if len(rows)>=a.target: break
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['text','source_url','wet_path']);w.writeheader();w.writerows(rows)
    print(f'Saved {len(rows)} UNLABELED Common Crawl excerpts: {a.output.resolve()}')
    print('These are actual downloaded excerpts, not verified sentiment reviews. Check them before using.')
    if not rows: print('No suitable text selected. Try --max-wet-files 5 --max-mb-per-file 50')

if __name__=='__main__': main()
