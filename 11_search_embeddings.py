"""Try local embedding search over a saved Common Crawl vector dataset.

Usage: python 11_search_embeddings.py --cache data/commoncrawl_raw_embeddings.pt --query "great service"
This is an educational similarity demo; GPT-2 mean pooling is not optimized for semantic search.
"""
import argparse
from pathlib import Path

import torch
from torch.nn.functional import normalize
from hybrid_model import LANGUAGE_MODEL_ID, load_language_model, encode_texts
from importlib import import_module
embed_utils = import_module("10_embed_dataset")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--query", type=str, required=True)
    p.add_argument("--top-k", type=int, default=5)
    a = p.parse_args()
    if a.top_k < 1 or not a.query.strip():
        p.error("top-k must be positive and query not blank")
    cache = torch.load(a.cache, map_location="cpu", weights_only=True)
    if cache.get("encoder_name") != LANGUAGE_MODEL_ID or cache.get("pooling") != embed_utils.POOLING:
        raise ValueError("Cache was generated with an incompatible embedding encoder")
    texts, vectors = cache["texts"], cache["embeddings"]
    if len(texts) != len(vectors) or len(texts) == 0:
        raise ValueError("Empty or invalid cache")
    tokenizer, encoder = load_language_model(LANGUAGE_MODEL_ID)
    query_vector = encode_texts([a.query], tokenizer, encoder).cpu()
    scores = normalize(vectors.float(), dim=1) @ normalize(query_vector.float(), dim=1).T
    values, indices = scores.squeeze(1).topk(min(a.top_k, len(texts)))
    print("Closest text excerpts by embedding cosine similarity (not sentiment predictions):")
    for rank, (index, score) in enumerate(zip(indices.tolist(), values.tolist()), start=1):
        print(f"{rank}. Similarity={score:.3f} | {texts[index][:250]}")
        if cache.get("source_urls") and cache["source_urls"][index]:
            print(f"   Source: {cache['source_urls'][index]}")


if __name__ == "__main__":
    main()
