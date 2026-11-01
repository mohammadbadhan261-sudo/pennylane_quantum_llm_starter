"""OpenAI hosted text and embedding API helpers. Requires OPENAI_API_KEY."""
import os
import torch
from openai import OpenAI

EMBED_MODEL = "text-embedding-3-small"  # 1536-dimensional by default
GPT_MODEL = os.getenv("OPENAI_GPT_MODEL", "gpt-6-luna")


def embed_texts(texts, model=EMBED_MODEL, batch_size=16):
    """Create feature vectors with a hosted embedding model; never trains it."""
    if not texts or any(not isinstance(s, str) or not s.strip() for s in texts):
        raise ValueError("Pass a nonempty list of nonempty strings")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    client = OpenAI()  # reads OPENAI_API_KEY from environment
    vectors = []
    for start in range(0, len(texts), batch_size):
        # A small request size suits beginner experiments. Length limit avoids
        # accidentally sending very long crawled pages and large API bills.
        batch = [x.strip()[:3000] for x in texts[start:start + batch_size]]
        result = client.embeddings.create(model=model, input=batch, encoding_format="float")
        items = sorted(result.data, key=lambda item: item.index)
        if len(items) != len(batch):
            raise RuntimeError("OpenAI embeddings count differs from input count")
        vectors.extend(item.embedding for item in items)
        print(f"Embedded {min(start + len(batch),len(texts))}/{len(texts)} texts")
    out = torch.tensor(vectors, dtype=torch.float32)
    if out.ndim != 2 or not torch.isfinite(out).all():
        raise RuntimeError("Invalid embedding vectors")
    return out


def classify_with_gpt(text, model=GPT_MODEL):
    """Comparison with hosted GPT; not part of differentiable quantum training."""
    client = OpenAI()
    response = client.responses.create(
        model=model,
        instructions="Classify the sentiment of the text. Reply with only POSITIVE, NEGATIVE, or NEUTRAL.",
        input=text[:3000],
    )
    return response.output_text.strip()
