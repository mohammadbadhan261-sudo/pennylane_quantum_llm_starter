# PennyLane + DistilGPT2: Common Crawl AI Embedding Project

This project downloads samples of genuine Common Crawl WET web text, turns them into numerical vectors using **a frozen DistilGPT2 encoder**, optionally applies local AI sentiment labeling using Ollama, and trains a **4-qubit PennyLane + PyTorch sentiment classifier** from a labeled embedding cache.

**Beginner setup and exact VS Code commands:** Read [EMBEDDINGS_GUIDE.md](EMBEDDINGS_GUIDE.md).

**Data pipeline**: Common Crawl WET → cleaned `commoncrawl_raw.csv` → `10_embed_dataset.py` → unlabeled embeddings (`N x 768`); for classification: Ollama suggested labels → manual review → `09_merge_training_data.py` → labeled `train_mixed.csv` → `10_embed_dataset.py` → training from the embedding cache → `03_app.py`.

The package does not contain actual downloaded Common Crawl data, model checkpoint or experimental accuracy measurements. The web-text sampling and end-to-end training must run on your computer with internet access. Source data and embedding caches are ignored by Git.
