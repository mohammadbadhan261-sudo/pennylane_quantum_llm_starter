# OpenAI API upgrade for your PennyLane + Common Crawl project

This is an ADD-ON to the prior PennyLane Common Crawl project. Copy the Python files
into the same directory as `07_fetch_commoncrawl.py`, `09_merge_training_data.py`,
and `data/sentences.csv`. Keep your `.venv` and other files unchanged.

**IMPORTANT:** GPT-4 (or GPT-6) cannot replace `AutoModel.from_pretrained("distilgpt2")`.
Instead, this add-on uses the OpenAI **embedding endpoint** for the frozen text
vectors, while GPT-4o / GPT-6 Luna can optionally label text or provide an
independent classification baseline through the Responses API.

## Windows / VS Code quickstart

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install "openai>=1.60,<4"
```

Create an API key at https://platform.openai.com/api-keys and configure billing
at https://platform.openai.com/settings/organization/billing . ChatGPT plans and
OpenAI API billing are separate. In the current PowerShell **terminal**:

```powershell
$env:OPENAI_API_KEY = "paste-your-own-secret-key-here"
```

Never paste a real key in source code, README, chat or GitHub. Never commit it.
The environment setting above disappears after the terminal closes.

Test your key and embedding access:

```powershell
python -c "from openai_text_api import embed_texts; print(embed_texts(['Quantum computing is interesting.']).shape)"
```

Expected shape (with text-embedding-3-small): `torch.Size([1, 1536])`.

Create dataset embeddings (use an existing CSV with `text,label`):

```powershell
python 12_prepare_openai_embeddings.py --input data/sentences.csv --output data/openai_embeddings.pt
```

Use `data/train_mixed.csv` instead to include reviewed Common Crawl texts:

```powershell
python 12_prepare_openai_embeddings.py --input data/train_mixed.csv --output data/openai_embeddings.pt
```

Train your PennyLane head (this never trains GPT-4, GPT-6 or the embedding model):

```powershell
python 13_train_openai_quantum.py --embeddings data/openai_embeddings.pt --epochs 20 --batch-size 8 --lr 0.003
python -m streamlit run 14_app_openai_quantum.py
```

To label newly downloaded Common Crawl text with an API GPT model, run:

```powershell
python 15_label_commoncrawl_openai.py --input data/commoncrawl_raw.csv --max-rows 30 --model gpt-6-luna
# Or if you explicitly want GPT-4 (legacy series, scheduled for deprecation):
python 15_label_commoncrawl_openai.py --input data/commoncrawl_raw.csv --max-rows 30 --model gpt-4o
```

Open `data/commoncrawl_labeled.csv`, verify or correct the suggestions and set
`reviewed=yes` only for accepted records. Then use the existing merge script:

```powershell
python 09_merge_training_data.py --base data/imdb_sentences.csv
```

Then re-embed, retrain and restart your Streamlit app using the previous commands.

## GitHub workflow

The previous project `.gitignore` ignores `.env`, `models/`, `*.pt`, and
`data/*.csv` except sample data. Preserve those exclusions. Stage only source code:

```powershell
git add openai_text_api.py openai_quantum_head.py 12_prepare_openai_embeddings.py 13_train_openai_quantum.py 14_app_openai_quantum.py 15_label_commoncrawl_openai.py requirements-openai.txt README_UPGRADE.md
git commit -m "Add OpenAI embeddings and GPT comparison to PennyLane"
git push
```

## Research cautions

- These are API-based embeddings and API-based GPT calls, not self-hosted GPT-4.
- Sending Common Crawl text to OpenAI is an external data transfer; review rights,
  privacy, and API usage costs. Never send passwords or unredacted personal data.
- Do not mistake positive score (e.g. 47%) for held-out accuracy.
- The quantum head has a **classical skip connection**; compare to a classical-only
  model before making quantum-advantage claims.
- The train/test split here is for learning. A research report should use an
  independent held-out evaluation dataset and record all preprocessing.
- Never share checkpoint files or crawled text unless distribution is permitted.
- Public deployment needs a secure OPENAI_API_KEY environment setting and a
  separately supplied trained checkpoint; `models/` is intentionally gitignored.
