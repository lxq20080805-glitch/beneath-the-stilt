# Beneath the Stilt

An AI guide to the stilt houses (吊脚楼) of Chongqing and the lives of the people who lived in them.
Visitors ask questions in English or Chinese; answers combine former residents' recollections
(retrieved from `corpus/`) with the model's general historical knowledge.

```
app.py            Streamlit website
rag.py            chunking, local embeddings, FAISS index, search (+ CLI)
prompts.py        system prompt and message assembly
corpus/           oral-history texts, one UTF-8 .txt per narrator (private, git-ignored)
index/            FAISS index built from corpus/ (git-ignored)
tests/            offline checks using SYNTHETIC text only
.streamlit/       theme (config.toml) and API key (secrets.toml, git-ignored)
```

## 1. Install

Requires Python 3.10+.

```bash
cd "Beneath the Stilt"
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The first run downloads the multilingual embedding model
(`paraphrase-multilingual-MiniLM-L12-v2`, ~470 MB) and caches it.

## 2. Set the API key

Edit `.streamlit/secrets.toml` (template: `.streamlit/secrets.toml.example`), uncomment the line
and paste your key:

```toml
OPENAI_API_KEY = "sk-..."
```

Keep the line commented out until you have a key: Streamlit copies secrets into environment
variables, so an empty value would override an `OPENAI_API_KEY` set in your shell.

Never commit this file — it is already in `.gitignore`. Without a key the site still opens and shows
a configuration notice; it never produces fake answers.

Optional settings (same file, or environment variables): `OPENAI_MODEL` (default `gpt-4o-mini`),
`MAX_OUTPUT_TOKENS`, `TOP_K`, `HIGH_THRESHOLD`, `LOW_THRESHOLD`, `HISTORY_TURNS`,
`SESSION_MESSAGE_LIMIT`, `DAILY_MESSAGE_LIMIT`, `MIN_SECONDS_BETWEEN_MESSAGES`, `DEBUG_SOURCES`.

## 3. Run

```bash
streamlit run app.py
```

## 4. Add a transcript

1. Save it as UTF-8 text in `corpus/`, e.g. `corpus/interview_005.txt`. Never overwrite existing files.
2. Recommended format (plain paragraphs also work):

   ```text
   TITLE: Short descriptive title
   SOURCE: where it came from
   NARRATOR: who is speaking (no name if not authorised)
   TIME PERIOD: ...
   PLACES: ...
   KEYWORDS: ...

   ## Section heading

   One self-contained paragraph per idea, separated by blank lines.
   ```

   Header lines are used for the title only; they are not embedded. Each paragraph becomes a
   retrievable chunk, so write paragraphs that make sense on their own (repeat who/where).
3. The app notices changed files and rebuilds the index automatically on the next question.
   To rebuild by hand:

   ```bash
   python rag.py --check      # is the index up to date?
   python rag.py --rebuild    # rebuild it
   ```

Only add transcripts you have permission to use.

## 5. How answers are chosen

For each question the app retrieves the `TOP_K` most similar paragraphs and looks at the best score:

| best score | strategy |
|---|---|
| ≥ `HIGH_THRESHOLD` | answer mainly from residents' memories |
| between the two | combine memories with general knowledge |
| < `LOW_THRESHOLD` | answer from general knowledge; no excerpts are sent |

Scores depend on the embedding model and the corpus, so the defaults (0.65 / 0.30) are a starting
point. They were set from test queries on the current corpus: relevant questions scored
0.46–0.79 (Chinese questions score lower against the English corpus), unrelated ones 0.17–0.23.
A high score does not guarantee relevance, so the prompt also tells the model to ignore excerpts
that do not really answer the question. Check real questions and adjust:

```bash
python rag.py --query "吊脚楼里怎么做饭？"
```

Sources and scores are kept internally (in session state, and in logs as file names + scores only
when `DEBUG_SOURCES = true`) and are never shown on the website.

## 6. Tests

```bash
python tests/test_pipeline.py
```

Runs chunking, staleness detection, English/Chinese retrieval, relevance tiers and prompt assembly
on invented test text in `tests/synthetic_corpus/`. No API key needed.

## 7. Deploying to Streamlit Community Cloud

1. Push the repo to GitHub (secrets, `corpus/` and `index/` are git-ignored).
2. On share.streamlit.io create an app pointing to `app.py`, and paste the key into **Settings → Secrets**.
3. The corpus is private by default, so a public deployment answers from general knowledge only.
   To include transcripts, publish only those you are authorised to share (e.g. remove `corpus/`
   from `.gitignore` for the approved files, or use a private repo).
4. Keep the rate limits on (`SESSION_MESSAGE_LIMIT`, `DAILY_MESSAGE_LIMIT`) and set a monthly
   spending limit in your OpenAI account.
