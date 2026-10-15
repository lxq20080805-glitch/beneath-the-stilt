"""Retrieval for Beneath the Stilt: chunk the corpus, embed locally, search with FAISS.

CLI:
    python rag.py --rebuild          # rebuild the index from corpus/
    python rag.py --check            # report whether the index is out of date
    python rag.py --query "问题"      # show top matches (file, section, score) for testing
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np

BASE_DIR = Path(__file__).resolve().parent
CORPUS_DIR = Path(os.environ.get("CORPUS_DIR", BASE_DIR / "corpus"))
INDEX_DIR = Path(os.environ.get("INDEX_DIR", BASE_DIR / "index"))
EMBEDDING_MODEL = os.environ.get(
    "EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

# Bump when the chunking logic changes so old indexes are detected as stale.
CHUNKER_VERSION = "1"
TARGET_CHARS = 900   # merge small paragraphs up to roughly this size
MAX_CHARS = 1400     # split paragraphs longer than this at sentence boundaries
MIN_CHARS = 200      # paragraphs shorter than this are merged with the next one

HEADER_RE = re.compile(r"^([A-Z][A-Z /]+):\s*(.*)$")
SENTENCE_RE = re.compile(r"(?<=[。！？!?\.])\s*")


@dataclass
class Chunk:
    id: int
    source: str   # file name, kept for internal debugging only
    title: str
    section: str
    text: str


# ---------------------------------------------------------------- chunking

def _parse_file(path: Path) -> tuple[str, list[tuple[str, str]]]:
    """Return (title, [(section, paragraph), ...]) for one corpus file.

    Optional header lines like ``TITLE: ...`` at the top are read as metadata;
    ``## Heading`` lines start a new section. Plain transcripts without either
    still work: every blank-line-separated paragraph becomes a unit.
    """
    lines = path.read_text(encoding="utf-8").splitlines()
    title = path.stem
    i = 0
    while i < len(lines) and (HEADER_RE.match(lines[i]) or not lines[i].strip()):
        m = HEADER_RE.match(lines[i])
        if m and m.group(1) == "TITLE":
            title = m.group(2).strip()
        i += 1

    section, buf, units = "", [], []

    def flush():
        para = " ".join(s.strip() for s in buf if s.strip())
        if para:
            units.append((section, para))
        buf.clear()

    for line in lines[i:]:
        if line.startswith("## "):
            flush()
            section = line[3:].strip()
        elif not line.strip():
            flush()
        else:
            buf.append(line)
    flush()
    return title, units


def _split_long(text: str) -> list[str]:
    if len(text) <= MAX_CHARS:
        return [text]
    parts, cur = [], ""
    for sent in SENTENCE_RE.split(text):
        if cur and len(cur) + len(sent) > TARGET_CHARS:
            parts.append(cur.strip())
            cur = ""
        cur += sent + " "
    if cur.strip():
        parts.append(cur.strip())
    return parts


def chunk_corpus(corpus_dir: Path = CORPUS_DIR) -> list[Chunk]:
    chunks: list[Chunk] = []
    for path in sorted(corpus_dir.glob("*.txt")):
        title, units = _parse_file(path)
        pending_section, pending = None, ""
        for section, para in units:
            for piece in _split_long(para):
                same_section = pending_section == section
                if pending and same_section and len(pending) < MIN_CHARS and len(pending) + len(piece) <= TARGET_CHARS:
                    pending = f"{pending}\n{piece}"
                    continue
                if pending:
                    chunks.append(Chunk(len(chunks), path.name, title, pending_section or "", pending))
                pending_section, pending = section, piece
        if pending:
            chunks.append(Chunk(len(chunks), path.name, title, pending_section or "", pending))
    return chunks


def _embed_text(c: Chunk) -> str:
    head = f"{c.title} — {c.section}" if c.section else c.title
    return f"{head}\n{c.text}"


# ---------------------------------------------------------------- index files

def corpus_fingerprint(corpus_dir: Path = CORPUS_DIR, model_name: str = EMBEDDING_MODEL) -> str:
    h = hashlib.sha256(f"{model_name}|{CHUNKER_VERSION}".encode())
    for path in sorted(corpus_dir.glob("*.txt")):
        h.update(path.name.encode())
        h.update(hashlib.sha256(path.read_bytes()).digest())
    return h.hexdigest()


def index_is_stale(corpus_dir: Path = CORPUS_DIR, index_dir: Path = INDEX_DIR,
                   model_name: str = EMBEDDING_MODEL) -> bool:
    manifest = index_dir / "manifest.json"
    if not manifest.exists() or not (index_dir / "faiss.index").exists():
        return True
    saved = json.loads(manifest.read_text(encoding="utf-8")).get("fingerprint")
    return saved != corpus_fingerprint(corpus_dir, model_name)


def load_embedder(model_name: str = EMBEDDING_MODEL):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(model_name)


def _encode(embedder, texts: list[str]) -> np.ndarray:
    vecs = embedder.encode(texts, normalize_embeddings=True, convert_to_numpy=True,
                           show_progress_bar=False)
    return np.asarray(vecs, dtype="float32")


def build_index(embedder, corpus_dir: Path = CORPUS_DIR, index_dir: Path = INDEX_DIR,
                model_name: str = EMBEDDING_MODEL) -> int:
    """Rebuild the FAISS index. Returns the number of chunks indexed."""
    import faiss

    chunks = chunk_corpus(corpus_dir)
    index_dir.mkdir(parents=True, exist_ok=True)
    dim = embedder.get_sentence_embedding_dimension()
    index = faiss.IndexFlatIP(dim)  # inner product on normalized vectors = cosine
    if chunks:
        index.add(_encode(embedder, [_embed_text(c) for c in chunks]))
    faiss.write_index(index, str(index_dir / "faiss.index"))
    (index_dir / "chunks.json").write_text(
        json.dumps([asdict(c) for c in chunks], ensure_ascii=False, indent=1), encoding="utf-8")
    (index_dir / "manifest.json").write_text(json.dumps({
        "fingerprint": corpus_fingerprint(corpus_dir, model_name),
        "model": model_name,
        "chunks": len(chunks),
    }, indent=1), encoding="utf-8")
    return len(chunks)


# ---------------------------------------------------------------- search

class Retriever:
    def __init__(self, embedder, index_dir: Path = INDEX_DIR):
        import faiss

        self.embedder = embedder
        self.index = faiss.read_index(str(index_dir / "faiss.index"))
        raw = json.loads((index_dir / "chunks.json").read_text(encoding="utf-8"))
        self.chunks = [Chunk(**c) for c in raw]

    def search(self, query: str, k: int = 4) -> list[tuple[Chunk, float]]:
        if not self.chunks or not query.strip():
            return []
        scores, ids = self.index.search(_encode(self.embedder, [query]), min(k, len(self.chunks)))
        return [(self.chunks[i], float(s)) for i, s in zip(ids[0], scores[0]) if i >= 0]


def relevance_tier(top_score: float | None, high: float, low: float) -> str:
    """Map the best similarity score to an answering strategy: high / medium / low."""
    if top_score is None or top_score < low:
        return "low"
    return "high" if top_score >= high else "medium"


# ---------------------------------------------------------------- CLI

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rebuild", action="store_true", help="rebuild the index")
    ap.add_argument("--check", action="store_true", help="check whether the index is stale")
    ap.add_argument("--query", help="run a test query and print file/section/score")
    ap.add_argument("-k", type=int, default=4)
    args = ap.parse_args()

    if args.check:
        print("Index is OUT OF DATE." if index_is_stale() else "Index is up to date.")
        return
    embedder = load_embedder()
    if args.rebuild or index_is_stale():
        n = build_index(embedder)
        print(f"Indexed {n} chunks from {CORPUS_DIR} into {INDEX_DIR}")
    if args.query:
        for chunk, score in Retriever(embedder).search(args.query, args.k):
            print(f"{score:.3f}  {chunk.source}  [{chunk.section}]")


if __name__ == "__main__":
    main()
