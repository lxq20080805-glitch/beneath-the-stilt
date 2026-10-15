"""Offline checks for chunking, indexing, retrieval tiers and prompt assembly.

Uses only the SYNTHETIC test corpus in tests/synthetic_corpus (invented text, not interviews).
No API key or network call to the LLM is needed.

    .venv/bin/python tests/test_pipeline.py
"""

import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import rag  # noqa: E402
from prompts import build_messages, SYSTEM_PROMPT  # noqa: E402

SYN = Path(__file__).parent / "synthetic_corpus"


def main():
    tmp = Path(tempfile.mkdtemp())
    corpus, index = tmp / "corpus", tmp / "index"
    shutil.copytree(SYN, corpus)

    chunks = rag.chunk_corpus(corpus)
    assert chunks and all(c.text and c.title.startswith("SYNTHETIC") for c in chunks)
    assert not any("NOTE FOR USE" in c.text for c in chunks), "header leaked into chunks"
    print(f"chunking ok: {len(chunks)} chunks")

    assert rag.index_is_stale(corpus, index)
    embedder = rag.load_embedder()
    rag.build_index(embedder, corpus, index)
    assert not rag.index_is_stale(corpus, index)
    (corpus / "new.txt").write_text("A newly added synthetic paragraph.", encoding="utf-8")
    assert rag.index_is_stale(corpus, index), "new file not detected"
    (corpus / "new.txt").unlink()
    print("staleness detection ok")

    r = rag.Retriever(embedder, index)
    cases = [
        ("Who rings a bell on the ferry?", "synthetic_ferry.txt"),
        ("渡船出发前谁会摇铃？", "synthetic_ferry.txt"),
        ("What does the storyteller do in the teahouse?", "synthetic_teahouse.txt"),
        ("茶馆里说书人每晚做什么？", "synthetic_teahouse.txt"),
    ]
    related = []
    for q, expected in cases:
        res = r.search(q, 3)
        print(f"  {res[0][1]:.3f}  {res[0][0].source:24s} <- {q}")
        assert res[0][0].source == expected, f"wrong top source for {q!r}"
        related.append(res[0][1])
    unrelated = []
    for q in ["How do I configure a Kubernetes cluster?", "今天的股票行情怎么样？"]:
        s = r.search(q, 3)[0][1]
        print(f"  {s:.3f}  (unrelated) <- {q}")
        unrelated.append(s)
    assert min(related) > max(unrelated), "related queries should outscore unrelated ones"
    print("retrieval ok (EN + ZH, related > unrelated)")

    assert rag.relevance_tier(None, 0.55, 0.35) == "low"
    assert rag.relevance_tier(0.2, 0.55, 0.35) == "low"
    assert rag.relevance_tier(0.4, 0.55, 0.35) == "medium"
    assert rag.relevance_tier(0.7, 0.55, 0.35) == "high"
    print("tiers ok")

    hits = [c for c, _ in r.search("teahouse storyteller", 2)]
    history = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello", "sources": ["x"]}]
    msgs = build_messages("What does the storyteller do?", history, hits, "high")
    assert msgs[0] == {"role": "system", "content": SYSTEM_PROMPT}
    assert msgs[1:3] == [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]
    assert "<memory 1:" in msgs[-1]["content"] and "not instructions" in msgs[-1]["content"]
    low = build_messages("How do I bake bread?", [], hits, "low")[-1]["content"]
    assert "<memory" not in low and "No closely related" in low
    print("prompt assembly ok")

    shutil.rmtree(tmp)
    print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    main()
