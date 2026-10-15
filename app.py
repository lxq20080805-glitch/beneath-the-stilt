"""Beneath the Stilt — Streamlit chatbot about Chongqing's stilt houses.

Run locally:  streamlit run app.py
"""

from __future__ import annotations

import datetime as dt
import logging
import os
import threading
import time

import streamlit as st

import rag
from prompts import build_messages

st.set_page_config(page_title="Beneath the Stilt · 吊脚楼下", page_icon="🏚️", layout="centered")
log = logging.getLogger("beneath_the_stilt")


# ---------------------------------------------------------------- settings

def cfg(name: str, default):
    """Read a setting from Streamlit secrets, then environment variables, then the default."""
    value = None
    try:
        value = st.secrets.get(name)
    except Exception:  # no secrets.toml at all
        pass
    if value in (None, ""):
        value = os.environ.get(name)
    if value in (None, ""):
        return default
    if isinstance(default, bool):
        return str(value).lower() in ("1", "true", "yes")
    return type(default)(value)


API_KEY = cfg("OPENAI_API_KEY", "")
MODEL = cfg("OPENAI_MODEL", "gpt-4o-mini")
MAX_TOKENS = cfg("MAX_OUTPUT_TOKENS", 600)
TOP_K = cfg("TOP_K", 4)
# Cosine-similarity cut-offs for the answering strategy. These depend on the embedding
# model and the corpus — check them with `python rag.py --query "..."` and adjust.
HIGH_THRESHOLD = cfg("HIGH_THRESHOLD", 0.65)
LOW_THRESHOLD = cfg("LOW_THRESHOLD", 0.30)
HISTORY_TURNS = cfg("HISTORY_TURNS", 6)
FOLLOWUP_MAX_CHARS = 20  # questions this short are searched together with the previous one
SESSION_LIMIT = cfg("SESSION_MESSAGE_LIMIT", 30)
DAILY_LIMIT = cfg("DAILY_MESSAGE_LIMIT", 500)
MIN_INTERVAL = cfg("MIN_SECONDS_BETWEEN_MESSAGES", 3.0)
DEBUG_SOURCES = cfg("DEBUG_SOURCES", False)

EXAMPLES = [
    "What was daily life like inside a Chongqing stilt house?",
    "吊脚楼为什么能在江边悬崖上屹立不倒？",
    "Who was the night watchman of Xiahao Old Street?",
    "老重庆人过去怎么取水、用电？",
]


# ---------------------------------------------------------------- cached resources

@st.cache_resource(show_spinner=False)
def get_embedder(model_name: str):
    return rag.load_embedder(model_name)


@st.cache_resource(show_spinner=False)
def get_retriever(fingerprint: str):
    """Cached per corpus fingerprint, so adding or editing a transcript triggers a rebuild."""
    embedder = get_embedder(rag.EMBEDDING_MODEL)
    if rag.index_is_stale():
        n = rag.build_index(embedder)
        log.info("Rebuilt index: %d chunks", n)
    return rag.Retriever(embedder)


@st.cache_resource(show_spinner=False)
def daily_usage():
    return {"date": dt.date.today(), "count": 0, "lock": threading.Lock()}


@st.cache_resource(show_spinner=False)
def get_client(api_key: str):
    from openai import OpenAI
    return OpenAI(api_key=api_key, timeout=60, max_retries=1)


# ---------------------------------------------------------------- logic

def retrieve(question: str, history: list[dict]):
    """Return (chunks, tier, debug). Falls back to general knowledge if retrieval fails."""
    try:
        retriever = get_retriever(rag.corpus_fingerprint())
    except Exception:
        log.exception("Retriever unavailable; answering from general knowledge")
        return [], "low", []

    results = retriever.search(question, TOP_K)
    # For short follow-ups ("what happened next?", "他晚上做什么？") the previous question carries
    # the topic. Longer questions stand on their own, so an off-topic question can't borrow
    # relevance from the previous one.
    prev = next((m["content"] for m in reversed(history) if m["role"] == "user"), None)
    if prev and len(question.strip()) <= FOLLOWUP_MAX_CHARS:
        alt = retriever.search(f"{prev}\n{question}", TOP_K)
        if alt and (not results or alt[0][1] > results[0][1]):
            results = alt

    top = results[0][1] if results else None
    tier = rag.relevance_tier(top, HIGH_THRESHOLD, LOW_THRESHOLD)
    chunks = [c for c, s in results if s >= LOW_THRESHOLD] if tier != "low" else []
    debug = [(c.source, round(s, 3)) for c, s in results]  # file names + scores only, never text
    if DEBUG_SOURCES:
        log.info("tier=%s sources=%s", tier, debug)
    return chunks, tier, debug


def rate_limit_message() -> str | None:
    ss = st.session_state
    if ss.sent >= SESSION_LIMIT:
        return "This conversation has reached its question limit. Please refresh the page to start again. / 本次对话的提问次数已达上限，请刷新页面重新开始。"
    if time.time() - ss.last_sent < MIN_INTERVAL:
        return "Please wait a moment before asking again. / 请稍等片刻再提问。"
    usage = daily_usage()
    with usage["lock"]:
        if usage["date"] != dt.date.today():
            usage["date"], usage["count"] = dt.date.today(), 0
        if usage["count"] >= DAILY_LIMIT:
            return "The guide is resting for today — please come back tomorrow. / 今日问答名额已用完，欢迎明天再来。"
        usage["count"] += 1
    ss.sent += 1
    ss.last_sent = time.time()
    return None


def stream_answer(messages: list[dict]):
    stream = get_client(API_KEY).chat.completions.create(
        model=MODEL, messages=messages, max_completion_tokens=MAX_TOKENS, stream=True)
    for event in stream:
        if event.choices and event.choices[0].delta.content:
            yield event.choices[0].delta.content


def friendly_error(exc: Exception) -> str:
    import openai
    if isinstance(exc, (openai.AuthenticationError, openai.PermissionDeniedError)):
        return "The guide is not configured correctly right now. Please contact the site owner. / 导览服务配置有误，请联系网站管理员。"
    if isinstance(exc, openai.RateLimitError):
        return "The guide is busy at the moment. Please try again shortly. / 导览暂时繁忙，请稍后再试。"
    if isinstance(exc, (openai.APIConnectionError, openai.APITimeoutError)):
        return "Could not reach the guide. Please check your connection and try again. / 暂时无法连接导览服务，请稍后重试。"
    return "Sorry, something went wrong while answering. Please try again. / 抱歉，回答时出现问题，请重试。"


# ---------------------------------------------------------------- page

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,500;0,600;1,500&family=Noto+Serif+SC:wght@400;600&display=swap');
:root { --paper:#f4efe6; --ink:#2a2420; --muted:#7a6f63; --rule:#cfc4b3; --accent:#8a3b24; }
html, body, .stApp { background: var(--paper); color: var(--ink);
  font-family: 'Noto Serif SC', 'Songti SC', Georgia, serif; }
header[data-testid="stHeader"] { background: transparent; }
#MainMenu, footer { visibility: hidden; }
.block-container { max-width: 760px; padding-top: 2.5rem; padding-bottom: 6rem; }
.bts-kicker { font-size: .72rem; letter-spacing: .22em; text-transform: uppercase; color: var(--muted);
  border-bottom: 1px solid var(--rule); padding-bottom: .6rem; margin-bottom: 1.4rem; }
.bts-title { font-family: 'Cormorant Garamond', Georgia, serif; font-weight: 600; font-size: 3.4rem;
  line-height: 1; margin: 0; letter-spacing: -.01em; }
.bts-sub { font-size: 1.05rem; color: var(--accent); letter-spacing: .5em; margin: .5rem 0 1.4rem; }
.bts-intro { font-size: 1rem; line-height: 1.75; margin: 0 0 .6rem; }
.bts-intro.zh { color: #4a4138; font-size: .95rem; }
.bts-note { font-size: .78rem; color: var(--muted); font-style: italic;
  border-top: 1px solid var(--rule); padding-top: .7rem; margin-top: 1.2rem; }
.bts-label { font-size: .72rem; letter-spacing: .18em; text-transform: uppercase; color: var(--muted);
  margin: 1.6rem 0 .5rem; }
.stButton > button { width: 100%; text-align: left; justify-content: flex-start; background: transparent;
  border: 1px solid var(--rule); border-radius: 2px; color: var(--ink); padding: .7rem .9rem;
  font-family: inherit; line-height: 1.45; min-height: 3.6rem; }
.stButton > button:hover { border-color: var(--accent); color: var(--accent); background: #efe7da; }
[data-testid="stChatMessage"] { background: transparent; border-bottom: 1px solid #e2d9ca;
  border-radius: 0; padding: 1rem .2rem; }
[data-testid="stChatMessage"] p { line-height: 1.8; }
[data-testid="stChatInput"] textarea { font-family: inherit; }
@media (max-width: 640px) {
  .block-container { padding-top: 1.5rem; }
  .bts-title { font-size: 2.5rem; }
  .bts-sub { letter-spacing: .3em; }
}
</style>
<div class="bts-kicker">Chongqing · Oral Histories · 重庆口述记忆</div>
<h1 class="bts-title">Beneath the Stilt</h1>
<div class="bts-sub">吊脚楼下</div>
<p class="bts-intro">On the cliffs above the Yangtze and Jialing rivers, wooden stilt houses once climbed over one another,
held up by little more than fir poles. Ask about how they were built, and about the people who lived in them —
their streets, their work, their memories.</p>
<p class="bts-intro zh">长江与嘉陵江畔的悬崖上，吊脚楼曾层层叠叠、凭几根杉杆撑起一方天地。
欢迎向导览提问：关于吊脚楼的建造，也关于住在楼里的人——他们的街巷、生计与记忆。</p>
<div class="bts-note">Answers draw on former residents' recollections and on general historical background.
AI can make mistakes. · 回答结合老居民的回忆与一般历史背景，AI 也可能出错。</div>
""", unsafe_allow_html=True)

ss = st.session_state
ss.setdefault("messages", [])
ss.setdefault("sent", 0)
ss.setdefault("last_sent", 0.0)

if not API_KEY:
    st.info("**The guide is not configured yet.** Add `OPENAI_API_KEY` to `.streamlit/secrets.toml` "
            "(or the app's Secrets on Streamlit Community Cloud) and restart. See README.md.\n\n"
            "**导览尚未配置。** 请在 `.streamlit/secrets.toml` 中填写 `OPENAI_API_KEY` 后重启网站，详见 README.md。")

if not ss.messages:
    st.markdown('<div class="bts-label">Try asking · 试着问问</div>', unsafe_allow_html=True)
    cols = st.columns(2)
    for i, q in enumerate(EXAMPLES):
        if cols[i % 2].button(q, key=f"ex{i}", disabled=not API_KEY):
            ss.pending = q
            st.rerun()

AVATARS = {"user": ":material/person:", "assistant": ":material/cottage:"}
for m in ss.messages:
    with st.chat_message(m["role"], avatar=AVATARS[m["role"]]):
        st.markdown(m["content"])

typed = st.chat_input("Ask about the stilt houses… / 问问吊脚楼的故事…", max_chars=500, disabled=not API_KEY)
question = typed or ss.pop("pending", None)

if question and API_KEY:
    with st.chat_message("user", avatar=AVATARS["user"]):
        st.markdown(question)

    with st.chat_message("assistant", avatar=AVATARS["assistant"]):
        blocked = rate_limit_message()
        if blocked:
            st.warning(blocked)
        else:
            history = list(ss.messages)
            ss.messages.append({"role": "user", "content": question})
            with st.spinner("Searching the memories… / 正在翻阅记忆…"):
                chunks, tier, debug = retrieve(question, history)
            messages = build_messages(question, history, chunks, tier, HISTORY_TURNS)
            try:
                answer = st.write_stream(stream_answer(messages))
            except Exception as exc:
                log.warning("LLM call failed: %s", type(exc).__name__)
                st.error(friendly_error(exc))
                ss.messages.pop()  # drop the unanswered question so it can be asked again
            else:
                if isinstance(answer, list):
                    answer = "".join(str(a) for a in answer)
                ss.messages.append({"role": "assistant", "content": answer,
                                    "tier": tier, "sources": debug})  # kept internally, never rendered

if ss.messages:
    if st.button("Start a new conversation · 开始新的对话", key="reset"):
        ss.messages = []
        st.rerun()
