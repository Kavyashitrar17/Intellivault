import streamlit as st
import requests
import os
import time
from datetime import datetime

st.set_page_config(
    page_title="IntelliVault — Vault",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Auth guard ────────────────────────────────────────────────────────────────
if not st.session_state.get("logged_in", False):
    st.switch_page("app.py")

# ── Load CSS ──────────────────────────────────────────────────────────────────
css_path = os.path.join(os.path.dirname(__file__), "..", "style.css")
with open(css_path) as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

BACKEND_URL = "http://127.0.0.1:8000"

# ── Session State ─────────────────────────────────────────────────────────────
defaults = {
    "chat_history":   [],
    "uploaded_docs": {},
    "selected_docs": [],
    "show_chunks":   True,
    "ask_in_progress": False,
    "scroll_to_latest": False,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── Helpers ───────────────────────────────────────────────────────────────────
def upload_to_backend(file):
    try:
        res = requests.post(
            f"{BACKEND_URL}/upload",
            files={"file": (file.name, file.getvalue(), file.type)},
            timeout=30
        )
        if res.status_code == 200:
            return res.json()
        return {"error": f"Backend upload failed ({res.status_code}): {res.text}"}
    except Exception as exc:
        return {"error": str(exc)}


def ask_backend(question, top_k, selected_docs):
    try:
        res = requests.post(
            f"{BACKEND_URL}/query",
            json={"question": question, "top_k": top_k, "docs": selected_docs},
            timeout=30
        )
        if res.status_code == 200:
            return res.json()
        return {"error": f"Backend query failed ({res.status_code}): {res.text}"}
    except Exception as exc:
        return {"error": str(exc)}

def backend_alive():
    try:
        r = requests.get(f"{BACKEND_URL}/health", timeout=2)
        return r.status_code == 200
    except Exception:
        return False

def build_export():
    lines = ["IntelliVault — Q&A Export", "=" * 60, ""]
    for i, e in enumerate(st.session_state.chat_history, 1):
        lines += [f"Q{i}: {e['question']}", f"A{i}: {e['answer']}",
                  f"Source: {e.get('source','N/A')}", f"Time: {e.get('ts','')}", "-"*60, ""]
    return "\n".join(lines)

# ═══════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════
with st.sidebar:
    username = st.session_state.get("username", "user").capitalize()
    st.markdown(f"""
    <div class="brand-block">
        <div class="brand-icon">🧠</div>
        <div>
            <div class="brand-name">IntelliVault</div>
            <div class="brand-sub">Signed in as <strong>{username}</strong></div>
        </div>
    </div>""", unsafe_allow_html=True)

    alive = backend_alive()
    if alive:
        st.markdown('<div class="status-badge live">● Backend Connected</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="status-badge demo">● Demo Mode  (backend offline)</div>', unsafe_allow_html=True)

    st.markdown("---")

    # ── Upload ────────────────────────────────────────────────
    st.markdown('<div class="sb-title">📁 Upload Documents</div>', unsafe_allow_html=True)
    uploaded_files = st.file_uploader(
        "Files", type=["pdf", "txt"],
        accept_multiple_files=True, label_visibility="collapsed"
    )

    if uploaded_files:
        if st.button("⚡ Chunk & Embed", use_container_width=True):
            new = [f for f in uploaded_files if f.name not in st.session_state.uploaded_docs]
            if not new:
                st.info("All files already processed.")
            for f in new:
                bar = st.progress(0, text=f"Processing {f.name}…")
                t0 = time.time()
                for p in range(0, 85, 20):
                    time.sleep(0.07)
                    bar.progress(p, text=f"Chunking {f.name}…")
                result = upload_to_backend(f)
                bar.progress(100, text="✓ Done!")
                time.sleep(0.3)
                bar.empty()
                elapsed = round(time.time() - t0, 2)

                if result is None or (isinstance(result, dict) and result.get("error")):
                    err_msg = "Unknown error while uploading."
                    if isinstance(result, dict):
                        err_msg = result.get("error", err_msg)
                    st.error(f"Failed to upload {f.name}: {err_msg}")
                    meta = {
                        "chunks":     "~0",
                        "size_kb":    round(f.size / 1024, 1),
                        "embed_time": elapsed,
                        "live":       False,
                    }
                else:
                    meta = {
                        "chunks":     result.get("chunks", "~40"),
                        "size_kb":    round(f.size / 1024, 1),
                        "embed_time": elapsed,
                        "live":       True,
                    }

                st.session_state.uploaded_docs[f.name] = meta
                if f.name not in st.session_state.selected_docs:
                    st.session_state.selected_docs.append(f.name)

                if result is None or (isinstance(result, dict) and result.get("error")):
                    st.warning(f"Demo — {f.name} registered (start backend for full RAG)")
                else:
                    st.success(f"✓ {f.name} — {meta['chunks']} chunks")

    st.markdown("---")

    # ── Document Library ──────────────────────────────────────
    if st.session_state.uploaded_docs:
        st.markdown('<div class="sb-title">📚 Document Library</div>', unsafe_allow_html=True)
        for fname, meta in list(st.session_state.uploaded_docs.items()):
            icon = "📕" if fname.lower().endswith(".pdf") else "📄"
            dot  = "🟢" if meta["live"] else "🟡"
            c1, c2 = st.columns([7, 1])
            with c1:
                sel = st.checkbox(
                    f"{icon} {fname}",
                    value=(fname in st.session_state.selected_docs),
                    key=f"sel_{fname}"
                )
                if sel and fname not in st.session_state.selected_docs:
                    st.session_state.selected_docs.append(fname)
                elif not sel and fname in st.session_state.selected_docs:
                    st.session_state.selected_docs.remove(fname)
                st.markdown(
                    f'<div class="doc-meta">{dot} {meta["chunks"]} chunks · '
                    f'{meta["size_kb"]} KB · {meta["embed_time"]}s</div>',
                    unsafe_allow_html=True
                )
            with c2:
                if st.button("✕", key=f"del_{fname}"):
                    del st.session_state.uploaded_docs[fname]
                    if fname in st.session_state.selected_docs:
                        st.session_state.selected_docs.remove(fname)
                    st.rerun()

        ca, cb = st.columns(2)
        with ca:
            if st.button("Select All", use_container_width=True):
                st.session_state.selected_docs = list(st.session_state.uploaded_docs.keys())
                st.rerun()
        with cb:
            if st.button("Deselect All", use_container_width=True):
                st.session_state.selected_docs = []
                st.rerun()
        st.markdown("---")

    # ── Settings ──────────────────────────────────────────────
    st.markdown('<div class="sb-title">⚙️ Settings</div>', unsafe_allow_html=True)
    top_k = st.slider("Top-K chunks", 1, 10, 3)
    st.session_state.show_chunks = st.toggle(
        "Show RAG context window", value=st.session_state.show_chunks
    )

    if st.session_state.chat_history:
        st.markdown("---")
        ec, cc = st.columns(2)
        with ec:
            st.download_button(
                "⬇ Export", data=build_export(),
                file_name=f"intellivault_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
                mime="text/plain", use_container_width=True
            )
        with cc:
            if st.button("🗑 Clear", use_container_width=True):
                st.session_state.chat_history = []
                st.rerun()

    st.markdown("---")
    if st.button("🚪 Sign Out", use_container_width=True):
        for key in ["logged_in", "username", "show_welcome",
                    "chat_history", "uploaded_docs", "selected_docs"]:
            if key in st.session_state:
                del st.session_state[key]
        st.switch_page("app.py")


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════
total_chunks = sum(
    int(str(m["chunks"]).replace("~", ""))
    for m in st.session_state.uploaded_docs.values()
    if str(m["chunks"]).replace("~", "").isdigit()
)

# ── Hero ──────────────────────────────────────────────────────
st.markdown(f"""
<div class="hero">
    <div class="hero-left">
        <div class="hero-eyebrow">RAG-Powered · Semantic Search · Source-Cited</div>
        <h1 class="hero-title">Ask Your Documents.<br>Get Grounded Answers.</h1>
        <p class="hero-desc">
            Upload your PDFs and notes. IntelliVault chunks, embeds, and indexes them
            into a <strong>FAISS vector database</strong>. Ask any question —
            it retrieves the most relevant passages and uses <strong>GPT-2</strong>
            to generate a precise, hallucination-free answer.
        </p>
    </div>
    <div class="hero-metrics">
        <div class="metric-box">
            <div class="metric-val">{len(st.session_state.uploaded_docs)}</div>
            <div class="metric-lbl">Documents</div>
        </div>
        <div class="metric-box">
            <div class="metric-val">{len(st.session_state.selected_docs)}</div>
            <div class="metric-lbl">Active</div>
        </div>
        <div class="metric-box">
            <div class="metric-val">{total_chunks}</div>
            <div class="metric-lbl">Chunks</div>
        </div>
        <div class="metric-box">
            <div class="metric-val">{len(st.session_state.chat_history)}</div>
            <div class="metric-lbl">Queries</div>
        </div>
    </div>
</div>""", unsafe_allow_html=True)

# ── Pipeline ──────────────────────────────────────────────────
st.markdown("""
<div class="pipeline">
    <div class="pipe-node"><div class="pipe-ico">📄</div><div class="pipe-lbl">Upload<br>Document</div></div>
    <div class="pipe-arr">›</div>
    <div class="pipe-node"><div class="pipe-ico">✂️</div><div class="pipe-lbl">Chunk &<br>Split</div></div>
    <div class="pipe-arr">›</div>
    <div class="pipe-node"><div class="pipe-ico">🔢</div><div class="pipe-lbl">Sentence<br>Embeddings</div></div>
    <div class="pipe-arr">›</div>
    <div class="pipe-node"><div class="pipe-ico">🗄️</div><div class="pipe-lbl">FAISS<br>Index</div></div>
    <div class="pipe-arr">›</div>
    <div class="pipe-node"><div class="pipe-ico">🔍</div><div class="pipe-lbl">Semantic<br>Retrieval</div></div>
    <div class="pipe-arr">›</div>
    <div class="pipe-node"><div class="pipe-ico">🤖</div><div class="pipe-lbl">GPT-2<br>Answer</div></div>
    <div class="pipe-arr">›</div>
    <div class="pipe-node pipe-end"><div class="pipe-ico">💬</div><div class="pipe-lbl">Cited<br>Response</div></div>
</div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Empty State ───────────────────────────────────────────────
if not st.session_state.uploaded_docs:
    st.markdown("""
    <div class="empty-card">
        <div class="empty-ico">📂</div>
        <div class="empty-title">Your vault is empty</div>
        <div class="empty-body">Upload documents from the sidebar to begin querying your knowledge vault.</div>
        <div class="how-steps">
            <div class="how-step"><div class="step-num">1</div><div class="step-txt">Upload PDF or TXT from the sidebar</div></div>
            <div class="how-step"><div class="step-num">2</div><div class="step-txt">Click "Chunk & Embed" to index your documents</div></div>
            <div class="how-step"><div class="step-num">3</div><div class="step-txt">Ask any question — get a cited, grounded answer</div></div>
        </div>
    </div>""", unsafe_allow_html=True)

else:
    # ── Ask Section ───────────────────────────────────────────
    st.markdown('<div class="sec-hdr">💬 Ask a Question</div>', unsafe_allow_html=True)

    if not st.session_state.selected_docs:
        st.warning("⚠️ No documents selected. Tick at least one in the sidebar.")

    with st.form("ask_form", clear_on_submit=True):
        question = st.text_area(
            "q", height=95, label_visibility="collapsed",
            placeholder=(
                "e.g.  What are the four necessary conditions for deadlock?\n"
                "e.g.  Explain normalization in DBMS.\n"
                "e.g.  What is the difference between process and thread?"
            )
        )

        if st.session_state.ask_in_progress:
            st.info("⏳ Please wait, your previous question is being answered...")

        _, btn_col = st.columns([5, 1])
        with btn_col:
            submitted = st.form_submit_button(
                "Ask →",
                use_container_width=True,
                disabled=st.session_state.ask_in_progress,
            )

    if submitted:
        q = question.strip()
        if not q:
            st.warning("Please enter a question.")
        elif not st.session_state.selected_docs:
            st.error("Select at least one document first.")
        else:
            st.session_state.ask_in_progress = True
            with st.spinner("Retrieving context and generating answer…"):
                result = ask_backend(q, top_k, st.session_state.selected_docs)
            st.session_state.ask_in_progress = False

            if result and isinstance(result, dict) and result.get("error"):
                st.error(f"Backend error: {result['error']}")

            if result and isinstance(result, dict) and not result.get("error"):
                entry = {
                    "question": q,
                    "answer":   result.get("answer", "No answer returned."),
                    "source":   result.get("source", "Unknown"),
                    "chunks":   result.get("chunks", []),
                    "ts":       datetime.now().strftime("%H:%M"),
                    "mode":     "live",
                }
            else:
                entry = {
                    "question": q,
                    "answer": (
                        "Backend is offline — running in demo mode.\n\n"
                        "Start FastAPI with:\n    uvicorn backend.main:app --reload"
                    ),
                    "source": "N/A (demo)",
                    "chunks": [
                        {"text": "Sample chunk 1: This would be the most relevant passage retrieved by FAISS.",
                         "score": 0.91, "source": "document.pdf", "chunk_id": "chunk_007"},
                        {"text": "Sample chunk 2: Second most relevant passage used as context for GPT-2.",
                         "score": 0.78, "source": "document.pdf", "chunk_id": "chunk_014"},
                        {"text": "Sample chunk 3: Additional supporting context for the answer.",
                         "score": 0.62, "source": "document.pdf", "chunk_id": "chunk_021"},
                    ],
                    "ts":   datetime.now().strftime("%H:%M"),
                    "mode": "demo",
                }

            st.session_state.chat_history.append(entry)
            st.session_state.scroll_to_latest = True
            st.rerun()
                entry = {
                    "question": q,
                    "answer": (
                        "Backend is offline — running in demo mode.\n\n"
                        "Start FastAPI with:\n    uvicorn backend.main:app --reload"
                    ),
                    "source": "N/A (demo)",
                    "chunks": [
                        {"text": "Sample chunk 1: This would be the most relevant passage retrieved by FAISS.",
                         "score": 0.91, "source": "document.pdf", "chunk_id": "chunk_007"},
                        {"text": "Sample chunk 2: Second most relevant passage used as context for GPT-2.",
                         "score": 0.78, "source": "document.pdf", "chunk_id": "chunk_014"},
                        {"text": "Sample chunk 3: Additional supporting context for the answer.",
                         "score": 0.62, "source": "document.pdf", "chunk_id": "chunk_021"},
                    ],
                    "ts":   datetime.now().strftime("%H:%M"),
                    "mode": "demo",
                }
            st.session_state.chat_history.append(entry)
            st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Chat History ──────────────────────────────────────────
    if st.session_state.chat_history:
        n_q = len(st.session_state.chat_history)
        st.markdown(
            f'<div class="sec-hdr">🕑 Conversation History '
            f'<span class="hist-badge">{n_q} {"query" if n_q==1 else "queries"}</span></div>',
            unsafe_allow_html=True
        )

        for idx, entry in enumerate(reversed(st.session_state.chat_history)):
            n    = len(st.session_state.chat_history) - idx
            mode = entry.get("mode", "demo")

            st.markdown(f"""
            <div class="q-card">
                <div class="q-left"><span class="q-num">Q{n}</span></div>
                <div class="q-right">
                    <div class="q-text">{entry['question']}</div>
                    <div class="q-foot">{entry.get('ts','')} · {'🟢 Live' if mode=='live' else '🟡 Demo'}</div>
                </div>
            </div>""", unsafe_allow_html=True)

            ans = entry['answer'].replace('\n', '<br>')
            st.markdown(f"""
            <div class="ans-card">
                <div class="ans-header">
                    <span class="ans-badge">🧠 Answer</span>
                    <span class="ans-src">📎 {entry.get('source','N/A')}</span>
                </div>
                <div class="ans-body">{ans}</div>
            </div>""", unsafe_allow_html=True)

            chunks = entry.get("chunks", [])
            if chunks and st.session_state.show_chunks:
                with st.expander(
                    f"🔍  RAG Context Window — {len(chunks)} chunk(s) retrieved",
                    expanded=False
                ):
                    st.markdown("""
                    <div class="rag-intro">
                      Passages retrieved via <strong>semantic similarity</strong>
                      (all-MiniLM-L6-v2 + FAISS). GPT-2 generates its answer
                      <em>only</em> from this context — no hallucination.
                    </div>""", unsafe_allow_html=True)

                    for ci, ch in enumerate(chunks):
                        score  = ch.get("score", None)
                        c_text = ch.get("text", "")
                        c_src  = ch.get("source", "")
                        c_id   = ch.get("chunk_id", f"chunk_{ci+1}")

                        if isinstance(score, float):
                            pct = int(score * 100)
                            s_str = f"{score:.3f}"
                            if score >= 0.85:   bar_clr, rel, rel_c = "#16a34a","High relevance","rel-hi"
                            elif score >= 0.70: bar_clr, rel, rel_c = "#d97706","Medium relevance","rel-md"
                            else:               bar_clr, rel, rel_c = "#dc2626","Low relevance","rel-lo"
                        else:
                            pct, s_str = 50, "N/A"
                            bar_clr, rel, rel_c = "#94a3b8","Unknown","rel-md"

                        st.markdown(f"""
                        <div class="chunk-card">
                            <div class="chunk-top">
                                <div class="chunk-left">
                                    <span class="chunk-badge">#{ci+1}</span>
                                    <span class="chunk-id">{c_id}</span>
                                    <span class="chunk-src">📄 {c_src}</span>
                                </div>
                                <div class="chunk-right">
                                    <span class="sim-val">sim: {s_str}</span>
                                    <span class="rel-tag {rel_c}">{rel}</span>
                                </div>
                            </div>
                            <div class="sim-track">
                                <div class="sim-fill" style="width:{pct}%;background:{bar_clr}"></div>
                            </div>
                            <div class="chunk-txt">{c_text}</div>
                        </div>""", unsafe_allow_html=True)

            st.markdown("<div class='q-sep'></div>", unsafe_allow_html=True)

        if st.session_state.scroll_to_latest:
            st.markdown("<div id='scroll-anchor'></div>", unsafe_allow_html=True)
            st.markdown(
                """
                <script>
                    const el = document.getElementById('scroll-anchor');
                    if (el) { el.scrollIntoView({behavior: 'smooth', block: 'end'}); }
                </script>
                """,
                unsafe_allow_html=True,
            )
            st.session_state.scroll_to_latest = False

    else:
        st.markdown("""
        <div class="no-q">
            <div style="font-size:30px;margin-bottom:8px">💬</div>
            <strong>No questions yet.</strong><br>
            Type a question above and press Ask →
        </div>""", unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
    <div class="footer-title">IntelliVault · RAG Architecture</div>
    <div class="footer-chips">
        <span class="chip">🐍 Python</span><span class="chip">⚡ FastAPI</span>
        <span class="chip">🎈 Streamlit</span><span class="chip">🗄️ FAISS</span>
        <span class="chip">🤗 SentenceTransformers</span><span class="chip">🤖 GPT-2</span>
    </div>
</div>""", unsafe_allow_html=True)