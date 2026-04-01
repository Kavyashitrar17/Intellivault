"""
pages/vault.py  — clean output only
"""

import requests
import streamlit as st
import re
from typing import Dict, List

API_BASE = "http://127.0.0.1:8000"


def _query(question: str) -> Dict:
    resp = requests.post(f"{API_BASE}/query", json={"query": question}, timeout=30)
    resp.raise_for_status()
    return resp.json()


def _upload(file) -> Dict:
    resp = requests.post(
        f"{API_BASE}/upload",
        files={"file": (file.name, file.getvalue(), file.type)},
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()


def _reset_index():
    requests.delete(f"{API_BASE}/reset", timeout=10)


_CONFIDENCE_META = {
    "high":   ("🟢", "High confidence"),
    "medium": ("🟡", "Medium confidence"),
    "low":    ("🔴", "Low confidence"),
    "none":   ("⚫", "No match found"),
}


def _format_answer(text: str) -> str:
    """Convert answer text to clean bullet list or paragraph."""
    text = text.replace("\\n", "\n").replace("\\•", "•")
    lines, html, in_list = text.strip().splitlines(), [], False
    for line in lines:
        s = line.strip()
        if not s:
            if in_list:
                html.append("</ul>")
                in_list = False
            continue
        if re.match(r"^[-•*]\s+", s):
            if not in_list:
                html.append('<ul class="answer-list">')
                in_list = True
            html.append(f"<li>{re.sub(r'^[-•*]\\s+', '', s)}</li>")
        else:
            if in_list:
                html.append("</ul>")
                in_list = False
            html.append(f"<p>{s}</p>")
    if in_list:
        html.append("</ul>")
    return "\n".join(html)


def _render_message(role: str, content: Dict):
    if role == "user":
        st.markdown(
            f'<div class="chat-row user-row">'
            f'<div class="bubble user-bubble"><p>{content["text"]}</p></div>'
            f'<div class="avatar user-avatar">You</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
        return

    # Answer only — no source HTML, no previews
    answer_html = _format_answer(content.get("answer", ""))
    icon, label = _CONFIDENCE_META.get(content.get("confidence", "none"), ("⚫", "No match"))
    files       = content.get("source_files", [])
    files_text  = "  ·  ".join(f"📄 {f}" for f in files) if files else ""

    st.markdown(
        f'<div class="chat-row assistant-row">'
        f'<div class="avatar assistant-avatar">IV</div>'
        f'<div class="bubble assistant-bubble">'
        f'<div class="answer-body">{answer_html}</div>'
        f'<div class="answer-meta">'
        f'<span class="badge badge-{content.get("confidence","none")}">{icon} {label}</span>'
        f'<span class="source-chip">{files_text}</span>'
        f'</div>'
        f'</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


def render_chat():
    if "messages"  not in st.session_state: st.session_state.messages  = []
    if "thinking"  not in st.session_state: st.session_state.thinking  = False

    st.markdown(
        '<div class="page-header">'
        '<h1 class="page-title">Ask your documents</h1>'
        '<p class="page-subtitle">Upload a PDF or TXT, then ask anything about it.</p>'
        '</div>',
        unsafe_allow_html=True,
    )

    _, col_btn = st.columns([9, 1])
    with col_btn:
        if st.button("Clear"):
            st.session_state.messages = []
            st.rerun()

    st.markdown('<div class="chat-container">', unsafe_allow_html=True)

    if not st.session_state.messages:
        st.markdown(
            '<div class="empty-state">'
            '<div class="empty-icon">💬</div>'
            '<p>No conversation yet. Ask a question below.</p>'
            '</div>',
            unsafe_allow_html=True,
        )
    else:
        for msg in st.session_state.messages:
            _render_message(msg["role"], msg["content"])

    if st.session_state.thinking:
        st.markdown(
            '<div class="chat-row assistant-row">'
            '<div class="avatar assistant-avatar">IV</div>'
            '<div class="bubble assistant-bubble thinking-bubble">'
            '<div class="thinking-dots"><span></span><span></span><span></span></div>'
            '</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="input-bar">', unsafe_allow_html=True)
    with st.form(key="chat_form", clear_on_submit=True):
        cols = st.columns([11, 1])
        with cols[0]:
            user_input = st.text_input(
                "q", placeholder="Ask a question about your documents...",
                label_visibility="collapsed",
            )
        with cols[1]:
            submitted = st.form_submit_button("➤")
    st.markdown("</div>", unsafe_allow_html=True)

    if submitted and user_input.strip():
        st.session_state.messages.append({"role": "user", "content": {"text": user_input.strip()}})
        st.session_state.thinking = True
        st.rerun()

    if st.session_state.thinking and st.session_state.messages:
        if st.session_state.messages[-1]["role"] == "user":
            try:
                result = _query(st.session_state.messages[-1]["content"]["text"])
                st.session_state.messages.append({"role": "assistant", "content": result})
            except requests.exceptions.ConnectionError:
                st.session_state.messages.append({"role": "assistant", "content": {
                    "answer": "⚠️ Cannot connect to backend. Is uvicorn running?",
                    "sources": [], "source_files": [], "confidence": "none",
                }})
            except Exception as e:
                st.session_state.messages.append({"role": "assistant", "content": {
                    "answer": f"⚠️ Error: {e}",
                    "sources": [], "source_files": [], "confidence": "none",
                }})
            finally:
                st.session_state.thinking = False
                st.rerun()


def render_upload():
    st.markdown(
        '<div class="page-header">'
        '<h1 class="page-title">Upload documents</h1>'
        '<p class="page-subtitle">Add PDFs or TXT files to your vault.</p>'
        '</div>',
        unsafe_allow_html=True,
    )

    uploaded = st.file_uploader(
        "Drop files here or click to browse",
        type=["pdf", "txt"],
        accept_multiple_files=True,
    )

    if uploaded:
        if st.button("📥  Index all files", use_container_width=True):
            for f in uploaded:
                with st.spinner(f"Processing {f.name}..."):
                    try:
                        r = _upload(f)
                        st.success(
                            f"✅ {f.name} — "
                            f"{r.get('chunks_added','?')} chunks, "
                            f"{r.get('total_vectors','?')} total vectors"
                        )
                    except Exception as e:
                        st.error(f"❌ {f.name}: {e}")

    st.markdown("<br>", unsafe_allow_html=True)
    with st.expander("⚠️ Danger zone"):
        st.warning("Deletes all indexed vectors and chunks.")
        if st.button("🗑️ Reset index", type="primary"):
            try:
                _reset_index()
                st.success("Index cleared.")
            except Exception as e:
                st.error(f"Reset failed: {e}")