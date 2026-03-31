"""
pages/vault.py
--------------
Two views:
  render_chat()   — ChatGPT-style Q&A interface
  render_upload() — Document upload with live feedback
"""

import requests
import streamlit as st
import time
import re
from typing import Dict, List


API_BASE = "http://127.0.0.1:8000"


# ═══════════════════════════════════════════════════════
# API helpers
# ═══════════════════════════════════════════════════════

def _query(question: str) -> Dict:
    """POST /query — returns full response dict or raises."""
    resp = requests.post(
        f"{API_BASE}/query",
        json={"query": question},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


def _upload(file) -> Dict:
    """POST /upload — returns response dict or raises."""
    resp = requests.post(
        f"{API_BASE}/upload",
        files={"file": (file.name, file.getvalue(), file.type)},
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()


def _reset_index() -> None:
    requests.delete(f"{API_BASE}/reset", timeout=10)


# ═══════════════════════════════════════════════════════
# Rendering helpers
# ═══════════════════════════════════════════════════════

_CONFIDENCE_META = {
    "high":   ("🟢", "high",   "High confidence"),
    "medium": ("🟡", "medium", "Medium confidence"),
    "low":    ("🔴", "low",    "Low confidence"),
    "none":   ("⚫", "none",   "No match found"),
}


def _format_answer(text: str) -> str:
    """
    Convert LLM answer text to clean HTML for st.markdown.
    Handles:  \\n  →  real newlines
              - item  →  <li> inside <ul>
              plain sentences  →  <p>
    """
    # Fix escaped newlines from JSON
    text = text.replace("\\n", "\n").replace("\\•", "•")

    lines   = text.strip().splitlines()
    html    = []
    in_list = False

    for line in lines:
        stripped = line.strip()
        if not stripped:
            if in_list:
                html.append("</ul>")
                in_list = False
            continue

        # Bullet line: starts with - or •
        if re.match(r"^[-•*]\s+", stripped):
            if not in_list:
                html.append('<ul class="answer-list">')
                in_list = True
            content = re.sub(r"^[-•*]\s+", "", stripped)
            html.append(f"<li>{content}</li>")
        else:
            if in_list:
                html.append("</ul>")
                in_list = False
            html.append(f"<p>{stripped}</p>")

    if in_list:
        html.append("</ul>")

    return "\n".join(html)


def _confidence_badge(level: str) -> str:
    icon, css_class, label = _CONFIDENCE_META.get(level, _CONFIDENCE_META["none"])
    return f'<span class="badge badge-{css_class}">{icon} {label}</span>'


def _source_files_html(files: List[str]) -> str:
    if not files:
        return ""
    chips = "".join(f'<span class="source-chip">📄 {f}</span>' for f in files)
    return f'<div class="source-chips">{chips}</div>'


def _render_message(role: str, content: Dict):
    """Render a single chat bubble."""
    if role == "user":
        st.markdown(f"""
        <div class="chat-row user-row">
            <div class="bubble user-bubble">
                <p>{content['text']}</p>
            </div>
            <div class="avatar user-avatar">You</div>
        </div>
        """, unsafe_allow_html=True)

    else:
        answer_html  = _format_answer(content.get("answer", ""))
        badge_html   = _confidence_badge(content.get("confidence", "none"))
        sources_html = _source_files_html(content.get("source_files", []))

        # Collapsible source previews
        previews_html = ""
        previews = content.get("sources", [])
        if previews:
            preview_items = "".join(
                f'<div class="preview-item">{p}</div>'
                for p in previews
            )
            previews_html = f"""
            <details class="source-details">
                <summary>View source excerpts ({len(previews)})</summary>
                <div class="preview-list">{preview_items}</div>
            </details>
            """

        st.markdown(f"""
        <div class="chat-row assistant-row">
            <div class="avatar assistant-avatar">IV</div>
            <div class="bubble assistant-bubble">
                <div class="answer-body">{answer_html}</div>
                <div class="answer-meta">
                    {badge_html}
                    {sources_html}
                </div>
                {previews_html}
            </div>
        </div>
        """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════
# Chat page
# ═══════════════════════════════════════════════════════

def render_chat():
    # ── Session state ──
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "thinking" not in st.session_state:
        st.session_state.thinking = False

    # ── Header ──
    st.markdown("""
    <div class="page-header">
        <h1 class="page-title">Ask your documents</h1>
        <p class="page-subtitle">Upload a PDF or TXT, then ask anything about it.</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Clear button ──
    col1, col2 = st.columns([8, 1])
    with col2:
        if st.button("Clear", key="clear_chat", help="Clear conversation"):
            st.session_state.messages = []
            st.rerun()

    # ── Chat history ──
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)

    if not st.session_state.messages:
        st.markdown("""
        <div class="empty-state">
            <div class="empty-icon">💬</div>
            <p>No conversation yet. Ask a question below.</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        for msg in st.session_state.messages:
            _render_message(msg["role"], msg["content"])

    # Thinking indicator
    if st.session_state.thinking:
        st.markdown("""
        <div class="chat-row assistant-row">
            <div class="avatar assistant-avatar">IV</div>
            <div class="bubble assistant-bubble thinking-bubble">
                <div class="thinking-dots">
                    <span></span><span></span><span></span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    # ── Input bar ──
    st.markdown('<div class="input-bar">', unsafe_allow_html=True)
    with st.form(key="chat_form", clear_on_submit=True):
        cols = st.columns([11, 1])
        with cols[0]:
            user_input = st.text_input(
                "question",
                placeholder="Ask a question about your documents...",
                label_visibility="collapsed",
                key="chat_input",
            )
        with cols[1]:
            submitted = st.form_submit_button("➤")
    st.markdown("</div>", unsafe_allow_html=True)

    # ── Handle submit ──
    if submitted and user_input.strip():
        # Add user message
        st.session_state.messages.append({
            "role":    "user",
            "content": {"text": user_input.strip()},
        })
        st.session_state.thinking = True
        st.rerun()

    # If thinking flag is set, make the actual API call on next render
    if st.session_state.thinking and st.session_state.messages:
        last = st.session_state.messages[-1]
        if last["role"] == "user":
            query_text = last["content"]["text"]
            try:
                result = _query(query_text)
                st.session_state.messages.append({
                    "role":    "assistant",
                    "content": result,
                })
            except requests.exceptions.ConnectionError:
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": {
                        "answer":       "⚠️ Cannot connect to backend. Is uvicorn running on port 8000?",
                        "sources":      [],
                        "source_files": [],
                        "confidence":   "none",
                    },
                })
            except requests.exceptions.Timeout:
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": {
                        "answer":       "⚠️ Request timed out. The backend took too long to respond.",
                        "sources":      [],
                        "source_files": [],
                        "confidence":   "none",
                    },
                })
            except Exception as e:
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": {
                        "answer":       f"⚠️ Error: {str(e)}",
                        "sources":      [],
                        "source_files": [],
                        "confidence":   "none",
                    },
                })
            finally:
                st.session_state.thinking = False
                st.rerun()


# ═══════════════════════════════════════════════════════
# Upload page
# ═══════════════════════════════════════════════════════

def render_upload():
    st.markdown("""
    <div class="page-header">
        <h1 class="page-title">Upload documents</h1>
        <p class="page-subtitle">Add PDFs or TXT files to your vault.</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Upload zone ──
    uploaded = st.file_uploader(
        "Drop files here or click to browse",
        type=["pdf", "txt"],
        accept_multiple_files=True,
        label_visibility="visible",
    )

    if uploaded:
        if st.button("📥  Index all files", use_container_width=True):
            for f in uploaded:
                with st.spinner(f"Processing **{f.name}**..."):
                    try:
                        result = _upload(f)
                        st.success(
                            f"✅ **{f.name}** — "
                            f"{result.get('chunks_added', '?')} chunks added, "
                            f"{result.get('total_vectors', '?')} total vectors"
                        )
                    except requests.exceptions.ConnectionError:
                        st.error("❌ Cannot reach backend. Is uvicorn running?")
                    except Exception as e:
                        st.error(f"❌ Failed to upload **{f.name}**: {e}")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Danger zone ──
    with st.expander("⚠️  Danger zone"):
        st.warning("This will delete all indexed vectors and chunks. Uploaded files are kept.")
        if st.button("🗑️  Reset index", type="primary"):
            try:
                _reset_index()
                st.success("Index cleared. Upload new documents to start fresh.")
            except Exception as e:
                st.error(f"Reset failed: {e}")