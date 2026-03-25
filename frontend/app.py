import streamlit as st
import requests
import os
import time

# ─── Page Config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="IntelliVault",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Load CSS ────────────────────────────────────────────────────────────────
def load_css():
    with open(os.path.join(os.path.dirname(__file__), "style.css")) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()

BACKEND_URL = "http://127.0.0.1:8000"

# ─── Session State Init ──────────────────────────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "uploaded_docs" not in st.session_state:
    st.session_state.uploaded_docs = []
if "show_chunks" not in st.session_state:
    st.session_state.show_chunks = False

# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="sidebar-header">
        <div class="logo-mark">IV</div>
        <div>
            <div class="logo-title">IntelliVault</div>
            <div class="logo-sub">Personal Knowledge Vault</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section-label">UPLOAD DOCUMENTS</div>', unsafe_allow_html=True)

    uploaded_files = st.file_uploader(
        "Drop PDF or TXT files",
        type=["pdf", "txt"],
        accept_multiple_files=True,
        label_visibility="collapsed"
    )

    if uploaded_files:
        if st.button("⚡ Process Documents", use_container_width=True, key="process_btn"):
            for file in uploaded_files:
                with st.spinner(f"Processing {file.name}..."):
                    try:
                        files = {"file": (file.name, file.getvalue(), file.type)}
                        response = requests.post(f"{BACKEND_URL}/upload", files=files)
                        if response.status_code == 200:
                            if file.name not in st.session_state.uploaded_docs:
                                st.session_state.uploaded_docs.append(file.name)
                            st.success(f"✓ {file.name}")
                        else:
                            st.error(f"✗ Failed: {file.name}")
                    except Exception as e:
                        st.error(f"Backend not reachable: {e}")

    # Document Library
    if st.session_state.uploaded_docs:
        st.markdown('<div class="sidebar-section-label">DOCUMENT LIBRARY</div>', unsafe_allow_html=True)
        for i, doc in enumerate(st.session_state.uploaded_docs):
            col1, col2 = st.columns([5, 1])
            with col1:
                st.markdown(f'<div class="doc-item">📄 {doc}</div>', unsafe_allow_html=True)
            with col2:
                if st.button("✕", key=f"del_{i}", help="Remove document"):
                    st.session_state.uploaded_docs.pop(i)
                    st.rerun()

    st.markdown("---")
    st.markdown('<div class="sidebar-section-label">SETTINGS</div>', unsafe_allow_html=True)
    show_raw_chunks = st.toggle("Show Retrieved Chunks", value=st.session_state.show_chunks)
    st.session_state.show_chunks = show_raw_chunks

    top_k = st.slider("Chunks to Retrieve (Top K)", min_value=1, max_value=10, value=3)

    if st.session_state.chat_history:
        st.markdown("---")
        if st.button("🗑 Clear Chat History", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()

# ─── Main Area ───────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-header">
    <h1 class="hero-title">Ask Your Documents</h1>
    <p class="hero-sub">Semantic search powered by RAG · Answers grounded in your content</p>
</div>
""", unsafe_allow_html=True)

# Stats bar
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown(f"""
    <div class="stat-card">
        <div class="stat-num">{len(st.session_state.uploaded_docs)}</div>
        <div class="stat-label">Documents Loaded</div>
    </div>""", unsafe_allow_html=True)
with col2:
    st.markdown(f"""
    <div class="stat-card">
        <div class="stat-num">{len(st.session_state.chat_history)}</div>
        <div class="stat-label">Questions Asked</div>
    </div>""", unsafe_allow_html=True)
with col3:
    st.markdown(f"""
    <div class="stat-card">
        <div class="stat-num">{top_k}</div>
        <div class="stat-label">Chunks Retrieved</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─── Chat History Display ─────────────────────────────────────────────────────
if st.session_state.chat_history:
    st.markdown('<div class="section-label">CONVERSATION HISTORY</div>', unsafe_allow_html=True)
    for entry in st.session_state.chat_history:
        # User question
        st.markdown(f"""
        <div class="chat-bubble user-bubble">
            <span class="bubble-icon">👤</span>
            <div class="bubble-text">{entry['question']}</div>
        </div>""", unsafe_allow_html=True)

        # Answer
        st.markdown(f"""
        <div class="chat-bubble answer-bubble">
            <span class="bubble-icon">🧠</span>
            <div class="bubble-content">
                <div class="bubble-text">{entry['answer']}</div>
                <div class="source-tag">📎 Source: {entry.get('source', 'N/A')}</div>
            </div>
        </div>""", unsafe_allow_html=True)

        # Show raw chunks if toggled
        if st.session_state.show_chunks and entry.get("chunks"):
            with st.expander(f"🔍 View Retrieved Chunks ({len(entry['chunks'])} found)"):
                for j, chunk in enumerate(entry["chunks"]):
                    score = chunk.get("score", "N/A")
                    text  = chunk.get("text", "")
                    src   = chunk.get("source", "")
                    st.markdown(f"""
                    <div class="chunk-card">
                        <div class="chunk-header">
                            <span>Chunk {j+1}</span>
                            <span class="chunk-score">Similarity: {score}</span>
                        </div>
                        <div class="chunk-source">📄 {src}</div>
                        <div class="chunk-text">{text}</div>
                    </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

# ─── Question Input ───────────────────────────────────────────────────────────
st.markdown('<div class="section-label">ASK A QUESTION</div>', unsafe_allow_html=True)

if not st.session_state.uploaded_docs:
    st.markdown("""
    <div class="empty-state">
        <div class="empty-icon">📂</div>
        <div class="empty-title">No documents loaded yet</div>
        <div class="empty-sub">Upload PDF or TXT files from the sidebar to get started</div>
    </div>""", unsafe_allow_html=True)
else:
    with st.form(key="question_form", clear_on_submit=True):
        question = st.text_area(
            "Type your question here...",
            placeholder="e.g. What is deadlock? Explain the main concepts in chapter 3...",
            height=100,
            label_visibility="collapsed"
        )
        col_a, col_b = st.columns([4, 1])
        with col_b:
            submitted = st.form_submit_button("Ask ⚡", use_container_width=True)

    if submitted and question.strip():
        with st.spinner("🔍 Retrieving context and generating answer..."):
            try:
                response = requests.post(
                    f"{BACKEND_URL}/query",
                    json={"question": question, "top_k": top_k}
                )
                if response.status_code == 200:
                    data = response.json()
                    entry = {
                        "question": question,
                        "answer": data.get("answer", "No answer returned."),
                        "source": data.get("source", "Unknown"),
                        "chunks": data.get("chunks", [])
                    }
                    st.session_state.chat_history.append(entry)
                    st.rerun()
                else:
                    st.error("Backend returned an error. Check your FastAPI server.")
            except Exception as e:
                # Demo mode — show mock answer when backend is offline
                mock_entry = {
                    "question": question,
                    "answer": "[Demo Mode] Backend not connected. Start your FastAPI server with: uvicorn backend.main:app --reload",
                    "source": "N/A",
                    "chunks": []
                }
                st.session_state.chat_history.append(mock_entry)
                st.rerun()
    elif submitted:
        st.warning("Please enter a question first.")

# ─── Footer ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
    IntelliVault · RAG-powered Knowledge Retrieval · Built with Streamlit + FastAPI + FAISS
</div>""", unsafe_allow_html=True)
