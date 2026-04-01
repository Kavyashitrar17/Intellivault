"""
app.py
------
IntelliVault frontend entry point.
Renders the sidebar navigation and routes to pages.
"""

import streamlit as st
# ── Load global CSS ──────────────────────────────────────
def _load_css():
    import os
    css_path = os.path.join(os.path.dirname(__file__), "style.css")
    if os.path.exists(css_path):
        with open(css_path) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

_load_css()

# ── Sidebar ──────────────────────────────────────────────
with st.sidebar:
    st.sidebar.markdown("""
<div style="
    font-size: 30px;
    font-weight: 800;
    background: linear-gradient(90deg, #7c3aed, #06b6d4);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 20px;
">
🔒 IntelliVault
</div>
""", unsafe_allow_html=True)

    page = st.radio(
        "Navigation",
        ["💬  Chat", "📂  Upload"],
        label_visibility="collapsed",
    )

    st.markdown("<div class='sidebar-divider'></div>", unsafe_allow_html=True)

    # Status indicator
    st.markdown("**System Status**")
    try:
        import requests
        r = requests.get("http://127.0.0.1:8000/status", timeout=2)
        if r.status_code == 200:
            d = r.json()
            st.markdown(f"""
            <div class="status-card ok">
                <span class="status-dot"></span> Backend online
            </div>
            <div class="status-meta">
                {d.get('total_vectors', 0)} vectors &nbsp;·&nbsp;
                {d.get('total_chunks', 0)} chunks
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown('<div class="status-card error"><span class="status-dot err"></span> Backend error</div>', unsafe_allow_html=True)
    except Exception:
        st.markdown('<div class="status-card error"><span class="status-dot err"></span> Backend offline</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.caption("IntelliVault v2.0 · RAG-powered")

# ── Route to page ─────────────────────────────────────────
if "💬" in page:
    from pages.vault import render_chat
    render_chat()
else:
    from pages.vault import render_upload
    render_upload()