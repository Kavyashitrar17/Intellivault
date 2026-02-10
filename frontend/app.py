import streamlit as st
import os
import time

st.set_page_config(
    page_title="IntelliVault — Login",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ── Load CSS ──────────────────────────────────────────────────────────────────
css_path = os.path.join(os.path.dirname(__file__), "style.css")
with open(css_path) as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Hide sidebar on login page
st.markdown("""
<style>
[data-testid="stSidebar"] { display: none !important; }
[data-testid="collapsedControl"] { display: none !important; }
</style>""", unsafe_allow_html=True)

# ── Credentials (from env, fallback) ───────────────────────────────────────────
def _load_users_from_env():
    raw = os.getenv("INTELLIVAULT_USERS", "")
    if not raw.strip():
        return {
            "prakriti": "intellivault123",
            "kavyashi": "intellivault123",
            "admin": "admin123",
        }
    users = {}
    for entry in raw.split(","):
        if ":" in entry:
            username, passwd = entry.split(":", 1)
            username = username.strip().lower()
            passwd = passwd.strip()
            if username and passwd:
                users[username] = passwd
    return users

USERS = _load_users_from_env()

if not USERS:
    st.warning(
        "No IntelliVault users found in INTELLIVAULT_USERS. "
        "Add env var in format user:pass,user2:pass2 for secure auth."
    )

# ── Session State ─────────────────────────────────────────────────────────────
if "logged_in"   not in st.session_state: st.session_state.logged_in   = False
if "username"    not in st.session_state: st.session_state.username    = ""
if "login_error" not in st.session_state: st.session_state.login_error = ""
if "show_welcome" not in st.session_state: st.session_state.show_welcome = False

# ── If already logged in, show welcome or redirect ───────────────────────────
if st.session_state.logged_in:
    if st.session_state.show_welcome:
        # ── Welcome Screen ────────────────────────────────────────────────────
        name = st.session_state.username.capitalize()
        st.markdown(f"""
        <div class="welcome-wrap">
            <div class="welcome-card">
                <div class="welcome-brain">🧠</div>
                <h1 class="welcome-title">Welcome back, {name}!</h1>
                <p class="welcome-sub">
                    Your personal knowledge vault is ready.<br>
                    Upload documents, ask questions, get grounded answers.
                </p>
                <div class="welcome-features">
                    <div class="wf-item">
                        <span class="wf-icon">📄</span>
                        <span class="wf-text">Upload PDF & TXT documents</span>
                    </div>
                    <div class="wf-item">
                        <span class="wf-icon">🔍</span>
                        <span class="wf-text">Semantic search with FAISS</span>
                    </div>
                    <div class="wf-item">
                        <span class="wf-icon">🤖</span>
                        <span class="wf-text">GPT-2 powered grounded answers</span>
                    </div>
                    <div class="wf-item">
                        <span class="wf-icon">📎</span>
                        <span class="wf-text">Source-cited responses, no hallucinations</span>
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🚀  Enter IntelliVault", use_container_width=True, key="enter_btn"):
                st.session_state.show_welcome = False
                st.switch_page("pages/vault.py")

        st.markdown("""
        <div style="text-align:center;margin-top:16px;">
            <a class="skip-link" href="#" onclick="void(0)">Skip welcome screen next time</a>
        </div>""", unsafe_allow_html=True)

    else:
        st.switch_page("pages/vault.py")

else:
    # ── LOGIN PAGE ────────────────────────────────────────────────────────────

    # Full-page layout
    st.markdown("""
    <div class="login-page">
        <div class="login-left">
            <div class="login-brand">
                <span class="login-brain">🧠</span>
                <span class="login-brand-name">IntelliVault</span>
            </div>
            <h1 class="login-headline">Search your files with confidence.<br>Get precise, source-cited answers.</h1>
            <p class="login-tagline">
                Upload PDFs & TXT, then use semantic retrieval + RAG to build a private knowledge vault.
                Works locally and securely with optional backend mode.
            </p>
            <div class="login-hero-steps">
                <div class="step-card">
                    <div class="step-num">1</div>
                    <div class="step-label">Upload your docs</div>
                </div>
                <div class="step-card">
                    <div class="step-num">2</div>
                    <div class="step-label">Chunk & embed</div>
                </div>
                <div class="step-card">
                    <div class="step-num">3</div>
                    <div class="step-label">Ask questions</div>
                </div>
            </div>
            <div class="trust-badges">
                <span>🔒 Local storage</span>
                <span>⚡ Instant demo</span>
                <span>✅ Source-cited outputs</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Login form — centered card
    st.markdown('<div class="login-card-wrap">', unsafe_allow_html=True)

    st.markdown("""
    <div class="login-card-header">
        <div class="lc-icon">🧠</div>
        <div class="lc-title">Sign in to IntelliVault</div>
        <div class="lc-sub">Enter your credentials to access your vault</div>
    </div>
    """, unsafe_allow_html=True)

    if st.session_state.login_error:
        st.markdown(
            f'<div class="login-error">⚠️ {st.session_state.login_error}</div>',
            unsafe_allow_html=True
        )

    with st.form("login_form"):
        st.markdown('<div class="field-label">Username</div>', unsafe_allow_html=True)
        username = st.text_input("u", placeholder="Enter your username",
                                  label_visibility="collapsed")

        st.markdown('<div class="field-label">Password</div>', unsafe_allow_html=True)
        password = st.text_input("p", placeholder="Enter your password",
                                  type="password", label_visibility="collapsed")

        st.markdown("<br>", unsafe_allow_html=True)
        login_btn = st.form_submit_button("Sign In →", use_container_width=True)

    if login_btn:
        u = username.strip().lower()
        p = password.strip()
        if not u or not p:
            st.session_state.login_error = "Please enter both username and password."
            st.rerun()
        elif u in USERS and USERS[u] == p:
            st.session_state.logged_in    = True
            st.session_state.username     = u
            st.session_state.login_error  = ""
            st.session_state.show_welcome = True
            st.rerun()
        else:
            st.session_state.login_error = "Incorrect username or password. Try again."
            st.rerun()

    if st.button("Try instant demo (no login)", use_container_width=True):
        st.session_state.logged_in = True
        st.session_state.username = "demo"
        st.session_state.login_error = ""
        st.session_state.show_welcome = True
        st.rerun()

    st.markdown("""
    <div class="login-hint">
        <strong>Demo credentials</strong><br>
        Username: <code>admin</code> &nbsp;·&nbsp; Password: <code>admin123</code>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    # Footer
    st.markdown("""
    <div class="login-footer">
        IntelliVault &nbsp;·&nbsp; RAG-Powered Document Intelligence &nbsp;·&nbsp;
        Built with Python · FastAPI · Streamlit · FAISS
    </div>""", unsafe_allow_html=True)