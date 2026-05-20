"""
Rant-Free Project - HITL UI
app.py - Streamlit interface for human-in-the-loop labeling
Reviewer: Rani
"""

import json
import os
import streamlit as st
import httpx

# ─── Config ──────────────────────────────────────────────────────────────────
HITL_BACKEND_URL = os.getenv("HITL_BACKEND_URL", "http://localhost:8001")
REVIEWER_NAME    = os.getenv("REVIEWER_NAME", "rani")

# ─── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Rant-Free-HITL",
    layout="wide",
)

# ─── Styling ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Press+Start+2P&family=VT323&family=Space+Grotesk:wght@400;700&display=swap');

:root {
    --red: #E60012;
    --red-shadow: #7A0009;
    --paper: #FBF7F0;
    --paper-2: #F0E9DC;
    --ink: #0A0A0A;
    --yellow: #FFD400;
    --font-display: 'Press Start 2P', monospace;
    --font-mono: 'VT323', monospace;
    --font-ui: 'Space Grotesk', system-ui, sans-serif;
    --border: 3px solid var(--ink);
    --shadow: 6px 6px 0 var(--ink);
}

html, body, [class*="css"] {
    font-family: var(--font-ui);
}

#MainMenu, footer, header { visibility: hidden; }

.stApp {
    background-color: var(--red);
    background-image:
        radial-gradient(circle at 0 0, var(--red-shadow) 1.2px, transparent 1.3px),
        radial-gradient(circle at 6px 6px, var(--red-shadow) 1.2px, transparent 1.3px);
    background-size: 12px 12px;
    background-position: 0 0, 6px 6px;
}

/* ── Header ── */
.hitl-header {
    background: var(--ink);
    color: var(--paper);
    border: var(--border);
    box-shadow: var(--shadow);
    padding: 14px 20px;
    margin-bottom: 24px;
}
.brand-name {
    font-family: var(--font-display);
    font-size: 16px;
    color: var(--paper);
}
.brand-tag {
    font-family: var(--font-mono);
    font-size: 20px;
    color: var(--yellow);
    margin-top: 4px;
}

/* ── Cards ── */
.text-card {
    background: var(--paper);
    border: var(--border);
    box-shadow: var(--shadow);
    padding: 20px 22px;
    margin-bottom: 20px;
}
.text-card-label {
    font-family: var(--font-mono);
    font-size: 18px;
    color: var(--ink);
    opacity: 0.5;
    margin-bottom: 10px;
    text-transform: uppercase;
}
.text-content {
    font-family: var(--font-mono);
    font-size: 26px;
    line-height: 1.4;
    color: var(--ink);
}
.request-id {
    font-family: var(--font-mono);
    font-size: 14px;
    color: var(--ink);
    opacity: 0.3;
    margin-top: 14px;
}

/* ── Instruction box ── */
.instruction-box {
    background: var(--paper-2);
    border: var(--border);
    box-shadow: var(--shadow);
    padding: 16px 18px;
    margin-bottom: 20px;
    font-family: var(--font-mono);
    font-size: 20px;
    line-height: 1.6;
    color: var(--ink);
}

/* ── Buttons ── */
div[data-testid="column"]:nth-child(1) .stButton > button {
    background: var(--red) !important;
    color: var(--paper) !important;
    border: var(--border) !important;
    box-shadow: 4px 4px 0 var(--red-shadow) !important;
    font-family: var(--font-mono) !important;
    font-size: 22px !important;
    font-weight: 700 !important;
    text-transform: uppercase !important;
    letter-spacing: 1px !important;
    padding: 12px 20px !important;
    border-radius: 0 !important;
    width: 100% !important;
    transition: all 0.08s linear !important;
}
div[data-testid="column"]:nth-child(1) .stButton > button:hover {
    transform: translate(-2px, -2px) !important;
    box-shadow: 6px 6px 0 var(--red-shadow) !important;
}

div[data-testid="column"]:nth-child(2) .stButton > button {
    background: var(--yellow) !important;
    color: var(--ink) !important;
    border: var(--border) !important;
    box-shadow: 4px 4px 0 var(--red-shadow) !important;
    font-family: var(--font-mono) !important;
    font-size: 22px !important;
    font-weight: 700 !important;
    text-transform: uppercase !important;
    letter-spacing: 1px !important;
    padding: 12px 20px !important;
    border-radius: 0 !important;
    width: 100% !important;
    transition: all 0.08s linear !important;
}
div[data-testid="column"]:nth-child(2) .stButton > button:hover {
    transform: translate(-2px, -2px) !important;
    box-shadow: 6px 6px 0 var(--red-shadow) !important;
}

div[data-testid="column"] button [data-testid="stMarkdownContainer"] p {
    font-family: 'VT323', monospace !important;
    font-size: 22px !important;
}

/* Reset button */
.stSidebar .stButton > button,
[data-testid="stSidebar"] .stButton > button,
[data-testid="stSidebar"] button {
    background: var(--ink) !important;
    color: var(--yellow) !important;
    border: var(--border) !important;
    box-shadow: 4px 4px 0 var(--red-shadow) !important;
    font-family: 'VT323', monospace !important;
    font-size: 22px !important;
    font-weight: 700 !important;
    text-transform: uppercase !important;
    border-radius: 0 !important;
    width: 100% !important;
}

/* ── Sidebar ── */
.stSidebar {
    background: var(--ink) !important;
}
.stSidebar * {
    color: var(--paper) !important;
}
section[data-testid="stSidebar"] .stButton > button p {
    font-family: 'VT323', monospace !important;
    font-size: 22px !important;
}

.stat-box {
    background: #1a1a1a;
    border: 2px solid #333;
    padding: 10px 14px;
    margin-bottom: 10px;
    font-family: var(--font-mono);
}
.stat-label {
    font-size: 14px;
    color: #888 !important;
    text-transform: uppercase;
}
.stat-value {
    font-size: 28px;
    color: var(--yellow) !important;
    line-height: 1.2;
}

.history-item {
    font-family: var(--font-mono);
    font-size: 16px;
    padding: 6px 0;
    border-bottom: 1px solid #222;
    color: #aaa !important;
}

/* ── Empty / done / error state ── */
.done-state {
    background: var(--paper);
    border: var(--border);
    box-shadow: var(--shadow);
    padding: 60px 40px;
    text-align: center;
}
.done-title {
    font-family: var(--font-display);
    font-size: 18px;
    color: var(--ink);
    margin-bottom: 16px;
}
.done-sub {
    font-family: var(--font-mono);
    font-size: 24px;
    color: var(--ink);
    opacity: 0.6;
}
.error-state {
    background: var(--paper);
    border: var(--border);
    box-shadow: var(--shadow);
    padding: 40px;
    text-align: center;
    font-family: var(--font-mono);
    font-size: 20px;
    color: var(--red);
}

/* ── Progress label ── */
.progress-label {
    font-family: var(--font-mono);
    font-size: 20px;
    color: var(--paper);
    margin-bottom: 6px;
}

/* ── Feedback ── */
.feedback-toxic {
    font-family: var(--font-mono);
    font-size: 20px;
    color: var(--red);
    background: var(--paper);
    border: var(--border);
    padding: 8px 14px;
    margin-top: 12px;
    text-align: center;
}
.feedback-not-toxic {
    font-family: var(--font-mono);
    font-size: 20px;
    color: #1a7a1a;
    background: var(--paper);
    border: var(--border);
    padding: 8px 14px;
    margin-top: 12px;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
button p {
    font-family: 'VT323', monospace !important;
    font-size: 22px !important;
}
</style>
""", unsafe_allow_html=True)


# ─── Backend helpers ──────────────────────────────────────────────────────────

def fetch_queue() -> dict | None:
    """
    GET /queue → {"items": [...], "total": N}
    Return None kalau backend tidak bisa dihubungi.
    """
    try:
        r = httpx.get(f"{HITL_BACKEND_URL}/queue", timeout=5.0)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None

def post_review(request_id: str, toxic: int) -> bool:
    """
    POST /review/{request_id}
    Body: {"reviewed_by": "rani", "toxic": 0|1}
    Return True kalau berhasil.
    """
    try:
        r = httpx.post(
            f"{HITL_BACKEND_URL}/review/{request_id}",
            json={"reviewed_by": REVIEWER_NAME, "toxic": toxic},
            timeout=5.0,
        )
        r.raise_for_status()
        return True
    except Exception:
        return False


# ─── Session State ────────────────────────────────────────────────────────────
if "queue_response" not in st.session_state:
    st.session_state.queue_response = None
if "current_idx" not in st.session_state:
    st.session_state.current_idx = 0
if "labeled" not in st.session_state:
    st.session_state.labeled = []
if "last_action" not in st.session_state:
    st.session_state.last_action = None
if "backend_error" not in st.session_state:
    st.session_state.backend_error = False

# Fetch queue saat pertama load
if st.session_state.queue_response is None:
    result = fetch_queue()
    if result is None:
        st.session_state.backend_error = True
    else:
        st.session_state.backend_error = False
        st.session_state.queue_response = result

queue           = st.session_state.queue_response["items"] if st.session_state.queue_response else []
total           = st.session_state.queue_response["total"] if st.session_state.queue_response else 0
done            = len(st.session_state.labeled)
remaining       = total - done
toxic_count     = sum(1 for x in st.session_state.labeled if x["toxic"] == 1)
not_toxic_count = done - toxic_count


# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="font-family:'Press Start 2P',monospace; font-size:10px; color:#FFD400;
                margin-bottom:20px; line-height:1.8;">
        HITL<br>REVIEW
    </div>
    """, unsafe_allow_html=True)

    for label, value in [
        ("Reviewed",  f"{done}/{total}"),
        ("Remaining", str(remaining)),
        ("Toxic",     str(toxic_count)),
        ("Not Toxic", str(not_toxic_count)),
    ]:
        st.markdown(f"""
        <div class="stat-box">
            <div class="stat-label">{label}</div>
            <div class="stat-value">{value}</div>
        </div>
        """, unsafe_allow_html=True)

    if done > 0:
        st.markdown('<div style="margin: 16px 0 8px; font-family:\'VT323\',monospace; font-size:18px; color:#888; text-transform:uppercase;">History</div>', unsafe_allow_html=True)
        for item in reversed(st.session_state.labeled[-5:]):
            badge   = "[TOXIC]" if item["toxic"] == 1 else "[NOT TOXIC]"
            color   = "#E60012" if item["toxic"] == 1 else "#FFD400"
            text    = item.get("text", "")
            preview = text[:40] + "..." if len(text) > 40 else text
            st.markdown(f"""
            <div class="history-item">
                <span style="color:{color}; font-weight:700;">{badge}</span> {preview}
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("↺ REFRESH", key="refresh"):
        result = fetch_queue()
        if result is None:
            st.session_state.backend_error = True
        else:
            st.session_state.backend_error = False
            st.session_state.queue_response = result
            st.session_state.current_idx    = 0
            st.session_state.labeled        = []
            st.session_state.last_action    = None
        st.rerun()


# ─── Header ──────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hitl-header">
    <div class="brand-name">RANT FREE</div>
    <div class="brand-tag">verify it hooman.</div>
</div>
""", unsafe_allow_html=True)


# ─── Backend error state ──────────────────────────────────────────────────────
if st.session_state.backend_error:
    st.markdown(f"""
    <div class="error-state">
        ⚠ CANNOT CONNECT TO BACKEND<br><br>
        <span style="font-size:16px; color:#555;">
            {HITL_BACKEND_URL} tidak bisa dihubungi.<br>
            Pastikan hitl-backend sudah berjalan.
        </span>
    </div>
    """, unsafe_allow_html=True)
    st.stop()


# ─── Empty queue ─────────────────────────────────────────────────────────────
if total == 0:
    st.markdown("""
    <div class="done-state">
        <div class="done-title">QUEUE EMPTY</div>
        <div class="done-sub">tidak ada item untuk direview saat ini.</div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()


# ─── All Done ────────────────────────────────────────────────────────────────
if done >= total:
    st.markdown("""
    <div class="done-state">
        <div class="done-title">ALL DONE!</div>
        <div class="done-sub">queue cleared. nice work, rani.</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    result_json = json.dumps(st.session_state.labeled, indent=2, ensure_ascii=False)
    st.download_button(
        label="⬇ DOWNLOAD LABELS (JSON)",
        data=result_json,
        file_name="hitl_labels.json",
        mime="application/json",
    )
    st.stop()


# ─── Progress ────────────────────────────────────────────────────────────────
st.markdown(f'<div class="progress-label">{done} of {total} reviewed</div>',
            unsafe_allow_html=True)
st.progress(done / total if total > 0 else 0)
st.markdown("<br>", unsafe_allow_html=True)


# ─── Current Item ─────────────────────────────────────────────────────────────
idx  = st.session_state.current_idx
item = queue[idx]

# Ambil data dari feast enrichment
feast      = item.get("feast") or {}
text       = feast.get("text") or "(teks tidak tersedia)"
confidence = item["confidence"]
request_id = item["request_id"]

col_main, col_action = st.columns([3, 2], gap="large")

with col_main:
    st.markdown(f"""
    <div class="text-card">
        <div class="text-card-label">rant #{idx + 1} of {total}</div>
        <div class="text-content">{text}</div>
        <div class="request-id">id: {request_id} &nbsp;·&nbsp; confidence: {confidence:.4f}</div>
    </div>
    """, unsafe_allow_html=True)

with col_action:
    st.markdown("""
    <div class="instruction-box">
        read the rant.<br><br>
        <span style="color:#E60012; font-weight:700;">■ TOXIC</span> — hate, threats,
        slurs, harassment.<br><br>
        <span style="color:#1a7a1a; font-weight:700;">■ NOT TOXIC</span> — criticism,
        opinion, normal talk.<br><br>
        you must pick one.
    </div>
    """, unsafe_allow_html=True)

    col_t, col_c = st.columns(2)
    with col_t:
        toxic_clicked = st.button("\u200bTOXIC", key="btn_toxic", use_container_width=True)
    with col_c:
        nottoxic_clicked = st.button("\u200bNOT TOXIC", key="btn_nottoxic", use_container_width=True)

    if st.session_state.last_action == "toxic":
        st.markdown('<div class="feedback-toxic">✓ LABELED: TOXIC</div>', unsafe_allow_html=True)
    elif st.session_state.last_action == "not_toxic":
        st.markdown('<div class="feedback-not-toxic">✓ LABELED: NOT TOXIC</div>', unsafe_allow_html=True)


# ─── Handle Actions ───────────────────────────────────────────────────────────
def submit_label(toxic: int, action: str):
    success = post_review(request_id, toxic)
    if not success:
        st.error("⚠ Gagal menyimpan label ke backend. Coba lagi.")
        return

    st.session_state.labeled.append({
        "request_id": request_id,
        "text":       text,
        "toxic":      toxic,
        "confidence": confidence,
        "status":     "reviewed",
    })
    st.session_state.current_idx = min(idx + 1, total - 1)
    st.session_state.last_action = action

if toxic_clicked:
    submit_label(1, "toxic")
    st.rerun()

if nottoxic_clicked:
    submit_label(0, "not_toxic")
    st.rerun()


# ─── Nav Dots ─────────────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
dots_html = ""
for i in range(total):
    is_current = i == idx
    is_done    = i < done
    if is_current:
        bg, border = "#0A0A0A", "3px solid #0A0A0A"
    elif is_done:
        bg, border = "#FFD400", "3px solid #0A0A0A"
    else:
        bg, border = "#F0E9DC", "3px solid #0A0A0A"
    dots_html += f'<div style="flex:1; height:14px; background:{bg}; border:{border}; margin:0 3px;"></div>'

st.markdown(f'<div style="display:flex; gap:4px;">{dots_html}</div>', unsafe_allow_html=True)
