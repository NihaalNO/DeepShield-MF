"""DeepShield-MF analysis console (Streamlit front end for the Flask backend).

Tabs
  Analyze           upload an image or video, stream frame verdicts from the backend
  History           analyses of this browser session
  Research results  offline evaluation of the research pipeline, read from results/ (if present)
  About             what is served, what was evaluated, and the limits of both
"""
import csv
import json
import os
import statistics as st_stats
from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from plotly.subplots import make_subplots

ROOT = Path(__file__).resolve().parent

try:
    from dotenv import load_dotenv

    # Same .env file as the backend (project root); real env vars win.
    load_dotenv(ROOT / ".env", override=False)
except ImportError:
    pass

# Flask backend address. Must match the backend's HOST/PORT (default http://127.0.0.1:5000).
BACKEND_URL = os.getenv("DEEPSHIELD_BACKEND_URL", "http://127.0.0.1:5000").rstrip("/")
PREDICT_URL = f"{BACKEND_URL}/predict"
HEALTH_URL = f"{BACKEND_URL}/health"
RESULTS_DIR = Path(os.getenv("DEEPSHIELD_RESULTS_DIR") or ROOT / "results")

st.set_page_config(page_title="DeepShield-MF Console", page_icon=None,
                   layout="wide", initial_sidebar_state="collapsed")

# ============================================================
# DESIGN TOKENS
# ============================================================
C = {
    # taste-skill deal: palette lunar-grey-bronze, type instrument, radius soft, buttons solid
    "bg": "#111111", "surface": "#1c1c1c", "surface2": "#232323", "line": "rgba(242,242,242,0.10)",
    "line2": "rgba(242,242,242,0.16)", "border": "rgba(242,242,242,0.10)",
    "text": "#f2f2f2", "body": "#d6d6d6", "muted": "#8f8f8f", "faint": "#5c5c5c",
    "accent": "#a08672", "accent_ink": "#111111",
    "real": "#3fbf6f", "fake": "#e5484d", "warn": "#d9a441",
    "series2": "#8f8f8f", "series3": "#5f5f5f", "grid": "rgba(242,242,242,0.07)",
}
FONT = "Instrument Sans, system-ui, sans-serif"
MONO = "DM Mono, ui-monospace, Consolas, monospace"

# Lucide icons (MIT), inline SVG so no icon font or emoji is needed.
_ICON_PATHS = {
    "shield": '<path d="M20 13c0 5-3.5 7.5-7.66 8.95a1 1 0 0 1-.67-.01C7.5 20.5 4 18 4 13V6a1 1 0 0 1 1-1c2 0 4.5-1.2 6.24-2.72a1.17 1.17 0 0 1 1.52 0C14.51 3.81 17 5 19 5a1 1 0 0 1 1 1z"/><path d="m9 12 2 2 4-4"/>',
    "alert": '<path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3"/><path d="M12 9v4"/><path d="M12 17h.01"/>',
    "check": '<circle cx="12" cy="12" r="10"/><path d="m9 12 2 2 4-4"/>',
    "cpu": '<rect width="16" height="16" x="4" y="4" rx="2"/><rect width="6" height="6" x="9" y="9" rx="1"/><path d="M15 2v2M15 20v2M2 15h2M2 9h2M20 15h2M20 9h2M9 2v2M9 20v2"/>',
    "server": '<rect width="20" height="8" x="2" y="2" rx="2"/><rect width="20" height="8" x="2" y="14" rx="2"/><path d="M6 6h.01M6 18h.01"/>',
    "layers": '<path d="m12.83 2.18a2 2 0 0 0-1.66 0L2.6 6.08a1 1 0 0 0 0 1.83l8.58 3.91a2 2 0 0 0 1.66 0l8.58-3.9a1 1 0 0 0 0-1.83Z"/><path d="m22 17.65-9.17 4.16a2 2 0 0 1-1.66 0L2 17.65"/><path d="m22 12.65-9.17 4.16a2 2 0 0 1-1.66 0L2 12.65"/>',
    "film": '<rect width="18" height="18" x="3" y="3" rx="2"/><path d="M7 3v18M3 7.5h4M3 12h18M3 16.5h4M17 3v18M17 7.5h4M17 16.5h4"/>',
    "image": '<rect width="18" height="18" x="3" y="3" rx="2" ry="2"/><circle cx="9" cy="9" r="2"/><path d="m21 15-3.09-3.09a2 2 0 0 0-2.82 0L6 21"/>',
    "info": '<circle cx="12" cy="12" r="10"/><path d="M12 16v-4M12 8h.01"/>',
}


def icon(name, size=18, color="currentColor"):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" '
            f'stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">'
            f'{_ICON_PATHS[name]}</svg>')


# ============================================================
# STYLES
# ============================================================
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Instrument+Sans:wdth,wght@75..100,400..700&family=DM+Mono:wght@400;500&display=swap');
:root {{ --ease: cubic-bezier(.2,.8,.2,1); --r-sm: 8px; --r-md: 12px; --r-lg: 20px; }}
html, body, .stApp, .stMarkdown, button, input, textarea, [data-testid="stWidgetLabel"] {{
  font-family: 'Instrument Sans', system-ui, sans-serif; }}
.stApp {{ background: {C['bg']}; color: {C['text']}; }}
.block-container {{ padding-top: 1rem; padding-bottom: 4rem; max-width: 1280px; }}
header[data-testid="stHeader"] {{ background: transparent; }}
[data-testid="stDecoration"], footer {{ display: none; }}
h1, h2, h3, h4 {{ font-family: 'Instrument Sans', sans-serif; font-weight: 500; letter-spacing: -0.028em; }}
code {{ font-family: 'DM Mono', monospace; }}

/* top bar: flat, opaque, one hairline under it */
.topbar {{ display:flex; align-items:center; justify-content:space-between; gap:16px; flex-wrap:wrap;
          padding: 12px 0 16px; border-bottom: 1px solid {C['line']}; margin-bottom: 8px; }}
.brand {{ display:flex; align-items:center; gap:12px; }}
.brand-mark {{ width:36px; height:36px; border-radius:var(--r-sm); display:grid; place-items:center;
              background:{C['text']}; color:{C['bg']}; }}
.brand h1 {{ font-size: 20px; margin:0; padding:0; font-weight:500; color:{C['text']}; line-height:1.2; letter-spacing:-0.02em; }}
.brand p {{ margin:0; font-size:13px; font-weight:500; color:{C['muted']}; }}
.pills {{ display:flex; gap:8px; flex-wrap:wrap; }}
.pill {{ display:inline-flex; align-items:center; gap:8px; padding:6px 12px; border-radius:999px; font-size:13px;
        font-weight:500; border:1px solid {C['line']}; background:{C['surface']}; color:{C['body']}; white-space:nowrap; }}
.dot {{ width:7px; height:7px; border-radius:50%; display:inline-block; }}
.pill code {{ font-family:'DM Mono', monospace; font-size:12.5px; color:{C['muted']}; background:none; padding:0; }}

/* one card recipe: one step off the ground, a hairline, an inner top highlight, no shadow */
.card, .kpi {{ border:1px solid {C['line']}; border-radius:var(--r-lg); background:{C['surface']};
              box-shadow: inset 0 1px 0 rgba(255,255,255,0.05); }}
.card {{ padding:24px 28px; }}
.card-title {{ display:flex; align-items:center; gap:10px; font-size:18px; font-weight:500; color:{C['text']}; margin-bottom:8px; }}
.card-sub {{ font-size:16px; line-height:1.6; color:{C['body']}; margin:0; }}
.card-title svg {{ color:{C['muted']}; }}
.section-title {{ display:flex; align-items:baseline; gap:12px; flex-wrap:wrap; font-size:24px; font-weight:500;
                 letter-spacing:-0.02em; color:{C['text']}; margin: 40px 0 16px; }}
.section-title span.muted {{ font-weight:400; font-size:16px; letter-spacing:0; color:{C['muted']}; }}

/* figures */
.kpi {{ padding:20px 24px; height:100%; }}
.kpi .label {{ font-size:13px; font-weight:500; color:{C['muted']}; }}
.kpi .value {{ font-family:'Instrument Sans', sans-serif; font-size:38px; font-weight:500; letter-spacing:-0.03em;
              color:{C['text']}; line-height:1.1; margin-top:10px; font-variant-numeric: tabular-nums; }}
.kpi .value small {{ font-size:18px; color:{C['muted']}; font-weight:500; margin-left:2px; letter-spacing:0; }}
.kpi .foot {{ font-size:14px; color:{C['body']}; margin-top:8px; line-height:1.45; }}
.kpi .marker {{ display:inline-block; width:8px; height:8px; border-radius:50%; margin-right:8px; vertical-align:middle; }}
.kpi .delta {{ color:{C['text']}; font-weight:500; }}

/* verdict */
.verdict {{ display:flex; align-items:center; gap:20px; border-radius:var(--r-lg); padding:24px 28px; flex-wrap:wrap;
           background:{C['surface']}; border:1px solid {C['line']}; box-shadow: inset 0 1px 0 rgba(255,255,255,0.05); }}
.verdict .badge {{ width:52px; height:52px; border-radius:var(--r-md); display:grid; place-items:center; flex:none; }}
.verdict.fake .badge {{ background: rgba(229,72,77,0.14); color:{C['fake']}; }}
.verdict.real .badge {{ background: rgba(63,191,111,0.14); color:{C['real']}; }}
.verdict .eyebrow {{ font-size:13px; font-weight:500; color:{C['muted']}; }}
.verdict .headline {{ font-size:32px; font-weight:500; letter-spacing:-0.03em; color:{C['text']}; line-height:1.15; }}
.verdict .detail {{ font-size:16px; color:{C['body']}; margin-top:4px; }}
.verdict .score {{ margin-left:auto; text-align:right; }}
.verdict .score .num {{ font-size:48px; font-weight:500; letter-spacing:-0.04em; line-height:1; color:{C['text']};
                       font-variant-numeric: tabular-nums; }}
.verdict .score .cap {{ font-size:13px; font-weight:500; color:{C['muted']}; margin-top:6px; }}
.verdict .state {{ display:inline-flex; align-items:center; gap:8px; font-size:13px; font-weight:500; margin-top:10px;
                  padding:4px 10px; border-radius:999px; border:1px solid {C['line']}; color:{C['body']}; }}

.note {{ display:flex; gap:12px; align-items:flex-start; font-size:15px; line-height:1.55; color:{C['body']}; padding:14px 18px;
        border-radius:var(--r-md); border:1px solid {C['line']}; background:transparent; margin-top:16px; }}
.note svg {{ flex:none; margin-top:3px; color:{C['muted']}; }}
.note.warn svg {{ color:{C['warn']}; }}
.empty {{ text-align:center; padding:48px 24px; color:{C['body']}; border:1px dashed {C['line2']}; border-radius:var(--r-lg);
         font-size:16px; }}
.empty svg {{ color:{C['faint']}; margin-bottom:10px; }}
.meta {{ display:flex; gap:8px; flex-wrap:wrap; margin: 4px 0 12px; }}
.chip {{ font-family:'DM Mono', monospace; font-size:13px; padding:4px 10px; border-radius:var(--r-sm);
        background:{C['surface']}; color:{C['body']}; border:1px solid {C['line']}; }}
.flow {{ display:flex; gap:8px; flex-wrap:wrap; align-items:center; margin:16px 0 4px; }}
.flow .step {{ padding:8px 12px; border-radius:var(--r-sm); border:1px solid {C['line']}; background:{C['bg']};
              font-size:14px; color:{C['text']}; }}
.flow .arrow {{ color:{C['faint']}; }}
.badge-tag {{ display:inline-block; font-size:13px; font-weight:500; padding:3px 10px; border-radius:999px;
             border:1px solid {C['line2']}; color:{C['body']}; margin-left:auto; }}
.findings {{ margin:0; padding-left:20px; color:{C['body']}; font-size:16px; line-height:1.75; }}

/* native widgets */
[data-testid="stFileUploaderDropzone"] {{ background:{C['surface']}; border:1px dashed {C['line2']}; border-radius:var(--r-md);
                                         transition: border-color 150ms var(--ease); }}
[data-testid="stFileUploaderDropzone"]:hover {{ border-color:{C['muted']}; }}
.stButton > button {{ border-radius:var(--r-sm); font-weight:500; font-size:16px; min-height:44px; padding:0 24px;
                     background:transparent; border:1px solid {C['line2']}; color:{C['text']};
                     transition: transform 150ms var(--ease), background-color 150ms var(--ease), border-color 150ms var(--ease); }}
.stButton > button:hover {{ background: rgba(242,242,242,0.05); border-color:{C['muted']}; color:{C['text']}; transform: translateY(-1px); }}
.stButton > button[kind="primary"] {{ background:{C['accent']}; border:1px solid {C['accent']}; color:{C['accent_ink']}; }}
.stButton > button[kind="primary"]:hover {{ background: color-mix(in oklab, {C['accent']} 86%, {C['text']}); color:{C['accent_ink']};
                                           border-color: color-mix(in oklab, {C['accent']} 86%, {C['text']}); }}
.stButton > button:active {{ transform: translateY(1px) scale(.985); transition-duration: 90ms; }}
.stButton > button:disabled, .stButton > button[kind="primary"]:disabled {{ background:{C['surface']}; color:{C['faint']};
    border:1px solid {C['line']}; transform:none; cursor:not-allowed; }}
.stButton > button:focus-visible {{ outline: 2px solid {C['accent']}; outline-offset: 2px; }}
[data-testid="stTabs"] [role="tablist"] {{ gap: 4px; border-bottom: 1px solid {C['line']}; padding-bottom: 8px; }}
[data-testid="stTabs"] [role="tab"] {{ padding: 8px 16px; border-radius: 999px; min-height:40px;
                                      transition: background-color 240ms var(--ease), color 240ms var(--ease); }}
[data-testid="stTabs"] [role="tab"] p {{ font-weight:500; font-size:15px; color:{C['muted']}; }}
[data-testid="stTabs"] [role="tab"]:hover p {{ color:{C['text']}; }}
[data-testid="stTabs"] [role="tab"][aria-selected="true"] {{ background:{C['text']}; }}
[data-testid="stTabs"] [role="tab"][aria-selected="true"] p {{ color:{C['bg']}; }}
[data-testid="stTabs"] [data-baseweb="tab-highlight"], [data-testid="stTabs"] [data-baseweb="tab-border"] {{ display:none; }}
[data-testid="stProgress"] > div > div > div > div {{ background:{C['accent']}; }}
[data-testid="stDataFrame"] {{ border:1px solid {C['line']}; border-radius:var(--r-md); }}
[data-testid="stExpander"] details {{ border:1px solid {C['line']}; border-radius:var(--r-md); background:{C['surface']}; }}
@media (prefers-reduced-motion: reduce) {{ * {{ transition: none !important; animation: none !important; }} }}
@media (max-width: 640px) {{
  .verdict .score {{ margin-left:0; text-align:left; }} .kpi .value {{ font-size:28px; }}
  .verdict .headline {{ font-size:26px; }} .stButton > button {{ width:100%; }} .kpi {{ margin-bottom:12px; }} }}
</style>
""", unsafe_allow_html=True)


def html(markup):
    st.markdown(markup, unsafe_allow_html=True)


def kpi(label, value, unit="", foot="", accent=None):
    unit_html = f"<small>{unit}</small>" if unit else ""
    dot = f'<span class="marker" style="background:{accent}"></span>' if accent else ""
    return (f'<div class="kpi"><div class="label">{dot}{label}</div>'
            f'<div class="value">{value}{unit_html}</div><div class="foot">{foot}</div></div>')


def kpi_row(items):
    cols = st.columns(len(items), gap="small")
    for col, item in zip(cols, items):
        with col:
            html(kpi(**item))


def section(title, sub="", icon_name=None):
    ic = ""
    html(f'<div class="section-title">{ic}{title} <span class="muted">{sub}</span></div>')


def note(text, kind="info"):
    html(f'<div class="note {kind}">{icon("alert" if kind == "warn" else "info", 16)}<div>{text}</div></div>')


def style_fig(fig, height=320):
    fig.update_layout(
        height=height, margin=dict(l=8, r=8, t=40, b=8), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT, color=C["body"], size=13),
        legend=dict(orientation="h", yanchor="top", y=-0.2, xanchor="left", x=0, font=dict(color=C["muted"])),
        hoverlabel=dict(bgcolor=C["surface2"], bordercolor=C["line2"], font=dict(family=FONT, color=C["text"])),
    )
    fig.update_xaxes(gridcolor=C["grid"], zerolinecolor=C["grid"], linecolor=C["border"], tickfont=dict(color=C["muted"]))
    fig.update_yaxes(gridcolor=C["grid"], zerolinecolor=C["grid"], linecolor=C["border"], tickfont=dict(color=C["muted"]))
    return fig


PLOT_CONFIG = {"displayModeBar": False, "responsive": True}

# ============================================================
# STATE
# ============================================================
st.session_state.setdefault("prediction", None)
st.session_state.setdefault("history", [])


@st.cache_data(ttl=5, show_spinner=False)
def backend_health():
    try:
        r = requests.get(HEALTH_URL, timeout=3)
        if r.status_code == 200:
            return {"online": True, **r.json()}
        return {"online": False, "error": f"HTTP {r.status_code}"}
    except requests.exceptions.RequestException as e:
        return {"online": False, "error": type(e).__name__}


health = backend_health()
model_ready = health.get("online") and health.get("model_loaded")
FINETUNED = health.get("serve_model", "vit") == "finetuned"
SERVED_NAME = "ResNet50 + ViT, fine-tuned" if FINETUNED else str(health.get("model_arch") or "vit_base_patch16_224")

# ============================================================
# TOP BAR
# ============================================================


def pill(color, text, code=""):
    code_html = f"<code>{code}</code>" if code else ""
    return f'<span class="pill"><span class="dot" style="background:{color}"></span>{text}{code_html}</span>'


if health.get("online"):
    pills = [pill(C["real"], "Backend online"),
             pill(C["real"] if health.get("model_loaded") else C["warn"],
                  "Model loaded" if health.get("model_loaded") else "No model weights"),
             pill(C["muted"], "Device", str(health.get("device", "?"))),
             pill(C["muted"], "Model", SERVED_NAME)]
else:
    pills = [pill(C["fake"], "Backend offline", BACKEND_URL)]

html(f"""
<div class="topbar">
  <div class="brand">
    <div class="brand-mark">{icon("shield", 20)}</div>
    <div><h1>DeepShield-MF</h1><p>Deepfake analysis console</p></div>
  </div>
  <div class="pills">{''.join(pills)}</div>
</div>
""")

# ============================================================
# SIDEBAR (connection details)
# ============================================================
with st.sidebar:
    st.markdown("#### Connection")
    st.caption("Backend URL")
    st.code(BACKEND_URL, language=None)
    if health.get("online"):
        st.caption("Checkpoint")
        st.code(str(health.get("model_path") or "-"), language=None)
        st.caption(f"Serving `{health.get('serve_model', 'vit')}` · class order `{health.get('class_names')}` · "
                   f"upload limit `{health.get('max_upload_mb', 100)} MB`")
        if health.get("threshold") is not None:
            st.caption(f"Decision threshold `{health['threshold']:.4f}` (chosen on FF++ validation)")
        if health.get("model_error"):
            st.warning(health["model_error"])
    else:
        st.error(f"Cannot reach the backend ({health.get('error')}). Start it with `python backend/app.py`.")
    if st.button("Re-check backend", width="stretch"):
        backend_health.clear()
        st.rerun()

tab_analyze, tab_history, tab_research, tab_about = st.tabs(["Analyze", "History", "Research results", "About"])

# ============================================================
# ANALYZE
# ============================================================


def run_analysis(uploaded_file):
    """POST the file to the backend and read the server-sent events. Returns the result dict or None."""
    progress = st.progress(0, text="Uploading to the backend...")
    timeline = []  # cumulative (frames analysed, fake frames) after every sampled frame
    latest = None
    try:
        uploaded_file.seek(0)
        files = {"file": (uploaded_file.name, uploaded_file, uploaded_file.type)}
        response = requests.post(PREDICT_URL, files=files, stream=True, timeout=600)
        if response.status_code != 200:
            progress.empty()
            try:
                detail = response.json()
            except ValueError:
                detail = response.text
            st.error(f"Backend returned HTTP {response.status_code}: {detail}")
            return None
        progress.progress(15, text="Analyzing frames...")
        for line in response.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data:"):
                continue
            try:
                data = json.loads(line[len("data:"):].strip())
            except json.JSONDecodeError:
                continue
            if "error" in data:
                progress.empty()
                if data.get("kind") == "no_face":
                    st.warning(data["error"] + " The fine-tuned model judges faces only.")
                else:
                    st.error(f"Backend error: {data['error']}")
                return None
            stage = data.get("stage")
            if stage == "plan":
                progress.progress(8, text=f"Reading the whole video: {data['segments']} segment"
                                         f"{'s' if data['segments'] != 1 else ''} over {data['video_seconds']:.0f} s...")
                continue
            if stage == "segment":
                sc = data.get("score")
                progress.progress(8 + int(87 * data["done"] / data["of"]),
                                  text=f"Segment {data['done']} of {data['of']} · "
                                       + (f"score {sc:.2f}" if sc is not None else "no face in this part"))
                continue
            if stage == "decoding":
                progress.progress(20, text=f"Decoding {data['frames_wanted']} frames from the middle of the clip...")
                continue
            if stage == "faces":
                progress.progress(20 + int(60 * data["done"] / data["of"]),
                                  text=f"Tracking the face · {data['done']} of {data['of']} frames")
                continue
            if stage == "scoring":
                progress.progress(88, text=f"Scoring {data['frames']} face crops in 8-frame windows...")
                continue
            latest = data
            if stage == "done":
                continue
            timeline.append((data.get("total_frames", 0), data.get("fake_count", 0)))
            n = data.get("total_frames", 0)
            progress.progress(min(95, 15 + int(min(n, 120) / 120 * 80)),
                              text=f"Analyzed {n} frame{'s' if n != 1 else ''} · "
                                   f"{data.get('fake_percentage', 0):.1f}% classified fake so far")
    except requests.exceptions.Timeout:
        progress.empty()
        st.error("The request timed out. The video may be too long for the 10-minute limit.")
        return None
    except requests.exceptions.ConnectionError:
        progress.empty()
        st.error(f"Could not connect to the backend at {BACKEND_URL}.")
        return None
    if latest is None:
        progress.empty()
        st.error("The backend returned no prediction.")
        return None
    progress.progress(100, text="Analysis complete")
    kind = "image" if (uploaded_file.type or "").startswith("image/") else "video"
    return {**latest, "timeline": timeline, "file_name": uploaded_file.name, "kind": kind,
            "size_mb": uploaded_file.size / (1024 * 1024), "time": datetime.now().strftime("%H:%M:%S")}


def verdict_of(p):
    if p.get("model_kind") == "finetuned":
        fake = p.get("label") == "fake"
    else:
        fake = p.get("fake_percentage", 0) > p.get("real_percentage", 0)
    return ("fake", "Manipulation detected") if fake else ("real", "No manipulation detected")


def score_pct(p):
    """(score, threshold) on a 0-100 scale for either model."""
    if p.get("model_kind") == "finetuned":
        return 100 * p["score"], 100 * p["threshold"]
    return p.get("fake_percentage", 0.0), 50.0


def score_gauge(score, threshold, cls):
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=score,
        number=dict(valueformat=".2f", font=dict(family=FONT, size=44, color=C["text"])),
        title=dict(text=f"Video fake score · threshold {threshold:.2f}", font=dict(size=13, color=C["muted"])),
        gauge=dict(
            shape="angular", axis=dict(range=[0, 1], tickwidth=1, tickcolor=C["faint"], tickfont=dict(color=C["muted"]),
                                        tickvals=[0, 0.25, 0.5, 0.75, 1]),
            bar=dict(color=C["fake"] if cls == "fake" else C["real"], thickness=0.28),
            bgcolor="rgba(0,0,0,0)", borderwidth=0,
            steps=[dict(range=[0, threshold], color="rgba(242,242,242,0.05)"),
                   dict(range=[threshold, 1], color="rgba(242,242,242,0.10)")],
            threshold=dict(line=dict(color=C["text"], width=3), thickness=0.85, value=threshold),
        )))
    return style_fig(fig, 300)


def segment_timeline(segs, threshold, decider, middle):
    """One bar per segment along the video's time axis; height = segment score."""
    fig = go.Figure()
    xs = [(x["start_s"] + x["end_s"]) / 2 for x in segs]
    widths = [max(0.2, (x["end_s"] - x["start_s"]) * 0.92) for x in segs]
    ys = [x["score"] if x["score"] is not None else 0.03 for x in segs]
    colors = [C["faint"] if x["score"] is None else (C["fake"] if x["score"] >= threshold else C["real"]) for x in segs]
    text = ["no face" if x["score"] is None else f"{x['score']:.2f}" for x in segs]
    fig.add_trace(go.Bar(x=xs, y=ys, width=widths, marker_color=colors, marker_line_width=0, text=text,
                         textposition="outside", cliponaxis=False, textfont=dict(family=FONT, size=12, color=C["text"]),
                         customdata=[[x["start_s"], x["end_s"]] for x in segs],
                         hovertemplate="%{customdata[0]:.1f}-%{customdata[1]:.1f} s: %{text}<extra></extra>"))
    fig.add_hline(y=threshold, line=dict(color=C["text"], width=1.5, dash="dot"))
    fig.add_annotation(x=1, xref="paper", y=threshold, text="threshold", showarrow=False, xanchor="right", yshift=10,
                       font=dict(size=12, color=C["muted"]))
    m = segs[middle]
    fig.add_annotation(x=(m["start_s"] + m["end_s"]) / 2, y=0, yref="paper", yshift=-2, text="middle (evaluated)",
                       showarrow=False, yanchor="top", font=dict(size=11, color=C["muted"]))
    fig.update_yaxes(range=[0, 1.15], tickvals=[0, 0.5, 1], title_text="Segment score", title_font=dict(color=C["muted"], size=12))
    fig.update_xaxes(title_text="Time in the video (s)", title_font=dict(color=C["muted"], size=12), ticksuffix=" s")
    fig.update_layout(bargap=0.05, showlegend=False,
                      title=dict(text="Whole video: score of every segment (32 frames, about 8-11 s each)", font=dict(size=13, color=C["muted"]), x=0.01))
    return style_fig(fig, 300)


EXPLAIN_URL = f"{BACKEND_URL}/explain"
# deletion test on validation (results/final/heatmap_deletion_val.json, docs/final_evaluation_protocol.md)
HEATMAP_RELIABILITY = {
    "real": "On validation videos judged real, hiding each frame's 5 hottest regions weakened the verdict more than "
            "hiding 5 random regions in 94% of videos, so these maps show what the model relied on.",
    "fake": "On validation videos judged fake, the maps passed the check only narrowly: the hottest regions mattered "
            "more than random ones in 65% of videos, but hiding them barely changed the fake evidence on average. "
            "Read them as a hint, not as the place where the manipulation is.",
}


def render_heatmaps(p):
    """'Why' section: per-frame Grad-CAM heatmaps from POST /explain, loaded on request."""
    section("Why the model decided this", "per-frame heatmaps of the scored segment", "info")
    key = f"explain_{p['explain_id']}"
    if key not in st.session_state:
        if st.button("Show per-frame heatmaps", key=f"btn_{key}"):
            with st.spinner("Computing heatmaps for every frame..."):
                try:
                    r = requests.post(EXPLAIN_URL, json={"id": p["explain_id"]}, timeout=300)
                    st.session_state[key] = r.json() if r.status_code == 200 else {"error": r.json().get("error", r.text)}
                except requests.exceptions.RequestException as exc:
                    st.session_state[key] = {"error": f"Could not reach the backend ({type(exc).__name__})."}
            st.rerun()
        return
    e = st.session_state[key]
    if "error" in e:
        st.error(e["error"])
        return
    frames = e["frames"]
    show = st.radio("Show", ["Heatmap", "Original frames"], horizontal=True, key=f"view_{key}", label_visibility="collapsed")
    imgs = "".join(f'<figure style="margin:0"><img src="{f["overlay"] if show == "Heatmap" else f["image"]}" alt="Frame {i + 1}" '
                   f'style="width:100%;border-radius:8px;display:block"><figcaption style="font-size:13px;color:{C["muted"]};'
                   f'margin-top:4px">Frame {i + 1}</figcaption></figure>' for i, f in enumerate(frames))
    ncol = 8 if len(frames) > 1 else 1
    width = "160px" if len(frames) == 1 else "1fr"
    html(f'<div class="card"><div class="card-title">Toward "{e["verdict"]}": brighter = pushed the model toward this verdict'
         f'</div><div style="display:grid;grid-template-columns:repeat({ncol},minmax(0,{width}));gap:10px;margin-top:10px">'
         f'{imgs}</div></div>')
    if len(frames) > 1:
        fig = go.Figure(go.Bar(x=[i + 1 for i in range(len(frames))], y=[f["heat_mean"] for f in frames],
                               marker_color=C["accent"], marker_line_width=0,
                               hovertemplate="Frame %{x}: mean heat %{y:.3f}<extra></extra>"))
        fig.update_xaxes(title_text="Frame of the scored segment", title_font=dict(color=C["muted"], size=12), dtick=4)
        fig.update_yaxes(title_text="Mean heat", title_font=dict(color=C["muted"], size=12))
        fig.update_layout(title=dict(text="How much each frame drove the verdict", font=dict(size=13, color=C["muted"]), x=0.01))
        st.plotly_chart(style_fig(fig, 240), width="stretch", config=PLOT_CONFIG)
    note("Grad-CAM of both branches (ResNet50 and ViT) toward the verdict, on one scale for all frames. "
         + HEATMAP_RELIABILITY[e["verdict"]], "info" if e["verdict"] == "real" else "warn")


def window_chart(probs, threshold):
    xs = [f"W{i + 1}" for i in range(len(probs))]
    fig = go.Figure(go.Bar(
        x=xs, y=probs, marker_color=[C["fake"] if v >= threshold else C["real"] for v in probs], marker_line_width=0,
        text=[f"{v:.2f}" for v in probs], textposition="outside", cliponaxis=False,
        textfont=dict(family=FONT, color=C["text"], size=12),
        customdata=[f"frames {4 * i + 1}-{4 * i + 8}" for i in range(len(probs))],
        hovertemplate="%{x} (%{customdata}): P(fake) %{y:.3f}<extra></extra>"))
    fig.add_hline(y=threshold, line=dict(color=C["text"], width=1.5, dash="dot"))
    fig.add_annotation(x=1, xref="paper", y=threshold, text="threshold", showarrow=False, xanchor="right", yshift=10,
                       font=dict(size=12, color=C["muted"]))
    fig.update_yaxes(range=[0, 1.12], tickvals=[0, 0.5, 1], title_text="P(fake)", title_font=dict(color=C["muted"], size=12))
    fig.update_xaxes(title_text="8-frame windows along the face track (50% overlap)", title_font=dict(color=C["muted"], size=12))
    fig.update_layout(bargap=0.3, title=dict(text="Window scores", font=dict(size=13, color=C["muted"]), x=0.01))
    return style_fig(fig, 300)


def gauge(fake_share, cls):
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=fake_share,
        number=dict(suffix="%", valueformat=".1f", font=dict(family=FONT, size=44, color=C["text"])),
        title=dict(text="Fake-frame share", font=dict(size=13, color=C["muted"])),
        gauge=dict(
            shape="angular", axis=dict(range=[0, 100], tickwidth=1, tickcolor=C["faint"], tickfont=dict(color=C["muted"]),
                                        tickvals=[0, 25, 50, 75, 100]),
            bar=dict(color=C["fake"] if cls == "fake" else C["real"], thickness=0.28),
            bgcolor="rgba(0,0,0,0)", borderwidth=0,
            steps=[dict(range=[0, 50], color="rgba(242,242,242,0.05)"), dict(range=[50, 100], color="rgba(242,242,242,0.10)")],
            threshold=dict(line=dict(color=C["text"], width=3), thickness=0.85, value=50),
        )))
    fig.add_annotation(x=0.18, y=0.0, text="real majority", showarrow=False, font=dict(size=11, color=C["muted"]))
    fig.add_annotation(x=0.82, y=0.0, text="fake majority", showarrow=False, font=dict(size=11, color=C["muted"]))
    return style_fig(fig, 300)


def frame_timeline(timeline):
    frames = [t for t, _ in timeline]
    fakes = [f for _, f in timeline]
    per_frame = [fakes[0]] + [b - a for a, b in zip(fakes, fakes[1:])]
    running = [100 * f / t for t, f in timeline]
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.72, 0.28], vertical_spacing=0.08)
    fig.add_trace(go.Scatter(x=frames, y=running, mode="lines", line=dict(color=C["accent"], width=2, shape="spline"),
                             fill="tozeroy", fillcolor="rgba(160,134,114,0.12)", name="Running fake share",
                             hovertemplate="After %{x} frames: %{y:.1f}% fake<extra></extra>"), row=1, col=1)
    fig.add_hline(y=50, line=dict(color=C["faint"], width=1, dash="dot"), row=1, col=1)
    fig.add_trace(go.Bar(x=frames, y=[1] * len(frames), marker_color=[C["fake"] if v else C["real"] for v in per_frame],
                         marker_line_width=0, name="Frame verdict", showlegend=False,
                         customdata=["fake" if v else "real" for v in per_frame],
                         hovertemplate="Frame %{x}: %{customdata}<extra></extra>"), row=2, col=1)
    fig.update_yaxes(range=[0, 105], tickvals=[0, 50, 100], ticksuffix="%", title_text="Fake share", title_font=dict(color=C["muted"], size=12),
                     row=1, col=1)
    fig.update_yaxes(visible=False, row=2, col=1)
    fig.update_xaxes(title_text="Sampled frame (about 2 per second)", title_font=dict(color=C["muted"], size=12), row=2, col=1)
    fig.update_layout(bargap=0.08, showlegend=False,
                      title=dict(text="Frame-by-frame verdicts", font=dict(size=13, color=C["muted"]), x=0.01))
    return style_fig(fig, 300)


with tab_analyze:
    left, right = st.columns([5, 7], gap="large")
    with left:
        how = ("The model reads the whole video in segments of about 10 s, tracking the face in each; the verdict uses "
               "the middle segment."
               if FINETUNED else "Videos are sampled at about two frames per second.")
        html(f'<div class="card"><div class="card-title">{icon("layers", 18)}Upload media</div>'
             f'<p class="card-sub">Images (JPG, PNG, BMP, WebP) or videos (MP4, AVI, MOV, MKV). {how}</p></div>')
        uploaded = st.file_uploader("Image or video", type=["jpg", "jpeg", "png", "bmp", "webp", "mp4", "avi", "mov", "mkv"],
                                    accept_multiple_files=False, label_visibility="collapsed")
        if uploaded is not None:
            kind = "image" if (uploaded.type or "").startswith("image/") else "video"
            html(f'<div class="meta"><span class="chip">{kind}</span><span class="chip">{uploaded.size / (1024 * 1024):.2f} MB</span>'
                 f'<span class="chip">{uploaded.type or "unknown type"}</span></div>')
        disabled = uploaded is None or not model_ready
        clicked = st.button("Analyze media", type="primary", width="stretch", disabled=disabled)
        if not model_ready:
            note("Analysis is disabled: " + ("the backend has no model weights loaded."
                                             if health.get("online") else f"the backend at <code>{BACKEND_URL}</code> is offline."),
                 "warn")
        if clicked and uploaded is not None:
            if uploaded.size == 0:
                st.error("The uploaded file is empty.")
            else:
                result = run_analysis(uploaded)
                if result:
                    st.session_state.prediction = result
                    st.session_state.history.insert(0, result)
                    st.rerun()
    with right:
        if uploaded is not None:
            if (uploaded.type or "").startswith("image/"):
                st.image(uploaded, width="stretch")
            else:
                st.video(uploaded)
        else:
            html(f'<div class="empty">{icon("film", 34)}<div style="font-weight:600;color:{C["text"]}">No media selected</div>'
                 f'<div style="font-size:0.86rem">Drop a file on the left to preview it here.</div></div>')

    p = st.session_state.prediction
    if p and p.get("model_kind") == "finetuned":
        cls, headline = verdict_of(p)
        score, thr = p["score"], p["threshold"]
        side = "above" if score >= thr else "below"
        section("Result", f"{p['file_name']} · analyzed at {p['time']}", "shield")
        if p.get("mode") == "image":
            detail = f"Score {score:.2f} is {side} the threshold {thr:.2f} · single image, face repeated as one window"
        else:
            if p.get("segments"):
                rule_txt = {"middle": "the middle segment", "mean": "the mean over segments", "max": "the highest segment"}[p.get("rule", "middle")]
                n_fake = sum(1 for x in p["segments"] if x["score"] is not None and x["score"] >= thr)
                detail = (f"Score {score:.2f} ({rule_txt}) is {side} the threshold {thr:.2f} · "
                          f"{n_fake} of {p['segments_scored']} segments above it over {p['video_seconds']:.0f} s")
            else:
                detail = f"Score {score:.2f} is {side} the threshold {thr:.2f} · {p['windows_fake']} of {len(p['window_probs'])} windows above it"
        html(f"""
        <div class="verdict {cls}">
          <div class="badge">{icon("alert" if cls == "fake" else "check", 30)}</div>
          <div><div class="eyebrow">Verdict · fine-tuned ResNet50 + ViT</div><div class="headline">{headline}</div>
               <div class="detail">{detail}</div></div>
          <div class="score"><div class="num">{score:.2f}</div><div class="cap">video fake score</div></div>
        </div>""")
        st.write("")
        if p.get("mode") == "video":
            a, b = p["segment_seconds"]
            face_foot = f"of {p['frames_analyzed']} frames, {a:.1f}&ndash;{b:.1f} s of {p['video_seconds']:.1f} s"
        else:
            face_foot = "one still image"
        pc = p.get("p_calibrated")
        second = (dict(label="Calibrated probability fake", value=f"{100 * pc:.0f}", unit="%",
                       foot="if real and fake were equally likely; typically off by about 10 points")
                  if pc is not None else dict(label="Threshold", value=f"{thr:.3f}", foot="chosen on validation, then frozen"))
        kpi_row([
            dict(label="Video fake score", value=f"{score:.3f}", foot=f"mean P(fake) over the windows · threshold {thr:.3f}",
                 accent=C["fake"] if cls == "fake" else C["real"]),
            second,
            dict(label="Windows above threshold", value=f"{p['windows_fake']}", unit=f"/{len(p['window_probs'])}",
                 foot="8-frame windows, 50% overlap"),
            dict(label="Face found", value=f"{100 * p['face_rate']:.0f}", unit="%", foot=face_foot),
        ])
        if p.get("segments") and len(p["segments"]) > 1:
            st.plotly_chart(segment_timeline(p["segments"], thr, p["decider"], p["middle"]), width="stretch", config=PLOT_CONFIG)
            flagged = [x for k, x in enumerate(p["segments"])
                       if k != p["decider"] and x["score"] is not None and x["score"] >= thr]
            if flagged and cls == "real":
                spans = ", ".join(f"{x['start_s']:.0f}&ndash;{x['end_s']:.0f} s" for x in flagged[:4])
                note(f"{len(flagged)} other segment{'s' if len(flagged) > 1 else ''} score{'' if len(flagged) > 1 else 's'} "
                     f"above the threshold ({spans}). The verdict uses the {p.get('rule', 'middle')} segment, the reading the model "
                     "was evaluated on; check those passages yourself.", "warn")
        g, t = st.columns([4, 7], gap="medium")
        with g:
            st.plotly_chart(score_gauge(score, thr, cls), width="stretch", config=PLOT_CONFIG)
        with t:
            if p.get("mode") == "video":
                st.plotly_chart(window_chart(p["window_probs"], thr), width="stretch", config=PLOT_CONFIG)
            else:
                html(f'<div class="empty" style="margin-top:30px">{icon("image", 30)}<div style="font-weight:600;color:{C["text"]}">'
                     f'Single image</div><div style="font-size:16px">The model reads 8-frame windows; a still is one window '
                     f'of the same face, a case it was never tested on.</div></div>')
        thumbs = "".join(f'<img src="{u}" alt="Face crop {i + 1}" style="width:100%;border-radius:8px;display:block">'
                         for i, u in enumerate(p.get("thumbs", [])))
        cols = max(1, len(p.get("thumbs", [])))
        html(f'<div class="card" style="margin-top:8px"><div class="card-title">Face crops the model scored</div>'
             f'<div style="display:grid;grid-template-columns:repeat({8 if cols > 1 else 1},minmax(0,{96 if cols == 1 else 999}px));'
             f'gap:10px;margin-top:8px">{thumbs}</div></div>')
        if p.get("explain_id"):
            render_heatmaps(p)
        note("The score is the model's mean P(fake) over 8-frame windows of the tracked face. It is not a calibrated "
             "probability. Trained on FaceForensics++ and Celeb-DF, it reached 0.874 AUC on the FF++ test split and 0.956 on "
             "the Celeb-DF v2 test list. On DeepFakeDetection, a dataset it never saw, it reached 0.814 AUC but caught only "
             "65% of the fakes and called 18% of the real videos fake. Treat a verdict as a signal, not proof."
             + (" Stills were never part of its evaluation." if p.get("mode") == "image" else ""))
    elif p:
        cls, headline = verdict_of(p)
        fake_share = p.get("fake_percentage", 0.0)
        n, nf, nr = p.get("total_frames", 0), p.get("fake_count", 0), p.get("real_count", 0)
        unit = "frame" if n == 1 else "frames"
        section("Result", f"{p['file_name']} · analyzed at {p['time']}", "shield")
        html(f"""
        <div class="verdict {cls}">
          <div class="badge">{icon("alert" if cls == "fake" else "check", 30)}</div>
          <div><div class="eyebrow">Verdict</div><div class="headline">{headline}</div>
               <div class="detail">{nf} of {n} sampled {unit} classified as fake</div></div>
          <div class="score"><div class="num">{fake_share:.1f}%</div><div class="cap">fake-frame share</div></div>
        </div>""")
        st.write("")
        kpi_row([
            dict(label="Frames analyzed", value=f"{n}", foot="sampled from the upload"),
            dict(label="Classified fake", value=f"{nf}", foot=f"{fake_share:.1f}% of frames", accent=C["fake"]),
            dict(label="Classified real", value=f"{nr}", foot=f"{p.get('real_percentage', 0):.1f}% of frames", accent=C["real"]),
            dict(label="Decision rule", value="&gt;50", unit="%", foot="of frames fake for a fake verdict"),
        ])
        g, t = st.columns([4, 7], gap="medium")
        with g:
            st.plotly_chart(gauge(fake_share, cls), width="stretch", config=PLOT_CONFIG)
        with t:
            if p.get("kind") == "video" and len(p.get("timeline", [])) > 1:
                st.plotly_chart(frame_timeline(p["timeline"]), width="stretch", config=PLOT_CONFIG)
            else:
                html(f'<div class="empty" style="margin-top:30px">{icon("image", 30)}<div style="font-weight:600;color:{C["text"]}">'
                     f'Single-frame input</div><div style="font-size:0.86rem">A frame-by-frame timeline is shown for videos.</div></div>')
        note("The score is the share of sampled frames that the served ViT frame classifier labels fake. It is a vote "
             "count, not a calibrated probability, and the classifier was not evaluated on held-out videos. See "
             "<b>Research results</b> for the evaluated video-level models.")

# ============================================================
# HISTORY
# ============================================================
with tab_history:
    hist = st.session_state.history
    if not hist:
        html(f'<div class="empty">{icon("layers", 34)}<div style="font-weight:600;color:{C["text"]}">No analyses yet</div>'
             f'<div style="font-size:0.86rem">Results from this browser session appear here.</div></div>')
    else:
        n_fake = sum(verdict_of(h)[0] == "fake" for h in hist)
        kpi_row([
            dict(label="Analyses this session", value=str(len(hist)), foot="most recent first"),
            dict(label="Flagged as manipulated", value=str(n_fake), foot=f"{len(hist) - n_fake} judged real", accent=C["fake"]),
            dict(label="Frames analyzed", value=str(sum(h.get("frames_analyzed", h.get("total_frames", 0)) for h in hist)),
                 foot="across all uploads"),
        ])
        st.write("")
        df = pd.DataFrame([{
            "Time": h["time"], "File": h["file_name"], "Type": h["kind"],
            "Model": "Fine-tuned" if h.get("model_kind") == "finetuned" else "ViT vote",
            "Frames": h.get("frames_analyzed", h.get("total_frames", 0)),
            "Fake score (%)": round(score_pct(h)[0], 1),
            "Verdict": "Manipulated" if verdict_of(h)[0] == "fake" else "Real"} for h in hist])
        c1, c2 = st.columns([7, 5], gap="medium")
        with c1:
            st.dataframe(df, width="stretch", hide_index=True, column_config={
                "Fake score (%)": st.column_config.ProgressColumn("Fake score", min_value=0, max_value=100, format="%.1f%%")})
        with c2:
            fig = go.Figure(go.Bar(
                y=[f"{h['time']} · {h['file_name'][:22]}" for h in hist][::-1],
                x=[score_pct(h)[0] for h in hist][::-1], orientation="h",
                marker_color=[C["fake"] if verdict_of(h)[0] == "fake" else C["real"] for h in hist][::-1],
                text=[f"{score_pct(h)[0]:.1f}%" for h in hist][::-1], textposition="outside", cliponaxis=False,
                width=0.5, textfont=dict(family=FONT, color=C["text"], size=12),
                hovertemplate="%{y}: %{x:.1f}% fake<extra></extra>"))
            thresholds = {round(score_pct(h)[1], 2) for h in hist}
            if len(thresholds) == 1:
                fig.add_vline(x=thresholds.pop(), line=dict(color=C["faint"], dash="dot"))
            fig.update_xaxes(range=[0, 118], tickvals=[0, 50, 100], ticksuffix="%")
            fig.update_layout(title=dict(text="Fake score per upload", font=dict(size=13, color=C["muted"]), x=0.01))
            st.plotly_chart(style_fig(fig, max(200, 90 + 36 * len(hist))), width="stretch", config=PLOT_CONFIG)
        if st.button("Clear history"):
            st.session_state.history = []
            st.session_state.prediction = None
            st.rerun()

# ============================================================
# RESEARCH RESULTS (read from results/ written by the research scripts)
# ============================================================
DATASETS = [("ffpp_test", "FF++ c23 test"), ("celebdf_v1", "Celeb-DF v1"), ("celebdf_v2", "Celeb-DF v2"),
            ("dfd", "DFD (unseen)")]
MODELS = [("final_cdf", "Fine-tuned, FF++ + Celeb-DF (served)", C["accent"]), ("final", "Fine-tuned, FF++ only", C["series2"]),
          ("C", "C: ResNet50+ViT, frozen", C["series3"]), ("F", "F: full hybrid, frozen", C["faint"])]


def _read_json(path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


@st.cache_data(ttl=300, show_spinner=False)
def load_research(results_dir: str):
    root = Path(results_dir)
    final = {}
    for ds, _ in DATASETS:
        for key, _, _ in MODELS:
            files = [root / "final" / ds / f"{key}.json"] if key.startswith("final") else \
                    [root / "final" / ds / f"{key}_seed{s}.json" for s in range(3)]
            reps = [r for r in (_read_json(f) for f in files) if r]
            if reps:
                final[(ds, key)] = reps
    ablation = []
    abl_csv = root / "ffpp_c23" / "summary" / "ablation_table.csv"
    if abl_csv.exists():
        for r in csv.DictReader(open(abl_csv, encoding="utf-8")):
            if r["run"] in {f"{x}_faces_T8" for x in "ABCDEF"}:
                ablation.append(r)
    robust_c = {}
    rob_csv = root / "ffpp_c23" / "summary" / "robustness.csv"
    if rob_csv.exists():
        acc = {}
        for r in csv.DictReader(open(rob_csv, encoding="utf-8")):
            if r["run"] == "C_faces_T8":
                key = "clean" if r["condition"] == "clean" else f"{r['condition']}_{r['level']}"
                acc.setdefault(key, []).append(float(r["roc_auc"]))
        robust_c = {k: st_stats.mean(v) for k, v in acc.items()}
    robust_final = _read_json(root / "ffpp_c23" / "e2e_finetune_aug-frame_final" / "seed0" / "metrics" / "val_robustness.json") or {}
    val_final = _read_json(root / "ffpp_c23" / "e2e_finetune_aug-frame_final" / "seed0" / "metrics" / "val_metrics.json")
    return {"final": final, "ablation": ablation, "robust_c": robust_c,
            "robust_final": {k: v["roc_auc"] for k, v in robust_final.items()}, "val_final": val_final}


def mean_of(reps, key):
    return st_stats.mean(r[key] for r in reps)


with tab_research:
    R = load_research(str(RESULTS_DIR))
    if not R["final"]:
        html(f'<div class="empty">{icon("layers", 34)}<div style="font-weight:600;color:{C["text"]}">No research results found</div>'
             f'<div style="font-size:0.86rem">Expected files under <code>{RESULTS_DIR}</code>. Run the pipeline in the README '
             f'(Steps 1 to 8) to produce them.</div></div>')
    else:
        if FINETUNED:
            note("Offline evaluation of the <b>research pipeline</b> (video level, face crops, held-out test sets, each scored "
                 "once). The Analyze tab serves the model trained on <b>FF++ + Celeb-DF v2</b>; its Celeb-DF scores are "
                 "same-dataset results (unseen videos, but celebrities it also saw in training).")
        else:
            note("These are offline evaluations of the <b>research pipeline</b> (video level, face crops, source-family-disjoint "
                 "splits, each test set scored once). They are <b>not</b> results of the ViT frame classifier served on the "
                 "Analyze tab.", "warn")
        F = R["final"]

        def tile(ds, label):
            fin, base = F.get((ds, "final_cdf")), F.get((ds, "final"))
            if not fin:
                return dict(label=label, value="-", foot="not evaluated")
            r = fin[0]
            ci = r.get("ci95_family_bootstrap", {}).get("roc_auc", [None, None])
            delta = f'<br><span class="delta">{r["roc_auc"] - mean_of(base, "roc_auc"):+.3f}</span> vs FF++-only model' if base else ""
            return dict(label=f"{label} · AUC", value=f"{r['roc_auc']:.3f}",
                        foot=f"95% CI {ci[0]:.3f}&ndash;{ci[1]:.3f}{delta}")

        section("Served model on held-out data", "fine-tuned ResNet50 + ViT, trained on FF++ + Celeb-DF v2", "shield")
        ff = F.get(("ffpp_test", "final_cdf"), [{}])[0]
        kpi_row([tile("ffpp_test", "FF++ test"), tile("celebdf_v2", "Celeb-DF v2"), tile("dfd", "DFD, never seen"),
                 dict(label="FF++ test · balanced acc", value=f"{ff.get('balanced_accuracy', 0):.3f}",
                      foot=f"fake recall {ff.get('fake_recall', 0):.3f}<br>specificity {ff.get('specificity', 0):.3f}")])

        c1, c2 = st.columns(2, gap="medium")
        with c1:
            fig = go.Figure()
            for key, name, color in MODELS:
                xs, ys, err_lo, err_hi = [], [], [], []
                for ds, dlabel in DATASETS:
                    reps = F.get((ds, key))
                    if not reps:
                        continue
                    xs.append(dlabel)
                    m = mean_of(reps, "roc_auc")
                    ys.append(m)
                    if key.startswith("final"):
                        lo, hi = reps[0]["ci95_family_bootstrap"]["roc_auc"]
                        err_lo.append(m - lo)
                        err_hi.append(hi - m)
                    else:
                        sd = st_stats.stdev([r["roc_auc"] for r in reps]) if len(reps) > 1 else 0
                        err_lo.append(sd)
                        err_hi.append(sd)
                fig.add_trace(go.Bar(x=xs, y=ys, name=name, marker_color=color, marker_line_width=0,
                                     error_y=dict(type="data", symmetric=False, array=err_hi, arrayminus=err_lo,
                                                  color=C["muted"], thickness=1.2, width=4),
                                     text=[f"{v:.3f}" for v in ys], textposition="inside", insidetextanchor="end",
                                     textfont=dict(family=FONT, color=C["accent_ink"], size=11),
                                     hovertemplate=f"{name}<br>%{{x}}: AUC %{{y:.3f}}<extra></extra>"))
            fig.update_yaxes(range=[0.5, 1.0], title_text="Video-level AUC", title_font=dict(color=C["muted"], size=12))
            fig.update_layout(barmode="group", bargap=0.25, bargroupgap=0.06,
                              title=dict(text="Held-out and cross-dataset AUC", font=dict(size=13, color=C["muted"]), x=0.01))
            st.plotly_chart(style_fig(fig, 360), width="stretch", config=PLOT_CONFIG)
        with c2:
            conds = [("clean", "Clean"), ("jpeg_40", "JPEG q40"), ("blur_1", "Blur"), ("downscale_2", "Downscale ×2"),
                     ("noise_10", "Noise"), ("brightness_0.6", "Dim ×0.6")]
            fig = go.Figure()
            for data, name, color in [(R["robust_c"], "C: frozen features", C["series2"]),
                                      (R["robust_final"], "Fine-tuned (final)", C["accent"])]:
                pts = [(lab, data[k]) for k, lab in conds if k in data]
                if pts:
                    fig.add_trace(go.Scatter(x=[a for a, _ in pts], y=[b for _, b in pts], mode="lines+markers", name=name,
                                             line=dict(color=color, width=2.5), marker=dict(size=9, line=dict(color=C["bg"], width=2)),
                                             hovertemplate=f"{name}<br>%{{x}}: AUC %{{y:.3f}}<extra></extra>"))
            fig.update_yaxes(range=[0.6, 0.9], title_text="Validation AUC", title_font=dict(color=C["muted"], size=12))
            fig.update_layout(title=dict(text="Robustness to degradations (validation)", font=dict(size=13, color=C["muted"]), x=0.01))
            st.plotly_chart(style_fig(fig, 360), width="stretch", config=PLOT_CONFIG)

        c3, c4 = st.columns(2, gap="medium")
        with c3:
            if R["ablation"]:
                names = {"A": "A · ResNet50", "B": "B · ViT", "C": "C · ResNet50+ViT", "D": "D · ResNet50+BiLSTM",
                         "E": "E · ResNet50+ViT+BiLSTM", "F": "F · full hybrid (+FFT)"}
                rows = sorted(R["ablation"], key=lambda r: r["run"])[::-1]
                fig = go.Figure(go.Bar(
                    y=[names[r["run"][0]] for r in rows], x=[float(r["roc_auc_mean"]) for r in rows], orientation="h",
                    error_x=dict(type="data", array=[float(r["roc_auc_std"]) for r in rows], color=C["muted"], thickness=1.2, width=4),
                    marker_color=[C["accent"] if r["run"].startswith("C") else C["series3"] for r in rows],
                    marker_line_width=0,
                    customdata=[int(r["trainable_params"]) / 1e6 for r in rows],
                    text=[f"{float(r['roc_auc_mean']):.3f}" for r in rows], textposition="inside", insidetextanchor="end",
                    textfont=dict(family=FONT, size=11, color=C["accent_ink"]),
                    hovertemplate="%{y}<br>AUC %{x:.3f}<br>%{customdata:.2f}M trainable params<extra></extra>"))
                fig.update_xaxes(range=[0.6, 0.8], title_text="Validation AUC (3 seeds)", title_font=dict(color=C["muted"], size=12))
                fig.update_layout(title=dict(text="Ablation: which branches help", font=dict(size=13, color=C["muted"]), x=0.01))
                st.plotly_chart(style_fig(fig, 340), width="stretch", config=PLOT_CONFIG)
        with c4:
            meths = ["Deepfakes", "Face2Face", "FaceShifter", "FaceSwap", "NeuralTextures"]
            fig = go.Figure()
            for key, name, color in MODELS[:3][::-1]:
                reps = F.get(("ffpp_test", key))
                if reps:
                    vals = [st_stats.mean(r["per_method"][m]["roc_auc"] for r in reps) for m in meths]
                    fig.add_trace(go.Bar(y=meths[::-1], x=vals[::-1], orientation="h", name=name, marker_color=color,
                                         marker_line_width=0, text=[f"{v:.3f}" for v in vals[::-1]], textposition="inside",
                                         insidetextanchor="end", textfont=dict(family=FONT, color=C["accent_ink"], size=11),
                                         hovertemplate=f"{name}<br>%{{y}}: AUC %{{x:.3f}}<extra></extra>"))
            fig.update_xaxes(range=[0.6, 1.0], title_text="FF++ test AUC (method vs all reals)",
                             title_font=dict(color=C["muted"], size=12))
            fig.update_layout(barmode="group", bargap=0.28, bargroupgap=0.08,
                              title=dict(text="Per manipulation method (test split)", font=dict(size=13, color=C["muted"]), x=0.01))
            st.plotly_chart(style_fig(fig, 340), width="stretch", config=PLOT_CONFIG)

        rows = []
        for ds, dlabel in DATASETS:
            for key, name, _ in MODELS:
                reps = F.get((ds, key))
                if reps:
                    rows.append({"Dataset": dlabel, "Model": name, "Seeds": len(reps), "AUC": mean_of(reps, "roc_auc"),
                                 "Balanced acc": mean_of(reps, "balanced_accuracy"), "Fake recall": mean_of(reps, "fake_recall"),
                                 "Specificity": mean_of(reps, "specificity")})
        with st.expander("All final-evaluation numbers", expanded=False):
            st.dataframe(pd.DataFrame(rows).round(3), width="stretch", hide_index=True)
        note("Each model's threshold was chosen on its own validation data and frozen before testing. Thresholds "
             "transfer poorly between datasets (on DFD the served model catches 65% of fakes at its threshold), so AUC "
             "is the most comparable number across datasets.")

# ============================================================
# ABOUT
# ============================================================
with tab_about:
    a1, a2 = st.columns(2, gap="medium")
    vit_card = ("ViT frame classifier", "Whole frames, no face crop. Trained on a random frame split, so its reported 88% "
                "frame accuracy is not a held-out estimate.",
                '<span class="step">Upload</span><span class="arrow">&rarr;</span><span class="step">~2 frames/s</span>'
                '<span class="arrow">&rarr;</span><span class="step">ViT-B/16 per frame</span><span class="arrow">&rarr;</span>'
                '<span class="step">Majority vote</span>')
    ft_card = ("Fine-tuned ResNet50 + ViT", "Trained on FF++ and Celeb-DF v2: 0.874 AUC on the FF++ test split and 0.956 on "
               "the Celeb-DF v2 test list (unseen videos of a dataset it trained on). Every ~10 s segment of a video is scored "
               "and shown; the verdict uses the middle segment, the reading it was evaluated on.",
               '<span class="step">YuNet face track</span><span class="arrow">&rarr;</span><span class="step">32 frames</span>'
               '<span class="arrow">&rarr;</span><span class="step">ResNet50 + ViT</span><span class="arrow">&rarr;</span>'
               '<span class="step">8-frame windows</span><span class="arrow">&rarr;</span><span class="step">Video score</span>')
    served, other = (ft_card, vit_card) if FINETUNED else (vit_card, ft_card)
    with a1:
        html(f"""<div class="card">
          <div class="card-title">{icon("server", 18)}Served: {served[0]}<span class="badge-tag">Live</span></div>
          <p class="card-sub">Behind <code>/predict</code> on the Flask backend.</p>
          <div class="flow">{served[2]}</div>
          <p class="card-sub" style="margin-top:10px">{served[1]}</p></div>""")
    with a2:
        html(f"""<div class="card">
          <div class="card-title">{icon("layers", 18)}Also available: {other[0]}<span class="badge-tag">Off</span></div>
          <p class="card-sub">Switch with <code>SERVE_MODEL={'vit' if FINETUNED else 'finetuned'}</code> in <code>.env</code>
          and restart the backend.</p>
          <div class="flow">{other[2]}</div>
          <p class="card-sub" style="margin-top:10px">{other[1]}</p></div>""")
    section("What the research found", "details in the README and docs/", "info")
    html(f"""<div class="card"><ul class="findings">
      <li>Fine-tuning the last backbone layers is the change that matters: 0.853 test AUC vs 0.752 with frozen features.</li>
      <li>The full hybrid (FFT branch + BiLSTM) is no better than mean-pooled ResNet50 + ViT; the BiLSTM ignores frame order.</li>
      <li>Face cropping is worth about +0.17 AUC over whole frames.</li>
      <li>Unseen manipulation methods cost 0.07&ndash;0.14 AUC; raw scores are miscalibrated until Platt-scaled.</li>
    </ul></div>""")

html(f'<div style="text-align:center;color:{C["faint"]};font-size:0.78rem;margin-top:28px">'
     f'DeepShield-MF · analysis results are decision support, not proof of manipulation</div>')
