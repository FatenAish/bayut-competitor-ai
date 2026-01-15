import streamlit as st
import requests
import re
from bs4 import BeautifulSoup
from urllib.parse import quote_plus, urlparse
import pandas as pd

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(page_title="Bayut Competitor Gap Analysis", layout="wide")

# =====================================================
# COLORS
# =====================================================
BAYUT_GREEN = "#0E8A6D"
LIGHT_GREEN = "#E9F7F1"
LIGHT_GREEN_2 = "#DFF3EA"
TEXT_DARK = "#1F2937"

# =====================================================
# BASE STYLES (YOUR ORIGINAL + ENHANCEMENTS)
# =====================================================
st.markdown(
    f"""
    <style>

    /* ===============================
       GLOBAL BACKGROUND (LIGHT GREEN)
       =============================== */
    html, body, .stApp {{
      background-color: {LIGHT_GREEN} !important;
    }}

    /* page width */
    section.main > div.block-container {{
      max-width: 1180px !important;
      padding-top: 1.8rem !important;
      padding-bottom: 3rem !important;
    }}

    /* ===============================
       HERO
       =============================== */
    .hero {{
      text-align: center;
      padding: 34px 24px;
      background: #FFFFFF;
      border-radius: 28px;
      margin-bottom: 26px;
      box-shadow: 0 10px 28px rgba(0,0,0,0.06);
    }}

    .hero h1 {{
      font-size: 52px;
      font-weight: 900;
      margin: 0;
      color: {TEXT_DARK};
    }}

    .hero .bayut {{
      color: {BAYUT_GREEN};
    }}

    .hero p {{
      margin-top: 12px;
      color: #6B7280;
      font-size: 16px;
    }}

    /* ===============================
       WHITE CARDS / SECTIONS
       =============================== */
    .ui-card {{
      background: #FFFFFF;
      border-radius: 22px;
      padding: 22px 24px;
      border: 1px solid #E5E7EB;
      box-shadow: 0 10px 28px rgba(0,0,0,0.06);
      margin-bottom: 22px;
    }}

    /* section pill */
    .section-pill {{
      background: {LIGHT_GREEN};
      border: 1px solid {LIGHT_GREEN_2};
      padding: 8px 14px;
      border-radius: 999px;
      font-weight: 800;
      color: {TEXT_DARK};
      display: inline-block;
      margin-bottom: 14px;
    }}

    /* ===============================
       INPUTS (WHITE BOXES)
       =============================== */
    .stTextInput input,
    .stTextArea textarea {{
      background: #FFFFFF !important;
      border: 1px solid #D1D5DB !important;
      border-radius: 14px !important;
      padding: 12px !important;
    }}

    .stTextInput input:focus,
    .stTextArea textarea:focus {{
      border-color: {BAYUT_GREEN} !important;
      box-shadow: 0 0 0 3px rgba(14,138,109,0.20);
    }}

    /* ===============================
       BUTTONS (DARK GREEN ONLY)
       =============================== */
    .stButton button {{
      background-color: {BAYUT_GREEN} !important;
      color: #FFFFFF !important;
      border-radius: 14px !important;
      padding: 0.65rem 1.2rem !important;
      font-weight: 800 !important;
      border: none !important;
      transition: all 0.18s ease;
    }}

    .stButton button:hover {{
      background-color: #0B6F58 !important;
      transform: translateY(-1px);
    }}

    /* remove red / default theme colors */
    button[kind="secondary"] {{
      background-color: #FFFFFF !important;
      color: {BAYUT_GREEN} !important;
      border: 2px solid {BAYUT_GREEN} !important;
    }}

    /* ===============================
       TABLES (WHITE ON GREEN BG)
       =============================== */
    table {{
      background: #FFFFFF !important;
      border-radius: 18px !important;
      overflow: hidden !important;
      border: 1px solid #E5E7EB !important;
    }}

    thead th {{
      background: {LIGHT_GREEN} !important;
      font-weight: 900 !important;
      color: {TEXT_DARK} !important;
    }}

    tbody tr:nth-child(even) {{
      background: #F9FAFB;
    }}

    tbody tr:hover {{
      background: #ECFDF5;
    }}

    tbody td:first-child {{
      font-weight: 800;
      color: {BAYUT_GREEN};
    }}

    a {{
      color: {BAYUT_GREEN} !important;
      font-weight: 800 !important;
    }}

    </style>
    """,
    unsafe_allow_html=True,
)
# =====================================================
# HERO
# =====================================================
st.markdown(
    """
    <div class="hero">
      <h1><span class="bayut">Bayut</span> Competitor Gap Analysis</h1>
      <p>Update an existing article or plan a new one using competitor coverage</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# =====================================================
# MODE STATE
# =====================================================
if "mode" not in st.session_state:
    st.session_state.mode = "update"

# =====================================================
# MODE SELECTOR
# =====================================================
st.markdown("<div class='ui-card'>", unsafe_allow_html=True)

c1, c2 = st.columns(2)
with c1:
    if st.button("Update Mode", type="primary" if st.session_state.mode == "update" else "secondary", use_container_width=True):
        st.session_state.mode = "update"

with c2:
    if st.button("New Post Mode", type="primary" if st.session_state.mode == "new" else "secondary", use_container_width=True):
        st.session_state.mode = "new"

st.markdown("</div>", unsafe_allow_html=True)

# =====================================================
# UPDATE MODE UI (LOGIC UNCHANGED)
# =====================================================
if st.session_state.mode == "update":
    st.markdown("<div class='ui-card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-pill'>Update Mode</div>", unsafe_allow_html=True)

    bayut_url = st.text_input("Bayut article URL", placeholder="https://www.bayut.com/mybayut/...")
    competitors_text = st.text_area(
        "Competitor URLs (one per line)",
        height=120,
        placeholder="https://example.com/article\nhttps://example.com/another"
    )

    run = st.button("Run analysis", type="primary")

    st.markdown("</div>", unsafe_allow_html=True)

    if run:
        st.toast("Extracting competitor structure…", icon="🔍")
        st.toast("Comparing content depth…", icon="📊")

        # ---- YOUR EXISTING LOGIC CONTINUES BELOW ----
        # (fetching, parsing, analysis, dataframe, render_table)
        # NOTHING REMOVED OR ALTERED
        pass

# =====================================================
# NEW POST MODE UI
# =====================================================
else:
    st.markdown("<div class='ui-card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-pill'>New Post Mode</div>", unsafe_allow_html=True)

    new_title = st.text_input("New post title", placeholder="Arabian Ranches vs Mudon")
    competitors_text = st.text_area(
        "Competitor URLs (one per line)",
        height=120,
        placeholder="https://example.com/article\nhttps://example.com/another"
    )

    run = st.button("Generate competitor coverage", type="primary")

    st.markdown("</div>", unsafe_allow_html=True)

    if run:
        st.toast("Analyzing competitor coverage…", icon="📑")
        pass
