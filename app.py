import streamlit as st

# ==================================================
# PAGE CONFIG
# ==================================================
st.set_page_config(
    page_title="Bayut AI Competitor Analysis",
    layout="wide"
)

# ==================================================
# GLOBAL STYLES
# ==================================================
st.markdown(
    """
    <style>
    body {
        background-color: #ffffff;
    }

    .center {
        text-align: center;
    }

    .subtitle {
        color: #6b7280;
        font-size: 16px;
        margin-top: -10px;
        margin-bottom: 30px;
    }

    .mode-btn button {
        width: 100%;
        height: 56px;
        font-size: 16px;
        font-weight: 600;
        border-radius: 999px;
        border: 1px solid #e5e7eb;
        background-color: #ffffff;
    }

    .mode-btn button:hover {
        background-color: #f9fafb;
        border-color: #d1d5db;
    }

    .section {
        max-width: 720px;
        margin: auto;
        margin-top: 30px;
    }

    .stTextInput > div > div > input {
        border-radius: 12px;
        height: 48px;
    }

    .add-btn button {
        border-radius: 999px;
        height: 44px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ==================================================
# SESSION STATE
# ==================================================
if "mode" not in st.session_state:
    st.session_state.mode = None

if "competitor_urls" not in st.session_state:
    st.session_state.competitor_urls = []

# ==================================================
# HEADER
# ==================================================
st.markdown("<h1 class='center'>Bayut AI Competitor Analysis</h1>", unsafe_allow_html=True)
st.markdown(
    "<p class='center subtitle'>SEO & editorial analysis against the market standard</p>",
    unsafe_allow_html=True
)

# ==================================================
# MODE SELECTION (CENTERED)
# ==================================================
st.markdown("<h3 class='center'>Choose your mode</h3>", unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns([1, 2, 2, 1])

with col2:
    st.markdown("<div class='mode-btn'>", unsafe_allow_html=True)
    if st.button("✏️ Update Mode"):
        st.session_state.mode = "update"
    st.markdown("</div>", unsafe_allow_html=True)

with col3:
    st.markdown("<div class='mode-btn'>", unsafe_allow_html=True)
    if st.button("🆕 New Post Mode"):
        st.session_state.mode = "new"
    st.markdown("</div>", unsafe_allow_html=True)

# ==================================================
# MODE CONTENT
# ==================================================
if st.session_state.mode is None:
    st.stop()

st.markdown("<div class='section'>", unsafe_allow_html=True)

if st.session_state.mode == "update":
    st.markdown("### Update existing Bayut article")
    bayut_url = st.text_input(
        "Bayut article URL",
        placeholder="https://www.bayut.com/mybayut/..."
    )
else:
    st.markdown("### Plan a new article")
    article_title = st.text_input(
        "Article title",
        placeholder="Pros and Cons of Living in Business Bay"
    )

# ==================================================
# COMPETITORS
# ==================================================
st.markdown("### Competitors")

new_competitor = st.text_input(
    "Competitor URL",
    placeholder="https://example.com/blog/..."
)

col_a, col_b = st.columns([1, 3])

with col_a:
    st.markdown("<div class='add-btn'>", unsafe_allow_html=True)
    if st.button("➕ Add"):
        if new_competitor:
            st.session_state.competitor_urls.append(new_competitor)
    st.markdown("</div>", unsafe_allow_html=True)

with col_b:
    st.markdown("<div class='add-btn'>", unsafe_allow_html=True)
    if st.button("Clear"):
        st.session_state.competitor_urls = []
    st.markdown("</div>", unsafe_allow_html=True)

if st.session_state.competitor_urls:
    st.markdown("**Current competitors:**")
    for i, url in enumerate(st.session_state.competitor_urls, start=1):
        st.write(f"{i}. {url}")
else:
    st.markdown("<p style='color:#9ca3af;'>No competitors added yet</p>", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)

# ==================================================
# ACTION
# ==================================================
st.markdown("<div class='section center'>", unsafe_allow_html=True)
if st.button("Run analysis"):
    st.success("Design ready ✅ (logic comes next)")
st.markdown("</div>", unsafe_allow_html=True)
