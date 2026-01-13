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
    .center {
        text-align: center;
    }
    .mode-btn button {
        width: 100%;
        height: 60px;
        font-size: 18px;
        font-weight: 600;
        border-radius: 14px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ==================================================
# HEADER
# ==================================================
st.markdown("<h1 class='center'>Bayut AI Competitor Analysis</h1>", unsafe_allow_html=True)
st.markdown(
    "<p class='center'>SEO & editorial analysis against the market standard</p>",
    unsafe_allow_html=True
)

st.divider()

# ==================================================
# SESSION STATE
# ==================================================
if "mode" not in st.session_state:
    st.session_state.mode = None

if "competitor_urls" not in st.session_state:
    st.session_state.competitor_urls = []

# ==================================================
# MODE SELECTION (CENTERED BUTTONS)
# ==================================================
st.markdown("<h3 class='center'>Choose your mode</h3>", unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns([1, 2, 2, 1])

with col2:
    st.markdown("<div class='mode-btn'>", unsafe_allow_html=True)
    if st.button("🛠 Update Mode"):
        st.session_state.mode = "update"
    st.markdown("</div>", unsafe_allow_html=True)

with col3:
    st.markdown("<div class='mode-btn'>", unsafe_allow_html=True)
    if st.button("🆕 New Post Mode"):
        st.session_state.mode = "new"
    st.markdown("</div>", unsafe_allow_html=True)

st.divider()

# ==================================================
# MODE-SPECIFIC INPUTS
# ==================================================
if st.session_state.mode == "update":
    st.subheader("Update existing Bayut article")

    bayut_url = st.text_input(
        "Bayut article URL",
        placeholder="https://www.bayut.com/mybayut/..."
    )

elif st.session_state.mode == "new":
    st.subheader("Plan new article from title")

    article_title = st.text_input(
        "Article title",
        placeholder="Pros and Cons of Living in Business Bay"
    )

else:
    st.info("Please choose a mode to continue.")
    st.stop()

# ==================================================
# COMPETITOR INPUT
# ==================================================
st.divider()
st.subheader("Competitors")

new_competitor = st.text_input(
    "Add competitor URL",
    placeholder="https://example.com/blog/..."
)

col_a, col_b = st.columns([1, 4])

with col_a:
    if st.button("➕ Add competitor"):
        if new_competitor:
            st.session_state.competitor_urls.append(new_competitor)

with col_b:
    if st.button("🧹 Clear competitors"):
        st.session_state.competitor_urls = []

if st.session_state.competitor_urls:
    st.markdown("**Current competitors:**")
    for i, url in enumerate(st.session_state.competitor_urls, start=1):
        st.write(f"{i}. {url}")
else:
    st.info("No competitors added yet")

# ==================================================
# DEBUG (TEMP)
# ==================================================
st.divider()
if st.button("Run analysis"):
    st.subheader("Debug – Inputs")

    st.write("Mode:", st.session_state.mode)

    if st.session_state.mode == "update":
        st.write("Bayut URL:")
        st.code(bayut_url or "❌ Not provided")
    else:
        st.write("Article title:")
        st.code(article_title or "❌ Not provided")

    st.write("Competitors:")
    for u in st.session_state.competitor_urls:
        st.code(u)
