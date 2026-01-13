import streamlit as st

st.set_page_config(
    page_title="Bayut Competitor Multi-Agent System",
    layout="wide"
)

st.title("Bayut Competitor Multi-Agent System")
st.caption("SEO & editorial analysis against the market standard")

st.divider()

# ==================================================
# SESSION STATE INIT
# ==================================================
if "competitor_urls" not in st.session_state:
    st.session_state.competitor_urls = []

# ==================================================
# MODE SELECTION
# ==================================================
st.subheader("Select tool")

mode = st.radio(
    "What do you want to do?",
    [
        "Update existing Bayut article",
        "Plan a new article (title only)"
    ]
)

st.divider()

# ==================================================
# MODE-SPECIFIC INPUTS
# ==================================================
if mode == "Update existing Bayut article":
    st.subheader("Update published article")

    bayut_url = st.text_input(
        "Bayut article URL",
        placeholder="https://www.bayut.com/mybayut/..."
    )

else:
    st.subheader("Plan new article from title")

    article_title = st.text_input(
        "Article title",
        placeholder="Pros and Cons of Living in Business Bay"
    )

st.divider()

# ==================================================
# COMPETITOR INPUT (DYNAMIC)
# ==================================================
st.subheader("Competitors")

new_competitor = st.text_input(
    "Add competitor URL",
    placeholder="https://example.com/blog/..."
)

col1, col2 = st.columns([1, 5])

with col1:
    if st.button("➕ Add"):
        if new_competitor:
            st.session_state.competitor_urls.append(new_competitor)

with col2:
    if st.button("🧹 Clear all"):
        st.session_state.competitor_urls = []

# Show current competitors
if st.session_state.competitor_urls:
    st.markdown("**Current competitors:**")
    for i, url in enumerate(st.session_state.competitor_urls, start=1):
        st.write(f"{i}. {url}")
else:
    st.info("No competitors added yet")

st.divider()

# ==================================================
# DEBUG OUTPUT (TEMPORARY)
# ==================================================
if st.button("Run analysis"):
    st.subheader("Debug – Current inputs")

    st.write("Mode:", mode)

    if mode == "Update existing Bayut article":
        st.write("Bayut URL:")
        st.code(bayut_url or "❌ Not provided")
    else:
        st.write("Article title:")
        st.code(article_title or "❌ Not provided")

    st.write("Competitors:")
    if st.session_state.competitor_urls:
        for u in st.session_state.competitor_urls:
            st.code(u)
    else:
        st.write("❌ No competitors provided")
