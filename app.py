import streamlit as st

st.set_page_config(
    page_title="Bayut Competitor Multi-Agent System",
    layout="wide"
)

st.title("Bayut Competitor Multi-Agent System")
st.caption("SEO & editorial analysis against the market standard")

st.divider()

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
# MODE 1: UPDATE EXISTING ARTICLE
# ==================================================

if mode == "Update existing Bayut article":
    st.subheader("Update published article")

    bayut_url = st.text_input(
        "Bayut article URL",
        placeholder="https://www.bayut.com/mybayut/..."
    )

    st.markdown("### Competitors (up to 5)")
    competitor_urls = []

    for i in range(5):
        url = st.text_input(
            f"Competitor URL {i + 1}",
            placeholder="https://example.com/blog/..."
        )
        if url:
            competitor_urls.append(url)

# ==================================================
# MODE 2: PLAN NEW ARTICLE
# ==================================================

else:
    st.subheader("Plan new article from title")

    article_title = st.text_input(
        "Article title",
        placeholder="Pros and Cons of Living in Business Bay"
    )

    st.markdown("### Competitors (up to 5)")
    competitor_urls = []

    for i in range(5):
        url = st.text_input(
            f"Competitor URL {i + 1}",
            placeholder="https://example.com/blog/..."
        )
        if url:
            competitor_urls.append(url)

st.divider()

# ==================================================
# DEBUG OUTPUT (TEMPORARY)
# ==================================================

if st.button("Run analysis"):
    st.subheader("Debug – Current mode & inputs")

    st.write("Mode:", mode)

    if mode == "Update existing Bayut article":
        st.write("Bayut URL:")
        st.code(bayut_url or "❌ Not provided")
    else:
        st.write("Article title:")
        st.code(article_title or "❌ Not provided")

    st.write("Competitor URLs:")
    if competitor_urls:
        for u in competitor_urls:
            st.code(u)
    else:
        st.write("❌ No competitors provided")
