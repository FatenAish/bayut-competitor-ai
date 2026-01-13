import streamlit as st
import re
import requests
from bs4 import BeautifulSoup

# ==================================================
# PAGE CONFIG
# ==================================================
st.set_page_config(
    page_title="Bayut AI Competitor Analysis",
    layout="wide"
)

# ==================================================
# SESSION STATE
# ==================================================
if "mode" not in st.session_state:
    st.session_state.mode = None

if "competitor_urls" not in st.session_state:
    st.session_state.competitor_urls = []

# ==================================================
# INGESTION HELPERS
# ==================================================
IGNORE_TAGS = {"nav", "footer", "header", "aside", "form", "noscript", "script", "style"}

def _clean_text(s: str) -> str:
    s = re.sub(r"\s+", " ", (s or "")).strip()
    return s

def extract_main_text(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")

    for tag in soup.find_all(list(IGNORE_TAGS)):
        tag.decompose()

    article = soup.find("article")
    if article:
        text = article.get_text(" ")
    else:
        text = soup.get_text(" ")

    return _clean_text(text)

@st.cache_data(show_spinner=False, ttl=60 * 60)
def fetch_html(url: str, timeout: int = 25) -> str:
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0 Safari/537.36"
        )
    }
    r = requests.get(url, headers=headers, timeout=timeout)
    r.raise_for_status()
    return r.text

def ingest_url(url: str) -> dict:
    html = fetch_html(url)
    text = extract_main_text(html)
    return {
        "url": url,
        "text": text,
        "text_len": len(text),
        "preview": text[:700] + ("..." if len(text) > 700 else "")
    }

def normalize_competitors(urls: list[str], max_n: int = 5) -> list[str]:
    clean = []
    seen = set()
    for u in urls:
        u = (u or "").strip()
        if not u or u in seen:
            continue
        seen.add(u)
        clean.append(u)
    return clean[:max_n]

# ==================================================
# HEADER
# ==================================================
st.markdown("<h1 style='text-align:center;'>Bayut AI Competitor Analysis</h1>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align:center;color:#6b7280;'>SEO & editorial analysis against the market standard</p>",
    unsafe_allow_html=True
)

# ==================================================
# MODE SELECTION
# ==================================================
st.markdown("<h3 style='text-align:center;'>Choose your mode</h3>", unsafe_allow_html=True)

col1, col2, col3 = st.columns([3, 2, 2])

with col2:
    if st.button("✏️ Update Mode"):
        st.session_state.mode = "update"

with col3:
    if st.button("🆕 New Post Mode"):
        st.session_state.mode = "new"

if st.session_state.mode is None:
    st.stop()

# ==================================================
# MODE INPUTS
# ==================================================
st.markdown("---")

if st.session_state.mode == "update":
    st.subheader("Update existing Bayut article")
    bayut_url = st.text_input(
        "Bayut article URL",
        placeholder="https://www.bayut.com/mybayut/..."
    )
else:
    st.subheader("Plan a new article")
    article_title = st.text_input(
        "Article title",
        placeholder="Pros and Cons of Living in Business Bay"
    )

# ==================================================
# COMPETITORS
# ==================================================
st.subheader("Competitors")

new_competitor = st.text_input(
    "Add competitor URL",
    placeholder="https://example.com/blog/..."
)

col_a, col_b = st.columns([1, 3])

with col_a:
    if st.button("➕ Add competitor"):
        if new_competitor:
            st.session_state.competitor_urls.append(new_competitor)

with col_b:
    if st.button("🧹 Clear competitors"):
        st.session_state.competitor_urls = []

if st.session_state.competitor_urls:
    for i, url in enumerate(st.session_state.competitor_urls, start=1):
        st.write(f"{i}. {url}")
else:
    st.info("No competitors added yet")

# ==================================================
# RUN ANALYSIS — INGESTION
# ==================================================
st.markdown("---")

if st.button("Run analysis"):
    competitor_urls = normalize_competitors(st.session_state.competitor_urls, max_n=5)

    if st.session_state.mode == "update":
        if not bayut_url or not bayut_url.strip():
            st.error("Please provide the Bayut article URL.")
            st.stop()

    if not competitor_urls:
        st.error("Please add at least one competitor (max 5).")
        st.stop()

    results = {"mode": st.session_state.mode}

    with st.spinner("Fetching & cleaning content..."):
        # Bayut (update mode)
        if st.session_state.mode == "update":
            try:
                results["bayut"] = ingest_url(bayut_url.strip())
            except Exception as e:
                st.error(f"Failed to fetch Bayut article: {e}")
                st.stop()
        else:
            results["title"] = (article_title or "").strip()

        # Competitors
        comps = []
        for i, url in enumerate(competitor_urls, start=1):
            try:
                item = ingest_url(url)
                item["id"] = f"c{i}"
                comps.append(item)
            except Exception as e:
                comps.append({
                    "id": f"c{i}",
                    "url": url,
                    "error": str(e),
                    "text_len": 0,
                    "preview": ""
                })

        results["competitors"] = comps

    # ==================================================
    # OUTPUT — DEBUG VIEW
    # ==================================================
    st.success("Ingestion complete ✅")

    if results["mode"] == "update":
        st.subheader("Bayut article (cleaned)")
        st.write("Text length:", results["bayut"]["text_len"])
        st.text_area("Preview", results["bayut"]["preview"], height=200)
    else:
        st.subheader("New post title")
        st.write(results.get("title") or "❌ No title provided")

    st.subheader("Competitors (cleaned)")
    for c in results["competitors"]:
        st.markdown(f"**{c['id']}** — {c['url']}")
        if c.get("error"):
            st.error(c["error"])
        else:
            st.write("Text length:", c["text_len"])
            st.text_area(f"Preview ({c['id']})", c["preview"], height=180)
