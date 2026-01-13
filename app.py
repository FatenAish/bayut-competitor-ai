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
    return re.sub(r"\s+", " ", (s or "")).strip()

def extract_main_text(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")

    for tag in soup.find_all(list(IGNORE_TAGS)):
        tag.decompose()

    article = soup.find("article")
    text = article.get_text(" ") if article else soup.get_text(" ")
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
    clean, seen = [], set()
    for u in urls:
        u = (u or "").strip()
        if not u or u in seen:
            continue
        seen.add(u)
        clean.append(u)
    return clean[:max_n]

# ==================================================
# SEO COMPLIANCE AGENT
# ==================================================
def seo_compliance_check(text: str, title: str | None = None) -> dict:
    text_lower = text.lower()
    checks = {}

    # Rule 1: Minimum length
    checks["min_length"] = {
        "passed": len(text) >= 1200,
        "value": len(text),
        "required": 1200
    }

    # Rule 2: Headings / structure signals
    checks["has_sections"] = {
        "passed": any(k in text_lower for k in [
            "pros", "cons", "overview", "price", "prices",
            "cost", "transport", "metro", "family"
        ])
    }

    # Rule 3: Keyword coverage (from title)
    if title:
        terms = [w.lower() for w in title.split() if len(w) > 3]
        hits = sum(1 for w in terms if w in text_lower)
        checks["keyword_coverage"] = {
            "passed": hits >= max(2, len(terms) // 3),
            "hits": hits,
            "terms": terms
        }

    # Rule 4: Lists / scannability
    checks["uses_lists"] = {
        "passed": any(sym in text for sym in ["•", "-", "–"])
    }

    # Rule 5: Question answering
    checks["answers_questions"] = {
        "passed": "?" in text
    }

    passed = sum(1 for c in checks.values() if c["passed"])
    total = len(checks)
    score = round((passed / total) * 10, 1)

    return {
        "score": score,
        "passed": passed,
        "total": total,
        "checks": checks
    }

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
# RUN ANALYSIS
# ==================================================
st.markdown("---")

if st.button("Run analysis"):
    competitor_urls = normalize_competitors(st.session_state.competitor_urls)

    if st.session_state.mode == "update":
        if not bayut_url or not bayut_url.strip():
            st.error("Please provide the Bayut article URL.")
            st.stop()

    if not competitor_urls:
        st.error("Please add at least one competitor (max 5).")
        st.stop()

    results = {"mode": st.session_state.mode}

    with st.spinner("Fetching & cleaning content..."):
        if st.session_state.mode == "update":
            results["bayut"] = ingest_url(bayut_url.strip())
        else:
            results["title"] = (article_title or "").strip()

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
    # OUTPUT — INGESTION
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

    # ==================================================
    # OUTPUT — SEO COMPLIANCE
    # ==================================================
    st.markdown("---")
    st.subheader("SEO Compliance Results")

    if results["mode"] == "update":
        bayut_seo = seo_compliance_check(results["bayut"]["text"])
        st.markdown("### Bayut SEO Score")
        st.metric("Score (out of 10)", bayut_seo["score"])
        with st.expander("Bayut SEO checks"):
            st.json(bayut_seo["checks"])

    st.markdown("### Competitor SEO Scores")
    for c in results["competitors"]:
        if c.get("error"):
            continue
        comp_seo = seo_compliance_check(
            c["text"],
            title=results.get("title")
        )
        st.markdown(f"**{c['id']}** — {c['url']}")
        st.metric("Score (out of 10)", comp_seo["score"])
