import streamlit as st
import re
import requests
from bs4 import BeautifulSoup
from collections import Counter

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
# INGESTION AGENT
# ==================================================
IGNORE_TAGS = {"nav", "footer", "header", "aside", "form", "noscript", "script", "style"}

def clean_text(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()

def extract_main_text(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    for tag in soup.find_all(list(IGNORE_TAGS)):
        tag.decompose()
    article = soup.find("article")
    text = article.get_text(" ") if article else soup.get_text(" ")
    return clean_text(text)

@st.cache_data(show_spinner=False, ttl=3600)
def fetch_html(url: str) -> str:
    headers = {
        "User-Agent": "Mozilla/5.0 Chrome/120.0"
    }
    r = requests.get(url, headers=headers, timeout=25)
    r.raise_for_status()
    return r.text

def ingest_url(url: str) -> dict:
    html = fetch_html(url)
    text = extract_main_text(html)
    return {
        "url": url,
        "text": text,
        "length": len(text)
    }

def normalize_competitors(urls, max_n=5):
    seen, out = set(), []
    for u in urls:
        u = (u or "").strip()
        if u and u not in seen:
            seen.add(u)
            out.append(u)
    return out[:max_n]

# ==================================================
# INTENT & SECTION SIGNAL AGENT
# ==================================================
INTENT_KEYWORDS = {
    "overview": ["overview", "about", "introduction"],
    "pros": ["pros", "advantages", "benefits"],
    "cons": ["cons", "disadvantages", "drawbacks"],
    "pricing": ["price", "prices", "cost", "rent"],
    "transport": ["transport", "metro", "bus", "connectivity"],
    "family": ["family", "families", "schools", "nursery"],
    "lifestyle": ["lifestyle", "restaurants", "cafes", "shopping"]
}

def detect_intents(text: str) -> set:
    t = text.lower()
    intents = set()
    for intent, keywords in INTENT_KEYWORDS.items():
        if any(k in t for k in keywords):
            intents.add(intent)
    return intents

# ==================================================
# COMPETITOR NORMALIZER AGENT
# ==================================================
def build_market_baseline(competitor_intents: list[set]) -> dict:
    counter = Counter()
    for intents in competitor_intents:
        counter.update(intents)
    return dict(counter)

# ==================================================
# GAP DETECTION AGENT
# ==================================================
def detect_gaps(bayut_intents: set, market_baseline: dict, competitor_count: int):
    critical, optional = [], []

    for intent, freq in market_baseline.items():
        coverage_ratio = freq / competitor_count
        if intent not in bayut_intents:
            if coverage_ratio >= 0.6:
                critical.append((intent, freq))
            elif coverage_ratio >= 0.3:
                optional.append((intent, freq))

    return {
        "critical": critical,
        "optional": optional
    }

# ==================================================
# SEO COMPLIANCE AGENT
# ==================================================
def seo_compliance(text: str, title: str | None = None) -> dict:
    checks = {}

    checks["min_length"] = len(text) >= 1200
    checks["has_sections"] = any(k in text.lower() for k in [
        "pros", "cons", "price", "transport", "family"
    ])
    checks["uses_lists"] = any(sym in text for sym in ["•", "-", "–"])
    checks["answers_questions"] = "?" in text

    if title:
        terms = [w.lower() for w in title.split() if len(w) > 3]
        hits = sum(1 for w in terms if w in text.lower())
        checks["keyword_coverage"] = hits >= max(2, len(terms)//3)

    score = round((sum(checks.values()) / len(checks)) * 10, 1)

    return {"score": score, "checks": checks}

# ==================================================
# UI — HEADER
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

c1, c2, c3 = st.columns([3, 2, 2])
with c2:
    if st.button("✏️ Update Mode"):
        st.session_state.mode = "update"
with c3:
    if st.button("🆕 New Post Mode"):
        st.session_state.mode = "new"

if st.session_state.mode is None:
    st.stop()

# ==================================================
# MODE INPUTS
# ==================================================
st.markdown("---")

if st.session_state.mode == "update":
    bayut_url = st.text_input("Bayut article URL")
else:
    article_title = st.text_input("Article title")

# ==================================================
# COMPETITORS INPUT
# ==================================================
new_comp = st.text_input("Add competitor URL")

col_a, col_b = st.columns([1, 3])
with col_a:
    if st.button("Add competitor") and new_comp:
        st.session_state.competitor_urls.append(new_comp)
with col_b:
    if st.button("Clear competitors"):
        st.session_state.competitor_urls = []

for i, u in enumerate(st.session_state.competitor_urls, 1):
    st.write(f"{i}. {u}")

# ==================================================
# RUN ORCHESTRATOR
# ==================================================
if st.button("Run analysis"):
    competitors = normalize_competitors(st.session_state.competitor_urls)

    if not competitors:
        st.error("Add at least one competitor.")
        st.stop()

    with st.spinner("Running multi-agent analysis..."):
        comp_docs = [ingest_url(u) for u in competitors]
        comp_intents = [detect_intents(d["text"]) for d in comp_docs]

        market = build_market_baseline(comp_intents)

        if st.session_state.mode == "update":
            bayut = ingest_url(bayut_url)
            bayut_intents = detect_intents(bayut["text"])
            gaps = detect_gaps(bayut_intents, market, len(comp_docs))
            bayut_seo = seo_compliance(bayut["text"])

    # ==================================================
    # OUTPUT
    # ==================================================
    st.success("Analysis complete")

    st.subheader("Market standard (competitor coverage)")
    st.json(market)

    if st.session_state.mode == "update":
        st.subheader("Bayut gaps vs market")

        st.markdown("### 🔴 Critical gaps")
        for g in gaps["critical"]:
            st.write(f"- {g[0]} (covered by {g[1]} competitors)")

        st.markdown("### 🟡 Optional gaps")
        for g in gaps["optional"]:
            st.write(f"- {g[0]} (covered by {g[1]} competitors)")

        st.subheader("Bayut SEO score")
        st.metric("SEO score (out of 10)", bayut_seo["score"])
        with st.expander("SEO checks"):
            st.json(bayut_seo["checks"])

    st.subheader("Competitor SEO scores")
    for d in comp_docs:
        score = seo_compliance(d["text"], article_title if st.session_state.mode=="new" else None)
        st.markdown(f"**{d['url']}**")
        st.metric("Score", score["score"])
