import streamlit as st
import requests
import re
from bs4 import BeautifulSoup

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="Bayut Dynamic Content Gap Analyzer", layout="wide")

# =========================
# INGESTION (SAFE)
# =========================
IGNORE = {"nav","footer","header","aside","script","style"}

def clean(text):
    return re.sub(r"\s+", " ", text or "").strip()

def fetch_text_safe(url):
    try:
        r = requests.get(
            url,
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=20
        )

        if r.status_code != 200:
            return {
                "url": url,
                "ok": False,
                "error": f"HTTP {r.status_code}",
                "text": ""
            }

        soup = BeautifulSoup(r.text, "lxml")
        for t in soup.find_all(list(IGNORE)):
            t.decompose()

        article = soup.find("article")
        text = article.get_text(" ") if article else soup.get_text(" ")
        text = clean(text)

        if len(text) < 300:
            return {
                "url": url,
                "ok": False,
                "error": "Content too short / blocked",
                "text": ""
            }

        return {
            "url": url,
            "ok": True,
            "error": None,
            "text": text
        }

    except Exception as e:
        return {
            "url": url,
            "ok": False,
            "error": str(e),
            "text": ""
        }

def split_sentences(text):
    return [
        s.strip()
        for s in re.split(r"[.!?]", text)
        if len(s.strip()) > 40
    ]

# =========================
# SIMPLE SEMANTIC MATCHING
# =========================
def normalize(s):
    s = s.lower()
    s = re.sub(r"[^a-z0-9\s]", "", s)
    return s

def similarity(a, b):
    a_set = set(a.split())
    b_set = set(b.split())
    if not a_set or not b_set:
        return 0
    return len(a_set & b_set) / max(len(a_set), len(b_set))

# =========================
# UI
# =========================
st.title("Bayut Dynamic Content Gap Analysis")
st.caption("Competitor-driven content gaps (robust & editor-ready)")

bayut_url = st.text_input("Bayut article URL")

competitors = st.text_area(
    "Competitor URLs (one per line)",
    placeholder="https://example.com/article"
).splitlines()

if st.button("Run content gap analysis"):
    if not bayut_url or not any(c.strip() for c in competitors):
        st.error("Bayut URL and at least one competitor are required.")
        st.stop()

    with st.spinner("Fetching content safely..."):
        bayut = fetch_text_safe(bayut_url)

        if not bayut["ok"]:
            st.error(f"Bayut URL failed: {bayut['error']}")
            st.stop()

        bayut_sentences = [
            normalize(s)
            for s in split_sentences(bayut["text"])
        ]

        competitor_results = []
        for url in competitors:
            if not url.strip():
                continue
            competitor_results.append(fetch_text_safe(url))

    # Show fetch status
    st.subheader("Fetch status")
    for c in competitor_results:
        if c["ok"]:
            st.success(f"✔ {c['url']}")
        else:
            st.warning(f"⚠ {c['url']} — {c['error']}")

    valid_competitors = [c for c in competitor_results if c["ok"]]

    if not valid_competitors:
        st.error("No competitor content could be fetched. Try different URLs.")
        st.stop()

    # =========================
    # GAP ANALYSIS
    # =========================
    with st.spinner("Detecting competitor-driven gaps..."):
        competitor_sentences = []
        for c in valid_competitors:
            competitor_sentences.extend(
                split_sentences(c["text"])
            )

        normalized_comp = [normalize(s) for s in competitor_sentences]

        clusters = []
        for s in normalized_comp:
            placed = False
            for c in clusters:
                if similarity(s, c[0]) > 0.55:
                    c.append(s)
                    placed = True
                    break
            if not placed:
                clusters.append([s])

        rows = []
        for cluster in clusters:
            idea = cluster[0]

            found = False
            weak = False

            for b in bayut_sentences:
                sim = similarity(idea, b)
                if sim > 0.65:
                    found = True
                    break
                elif sim > 0.35:
                    weak = True

            if not found:
                rows.append({
                    "Competitor Topic": idea[:120] + "...",
                    "Bayut Coverage": "⚠️ Weak" if weak else "❌ Missing",
                    "Gap Type": "Expand" if weak else "Add",
                    "What to Add": "Add a clear paragraph addressing this point"
                })

    # =========================
    # OUTPUT
    # =========================
    if not rows:
        st.success("No competitor-driven content gaps detected.")
    else:
        st.subheader("Actionable Content Gaps")
        st.dataframe(rows, use_container_width=True)
