import streamlit as st
import requests
import re
from bs4 import BeautifulSoup
from collections import Counter

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="Bayut Content Gap Analyzer", layout="wide")

# =========================
# INGESTION
# =========================
IGNORE = {"nav","footer","header","aside","script","style"}

def clean(t):
    return re.sub(r"\s+", " ", t or "").strip()

def fetch_text(url):
    r = requests.get(url, headers={"User-Agent":"Mozilla/5.0"}, timeout=20)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "lxml")
    for t in soup.find_all(list(IGNORE)):
        t.decompose()
    article = soup.find("article")
    text = article.get_text(" ") if article else soup.get_text(" ")
    return clean(text)

def split_sentences(text):
    return [s.strip() for s in re.split(r"[.!?]", text) if len(s.strip()) > 40]

# =========================
# SEMANTIC NORMALIZATION
# =========================
def normalize_sentence(s):
    s = s.lower()
    s = re.sub(r"[^a-z0-9\s]", "", s)
    return s

def similar(a, b):
    a_set = set(a.split())
    b_set = set(b.split())
    return len(a_set & b_set) / max(len(a_set), 1)

# =========================
# UI
# =========================
st.title("Bayut Dynamic Content Gap Analysis")
st.caption("Competitor-driven content gaps (no fixed checklist)")

bayut_url = st.text_input("Bayut article URL")

competitors = st.text_area(
    "Competitor URLs (one per line)",
    placeholder="https://example.com/article"
).splitlines()

if st.button("Run content gap analysis"):
    if not bayut_url or len([c for c in competitors if c.strip()]) == 0:
        st.error("Bayut URL and at least one competitor are required.")
        st.stop()

    with st.spinner("Analyzing competitor-driven content gaps..."):
        bayut_text = fetch_text(bayut_url)
        bayut_sents = [normalize_sentence(s) for s in split_sentences(bayut_text)]

        competitor_sentences = []

        for url in competitors:
            if not url.strip():
                continue
            text = fetch_text(url)
            sents = split_sentences(text)
            competitor_sentences.extend(sents)

        # Normalize competitor ideas
        normalized = [normalize_sentence(s) for s in competitor_sentences]

        # Cluster similar competitor ideas
        clusters = []
        for s in normalized:
            placed = False
            for c in clusters:
                if similar(s, c[0]) > 0.5:
                    c.append(s)
                    placed = True
                    break
            if not placed:
                clusters.append([s])

        rows = []

        for cluster in clusters:
            idea = cluster[0]
            covered = False
            weak = False

            for b in bayut_sents:
                sim = similar(idea, b)
                if sim > 0.6:
                    covered = True
                    break
                elif sim > 0.3:
                    weak = True

            if not covered:
                rows.append({
                    "Competitor Topic": idea[:120] + "...",
                    "Found in Bayut": "⚠️ Weak" if weak else "❌ No",
                    "Gap Type": "Expand" if weak else "Missing",
                    "What Bayut Should Add": "Add a clear paragraph covering this point"
                })

    if not rows:
        st.success("No competitor-driven content gaps detected.")
    else:
        st.subheader("Actionable Content Gaps")
        st.dataframe(rows, use_container_width=True)
