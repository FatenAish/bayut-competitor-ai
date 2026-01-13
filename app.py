import streamlit as st
import requests
import re
from bs4 import BeautifulSoup

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="Bayut Content Gap Analyzer", layout="wide")

# =========================
# UTILITIES
# =========================
IGNORE = {"nav","footer","header","aside","script","style"}

def clean(text):
    return re.sub(r"\s+", " ", text or "").strip()

def extract_text(html):
    soup = BeautifulSoup(html, "lxml")
    for t in soup.find_all(list(IGNORE)):
        t.decompose()
    article = soup.find("article")
    return clean(article.get_text(" ") if article else soup.get_text(" "))

# =========================
# FETCH WITH FALLBACKS
# =========================
def fetch_with_fallback(url):
    headers = {"User-Agent": "Mozilla/5.0"}

    # 1️⃣ Normal fetch
    try:
        r = requests.get(url, headers=headers, timeout=15)
        if r.status_code == 200:
            text = extract_text(r.text)
            if len(text) > 500:
                return {"url": url, "ok": True, "source": "direct", "text": text}
    except:
        pass

    # 2️⃣ Textise fallback
    try:
        textise_url = f"https://textise.org/showtext.aspx?strURL={url}"
        r = requests.get(textise_url, headers=headers, timeout=15)
        if r.status_code == 200:
            soup = BeautifulSoup(r.text, "lxml")
            text = clean(soup.get_text(" "))
            if len(text) > 300:
                return {"url": url, "ok": True, "source": "textise", "text": text}
    except:
        pass

    # 3️⃣ Fail softly
    return {
        "url": url,
        "ok": False,
        "source": None,
        "text": ""
    }

# =========================
# GAP LOGIC (DYNAMIC)
# =========================
def split_sentences(text):
    return [s.strip() for s in re.split(r"[.!?]", text) if len(s.strip()) > 40]

def normalize(s):
    s = s.lower()
    return re.sub(r"[^a-z0-9\s]", "", s)

def similarity(a, b):
    a, b = set(a.split()), set(b.split())
    if not a or not b:
        return 0
    return len(a & b) / max(len(a), len(b))

# =========================
# UI
# =========================
st.title("Bayut Content Gap Analysis")
st.caption("Competitor-driven gaps • resilient ingestion • editor-ready")

bayut_url = st.text_input("Bayut article URL")

competitors = st.text_area(
    "Competitor URLs (one per line)",
    placeholder="https://example.com/article"
).splitlines()

if st.button("Run content gap analysis"):
    if not bayut_url or not any(c.strip() for c in competitors):
        st.error("Bayut URL and at least one competitor are required.")
        st.stop()

    with st.spinner("Fetching content (auto-resolving blocks)..."):
        bayut = fetch_with_fallback(bayut_url)

        if not bayut["ok"]:
            st.error("Bayut article could not be fetched.")
            st.stop()

        bayut_sents = [normalize(s) for s in split_sentences(bayut["text"])]

        comp_results = [fetch_with_fallback(c) for c in competitors if c.strip()]

    # Fetch status
    st.subheader("Fetch status")
    for c in comp_results:
        if c["ok"]:
            st.success(f"✔ {c['url']} ({c['source']})")
        else:
            st.warning(f"⚠ {c['url']} (blocked)")

    valid_comps = [c for c in comp_results if c["ok"]]

    if not valid_comps:
        st.error("No competitor content could be extracted.")
        st.stop()

    # =========================
    # GAP DETECTION
    # =========================
    with st.spinner("Detecting content gaps..."):
        comp_sentences = []
        for c in valid_comps:
            comp_sentences.extend(split_sentences(c["text"]))

        norm_comp = [normalize(s) for s in comp_sentences]

        clusters = []
        for s in norm_comp:
            for cluster in clusters:
                if similarity(s, cluster[0]) > 0.55:
                    cluster.append(s)
                    break
            else:
                clusters.append([s])

        rows = []
        for cluster in clusters:
            idea = cluster[0]
            found = False
            weak = False

            for b in bayut_sents:
                sim = similarity(idea, b)
                if sim > 0.65:
                    found = True
                    break
                elif sim > 0.35:
                    weak = True

            if not found:
                rows.append({
                    "Competitor Topic": idea[:140] + "...",
                    "Bayut Coverage": "⚠️ Weak" if weak else "❌ Missing",
                    "Gap Type": "Expand" if weak else "Add",
                    "What Bayut Should Add": "Add a clear paragraph covering this point"
                })

    # =========================
    # OUTPUT
    # =========================
    if rows:
        st.subheader("Actionable Content Gaps")
        st.dataframe(rows, use_container_width=True)
    else:
        st.success("No competitor-driven content gaps detected.")
