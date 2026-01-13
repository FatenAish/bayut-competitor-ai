import streamlit as st
import requests
import re
from bs4 import BeautifulSoup
from urllib.parse import quote_plus

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="Bayut Content Gap Analyzer", layout="wide")

# =========================
# TEXT HELPERS
# =========================
IGNORE = {"nav", "footer", "header", "aside", "script", "style", "noscript"}

def clean(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()

def extract_text_from_html(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    for t in soup.find_all(list(IGNORE)):
        t.decompose()
    article = soup.find("article")
    text = article.get_text(" ") if article else soup.get_text(" ")
    return clean(text)

def looks_blocked(text: str) -> bool:
    t = (text or "").lower()
    # common anti-bot / challenge pages
    return any(x in t for x in [
        "just a moment", "checking your browser", "verify you are human",
        "cloudflare", "access denied", "captcha"
    ])

# =========================
# FETCH FALLBACKS (ROBUST)
# =========================
DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Cache-Control": "no-cache",
    "Pragma": "no-cache",
}

@st.cache_data(show_spinner=False, ttl=60 * 30)
def fetch_url_raw(url: str, timeout: int = 20) -> tuple[int, str]:
    r = requests.get(url, headers=DEFAULT_HEADERS, timeout=timeout, allow_redirects=True)
    return r.status_code, r.text

@st.cache_data(show_spinner=False, ttl=60 * 30)
def fetch_jina(url: str, timeout: int = 20) -> tuple[int, str]:
    """
    r.jina.ai acts like a reader proxy and often succeeds where direct requests fail.
    Format: https://r.jina.ai/http(s)://example.com/page
    """
    if url.startswith("https://"):
        jina_url = "https://r.jina.ai/https://" + url[len("https://"):]
    elif url.startswith("http://"):
        jina_url = "https://r.jina.ai/http://" + url[len("http://"):]
    else:
        jina_url = "https://r.jina.ai/https://" + url

    r = requests.get(jina_url, headers=DEFAULT_HEADERS, timeout=timeout, allow_redirects=True)
    return r.status_code, r.text

def fetch_with_fallback(url: str) -> dict:
    """
    Returns:
      { url, ok, source, error, text }
    source in: direct | jina | textise | jina_textise
    """
    url = (url or "").strip()
    if not url:
        return {"url": url, "ok": False, "source": None, "error": "Empty URL", "text": ""}

    # ---- 1) DIRECT ----
    try:
        code, html = fetch_url_raw(url)
        if code == 200:
            text = extract_text_from_html(html)
            if len(text) > 600 and not looks_blocked(text):
                return {"url": url, "ok": True, "source": "direct", "error": None, "text": text}
    except Exception:
        pass

    # ---- 2) JINA READER ----
    try:
        code, content = fetch_jina(url)
        if code == 200:
            # Jina returns readable text-like content; just clean it.
            text = clean(content)
            if len(text) > 600 and not looks_blocked(text):
                return {"url": url, "ok": True, "source": "jina", "error": None, "text": text}
    except Exception:
        pass

    # ---- 3) TEXTISE (ENCODED) ----
    try:
        encoded = quote_plus(url)
        textise_url = f"https://textise.org/showtext.aspx?strURL={encoded}"
        code, html = fetch_url_raw(textise_url)
        if code == 200:
            text = clean(BeautifulSoup(html, "lxml").get_text(" "))
            if len(text) > 400 and not looks_blocked(text):
                return {"url": url, "ok": True, "source": "textise", "error": None, "text": text}
    except Exception:
        pass

    # ---- 4) JINA ON TEXTISE ----
    try:
        encoded = quote_plus(url)
        textise_url = f"https://textise.org/showtext.aspx?strURL={encoded}"
        code, content = fetch_jina(textise_url)
        if code == 200:
            text = clean(content)
            if len(text) > 400 and not looks_blocked(text):
                return {"url": url, "ok": True, "source": "jina_textise", "error": None, "text": text}
    except Exception:
        pass

    # Fail softly
    return {"url": url, "ok": False, "source": None, "error": "Blocked / unreachable", "text": ""}

# =========================
# GAP ENGINE (DYNAMIC)
# =========================
def split_sentences(text: str):
    return [s.strip() for s in re.split(r"[.!?]", text) if len(s.strip()) > 40]

def normalize(s: str) -> str:
    s = s.lower()
    return re.sub(r"[^a-z0-9\s]", "", s)

def similarity(a: str, b: str) -> float:
    a_set, b_set = set(a.split()), set(b.split())
    if not a_set or not b_set:
        return 0.0
    return len(a_set & b_set) / max(len(a_set), len(b_set))

# =========================
# UI
# =========================
st.title("Bayut Dynamic Content Gap Analysis")
st.caption("Competitor-driven gaps • automatic extraction fallbacks • table output")

bayut_url = st.text_input("Bayut article URL")

competitors_text = st.text_area(
    "Competitor URLs (one per line, up to 5 recommended)",
    placeholder="https://example.com/article\nhttps://example.com/another"
)
competitors = [c.strip() for c in competitors_text.splitlines() if c.strip()]

if st.button("Run content gap analysis"):
    if not bayut_url.strip():
        st.error("Bayut URL is required.")
        st.stop()
    if not competitors:
        st.error("At least one competitor URL is required.")
        st.stop()

    competitors = competitors[:5]  # keep it sane for performance

    with st.spinner("Fetching Bayut + competitors (auto-resolving blocks)..."):
        bayut = fetch_with_fallback(bayut_url.strip())
        if not bayut["ok"]:
            st.error(f"Bayut fetch failed: {bayut.get('error')}")
            st.stop()

        comp_results = [fetch_with_fallback(u) for u in competitors]

    st.subheader("Fetch status")
    for c in comp_results:
        if c["ok"]:
            st.success(f"✔ {c['url']}  — source: {c['source']}")
        else:
            st.warning(f"⚠ {c['url']}  — {c['error']}")

    valid_comps = [c for c in comp_results if c["ok"]]
    if not valid_comps:
        st.error("No competitor content could be extracted (all blocked).")
        st.info("If you need TRUE 100% coverage, the next step is adding Playwright or a scraping API (ZenRows/ScrapingBee).")
        st.stop()

    # Build Bayut sentence pool
    bayut_sents = [normalize(s) for s in split_sentences(bayut["text"])]

    # Build competitor idea pool
    all_comp_sents = []
    for c in valid_comps:
        all_comp_sents.extend(split_sentences(c["text"]))
    norm_comp = [normalize(s) for s in all_comp_sents]

    # Cluster competitor ideas (dedupe similar)
    clusters = []
    for s in norm_comp:
        placed = False
        for cluster in clusters:
            if similarity(s, cluster[0]) > 0.55:
                cluster.append(s)
                placed = True
                break
        if not placed:
            clusters.append([s])

    # Score clusters by frequency (how many times competitors repeat similar idea)
    clusters.sort(key=lambda cl: len(cl), reverse=True)

    rows = []
    for cluster in clusters[:120]:  # limit noise
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
                "Competitor idea (normalized)": idea[:160] + "...",
                "Bayut coverage": "⚠️ Weak" if weak else "❌ Missing",
                "Priority": "High" if len(cluster) >= 3 else "Med",
                "Evidence": f"Similar points found {len(cluster)} times across competitor text",
                "What to add": "Add a clear paragraph/section covering this point explicitly"
            })

    st.subheader("Actionable Content Gaps (competitor-driven)")
    if rows:
        st.dataframe(rows, use_container_width=True)
    else:
        st.success("No competitor-driven gaps detected.")
