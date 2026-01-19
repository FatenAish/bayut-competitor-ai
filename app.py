import re
import time
import json
import hashlib
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from urllib.parse import urlparse

import pandas as pd
import requests
import streamlit as st
from bs4 import BeautifulSoup

# Optional (recommended) JS rendering
try:
    from playwright.sync_api import sync_playwright
    PLAYWRIGHT_OK = True
except Exception:
    PLAYWRIGHT_OK = False


# =========================
# CONFIG
# =========================
st.set_page_config(page_title="Bayut Competitor Gap Analysis", layout="wide")

BAYUT_GREEN = "#0E8A6D"
DUBIZZLE_RED = "#D92C27"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120.0 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9",
}

IGNORE_TAGS = {"nav", "footer", "header", "aside", "form", "noscript", "script", "style"}
FAQ_HINTS = ("faq", "frequently asked", "people also ask", "common questions")

STOP = {
    "the","and","for","with","that","this","from","you","your","are","was","were","will","have","has","had",
    "but","not","can","may","more","most","into","than","then","they","them","their","our","out","about",
    "also","over","under","between","within","near","where","when","what","why","how","who","which",
    "a","an","to","of","in","on","at","as","is","it","be","or","by"
}


# =========================
# DATAFORSEO (NO SERPAPI)
# =========================
def _get_dataforseo_creds() -> Tuple[Optional[str], Optional[str]]:
    login = st.secrets.get("DATAFORSEO_LOGIN") if hasattr(st, "secrets") else None
    password = st.secrets.get("DATAFORSEO_PASSWORD") if hasattr(st, "secrets") else None
    return login, password


def dataforseo_post(path: str, payload: list) -> dict:
    """
    DataForSEO uses Basic Auth (login/password).
    """
    login, password = _get_dataforseo_creds()
    if not login or not password:
        return {"_error": "missing_dataforseo_credentials"}

    url = f"https://api.dataforseo.com{path}"
    try:
        r = requests.post(
            url,
            auth=(login, password),
            headers={"Content-Type": "application/json"},
            data=json.dumps(payload),
            timeout=60
        )
        return r.json()
    except Exception as e:
        return {"_error": f"dataforseo_request_failed: {e}"}


@st.cache_data(ttl=7 * 24 * 3600)
def dataforseo_get_google_locations() -> List[dict]:
    """
    Fetch all Google locations once, cache it.
    """
    login, password = _get_dataforseo_creds()
    if not login or not password:
        return []

    # Locations endpoint
    url = "https://api.dataforseo.com/v3/serp/google/locations"
    try:
        r = requests.get(url, auth=(login, password), timeout=60)
        j = r.json()
        tasks = j.get("tasks", [])
        if not tasks:
            return []
        res = tasks[0].get("result", [])
        return res if isinstance(res, list) else []
    except Exception:
        return []


def guess_uae_location_code(prefer: str = "Dubai") -> Optional[int]:
    locs = dataforseo_get_google_locations()
    if not locs:
        return None

    prefer = (prefer or "").strip().lower()
    # Try Dubai first, else UAE
    best = None
    for row in locs:
        name = (row.get("location_name") or "").lower()
        if "united arab emirates" in name and prefer in name:
            return row.get("location_code")
        if "united arab emirates" in name and best is None:
            best = row.get("location_code")
    return best


def dataforseo_google_rank_for_url(keyword: str, target_url: str, location_code: int, device: str) -> Dict[str, Optional[str]]:
    """
    Returns rank + AI visibility signals (best-effort) for a URL.
    """
    if not keyword or not target_url or not location_code:
        return {"rank": None, "ai_visibility": None}

    payload = [{
        "keyword": keyword,
        "language_code": "en",
        "location_code": int(location_code),
        "device": device,            # "desktop" or "mobile"
        "os": "windows" if device == "desktop" else "android",
        "depth": 50,
        "load_async_ai_overview": True
    }]

    j = dataforseo_post("/v3/serp/google/organic/live/advanced", payload)
    if j.get("_error"):
        return {"rank": None, "ai_visibility": None}

    tasks = j.get("tasks", [])
    if not tasks or not tasks[0].get("result"):
        return {"rank": None, "ai_visibility": None}

    result0 = tasks[0]["result"][0]
    items = result0.get("items", []) or []

    def norm(u: str) -> str:
        u = (u or "").strip().lower()
        u = u.replace("http://", "https://")
        if u.endswith("/"):
            u = u[:-1]
        return u

    tnorm = norm(target_url)

    rank = None
    ai_visibility = "Not detected"

    # AI overview detection (best effort)
    # Some responses include AI elements inside items list.
    for it in items:
        t = (it.get("type") or "").lower()
        if "ai" in t and ("overview" in t or "answer" in t):
            ai_visibility = "AI element present"
            break

    # Rank detection
    for it in items:
        if (it.get("type") or "").lower() != "organic":
            continue
        u = it.get("url") or it.get("domain") or ""
        if norm(u) == tnorm:
            rank = str(it.get("rank_absolute") or it.get("rank_group") or "")
            break

    return {"rank": rank, "ai_visibility": ai_visibility}


# =========================
# FETCH (FORCE RE-READ)
# =========================
@dataclass
class FetchResult:
    url: str
    ok: bool
    status: str
    html: str
    text: str
    used: str  # requests/playwright


def _requests_fetch(url: str) -> FetchResult:
    try:
        r = requests.get(url, headers=HEADERS, timeout=40, allow_redirects=True)
        html = r.text if r.text else ""
        soup = BeautifulSoup(html, "html.parser")
        for t in soup.find_all(list(IGNORE_TAGS)):
            t.decompose()
        text = re.sub(r"\s+", " ", soup.get_text(" ", strip=True))
        return FetchResult(url=url, ok=(r.status_code < 400 and len(text) > 200), status=str(r.status_code), html=html, text=text, used="requests")
    except Exception as e:
        return FetchResult(url=url, ok=False, status=f"requests_error:{e}", html="", text="", used="requests")


def _playwright_fetch(url: str) -> FetchResult:
    if not PLAYWRIGHT_OK:
        return FetchResult(url=url, ok=False, status="playwright_not_installed", html="", text="", used="playwright")

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            page.set_extra_http_headers(HEADERS)
            page.goto(url, wait_until="networkidle", timeout=60000)
            time.sleep(1.2)
            html = page.content()
            browser.close()

        soup = BeautifulSoup(html, "html.parser")
        for t in soup.find_all(list(IGNORE_TAGS)):
            t.decompose()
        text = re.sub(r"\s+", " ", soup.get_text(" ", strip=True))
        return FetchResult(url=url, ok=(len(text) > 200), status="200", html=html, text=text, used="playwright")
    except Exception as e:
        return FetchResult(url=url, ok=False, status=f"playwright_error:{e}", html="", text="", used="playwright")


def fetch_force(url: str) -> FetchResult:
    """
    Force an actual re-read:
    - try requests
    - if weak/blocked -> playwright
    """
    fr = _requests_fetch(url)
    if fr.ok and len(fr.html) > 1000:
        return fr
    pw = _playwright_fetch(url)
    if pw.ok:
        return pw
    return fr  # fallback (even if blocked)


# =========================
# PARSERS (TITLE/META/FAQ/VIDEO/TABLE)
# =========================
def extract_seo_fields(html: str, url: str) -> Dict[str, str]:
    out = {"slug": "", "seo_title": "", "meta_description": ""}
    out["slug"] = urlparse(url).path.strip("/") or "/"

    if not html:
        return out

    soup = BeautifulSoup(html, "html.parser")

    # Title
    t = soup.find("title")
    out["seo_title"] = (t.get_text(strip=True) if t else "")

    # Meta description
    md = soup.find("meta", attrs={"name": re.compile("^description$", re.I)})
    out["meta_description"] = (md.get("content", "").strip() if md else "")

    return out


def detect_real_faq(html: str, text: str) -> bool:
    """
    REAL FAQ only if:
    - FAQPage schema exists, OR
    - >= 3 question-like patterns in the page
    """
    if not (html or text):
        return False

    # Schema check
    if html:
        if re.search(r'"@type"\s*:\s*"FAQPage"', html, flags=re.I):
            return True

    # Question pattern check
    t = (text or "").lower()
    if not t:
        return False

    # If it contains FAQ hints and multiple question marks / Q patterns
    hint = any(h in t for h in FAQ_HINTS)
    qs = len(re.findall(r"\?\s", text)) + len(re.findall(r"\bq[:\-]\s", t))
    # Also detect “What is…”, “How to…”
    wh = len(re.findall(r"\b(what|how|why|where|when|who)\b", t))

    return (hint and (qs >= 2 or wh >= 12)) or (qs >= 4)


def count_tables(html: str) -> int:
    if not html:
        return 0
    soup = BeautifulSoup(html, "html.parser")
    return len(soup.find_all("table"))


def count_videos(html: str) -> int:
    if not html:
        return 0
    soup = BeautifulSoup(html, "html.parser")
    # youtube/vimeo iframes + html5 video tag
    vids = len(soup.find_all("video"))
    for iframe in soup.find_all("iframe"):
        src = (iframe.get("src") or "").lower()
        if "youtube" in src or "youtu.be" in src or "vimeo" in src:
            vids += 1
    return vids


def word_count(text: str) -> int:
    if not text:
        return 0
    return len(re.findall(r"\b\w+\b", text))


def keyword_usage_quality(text: str, keyword: str) -> Tuple[int, str]:
    """
    Not 'repeats' — quality-based:
    - Count mentions
    - Evaluate density + suspicious consecutive repetition
    """
    if not text or not keyword:
        return 0, "Not provided"

    t = text.lower()
    k = keyword.strip().lower()
    mentions = len(re.findall(re.escape(k), t))

    wc = max(1, word_count(text))
    density = mentions / wc

    # suspicious: same keyword appears 2+ times within 15 words window (crude)
    tokens = re.findall(r"\b\w+\b", t)
    k_tokens = re.findall(r"\b\w+\b", k)
    suspicious = False
    if k_tokens:
        joined = " ".join(tokens)
        suspicious = len(re.findall(rf"({re.escape(k)})\s+\S+\s+\S+\s+\S+\s+({re.escape(k)})", joined)) > 0

    if density > 0.03 or suspicious:
        return mentions, "Stuffing risk"
    if mentions == 0:
        return 0, "Missing"
    return mentions, "Natural"
# =========================
# HEADER-FIRST GAP LOGIC
# =========================
def extract_headings(html: str, text: str) -> List[str]:
    """
    Prefer HTML headings; fallback to text heuristics if needed.
    """
    headings: List[str] = []

    if html:
        soup = BeautifulSoup(html, "html.parser")
        for t in soup.find_all(list(IGNORE_TAGS)):
            t.decompose()

        for h in soup.find_all(["h1", "h2", "h3"]):
            s = re.sub(r"\s+", " ", h.get_text(" ", strip=True)).strip()
            if s and len(s) >= 3:
                headings.append(s)

    # Fallback if blocked html
    if not headings and text:
        # crude: lines that look like titles
        # (text fallback usually weak, but better than empty)
        candidates = re.findall(r"\b[A-Z][A-Za-z0-9 ,\-’'&]{8,80}\b", text)
        headings = list(dict.fromkeys(candidates[:20]))

    # Dedup preserve order
    seen = set()
    out = []
    for x in headings:
        nx = x.strip().lower()
        if nx not in seen:
            seen.add(nx)
            out.append(x)
    return out


def heading_match(a: str, b: str) -> float:
    a = (a or "").lower().strip()
    b = (b or "").lower().strip()
    if not a or not b:
        return 0.0
    # quick similarity
    from difflib import SequenceMatcher
    return SequenceMatcher(None, a, b).ratio()


def header_first_gaps(bayut_headings: List[str], comp_headings: List[str]) -> Tuple[List[str], List[Tuple[str, str]]]:
    """
    Returns:
    - missing_headers: competitor headers not covered by Bayut
    - undercovered: list of (matched_header, competitor_header) where Bayut has header but competitor's angle suggests extra parts
    """
    missing = []
    under = []

    for ch in comp_headings:
        best = ("", 0.0)
        for bh in bayut_headings:
            sc = heading_match(bh, ch)
            if sc > best[1]:
                best = (bh, sc)

        if best[1] >= 0.78:
            # same section, but competitor header may imply nuance -> mark under-covered
            if best[0].lower().strip() != ch.lower().strip():
                under.append((best[0], ch))
        else:
            missing.append(ch)

    return missing, under


def build_content_gaps_table(bayut: FetchResult, competitors: Dict[str, FetchResult]) -> pd.DataFrame:
    bayut_headings = extract_headings(bayut.html, bayut.text)
    rows = []

    for name, fr in competitors.items():
        comp_headings = extract_headings(fr.html, fr.text)
        missing_headers, undercovered = header_first_gaps(bayut_headings, comp_headings)

        # 1) Missing headers FIRST
        for mh in missing_headers:
            rows.append({
                "Header (Gap)": mh,
                "What to add": "Competitor covers this as a distinct section; Bayut is missing it.",
                "Source": name
            })

        # 2) Under-covered parts under matching header
        for (bh, ch) in undercovered:
            rows.append({
                "Header (Gap)": bh,
                "What to add": f"Under-covered: competitor frames this section as “{ch}” — expand Bayut with the missing angle/details.",
                "Source": name
            })

        # 3) FAQs (ONE row only if real FAQ exists on competitor but not on Bayut)
        bayut_has_faq = detect_real_faq(bayut.html, bayut.text)
        comp_has_faq = detect_real_faq(fr.html, fr.text)
        if comp_has_faq and not bayut_has_faq:
            rows.append({
                "Header (Gap)": "FAQs",
                "What to add": "Competitor includes a real FAQ section. Add a real FAQ block (not fake Qs) covering the core user questions.",
                "Source": name
            })

    if not rows:
        return pd.DataFrame(columns=["Header (Gap)", "What to add", "Source"])
    return pd.DataFrame(rows, columns=["Header (Gap)", "What to add", "Source"])


# =========================
# TABLES (SEO + CONTENT QUALITY)
# =========================
def build_seo_table(pages: Dict[str, FetchResult], focus_kw: str, location_code: Optional[int]) -> pd.DataFrame:
    rows = []

    for name, fr in pages.items():
        seo = extract_seo_fields(fr.html, fr.url)
        mentions, quality = keyword_usage_quality(fr.text, focus_kw)

        # UAE ranking (desktop + mobile)
        rank_desktop = None
        rank_mobile = None
        ai_desktop = None
        ai_mobile = None

        if focus_kw and location_code:
            d = dataforseo_google_rank_for_url(focus_kw, fr.url, location_code, device="desktop")
            m = dataforseo_google_rank_for_url(focus_kw, fr.url, location_code, device="mobile")
            rank_desktop, ai_desktop = d["rank"], d["ai_visibility"]
            rank_mobile, ai_mobile = m["rank"], m["ai_visibility"]

        rows.append({
            "Page": name,
            "Slug": seo["slug"],
            "SEO Title": seo["seo_title"],
            "Meta Description": seo["meta_description"],
            "FKW": focus_kw or "",
            "FKW usage (quality)": quality,
            "FKW mentions": mentions if focus_kw else "",
            "Google rank (UAE desktop)": rank_desktop or "Not found",
            "Google rank (UAE mobile)": rank_mobile or "Not found",
            "AI Visibility (desktop)": ai_desktop or "Not available",
            "AI Visibility (mobile)": ai_mobile or "Not available",
        })

    return pd.DataFrame(rows)


def build_content_quality_table(pages: Dict[str, FetchResult]) -> pd.DataFrame:
    rows = []
    for name, fr in pages.items():
        wc = word_count(fr.text)

        if not fr.html:
            # Don’t lie with "No" if we couldn't read HTML
            faq = "Not available (no HTML captured)"
            tables = "Not available (no HTML captured)"
            video = "Not available (no HTML captured)"
        else:
            faq = "Yes" if detect_real_faq(fr.html, fr.text) else "No"
            tables = count_tables(fr.html)
            video = count_videos(fr.html)

        rows.append({
            "Content Quality": "Word Count", name: wc
        })
        rows.append({
            "Content Quality": "FAQs", name: faq
        })
        rows.append({
            "Content Quality": "Tables", name: tables
        })
        rows.append({
            "Content Quality": "Video", name: video
        })

    # Convert rows to wide format (Content Quality as first col, sites as columns)
    df = pd.DataFrame(rows)
    df = df.groupby("Content Quality", as_index=False).first()
    # Ensure all page columns exist
    for site in pages.keys():
        if site not in df.columns:
            df[site] = ""
    return df[["Content Quality"] + list(pages.keys())]


# =========================
# UI
# =========================
st.title("Bayut Competitor Gap Analysis")
st.caption("Analyzes competitor articles to surface missing and under-covered sections.")

with st.form("inputs", clear_on_submit=False):
    bayut_url = st.text_input("Bayut article URL", placeholder="https://www.bayut.com/mybayut/...")
    comp_urls_raw = st.text_area("Competitor URLs (one per line, max 5)", height=90)

    focus_kw = st.text_input("Optional: Focus Keyword", placeholder="e.g., living in Dubai Marina")

    colA, colB = st.columns([1, 1])
    with colA:
        run = st.form_submit_button("Run analysis")
    with colB:
        st.write("")  # spacing

# ---- input signature (to kill stale session outputs)
sig = hashlib.sha1(
    (bayut_url.strip() + "\n" + comp_urls_raw.strip() + "\n" + (focus_kw or "").strip()).encode("utf-8")
).hexdigest()

if "last_sig" not in st.session_state:
    st.session_state.last_sig = None

if st.session_state.last_sig != sig:
    # clear stale outputs automatically when user changes URLs/keyword
    for k in ["gaps_df", "seo_df", "cq_df", "pages_cache", "comp_cache", "bayut_cache"]:
        if k in st.session_state:
            del st.session_state[k]
    st.session_state.last_sig = sig


# =========================
# RUN
# =========================
if run:
    bayut_url = bayut_url.strip()
    comp_urls = [u.strip() for u in comp_urls_raw.splitlines() if u.strip()]
    comp_urls = comp_urls[:5]

    if not bayut_url or not comp_urls:
        st.error("Please add a Bayut URL and at least 1 competitor URL.")
        st.stop()

    # Fetch (force re-read)
    bayut_fr = fetch_force(bayut_url)
    comps: Dict[str, FetchResult] = {}
    for i, u in enumerate(comp_urls, start=1):
        host = urlparse(u).netloc.replace("www.", "")
        name = host.split(".")[0].title() or f"Competitor {i}"
        comps[name] = fetch_force(u)

    st.session_state.bayut_cache = bayut_fr
    st.session_state.comp_cache = comps

    # Build gaps
    st.session_state.gaps_df = build_content_gaps_table(bayut_fr, comps)

    # Build pages dict for SEO + quality
    pages = {"Bayut": bayut_fr}
    pages.update(comps)

    # UAE location_code
    loc_code = guess_uae_location_code("Dubai")  # best-effort
    st.session_state.seo_df = build_seo_table(pages, focus_kw.strip(), loc_code)
    st.session_state.cq_df = build_content_quality_table(pages)


# =========================
# OUTPUTS
# =========================
st.subheader("Content Gaps Table")
gaps_df = st.session_state.get("gaps_df")
if isinstance(gaps_df, pd.DataFrame):
    st.dataframe(gaps_df, use_container_width=True, hide_index=True)
else:
    st.info("Run analysis to see results.")

st.subheader("Content Quality")
cq_df = st.session_state.get("cq_df")
if isinstance(cq_df, pd.DataFrame):
    st.dataframe(cq_df, use_container_width=True, hide_index=True)
else:
    st.info("Run analysis to see results.")

st.subheader("SEO Analysis")
seo_df = st.session_state.get("seo_df")
if isinstance(seo_df, pd.DataFrame):
    st.dataframe(seo_df, use_container_width=True, hide_index=True)
else:
    st.info("Run analysis to see results.")
