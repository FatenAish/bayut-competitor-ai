# =========================
# PART 1/2 — imports, config, helpers
# =========================

import re
import json
import time
import base64
import hashlib
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple
from urllib.parse import urlparse, urlunparse

import requests
import pandas as pd
import streamlit as st
from bs4 import BeautifulSoup

# Optional JS rendering tool (recommended for blocked pages)
# pip install playwright
# playwright install chromium
try:
    from playwright.sync_api import sync_playwright
    PLAYWRIGHT_OK = True
except Exception:
    PLAYWRIGHT_OK = False


# =====================================================
# PAGE CONFIG (MUST BE FIRST STREAMLIT CALL)
# =====================================================
st.set_page_config(page_title="Bayut Competitor Gap Analysis", layout="wide")

# =====================================================
# STYLE
# =====================================================
BAYUT_GREEN = "#0E8A6D"
LIGHT_GREEN = "#E9F7F1"
TEXT_DARK = "#1F2937"
PAGE_BG = "#F3FBF7"

st.markdown(
    f"""
    <style>
      .stApp {{
        background: {PAGE_BG};
        color: {TEXT_DARK};
      }}
      .block-container {{
        padding-top: 2rem;
        padding-bottom: 2rem;
      }}
      h1, h2, h3 {{
        color: {TEXT_DARK};
      }}
      .chip {{
        display:inline-block;
        padding: 8px 12px;
        border-radius: 999px;
        background: {LIGHT_GREEN};
        border: 1px solid rgba(0,0,0,0.06);
        font-weight: 600;
      }}
      .mode-row {{
        display:flex;
        gap:12px;
        align-items:center;
        justify-content:center;
        margin-top: 6px;
        margin-bottom: 10px;
      }}
      .note {{
        font-size: 0.95rem;
        opacity: 0.85;
      }}
      .small {{
        font-size: 0.9rem;
        opacity: 0.85;
      }}
      .table-title {{
        margin-top: 10px;
        margin-bottom: 6px;
        font-weight: 700;
        font-size: 1.05rem;
      }}
      .hr {{
        height:1px;
        background: rgba(0,0,0,0.06);
        margin: 12px 0;
      }}
      .stButton > button {{
        border-radius: 12px !important;
        padding: 0.55rem 0.9rem !important;
        border: 1px solid rgba(0,0,0,0.10) !important;
      }}
      .stButton > button:hover {{
        border: 1px solid rgba(0,0,0,0.18) !important;
      }}
    </style>
    """,
    unsafe_allow_html=True
)

# =====================================================
# UTILITIES
# =====================================================
UA_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}

IGNORE_TAGS = {"nav", "footer", "header", "aside", "form", "noscript", "script", "style"}


def normalize_url(url: str) -> str:
    url = (url or "").strip()
    if not url:
        return ""
    if not re.match(r"^https?://", url, flags=re.I):
        url = "https://" + url
    p = urlparse(url)
    # Remove fragments, keep query (some sites use it)
    clean = urlunparse((p.scheme, p.netloc, p.path, p.params, p.query, ""))
    return clean


def domain_of(url: str) -> str:
    try:
        return urlparse(url).netloc.lower().replace("www.", "")
    except Exception:
        return ""


def safe_text(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()


def hash_key(*parts: str) -> str:
    h = hashlib.sha256("||".join([p or "" for p in parts]).encode("utf-8")).hexdigest()
    return h[:16]


def looks_blocked(html: str) -> bool:
    if not html:
        return True
    low = html.lower()
    blockers = [
        "access denied", "forbidden", "cloudflare", "captcha", "unusual traffic",
        "verify you are a human", "request blocked", "attention required"
    ]
    return any(b in low for b in blockers) or len(low) < 800


def fetch_html_requests(url: str, timeout: int = 18) -> Tuple[Optional[str], Optional[str]]:
    try:
        r = requests.get(url, headers=UA_HEADERS, timeout=timeout)
        if r.status_code >= 400:
            return None, f"http_{r.status_code}"
        html = r.text or ""
        if looks_blocked(html):
            return html, "blocked_or_thin"
        return html, None
    except Exception as e:
        return None, f"requests_error:{type(e).__name__}"


def fetch_html_playwright(url: str, timeout_ms: int = 25000) -> Tuple[Optional[str], Optional[str]]:
    if not PLAYWRIGHT_OK:
        return None, "playwright_not_installed"
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)
            # small wait for lazy content
            page.wait_for_timeout(800)
            html = page.content() or ""
            browser.close()
        if looks_blocked(html):
            return html, "blocked_or_thin"
        return html, None
    except Exception as e:
        return None, f"playwright_error:{type(e).__name__}"


def fetch_html(url: str) -> Tuple[Optional[str], Optional[str], str]:
    """
    Returns (html, error_code, fetch_method)
    error_code is None if ok
    """
    html, err = fetch_html_requests(url)
    if html and err is None:
        return html, None, "requests"
    # If requests blocked, try playwright
    if PLAYWRIGHT_OK:
        html2, err2 = fetch_html_playwright(url)
        if html2 and err2 is None:
            return html2, None, "playwright"
        # even if blocked/thin, return what we got for diagnostics
        if html2:
            return html2, err2, "playwright"
    # Return requests result (maybe blocked/thin or None)
    return html, err, "requests"


def clean_soup(html: str) -> BeautifulSoup:
    soup = BeautifulSoup(html or "", "html.parser")
    for t in list(soup.find_all(IGNORE_TAGS)):
        t.decompose()
    # remove obvious cookie popups (best-effort)
    for sel in ["cookie", "consent", "gdpr", "newsletter", "subscribe"]:
        for div in soup.find_all(attrs={"class": re.compile(sel, re.I)}):
            # don't over-delete; keep it limited
            if div and len(div.get_text(" ", strip=True)) < 400:
                div.decompose()
    return soup


@dataclass
class PageExtract:
    url: str
    title: str
    meta_description: str
    h1: str
    slug: str
    images_count: int
    videos_count: int
    internal_links: int
    external_links: int
    headings: List[Tuple[str, str]]  # (tag, text)
    sections: Dict[str, str]         # heading -> content text
    faq_present: bool
    full_text: str


def extract_seo_meta(soup: BeautifulSoup, url: str) -> Tuple[str, str, str]:
    title = safe_text(soup.title.get_text()) if soup.title else ""
    meta_desc = ""
    md = soup.find("meta", attrs={"name": re.compile("^description$", re.I)})
    if md and md.get("content"):
        meta_desc = safe_text(md.get("content"))
    h1_tag = soup.find(["h1"])
    h1 = safe_text(h1_tag.get_text(" ", strip=True)) if h1_tag else ""
    slug = urlparse(url).path.strip("/")
    return title, meta_desc, h1, slug


def detect_faq(soup: BeautifulSoup) -> bool:
    # 1) JSON-LD FAQPage
    for script in soup.find_all("script", attrs={"type": "application/ld+json"}):
        try:
            data = json.loads(script.get_text() or "{}")
            # can be list or dict
            items = data if isinstance(data, list) else [data]
            for it in items:
                if isinstance(it, dict) and (it.get("@type") == "FAQPage" or it.get("['@type']") == "FAQPage"):
                    return True
        except Exception:
            continue
    # 2) Heading contains FAQ / Frequently Asked Questions with real Q/A patterns
    headings = [safe_text(h.get_text(" ", strip=True)) for h in soup.find_all(["h2", "h3", "h4"])]
    if any(re.search(r"\bfaq\b|frequently asked questions", h, flags=re.I) for h in headings):
        # Try to find question marks under that area (best-effort)
        text = soup.get_text(" ", strip=True)
        qmarks = text.count("?")
        return qmarks >= 3
    return False


def parse_headings_and_sections(soup: BeautifulSoup) -> Tuple[List[Tuple[str, str]], Dict[str, str]]:
    """
    Build a simple section map: each H2/H3 becomes a section header with subsequent text until next same/higher heading.
    """
    headings: List[Tuple[str, str]] = []
    sections: Dict[str, str] = {}

    # We prioritize H2 then H3 under it, but keep both for comparison
    content_nodes = soup.find_all(["h2", "h3", "p", "li"])
    current_header = None
    buffer: List[str] = []

    def flush():
        nonlocal current_header, buffer
        if current_header and buffer:
            existing = sections.get(current_header, "")
            merged = safe_text((existing + " " + " ".join(buffer)).strip())
            sections[current_header] = merged
        buffer = []

    for node in content_nodes:
        if node.name in ("h2", "h3"):
            # flush previous
            flush()
            hdr = safe_text(node.get_text(" ", strip=True))
            if hdr:
                current_header = hdr
                headings.append((node.name, hdr))
            else:
                current_header = None
        else:
            txt = safe_text(node.get_text(" ", strip=True))
            if current_header and txt:
                buffer.append(txt)

    flush()
    return headings, sections


def extract_media_and_links(soup: BeautifulSoup, url: str) -> Tuple[int, int, int, int]:
    imgs = len(soup.find_all("img"))
    # videos: iframe/youtube/video tags
    videos = len(soup.find_all("video")) + len(soup.find_all("iframe", src=re.compile("youtube|vimeo", re.I)))
    internal = 0
    external = 0
    base_dom = domain_of(url)
    for a in soup.find_all("a", href=True):
        href = a.get("href") or ""
        if href.startswith("#") or href.lower().startswith("javascript:"):
            continue
        # Normalize relative
        if href.startswith("/"):
            internal += 1
            continue
        if href.lower().startswith("http"):
            dom = domain_of(href)
            if dom and base_dom and dom.endswith(base_dom):
                internal += 1
            else:
                external += 1
    return imgs, videos, internal, external


def extract_page(url: str, html: str) -> PageExtract:
    soup = clean_soup(html or "")
    title, meta_desc, h1, slug = extract_seo_meta(soup, url)
    imgs, vids, internal, external = extract_media_and_links(soup, url)
    headings, sections = parse_headings_and_sections(soup)
    faq_present = detect_faq(soup)
    full_text = safe_text(soup.get_text(" ", strip=True))
    return PageExtract(
        url=url, title=title, meta_description=meta_desc, h1=h1, slug=slug,
        images_count=imgs, videos_count=vids,
        internal_links=internal, external_links=external,
        headings=headings, sections=sections, faq_present=faq_present, full_text=full_text
    )


# =====================================================
# KEYWORD USAGE (RELEVANCE, ANTI-STUFFING)
# =====================================================
def keyword_stats(text: str, kw: str) -> Tuple[int, int]:
    if not text or not kw:
        return 0, 0
    # exact phrase count
    pattern = re.compile(r"\b" + re.escape(kw.strip()) + r"\b", re.I)
    matches = list(pattern.finditer(text))
    count = len(matches)
    words = len(re.findall(r"\w+", text))
    return count, words


def keyword_usage_quality(page: PageExtract, kw: str) -> Tuple[str, str]:
    """
    Returns (grade, note)
    Grade focuses on relevant placement and avoids stuffing.
    """
    kw = (kw or "").strip()
    if not kw:
        return "Not provided", ""

    title_hit = bool(re.search(re.escape(kw), page.title or "", re.I))
    h1_hit = bool(re.search(re.escape(kw), page.h1 or "", re.I))
    meta_hit = bool(re.search(re.escape(kw), page.meta_description or "", re.I))

    body_count, body_words = keyword_stats(page.full_text, kw)
    density = (body_count / body_words) * 100 if body_words else 0

    # Detect awkward repetition: same phrase repeated closely (simple heuristic)
    close_rep = False
    if body_count >= 3:
        # check if kw appears twice within ~15 words window
        tokens = re.findall(r"\w+|\W+", page.full_text.lower())
        # crude: if kw string repeats with small gap in raw text
        close_rep = bool(re.search(re.escape(kw.lower()) + r".{0,120}" + re.escape(kw.lower()), page.full_text.lower()))

    # Grading rules: encourage natural placement, punish stuffing
    placements = sum([title_hit, h1_hit, meta_hit])
    if placements >= 2 and body_count >= 1 and density <= 1.2 and not close_rep:
        return "Good", "Used naturally (strong placement; no stuffing signals)."
    if placements >= 1 and body_count >= 1 and density <= 1.8 and not close_rep:
        return "OK", "Present and generally natural; minor improvements possible."
    if density > 2.2 or close_rep:
        return "Needs improvement", "Possible keyword stuffing / unnecessary repetition."
    if placements == 0 and body_count >= 1:
        return "Needs improvement", "Missing key placement (title/H1/meta) despite body usage."
    if body_count == 0:
        return "Needs improvement", "Focus keyword not found in body content."
    return "OK", "Mixed signals; check relevance and placement."


# =====================================================
# CONTENT GAPS (HEADER-FIRST + UNDER-COVERED)
# =====================================================
def normalize_heading(h: str) -> str:
    h = safe_text(h).lower()
    h = re.sub(r"[:\-–—]+$", "", h).strip()
    h = re.sub(r"\s+", " ", h)
    return h


def heading_set(page: PageExtract) -> Dict[str, str]:
    """
    Returns normalized -> original heading
    """
    out = {}
    for tag, h in page.headings:
        if not h:
            continue
        # ignore very short noisy headings
        if len(h) < 3:
            continue
        n = normalize_heading(h)
        out[n] = h
    return out


def extract_section_keywords(text: str, top_n: int = 10) -> List[str]:
    if not text:
        return []
    words = re.findall(r"[a-zA-Z]{4,}", text.lower())
    stop = {
        "this","that","with","from","your","they","them","their","have","more","most","into","than","then",
        "also","over","under","between","within","near","where","when","what","why","how","who","which",
        "about","there","here","such","some","many","much","very","will","would","could","should","been",
        "were","was","are","you","the","and","for","to","of","in","on","at","as","is","it","be","or","by"
    }
    words = [w for w in words if w not in stop]
    freq = {}
    for w in words:
        freq[w] = freq.get(w, 0) + 1
    ranked = sorted(freq.items(), key=lambda x: x[1], reverse=True)
    return [w for w, c in ranked[:top_n]]


def undercovered_notes(bayut_text: str, comp_text: str) -> str:
    """
    Summarize what's missing by comparing keyword coverage (best-effort).
    """
    bay_k = set(extract_section_keywords(bayut_text, 12))
    comp_k = set(extract_section_keywords(comp_text, 12))
    missing = list(comp_k - bay_k)
    missing = missing[:8]
    if not missing:
        # fallback: compare length difference
        if len(comp_text) > len(bayut_text) * 1.6 and len(bayut_text) > 120:
            return "Competitor covers this section in more depth (add supporting details)."
        return "Under-coverage detected (add missing detail where relevant)."
    return "Under-covered topics: " + ", ".join(missing)


def build_content_gaps_table(bayut: PageExtract, competitors: List[PageExtract]) -> pd.DataFrame:
    """
    Output:
    - Missing headers first
    - Then under-covered under matching headers
    - FAQs are one row ONLY if competitor has REAL FAQ and bayut does not
    """
    rows = []

    bay_map = heading_set(bayut)
    bay_norms = set(bay_map.keys())

    for comp in competitors:
        comp_map = heading_set(comp)
        comp_norms = set(comp_map.keys())

        # 1) Missing headers first
        missing_norms = [n for n in comp_norms if n not in bay_norms]

        # Optional: treat FAQ as single row (only if real FAQ)
        # If both have FAQ, no gap row.
        if comp.faq_present and not bayut.faq_present:
            rows.append({
                "Header (Gap)": "FAQs",
                "What to add": "Competitor includes a real FAQ section. Add ONE FAQ section only if it fits the page intent.",
                "Source": domain_of(comp.url)
            })

        for n in sorted(missing_norms, key=lambda x: comp_map.get(x, x)):
            hdr = comp_map.get(n, "")
            comp_text = comp.sections.get(hdr, "")
            # short actionable suggestion:
            kws = extract_section_keywords(comp_text, 8)
            note = "Add this section (missing)."
            if kws:
                note += f" Key topics: {', '.join(kws[:6])}."
            rows.append({
                "Header (Gap)": hdr,
                "What to add": note,
                "Source": domain_of(comp.url)
            })

        # 2) Under-covered under matching headers (only for headers that exist on both)
        common_norms = [n for n in comp_norms if n in bay_norms]
        # Pick only likely meaningful headers (avoid generic like "introduction")
        for n in sorted(common_norms, key=lambda x: comp_map.get(x, x)):
            comp_hdr = comp_map.get(n, "")
            bay_hdr = bay_map.get(n, bayut.h1)  # fallback
            comp_text = comp.sections.get(comp_hdr, "")
            bay_text = bayut.sections.get(bay_hdr, "")

            if not comp_text or not bay_text:
                continue

            # Under-covered heuristic: competitor section significantly longer or has different keywords
            if len(comp_text) > len(bay_text) * 1.7 and len(comp_text) > 350:
                rows.append({
                    "Header (Gap)": comp_hdr,
                    "What to add": undercovered_notes(bay_text, comp_text),
                    "Source": domain_of(comp.url)
                })
            else:
                # keyword-diff check
                bay_k = set(extract_section_keywords(bay_text, 10))
                comp_k = set(extract_section_keywords(comp_text, 10))
                if len(comp_k - bay_k) >= 5:
                    rows.append({
                        "Header (Gap)": comp_hdr,
                        "What to add": undercovered_notes(bay_text, comp_text),
                        "Source": domain_of(comp.url)
                    })

    if not rows:
        return pd.DataFrame(columns=["Header (Gap)", "What to add", "Source"])

    # Ensure missing headers appear before under-covered? We already append missing first.
    # Keep as-is to preserve header-first flow.
    return pd.DataFrame(rows, columns=["Header (Gap)", "What to add", "Source"])


# =====================================================
# SEO / RANKING (UAE: desktop + mobile)
# - Uses DataForSEO if credentials exist
# =====================================================
def get_secret(name: str) -> Optional[str]:
    try:
        v = st.secrets.get(name)
        return v if v else None
    except Exception:
        return None


def dataforseo_auth_header(login: str, password: str) -> str:
    token = base64.b64encode(f"{login}:{password}".encode("utf-8")).decode("utf-8")
    return f"Basic {token}"


def dataforseo_serp_rank(
    query: str,
    target_url: str,
    device: str = "desktop",
    country_code: str = "AE",
    location_name: str = "United Arab Emirates",
    language_code: str = "en",
    top_n: int = 100
) -> Tuple[Optional[int], str]:
    """
    Returns (rank_position or None, note)
    """
    login = get_secret("DATAFORSEO_LOGIN") or get_secret("DATAFORSEO_USERNAME")
    password = get_secret("DATAFORSEO_PASSWORD")

    if not login or not password:
        return None, "missing_dataforseo_credentials"

    endpoint = "https://api.dataforseo.com/v3/serp/google/organic/live/advanced"

    payload = [{
        "keyword": query,
        "location_name": location_name,
        "language_code": language_code,
        "device": device,
        "os": "windows" if device == "desktop" else "android",
        "depth": top_n,
        "target": target_url
    }]

    headers = {
        "Authorization": dataforseo_auth_header(login, password),
        "Content-Type": "application/json"
    }

    try:
        r = requests.post(endpoint, headers=headers, json=payload, timeout=30)
        if r.status_code >= 400:
            return None, f"dataforseo_http_{r.status_code}"
        data = r.json()
        tasks = data.get("tasks") or []
        if not tasks or "result" not in tasks[0]:
            return None, "dataforseo_no_result"

        result = tasks[0].get("result") or []
        # result can contain items in result[0]["items"]
        # We scan for exact url match (best effort)
        for block in result:
            items = block.get("items") or []
            for it in items:
                if it.get("type") != "organic":
                    continue
                url = it.get("url") or ""
                if url and normalize_url(url).rstrip("/") == normalize_url(target_url).rstrip("/"):
                    pos = it.get("rank_absolute") or it.get("rank_group")
                    return int(pos) if pos else None, "ok"

        return None, "not_found_in_top_results"
    except Exception as e:
        return None, f"dataforseo_error:{type(e).__name__}"


# =====================================================
# AI VISIBILITY (Google AI Overview)
# - Uses SerpAPI if key exists
# =====================================================
def serpapi_ai_overview(query: str, gl: str = "ae", hl: str = "en", location: str = "United Arab Emirates") -> Tuple[Optional[dict], str]:
    key = get_secret("SERPAPI_KEY") or get_secret("SERPAPI_API_KEY")
    if not key:
        return None, "missing_serpapi_key"

    params = {
        "engine": "google",
        "q": query,
        "gl": gl,
        "hl": hl,
        "location": location,
        "api_key": key,
        "num": 10
    }
    try:
        r = requests.get("https://serpapi.com/search.json", params=params, timeout=30)
        if r.status_code >= 400:
            return None, f"serpapi_http_{r.status_code}"
        return r.json(), "ok"
    except Exception as e:
        return None, f"serpapi_error:{type(e).__name__}"


def parse_ai_overview_and_citations(serp_json: dict) -> Tuple[bool, List[str]]:
    """
    Best-effort extraction. SerpAPI fields can vary.
    We try common keys: ai_overview, answer_box, etc.
    Returns: (ai_overview_present, cited_domains)
    """
    if not serp_json:
        return False, []

    cited = set()

    # 1) ai_overview block (if present)
    aio = serp_json.get("ai_overview")
    if isinstance(aio, dict):
        # citations can appear under 'sources' or 'citations' depending on schema
        for key in ["sources", "citations", "references"]:
            arr = aio.get(key)
            if isinstance(arr, list):
                for x in arr:
                    if isinstance(x, dict) and x.get("link"):
                        cited.add(domain_of(x["link"]))
                    if isinstance(x, str):
                        cited.add(domain_of(x))
        return True, sorted([c for c in cited if c])

    # 2) sometimes it’s nested in "answer_box" or "knowledge_graph"
    for container_key in ["answer_box", "knowledge_graph"]:
        box = serp_json.get(container_key)
        if isinstance(box, dict):
            # heuristics: if it has "sources" and large "snippet"
            arr = box.get("sources") or box.get("links") or box.get("references")
            if isinstance(arr, list):
                for x in arr:
                    if isinstance(x, dict) and x.get("link"):
                        cited.add(domain_of(x["link"]))
                    if isinstance(x, str):
                        cited.add(domain_of(x))
            # if it looks like AI overview style, treat as present
            if box.get("snippet") and len(str(box.get("snippet"))) > 120:
                return True, sorted([c for c in cited if c])

    return False, sorted([c for c in cited if c])


def build_ai_visibility_table(pages: List[Tuple[str, str]], query: str) -> pd.DataFrame:
    """
    pages: list of (label, url)
    """
    rows = []
    serp_json, note = serpapi_ai_overview(query)
    if not serp_json:
        for label, _url in pages:
            rows.append({
                "Page": label,
                "Query": query,
                "AI Overview present": "Not available",
                "Cited in AI Overview": "Not available",
                "AI Notes": note
            })
        return pd.DataFrame(rows)

    present, cited_domains = parse_ai_overview_and_citations(serp_json)
    cited_set = set([d for d in cited_domains if d])

    for label, url in pages:
        dom = domain_of(url)
        rows.append({
            "Page": label,
            "Query": query,
            "AI Overview present": "Yes" if present else "No",
            "Cited in AI Overview": "Yes" if (present and dom in cited_set) else "No",
            "AI Notes": "ok" if note == "ok" else note
        })

    return pd.DataFrame(rows, columns=["Page", "Query", "AI Overview present", "Cited in AI Overview", "AI Notes"])
# =========================
# PART 2/2 — UI + execution
# =========================

st.markdown("<h1>Bayut Competitor Editorial Gap Analysis</h1>", unsafe_allow_html=True)
st.markdown(
    '<div class="note">Analyzes competitor articles to surface missing and under-covered sections.</div>',
    unsafe_allow_html=True
)

st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

colA, colB = st.columns([1, 1])

with colA:
    bayut_url = st.text_input("Bayut article URL", value="").strip()

with colB:
    competitors_raw = st.text_area(
        "Competitor URLs (one per line, max 5)",
        value="",
        height=120
    ).strip()

# Label update per your request
focus_kw = st.text_input("Optional: Focus Keyword", value="").strip()

# Buttons rename per your request
btn_row1 = st.columns([1, 1, 3])
with btn_row1[0]:
    run_update_mode = st.button("Update Mode", use_container_width=True)
with btn_row1[1]:
    run_gaps_table = st.button("Content Gaps Table", use_container_width=True)

# Internal: run analysis if any button clicked
RUN = run_update_mode or run_gaps_table

# Normalize urls
bayut_url = normalize_url(bayut_url)
competitor_urls = [normalize_url(u) for u in competitors_raw.splitlines() if u.strip()]
competitor_urls = competitor_urls[:5]


def page_fetch_block(label: str, url: str) -> Tuple[Optional[PageExtract], str]:
    """
    Returns (PageExtract or None, status_note)
    - If blocked, provides a paste box without the removed "Tip" copy.
    """
    if not url:
        return None, "missing_url"

    html, err, method = fetch_html(url)
    if html and err is None:
        return extract_page(url, html), f"ok ({method})"

    # blocked / thin / errors -> allow paste HTML
    st.warning(f"{label} might be blocked or incomplete. If you have the page HTML, paste it below to ensure nothing is missing.")
    paste = st.text_area(f"Paste HTML for {label}", value="", height=160, key=f"paste_{hash_key(label, url)}").strip()
    if paste:
        return extract_page(url, paste), "ok (pasted_html)"
    # still return nothing
    return None, err or "blocked_or_failed"


def build_content_quality_table(bayut: PageExtract, competitors: List[PageExtract]) -> pd.DataFrame:
    """
    Content Quality is now the 2nd table.
    Simple, writer-useful signals.
    """
    def quality_row(label: str, p: PageExtract) -> dict:
        # readability (rough): avg sentence length
        sentences = re.split(r"[.!?]+", p.full_text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
        words = re.findall(r"\w+", p.full_text)
        avg_sent = (len(words) / max(1, len(sentences))) if words else 0

        structure = "Strong" if len(p.sections) >= 8 else ("OK" if len(p.sections) >= 5 else "Weak")
        media = "Good" if (p.images_count + p.videos_count) >= 6 else ("OK" if (p.images_count + p.videos_count) >= 3 else "Low")
        linking = "Good" if p.internal_links >= 6 else ("OK" if p.internal_links >= 3 else "Low")

        return {
            "Page": label,
            "Structure depth": structure,
            "Media richness": media,
            "Internal linking": linking,
            "Readability (avg words/sentence)": round(avg_sent, 1),
            "Notes": ""
        }

    rows = [quality_row("Bayut", bayut)]
    for c in competitors:
        rows.append(quality_row(domain_of(c.url).title(), c))
    return pd.DataFrame(rows)


def build_seo_table(bayut: PageExtract, competitors: List[PageExtract], focus_kw: str) -> pd.DataFrame:
    """
    SEO Analysis table (NO Headers count).
    - Includes KW usage quality (relevance/anti-stuffing)
    - Includes UAE ranking if focus_kw exists and DataForSEO credentials exist
    """
    def row(label: str, p: PageExtract) -> dict:
        kw_grade, kw_note = keyword_usage_quality(p, focus_kw)

        # UAE ranking only if focus_kw provided
        rank_d, note_d = (None, "no_focus_keyword") if not focus_kw else dataforseo_serp_rank(
            query=focus_kw, target_url=p.url, device="desktop"
        )
        rank_m, note_m = (None, "no_focus_keyword") if not focus_kw else dataforseo_serp_rank(
            query=focus_kw, target_url=p.url, device="mobile"
        )

        def fmt_rank(r, note):
            if r is not None:
                return int(r)
            # show clean "Not available" instead of noisy codes
            return "Not available"

        return {
            "Page": label,
            "Slug": p.slug,
            "SEO Title": p.title,
            "Meta description": p.meta_description,
            "H1": p.h1,
            "Media used": f"{p.images_count} images, {p.videos_count} videos",
            "KW usage (quality)": kw_grade,
            "KW usage notes": kw_note,
            "UAE rank (desktop)": fmt_rank(rank_d, note_d),
            "UAE rank (mobile)": fmt_rank(rank_m, note_m),
        }

    rows = [row("Bayut", bayut)]
    for c in competitors:
        rows.append(row(domain_of(c.url).title(), c))
    return pd.DataFrame(rows)


if RUN:
    if not bayut_url:
        st.error("Please enter the Bayut URL.")
        st.stop()
    if not competitor_urls:
        st.error("Please add at least 1 competitor URL.")
        st.stop()

    st.markdown('<div class="hr"></div>', unsafe_allow_html=True)
    st.info("Running analysis…")

    # Fetch Bayut
    bay_page, bay_status = page_fetch_block("Bayut", bayut_url)
    if not bay_page:
        st.error(f"Could not fetch Bayut page ({bay_status}). Paste HTML to proceed.")
        st.stop()

    # Fetch competitors
    comp_pages: List[PageExtract] = []
    for u in competitor_urls:
        label = domain_of(u).title() or "Competitor"
        p, status = page_fetch_block(label, u)
        if p:
            comp_pages.append(p)
        else:
            st.warning(f"Skipped {u} ({status}).")

    if not comp_pages:
        st.error("No competitor pages could be fetched. Paste HTML for at least one competitor.")
        st.stop()

    st.success("Analysis complete.")

    # =========================
    # TABLE 1 — CONTENT GAPS
    # =========================
    st.markdown('<div class="table-title">Content Gaps Table</div>', unsafe_allow_html=True)
    gaps_df = build_content_gaps_table(bay_page, comp_pages)
    st.dataframe(gaps_df, use_container_width=True, hide_index=True)

    # =========================
    # TABLE 2 — CONTENT QUALITY (must be 2nd)
    # =========================
    st.markdown('<div class="table-title">Content Quality</div>', unsafe_allow_html=True)
    cq_df = build_content_quality_table(bay_page, comp_pages)
    st.dataframe(cq_df, use_container_width=True, hide_index=True)

    # =========================
    # TABLE 3 — SEO ANALYSIS (NO headers count + new KW logic)
    # =========================
    st.markdown('<div class="table-title">SEO Analysis</div>', unsafe_allow_html=True)
    seo_df = build_seo_table(bay_page, comp_pages, focus_kw)
    st.dataframe(seo_df, use_container_width=True, hide_index=True)

    # =========================
    # TABLE 4 — AI VISIBILITY (Google AI Overview)
    # =========================
    # If no focus_kw, we still generate a query from Bayut title/H1 (best effort)
    query = focus_kw or (bay_page.h1 or bay_page.title or "").strip()
    if not query:
        query = "Living in Business Bay"  # fallback

    st.markdown('<div class="table-title">AI Visibility (Google AI Overview)</div>', unsafe_allow_html=True)
    pages_for_ai = [("Bayut", bay_page.url)]
    for c in comp_pages:
        pages_for_ai.append((domain_of(c.url).title(), c.url))

    ai_df = build_ai_visibility_table(pages_for_ai, query=query)
    st.dataframe(ai_df, use_container_width=True, hide_index=True)

else:
    # Idle screen hints (no "Tip" line, as requested)
    st.markdown('<div class="small">Enter URLs, optionally add a Focus Keyword, then run Update Mode or Content Gaps Table.</div>', unsafe_allow_html=True)
