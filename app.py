import base64
import html as html_lib
import streamlit as st
import requests
import re
from bs4 import BeautifulSoup
from urllib.parse import quote_plus, urlparse
import pandas as pd
import time, random, hashlib
from dataclasses import dataclass
from typing import Optional, Dict, Tuple, List
from difflib import SequenceMatcher
import json

# Optional (recommended): JS rendering tool
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
# STYLE (LIGHT GREEN BACKGROUND + CENTERED MODE BUTTONS)
# =====================================================
BAYUT_GREEN = "#0E8A6D"
LIGHT_GREEN = "#E9F7F1"
LIGHT_GREEN_2 = "#DFF3EA"
TEXT_DARK = "#1F2937"
PAGE_BG = "#F3FBF7"  # lighter green background

st.markdown(
    f"""
    <style>
      html, body, [data-testid="stAppViewContainer"] {{
        background: {PAGE_BG} !important;
      }}
      [data-testid="stHeader"] {{
        background: rgba(0,0,0,0) !important;
      }}
      section.main > div.block-container {{
        max-width: 1180px !important;
        padding-top: 1.6rem !important;
        padding-bottom: 2.4rem !important;
      }}
      .hero {{
        text-align:center;
        margin-top: 0.6rem;
        margin-bottom: 1.2rem;
      }}
      .hero h1 {{
        font-size: 52px;
        line-height: 1.08;
        margin: 0;
        color: {TEXT_DARK};
        font-weight: 800;
        letter-spacing: -0.02em;
      }}
      .hero .bayut {{
        color: {BAYUT_GREEN};
      }}
      .hero p {{
        margin: 10px 0 0 0;
        color: #6B7280;
        font-size: 16px;
      }}
      .section-pill {{
        background: {LIGHT_GREEN};
        border: 1px solid {LIGHT_GREEN_2};
        padding: 10px 14px;
        border-radius: 14px;
        font-weight: 900;
        color: {TEXT_DARK};
        display: inline-block;
      }}
      /* tighter + closer to table */
      .section-pill-tight {{
        margin: 6px 0 4px 0;
      }}
      .stTextInput input, .stTextArea textarea {{
        background: {LIGHT_GREEN} !important;
        border: 1px solid {LIGHT_GREEN_2} !important;
        border-radius: 12px !important;
      }}
      .stButton button {{
        border-radius: 14px !important;
        padding: 0.65rem 1rem !important;
        font-weight: 900 !important;
      }}
      .mode-wrap {{
        display:flex;
        justify-content:center;
        margin: 10px 0 6px 0;
      }}
      .mode-note {{
        text-align:center;
        color:#6B7280;
        font-size: 13px;
        margin-top: -2px;
        margin-bottom: 6px;
      }}
      table {{
        width: 100% !important;
        border-collapse: separate !important;
        border-spacing: 0 !important;
        overflow: hidden !important;
        border-radius: 14px !important;
        border: 1px solid #E5E7EB !important;
        background: white !important;
        margin-top: 0 !important;
      }}
      thead th {{
        background: {LIGHT_GREEN} !important;
        text-align: center !important;
        font-weight: 900 !important;
        color: {TEXT_DARK} !important;
        padding: 6px 14px !important;      /* smaller */
        border-bottom: 1px solid #E5E7EB !important;
      }}
      tbody td {{
        vertical-align: top !important;
        padding: 6px 6px !important;       /* smaller */
        border-bottom: 1px solid #F1F5F9 !important;
        color: {TEXT_DARK} !important;
        font-size: 12px !important;        /* smaller */
      }}
      tbody tr:last-child td {{
        border-bottom: 0 !important;
      }}
      a {{
        color: {BAYUT_GREEN} !important;
        font-weight: 900 !important;
        text-decoration: underline !important;
      }}
      code {{
        background: rgba(0,0,0,0.04);
        padding: 2px 6px;
        border-radius: 8px;
      }}
      details.details-card summary {{
        cursor: pointer;
        color: {TEXT_DARK};
        font-weight: 800;
        list-style: none;
        display: block;
        padding: 6px 10px;
        border: 1px solid #E5E7EB;
        border-radius: 10px;
        background: #F9FAFB;
      }}
      details.details-card summary:hover {{
        background: #F3F4F6;
      }}
      details.details-card[open] summary {{
        border-bottom-left-radius: 0;
        border-bottom-right-radius: 0;
      }}
      details.details-card summary::-webkit-details-marker {{
        display: none;
      }}
      .details-box {{
        margin-top: 6px;
        padding: 8px 10px;
        background: #F9FAFB;
        border: 1px solid #E5E7EB;
        border-radius: 10px;
        color: {TEXT_DARK};
      }}
      details.details-card .details-box {{
        margin-top: 0;
        border-top: 0;
        border-radius: 0 0 10px 10px;
      }}
      .ai-summary {{
        background: white;
        border: 1px solid #E5E7EB;
        border-radius: 14px;
        padding: 14px 14px;
        margin: 6px 0 10px 0;
      }}
      .muted {{
        color:#6B7280;
        font-size: 13px;
      }}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    f"""
    <div class="hero">
      <h1><span class="bayut">Bayut</span> Competitor Gap Analysis</h1>
      <p>Identifies missing sections and incomplete coverage against competitor articles.</p>
    </div>
    """,
    unsafe_allow_html=True,
)


# =====================================================
# FETCH (NO MISSING COMPETITORS — ENFORCED)
# =====================================================
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

IGNORE_TAGS = {"nav", "footer", "header", "aside", "script", "style", "noscript", "form"}
CONTENT_CONTAINER_RE = re.compile(r"(content|article|post|entry|blog|main|body|page|story|news|text|detail|details)", re.I)


def clean(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip()


def pick_main_content_root(soup: BeautifulSoup):
    body = soup.body or soup
    candidates = []
    for tag in body.find_all(["article", "main"]):
        candidates.append(tag)
    for tag in body.find_all(["div", "section"]):
        ident = f"{tag.get('id', '')} {' '.join(tag.get('class', []) or [])}"
        if CONTENT_CONTAINER_RE.search(ident):
            candidates.append(tag)
    if not candidates:
        return body

    def text_len(tag):
        return len(clean(tag.get_text(" ")))

    return max(candidates, key=text_len)


def looks_blocked(text: str) -> bool:
    t = (text or "").lower()
    return any(x in t for x in [
        "just a moment", "checking your browser", "verify you are human",
        "cloudflare", "access denied", "captcha", "forbidden", "service unavailable"
    ])


@dataclass
class FetchResult:
    ok: bool
    source: Optional[str]
    status: Optional[int]
    html: str
    text: str
    reason: Optional[str]


class FetchAgent:
    """
    Deterministic resolver:
    - direct HTML
    - optional JS render (Playwright)
    - Jina reader
    - Textise
    If all fail => app forces manual paste (hard gate).
    """

    def __init__(self, default_headers: dict, ignore_tags: set, clean_fn, looks_blocked_fn):
        self.default_headers = default_headers
        self.ignore_tags = ignore_tags
        self.clean = clean_fn
        self.looks_blocked = looks_blocked_fn

        self.user_agents = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_6) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.3 Safari/605.1.15",
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0 Safari/537.36",
        ]

    def _http_get(self, url: str, timeout: int = 25, tries: int = 3) -> Tuple[int, str]:
        last_code, last_text = 0, ""
        for i in range(tries):
            headers = dict(self.default_headers)
            headers["User-Agent"] = random.choice(self.user_agents)
            try:
                r = requests.get(url, headers=headers, timeout=timeout, allow_redirects=True)
                last_code, last_text = r.status_code, (r.text or "")

                if r.status_code in (429, 500, 502, 503, 504):
                    time.sleep(1.2 * (i + 1))
                    continue

                return last_code, last_text
            except Exception as e:
                last_code, last_text = 0, str(e)
                time.sleep(1.2 * (i + 1))
        return last_code, last_text

    def _jina_url(self, url: str) -> str:
        if url.startswith("https://"):
            return "https://r.jina.ai/https://" + url[len("https://"):]
        if url.startswith("http://"):
            return "https://r.jina.ai/http://" + url[len("http://"):]
        return "https://r.jina.ai/https://" + url

    def _textise_url(self, url: str) -> str:
        return f"https://textise.org/showtext.aspx?strURL={quote_plus(url)}"

    def _validate_text(self, text: str, min_len: int) -> bool:
        t = self.clean(text)
        if len(t) < min_len:
            return False
        if self.looks_blocked(t):
            return False
        return True

    def _extract_article_text_from_html(self, html: str) -> str:
        soup = BeautifulSoup(html, "html.parser")
        for t in soup.find_all(list(self.ignore_tags)):
            t.decompose()
        root = pick_main_content_root(soup)
        return self.clean(root.get_text(" "))

    def _fetch_playwright_html(self, url: str, timeout_ms: int = 25000) -> Tuple[bool, str]:
        if not PLAYWRIGHT_OK:
            return False, ""
        try:
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True, args=["--no-sandbox"])
                ctx = browser.new_context(user_agent=random.choice(self.user_agents))
                page = ctx.new_page()
                page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)
                page.wait_for_timeout(1400)
                html = page.content()
                browser.close()
            return True, html
        except Exception:
            return False, ""

    def resolve(self, url: str) -> FetchResult:
        url = (url or "").strip()
        if not url:
            return FetchResult(False, None, None, "", "", "empty_url")

        # 1) direct HTML
        code, html = self._http_get(url)
        if code == 200 and html:
            text = self._extract_article_text_from_html(html)
            if self._validate_text(text, min_len=500):
                return FetchResult(True, "direct", code, html, text, None)

        # 2) JS-rendered HTML
        ok, html2 = self._fetch_playwright_html(url)
        if ok and html2:
            text2 = self._extract_article_text_from_html(html2)
            if self._validate_text(text2, min_len=500):
                return FetchResult(True, "playwright", 200, html2, text2, None)

        # 3) Jina reader
        jurl = self._jina_url(url)
        code3, txt3 = self._http_get(jurl)
        if code3 == 200 and txt3:
            text3 = self.clean(txt3)
            if self._validate_text(text3, min_len=500):
                return FetchResult(True, "jina", code3, "", text3, None)

        # 4) Textise
        turl = self._textise_url(url)
        code4, html4 = self._http_get(turl)
        if code4 == 200 and html4:
            soup = BeautifulSoup(html4, "html.parser")
            text4 = self.clean(soup.get_text(" "))
            if self._validate_text(text4, min_len=350):
                return FetchResult(True, "textise", code4, "", text4, None)

        return FetchResult(False, None, code or None, "", "", "blocked_or_no_content")


agent = FetchAgent(
    default_headers=DEFAULT_HEADERS,
    ignore_tags=IGNORE_TAGS,
    clean_fn=clean,
    looks_blocked_fn=looks_blocked,
)


def _safe_key(prefix: str, url: str) -> str:
    h = hashlib.md5((url or "").encode("utf-8")).hexdigest()
    return f"{prefix}__{h}"


def resolve_all_or_require_manual(agent: FetchAgent, urls: List[str], st_key_prefix: str) -> Dict[str, FetchResult]:
    results: Dict[str, FetchResult] = {}
    failed: List[str] = []

    for u in urls:
        r = agent.resolve(u)
        results[u] = r
        if not r.ok:
            failed.append(u)
        time.sleep(0.25)

    if not failed:
        return results

    st.error("Some URLs could not be fetched automatically. Paste the article HTML/text for EACH failed URL to continue. (No missing URLs.)")

    for u in failed:
        with st.expander(f"Manual fallback required: {u}", expanded=True):
            pasted = st.text_area(
                "Paste the full article HTML OR readable article text:",
                key=_safe_key(st_key_prefix + "__paste", u),
                height=220,
            )
            if pasted and len(pasted.strip()) > 400:
                results[u] = FetchResult(True, "manual", 200, pasted.strip(), pasted.strip(), None)

    still_failed = [u for u in failed if not results[u].ok]
    if still_failed:
        st.stop()

    return results


# =====================================================
# HEADING TREE + FILTERS
# =====================================================
NOISE_PATTERNS = [
    r"\blooking to rent\b", r"\blooking to buy\b", r"\bexplore all available\b", r"\bview all\b",
    r"\bfind (a|an) (home|property|apartment|villa)\b", r"\bbrowse\b", r"\bsearch\b",
    r"\bproperties for (rent|sale)\b", r"\bavailable (rental|properties)\b", r"\bget in touch\b",
    r"\bcontact (us|agent)\b", r"\bcall (us|now)\b", r"\bwhatsapp\b", r"\benquire\b",
    r"\brequest a call\b", r"\brequest (a|an) callback\b", r"\bcall back\b", r"\bbook a call\b",
    r"\bcontact (an|our) expert\b", r"\bfree consultation\b",
    r"\binquire\b", r"\bbook a viewing\b",
    r"\bshare\b", r"\bshare this\b", r"\bfollow us\b", r"\blike\b", r"\bsubscribe\b",
    r"\bnewsletter\b", r"\bsign up\b", r"\blogin\b", r"\bregister\b",
    r"\brelated (posts|articles)\b", r"\byou may also like\b", r"\brecommended\b",
    r"\bpopular posts\b", r"\bmore articles\b", r"\blatest (blogs|blog|podcasts|podcast|insights)\b",
    r"\breal estate insights\b",
    r"\btable of contents\b", r"\bcontents\b", r"\bback to top\b", r"\bread more\b",
    r"\bnext\b", r"\bprevious\b", r"\bcomments\b",
    r"\bplease stand by\b", r"\bloading\b", r"\bjust a moment\b",
]

GENERIC_SECTION_HEADERS = {"introduction", "overview"}

STOP = {
    "the","and","for","with","that","this","from","you","your","are","was","were","will","have","has","had",
    "but","not","can","may","more","most","into","than","then","they","them","their","our","out","about",
    "also","over","under","between","within","near","where","when","what","why","how","who","which",
    "a","an","to","of","in","on","at","as","is","it","be","or","by","we","i","us"
}

GENERIC_STOP = {
    "dubai","uae","business","bay","community","area","living","pros","cons",
    "property","properties","rent","sale","apartments","villas","guide"
}

def norm_header(h: str) -> str:
    h = clean(h).lower()
    h = re.sub(r"[^a-z0-9\s]", "", h)
    h = re.sub(r"\s+", " ", h).strip()
    return h

def is_noise_header(h: str) -> bool:
    s = clean(h)
    if not s:
        return True
    if header_is_faq(s):
        return False
    hn = norm_header(s)
    if hn in GENERIC_SECTION_HEADERS:
        return True
    if len(hn) < 4:
        return True
    if len(s) > 95:
        return True
    if sum(1 for c in s if c.isalnum()) / max(len(s), 1) < 0.6:
        return True
    for pat in NOISE_PATTERNS:
        if re.search(pat, hn):
            return True
    return False

def level_of(tag_name: str) -> int:
    try:
        return int(tag_name[1])
    except Exception:
        return 9

def build_tree_from_html(html: str) -> List[dict]:
    soup = BeautifulSoup(html, "html.parser")
    for t in soup.find_all(list(IGNORE_TAGS)):
        t.decompose()

    root = pick_main_content_root(soup)
    headings = root.find_all(["h1", "h2", "h3", "h4"])

    nodes: List[dict] = []
    stack: List[dict] = []

    def pop_to_level(lvl: int):
        while stack and stack[-1]["level"] >= lvl:
            stack.pop()

    def add_node(node: dict):
        if stack:
            stack[-1]["children"].append(node)
        else:
            nodes.append(node)
        stack.append(node)

    for h in headings:
        header = clean(h.get_text(" "))
        if not header or len(header) < 3:
            continue
        if is_noise_header(header):
            continue

        lvl = level_of(h.name)
        pop_to_level(lvl)

        node = {"level": lvl, "header": header, "content": "", "children": []}
        add_node(node)

        content_parts = []
        for sib in h.find_all_next():
            if sib == h:
                continue
            if getattr(sib, "name", None) in ["h1", "h2", "h3", "h4"]:
                break
            if getattr(sib, "name", None) in ["p", "li"]:
                txt = clean(sib.get_text(" "))
                if txt:
                    content_parts.append(txt)

        node["content"] = clean(" ".join(content_parts))

    return nodes

# ... (File continues exactly as in the workspace)
