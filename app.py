# app.py (PART 1/2)
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
      .details-link summary {{
        cursor: pointer;
        color: {BAYUT_GREEN};
        text-decoration: underline;
        font-weight: 900;
        list-style: none;
      }}
      .details-link summary::-webkit-details-marker {{
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

IGNORE_TAGS = {"nav", "footer", "header", "aside", "script", "style", "noscript"}


def clean(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip()


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
        article = soup.find("article") or soup
        return self.clean(article.get_text(" "))

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

    root = soup.find("article") or soup
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

def build_tree_from_reader_text(text: str) -> List[dict]:
    lines = [l.rstrip() for l in (text or "").splitlines()]
    nodes: List[dict] = []
    stack: List[dict] = []

    def md_level(line: str):
        m = re.match(r"^(#{1,4})\s+(.*)$", line.strip())
        if not m:
            return None
        lvl = len(m.group(1))
        header = clean(m.group(2))
        return lvl, header

    def pop_to_level(lvl: int):
        while stack and stack[-1]["level"] >= lvl:
            stack.pop()

    def add_node(node: dict):
        if stack:
            stack[-1]["children"].append(node)
        else:
            nodes.append(node)
        stack.append(node)

    current = None
    for line in lines:
        s = line.strip()
        if not s:
            continue

        ml = md_level(s)
        if ml:
            lvl, header = ml
            if is_noise_header(header):
                current = None
                continue

            pop_to_level(lvl)
            node = {"level": lvl, "header": header, "content": "", "children": []}
            add_node(node)
            current = node
        else:
            if current:
                current["content"] += " " + s

    def walk(n: dict) -> dict:
        n["content"] = clean(n["content"])
        n["children"] = [walk(c) for c in n["children"]]
        return n

    return [walk(n) for n in nodes]

def build_tree_from_plain_text_heuristic(text: str) -> List[dict]:
    raw = (text or "").replace("\r", "")
    lines = [clean(l) for l in raw.split("\n")]
    lines = [l for l in lines if l]

    def looks_like_heading(line: str) -> bool:
        if len(line) < 5 or len(line) > 80:
            return False
        if line.endswith("."):
            return False
        if is_noise_header(line):
            return False
        words = line.split()
        if len(words) < 2 or len(words) > 12:
            return False
        caps_ratio = sum(1 for w in words if w[:1].isupper()) / max(len(words), 1)
        allcaps_ratio = sum(1 for c in line if c.isupper()) / max(sum(1 for c in line if c.isalpha()), 1)
        return (caps_ratio >= 0.6) or (allcaps_ratio >= 0.5)

    nodes: List[dict] = []
    current = None

    for line in lines:
        if looks_like_heading(line):
            current = {"level": 2, "header": line, "content": "", "children": []}
            nodes.append(current)
        else:
            if current is None:
                current = {"level": 2, "header": "Overview", "content": "", "children": []}
                nodes.append(current)
            current["content"] = clean(current["content"] + " " + line)

    return nodes

def get_tree_from_fetchresult(fr: FetchResult) -> dict:
    if not fr.ok:
        return {"ok": False, "source": None, "nodes": [], "status": fr.status}

    txt = fr.text or ""
    maybe_html = ("<html" in txt.lower()) or ("<article" in txt.lower()) or ("<h1" in txt.lower()) or ("<h2" in txt.lower())

    if fr.html:
        nodes = build_tree_from_html(fr.html)
    elif fr.source == "manual" and maybe_html:
        nodes = build_tree_from_html(txt)
    else:
        nodes = build_tree_from_reader_text(txt)
        if not nodes:
            nodes = build_tree_from_plain_text_heuristic(txt)

    return {"ok": True, "source": fr.source, "nodes": nodes, "status": fr.status}

def ensure_headings_or_require_repaste(urls: List[str], fr_map: Dict[str, FetchResult], st_key_prefix: str) -> Dict[str, dict]:
    tree_map: Dict[str, dict] = {}
    bad: List[str] = []

    for u in urls:
        tr = get_tree_from_fetchresult(fr_map[u])
        tree_map[u] = tr
        if not tr.get("nodes"):
            bad.append(u)

    if not bad:
        return tree_map

    st.error("Some URLs were fetched, but headings could not be extracted. Paste readable HTML (preferred) or clearly structured text for EACH URL below to continue.")

    for u in bad:
        with st.expander(f"Headings extraction required: {u}", expanded=True):
            repaste = st.text_area(
                "Paste readable HTML (preferred) OR structured text with headings:",
                key=_safe_key(st_key_prefix + "__repaste", u),
                height=240,
            )
            if repaste and len(repaste.strip()) > 400:
                fr_map[u] = FetchResult(True, "manual", 200, repaste.strip(), repaste.strip(), None)

    still_bad = []
    for u in bad:
        tr = get_tree_from_fetchresult(fr_map[u])
        tree_map[u] = tr
        if not tr.get("nodes"):
            still_bad.append(u)

    if still_bad:
        st.stop()

    return tree_map


# =====================================================
# HELPERS
# =====================================================
def site_name(url: str) -> str:
    try:
        host = urlparse(url).netloc.lower().replace("www.", "")
        base = host.split(":")[0]
        name = base.split(".")[0]
        return name[:1].upper() + name[1:]
    except Exception:
        return "Source"

def source_link(url: str) -> str:
    n = site_name(url)
    return f'<a href="{url}" target="_blank" rel="noopener noreferrer">{n}</a>'

def flatten(nodes: List[dict]) -> List[dict]:
    out = []
    def walk(n: dict, parent=None):
        out.append({
            "level": n["level"],
            "header": n.get("header",""),
            "content": n.get("content", ""),
            "parent": parent,
            "children": n.get("children", [])
        })
        for c in n.get("children", []):
            walk(c, n)
    for n in nodes:
        walk(n, None)
    return out

def strip_label(h: str) -> str:
    return clean(re.sub(r"\s*:\s*$", "", (h or "").strip()))

def format_gap_list(items: List[str], limit: int = 6) -> str:
    cleaned = []
    seen = set()
    skip = {
        "other", "other topics", "other faq topics", "faq topics", "other faq topic",
        "other faq", "general", "misc", "miscellaneous"
    }
    for item in items or []:
        it = clean(item)
        if not it:
            continue
        k = norm_header(it)
        if k in skip:
            continue
        if k in seen:
            continue
        seen.add(k)
        cleaned.append(it)
    if not cleaned:
        return ""
    if limit <= 0 or len(cleaned) <= limit:
        return ", ".join(cleaned)
    return ", ".join(cleaned[:limit]) + f", and {len(cleaned) - limit} more"

def headings_blob(nodes: List[dict]) -> str:
    hs = []
    for x in flatten(nodes):
        h = clean(x.get("header", ""))
        if h and not is_noise_header(h):
            hs.append(h)
    return clean(" | ".join(hs))

def get_first_h1(nodes: List[dict]) -> str:
    for x in flatten(nodes):
        if x.get("level") == 1:
            h = clean(x.get("header", ""))
            if h:
                return h
    for x in flatten(nodes):
        if x.get("level") == 2:
            h = clean(x.get("header", ""))
            if h:
                return h
    return "Not available"


# =====================================================
# STRICT FAQ DETECTION (REAL FAQ ONLY)
# =====================================================
FAQ_TITLES = {
    "faq",
    "faqs",
    "frequently asked questions",
    "frequently asked question",
}

def header_is_faq(header: str) -> bool:
    nh = norm_header(header)
    if not nh:
        return False
    if nh in FAQ_TITLES:
        return True
    if "faq" in nh:
        return True
    if "frequently asked" in nh:
        return True
    return False

def _looks_like_question(s: str) -> bool:
    s = clean(s)
    if not s or len(s) < 6:
        return False
    s_low = s.lower()
    if "?" in s:
        return True
    if re.match(r"^(what|where|when|why|how|who|which|can|is|are|do|does|did|should)\b", s_low):
        return True
    if any(p in s_low for p in ["what is", "how to", "is it", "are there", "can i", "should i"]):
        return True
    return False

def normalize_question(q: str) -> str:
    q = clean(q or "")
    q = re.sub(r"^\s*\d+[\.\)]\s*", "", q)
    q = re.sub(r"^\s*[-•]\s*", "", q)
    return q.strip()

def _has_faq_schema(html: str) -> bool:
    if not html:
        return False
    try:
        soup = BeautifulSoup(html, "html.parser")
        scripts = soup.find_all("script", attrs={"type": re.compile(r"ld\+json", re.I)})
        for s in scripts:
            raw = (s.string or s.get_text(" ") or "").strip()
            if not raw:
                continue
            try:
                j = json.loads(raw)
            except Exception:
                continue

            def walk(x):
                if isinstance(x, dict):
                    t = x.get("@type") or x.get("type")
                    if isinstance(t, str) and t.lower() == "faqpage":
                        return True
                    if isinstance(t, list) and any(str(z).lower() == "faqpage" for z in t):
                        return True
                    for v in x.values():
                        if walk(v):
                            return True
                elif isinstance(x, list):
                    for v in x:
                        if walk(v):
                            return True
                return False

            if walk(j):
                return True
    except Exception:
        return False
    return False

def _faq_questions_from_schema(html: str) -> List[str]:
    if not html:
        return []
    qs: List[str] = []
    try:
        soup = BeautifulSoup(html, "html.parser")
        scripts = soup.find_all("script", attrs={"type": re.compile(r"ld\+json", re.I)})
        for s in scripts:
            raw = (s.string or s.get_text(" ") or "").strip()
            if not raw:
                continue
            try:
                j = json.loads(raw)
            except Exception:
                continue

            def add_q(q: str):
                qn = normalize_question(q)
                if not qn or len(qn) < 6 or len(qn) > 180:
                    return
                qs.append(qn)

            def walk(x):
                if isinstance(x, dict):
                    t = x.get("@type") or x.get("type")
                    t_list = []
                    if isinstance(t, list):
                        t_list = [str(z).lower() for z in t]
                    elif isinstance(t, str):
                        t_list = [t.lower()]

                    if any("question" == z or z.endswith("question") for z in t_list):
                        name = x.get("name") or x.get("text") or ""
                        if name:
                            add_q(name)

                    for v in x.values():
                        walk(v)
                elif isinstance(x, list):
                    for v in x:
                        walk(v)

            walk(j)
    except Exception:
        return []

    seen = set()
    out = []
    for q in qs:
        k = norm_header(q)
        if not k or k in seen:
            continue
        seen.add(k)
        out.append(q)
    return out

def _faq_questions_from_html(html: str) -> List[str]:
    if not html:
        return []
    soup = BeautifulSoup(html, "html.parser")
    for t in soup.find_all(list(IGNORE_TAGS)):
        t.decompose()

    qs: List[str] = []
    qs.extend(_faq_questions_from_schema(html))

    candidates = []
    for tag in soup.find_all(True):
        id_attr = (tag.get("id") or "").lower()
        cls_attr = " ".join(tag.get("class", []) or []).lower()
        if re.search(r"\bfaq\b|\bfaqs\b|\baccordion\b|\bquestions\b", id_attr + " " + cls_attr):
            candidates.append(tag)

    for h in soup.find_all(["h1", "h2", "h3", "h4"]):
        if header_is_faq(h.get_text(" ")):
            candidates.append(h.parent or h)

    for c in candidates[:10]:
        for el in c.find_all(["summary", "button", "h3", "h4", "h5", "strong", "p", "li", "dt"]):
            txt = clean(el.get_text(" "))
            if not txt or len(txt) < 6 or len(txt) > 180:
                continue
            if _looks_like_question(txt):
                qs.append(normalize_question(txt))

    seen = set()
    out = []
    for q in qs:
        k = norm_header(q)
        if not k or k in seen:
            continue
        seen.add(k)
        out.append(q)
    return out

def _faq_heading_nodes(nodes: List[dict]) -> List[dict]:
    out = []
    for x in flatten(nodes):
        if x.get("level") in (2, 3, 4) and header_is_faq(x.get("header", "")):
            out.append(x)
    return out

def _question_heading_children(node: dict) -> List[str]:
    qs = []
    for c in node.get("children", []) or []:
        hdr = clean(c.get("header", ""))
        if hdr and _looks_like_question(hdr):
            qs.append(normalize_question(hdr))
    return qs

def page_has_real_faq(fr: FetchResult, nodes: List[dict]) -> bool:
    if fr and fr.html:
        if _has_faq_schema(fr.html):
            return True
        if len(_faq_questions_from_html(fr.html)) >= 2:
            return True

    faq_nodes = _faq_heading_nodes(nodes)
    if not faq_nodes:
        return False

    for fn in faq_nodes:
        if len(extract_questions_from_node(fn)) >= 2:
            return True
        txt = clean(fn.get("content", ""))
        if txt and txt.count("?") >= 2:
            return True

    return True

def extract_faq_questions(fr: FetchResult, nodes: List[dict]) -> List[str]:
    qs: List[str] = []
    if fr and fr.html:
        qs.extend(_faq_questions_from_html(fr.html))
    for fn in _faq_heading_nodes(nodes):
        qs.extend(extract_questions_from_node(fn))

    seen = set()
    out = []
    for q in qs:
        k = norm_header(q)
        if not k or k in seen:
            continue
        seen.add(k)
        out.append(q)
    return out

def extract_questions_from_node(node: dict) -> List[str]:
    qs: List[str] = []
    qh = _question_heading_children(node)
    qs.extend(qh)

    def add_from_text_block(txt: str):
        txt = clean(txt or "")
        if not txt:
            return
        chunks = re.split(r"[\n\r]+|(?<=[\.\?\!])\s+", txt)
        for ch in chunks[:80]:
            ch = clean(ch)
            if not ch or len(ch) > 160:
                continue
            if _looks_like_question(ch):
                qs.append(normalize_question(ch))

    if len(qs) < 3:
        add_from_text_block(node.get("content", ""))

    seen = set()
    out = []
    for q in qs:
        k = norm_header(q)
        if not k or k in seen:
            continue
        seen.add(k)
        out.append(q)
    return out[:25]

# =====================================================
# ✅ NEW: FAQ “topic labels” (brief, human) — forced
# =====================================================
FAQ_TOPIC_HINTS = [
    (["community", "neighborhood", "neighbourhood", "vibe", "atmosphere"], "Community"),
    (["amenities", "services", "facilities"], "Amenities and services"),
    (["rent", "rental", "price", "cost", "fees", "budget", "afford"], "Cost"),
    (["safe", "safety", "security", "crime"], "Safety"),
    (["schools", "nursery", "education"], "Schools"),
    (["parking", "traffic", "congestion"], "Traffic and parking"),
    (["transport", "metro", "monorail", "tram", "bus", "commute", "connect"], "Transport and commute"),
    (["freehold", "ownership", "buy", "purchase"], "Ownership"),
    (["pets", "pet"], "Pets"),
    (["beach", "waterfront"], "Beach access"),
]

FAQ_LABEL_STOP = STOP | {
    "palm", "jumeirah", "dubai", "uae", "residents", "resident", "people",
    "living", "live", "there", "here", "this", "that", "it", "in", "at", "on", "for", "to", "of"
}

def faq_brief_topic_label(question: str) -> str:
    q = normalize_question(question)
    q_low = q.lower()

    # 1) strong keyword hints (most reliable)
    for keys, label in FAQ_TOPIC_HINTS:
        if any(k in q_low for k in keys):
            return label

    # 2) generic fallback: strip question words + location phrases, keep 1–4 important words
    q_low = re.sub(r"[\?\.\!]+$", "", q_low).strip()
    q_low = re.sub(r"^(what|where|when|why|how|who|which|can|is|are|do|does|did|should|could|would|will)\b", "", q_low).strip()
    q_low = re.sub(r"\b(in|at|on|for|about|of)\b\s+(palm\s+jumeirah|dubai|uae)\b", "", q_low).strip()

    words = re.sub(r"[^a-z0-9\s]", " ", q_low)
    words = re.sub(r"\s+", " ", words).strip().split()
    keep = []
    for w in words:
        if w in FAQ_LABEL_STOP:
            continue
        if len(w) < 3:
            continue
        keep.append(w)
        if len(keep) >= 4:
            break

    if not keep:
        # last-resort: first non-empty token chunk
        return "General"

    # Title-case, small cleanup
    label = " ".join(keep)
    label = label.replace("  ", " ").strip()
    return label[:1].upper() + label[1:]


def faq_topics_from_questions_brief(questions: List[str], limit: int = 10) -> List[str]:
    out: List[str] = []
    seen = set()
    for q in questions:
        lab = faq_brief_topic_label(q)
        k = norm_header(lab)
        if not k or k in seen:
            continue
        seen.add(k)
        out.append(lab)
        if len(out) >= limit:
            break
    return out


def missing_faqs_row(
    bayut_nodes: List[dict],
    bayut_fr: FetchResult,
    comp_nodes: List[dict],
    comp_fr: FetchResult,
    comp_url: str
) -> Optional[dict]:
    # Only if competitor has a REAL FAQ
    if not page_has_real_faq(comp_fr, comp_nodes):
        return None

    comp_qs = extract_faq_questions(comp_fr, comp_nodes)
    comp_qs = [q for q in comp_qs if q and len(q) > 5]

    # If we can't parse topics, keep it neutral + human
    if not comp_qs:
        return {
            "Headers": "FAQs",
            "Description": "Competitor covers an FAQ section with additional questions.",
            "Source": source_link(comp_url),
        }

    bayut_has = page_has_real_faq(bayut_fr, bayut_nodes)
    bayut_qs = extract_faq_questions(bayut_fr, bayut_nodes) if bayut_has else []
    bayut_qs = [q for q in bayut_qs if q and len(q) > 5]

    def q_key(q: str) -> str:
        q2 = normalize_question(q)
        q2 = re.sub(r"[^a-z0-9\s]", "", q2.lower())
        q2 = re.sub(r"\s+", " ", q2).strip()
        return q2

    bayut_set = {q_key(q) for q in bayut_qs if q}

    # if Bayut has no FAQ or can't parse it, treat all competitor questions as "additional topics"
    if (not bayut_has) or (bayut_has and not bayut_qs):
        topics = faq_topics_from_questions_brief(comp_qs, limit=10)
        topic_list = format_gap_list(topics, limit=6)
        if topic_list:
            return {
                "Headers": "FAQs",
                "Description": f"Competitor covers additional FAQ topics: {topic_list}.",
                "Source": source_link(comp_url),
            }
        return {
            "Headers": "FAQs",
            "Description": "Competitor covers additional FAQ questions.",
            "Source": source_link(comp_url),
        }

    # Bayut has FAQ questions parsed => only topics for questions competitor has that Bayut doesn't
    missing_qs = [q for q in comp_qs if q_key(q) not in bayut_set]
    if not missing_qs:
        return None

    topics = faq_topics_from_questions_brief(missing_qs, limit=10)
    topic_list = format_gap_list(topics, limit=6)
    if topic_list:
        return {
            "Headers": "FAQs",
            "Description": f"Competitor covers additional FAQ topics: {topic_list}.",
            "Source": source_link(comp_url),
        }
    return {
        "Headers": "FAQs",
        "Description": "Competitor covers additional FAQ questions.",
        "Source": source_link(comp_url),
    }


# =====================================================
# SECTION EXTRACTION (HEADER-FIRST COMPARISON)
# =====================================================
def section_nodes(nodes: List[dict], levels=(2,3)) -> List[dict]:
    secs = []
    current_h2 = None
    for x in flatten(nodes):
        lvl = x["level"]
        h = strip_label(x.get("header",""))
        if not h or is_noise_header(h) or header_is_faq(h):
            continue
        if lvl == 2:
            current_h2 = h
        if lvl in levels:
            c = clean(x.get("content",""))
            secs.append({"level": lvl, "header": h, "content": c, "parent_h2": current_h2})

    seen = set()
    out = []
    for s in secs:
        k = norm_header(s["header"])
        if k in seen:
            continue
        seen.add(k)
        out.append(s)
    return out

def header_similarity(a: str, b: str) -> float:
    a_n = norm_header(a)
    b_n = norm_header(b)
    if not a_n or not b_n:
        return 0.0

    a_set = set(a_n.split())
    b_set = set(b_n.split())
    jacc = len(a_set & b_set) / max(len(a_set | b_set), 1) if a_set and b_set else 0.0
    seq = SequenceMatcher(None, a_n, b_n).ratio()
    return (0.55 * seq) + (0.45 * jacc)

def find_best_bayut_match(comp_header: str, bayut_sections: List[dict], min_score: float = 0.73) -> Optional[dict]:
    best = None
    best_score = 0.0
    for b in bayut_sections:
        sc = header_similarity(comp_header, b["header"])
        if sc > best_score:
            best_score = sc
            best = b
    if best and best_score >= min_score:
        return {"bayut_section": best, "score": best_score}
    return None

def dedupe_rows(rows: List[dict]) -> List[dict]:
    out = []
    seen = set()
    for r in rows:
        hk = norm_header(r.get("Headers", ""))
        sk = norm_header(re.sub(r"<[^>]+>", "", r.get("Source", "")))
        k = hk + "||" + sk
        if k in seen:
            continue
        seen.add(k)
        out.append(r)
    return out


# =====================================================
# ✅ NEW: CONTENT GAP DESCRIPTIONS (forced “Competitor covers …”)
# =====================================================
def theme_flags(text: str) -> set:
    t = (text or "").lower()
    flags = set()

    def has_any(words: List[str]) -> bool:
        return any(w in t for w in words)

    if has_any(["metro", "public transport", "commute", "connectivity", "access", "highway", "roads", "bus", "train", "monorail", "tram"]):
        flags.add("transport")
    if has_any(["parking", "traffic", "congestion", "rush hour", "gridlock"]):
        flags.add("traffic_parking")
    if has_any(["cost", "price", "pricing", "expensive", "afford", "budget", "rent", "fees", "charges"]):
        flags.add("cost")
    if has_any(["restaurants", "cafes", "nightlife", "vibe", "atmosphere", "social", "entertainment"]):
        flags.add("lifestyle")
    if has_any(["schools", "nursery", "kids", "family", "clinic", "hospital", "supermarket", "groceries", "pharmacy"]):
        flags.add("daily_life")
    if has_any(["safe", "safety", "security", "crime"]):
        flags.add("safety")
    if has_any(["pros", "cons", "advantages", "disadvantages", "weigh", "consider", "should you", "worth it"]):
        flags.add("decision_frame")
    if has_any(["compare", "comparison", "vs ", "versus", "alternative", "similar to"]):
        flags.add("comparison")

    return flags

def _human_theme_labels(flags: List[str], limit: int = 3) -> List[str]:
    human_map = {
        "transport": "commute & connectivity",
        "traffic_parking": "traffic/parking realities",
        "cost": "cost details",
        "lifestyle": "lifestyle & vibe",
        "daily_life": "day-to-day essentials",
        "safety": "safety/security",
        "decision_frame": "decision guidance",
        "comparison": "comparison context",
    }
    out = []
    for f in flags:
        lab = human_map.get(f, f)
        if lab not in out:
            out.append(lab)
        if len(out) >= limit:
            break
    return out

def summarize_missing_section_action(header: str, subheaders: Optional[List[str]], comp_content: str) -> str:
    # forced neutral phrasing
    bits = []

    if subheaders:
        sub_list = format_gap_list(subheaders, limit=4)
        if sub_list:
            bits.append(sub_list)

    flags = list(theme_flags(comp_content))
    theme_list = format_gap_list(_human_theme_labels(flags, limit=3), limit=3)
    if theme_list:
        bits.append(theme_list)

    if bits:
        # keep it one short sentence
        return f"Competitor covers {bits[0]}." if len(bits) == 1 else f"Competitor covers {bits[0]} and {bits[1]}."
    return "Competitor covers this section in more detail."

def summarize_content_gap_action(comp_content: str, bayut_content: str) -> str:
    c_txt = clean(comp_content or "")
    b_txt = clean(bayut_content or "")

    comp_flags = theme_flags(c_txt)
    bayut_flags = theme_flags(b_txt)

    extra = list(comp_flags - bayut_flags)
    extra_labels = _human_theme_labels(extra, limit=3)
    extra_list = format_gap_list(extra_labels, limit=3)

    if extra_list:
        return f"Competitor covers more detail on {extra_list}."
    return "Competitor covers more practical detail in this section."


# =====================================================
# UPDATE MODE ENGINE (Headers | Description | Source)
# =====================================================
def update_mode_rows_header_first(
    bayut_nodes: List[dict],
    bayut_fr: FetchResult,
    comp_nodes: List[dict],
    comp_fr: FetchResult,
    comp_url: str,
    max_missing_headers: Optional[int] = None
) -> List[dict]:
    rows_map: Dict[str, dict] = {}
    source = source_link(comp_url)

    def add_row(header: str, parts: List[str]):
        if not header or not parts:
            return
        key = norm_header(header) + "||" + norm_header(re.sub(r"<[^>]+>", "", source))
        if key not in rows_map:
            rows_map[key] = {"Headers": header, "DescriptionParts": [], "Source": source}
        for p in parts:
            p = clean(p)
            if not p:
                continue
            if not p.endswith("."):
                p = p + "."
            if p not in rows_map[key]["DescriptionParts"]:
                rows_map[key]["DescriptionParts"].append(p)

    def children_map(h3_list: List[dict]) -> Dict[str, List[dict]]:
        cmap: Dict[str, List[dict]] = {}
        for h3 in h3_list:
            parent = h3.get("parent_h2") or ""
            pk = norm_header(parent)
            if not pk:
                continue
            cmap.setdefault(pk, []).append(h3)
        return cmap

    def child_headers(cmap: Dict[str, List[dict]], parent_header: str) -> List[str]:
        pk = norm_header(parent_header)
        return [c.get("header", "") for c in cmap.get(pk, [])]

    def combined_h2_content(h2_header: str, h2_list: List[dict], cmap: Dict[str, List[dict]]) -> str:
        pk = norm_header(h2_header)
        h2_content = ""
        for h2 in h2_list:
            if norm_header(h2.get("header", "")) == pk:
                h2_content = h2.get("content", "")
                break
        child_content = " ".join(c.get("content", "") for c in cmap.get(pk, []))
        return clean(" ".join([h2_content, child_content]))

    def missing_children(comp_children: List[str], bayut_children: List[str]) -> List[str]:
        missing = []
        for ch in comp_children:
            if not any(header_similarity(ch, bh) >= 0.73 for bh in bayut_children):
                missing.append(ch)
        return missing

    def depth_gap_summary(comp_text: str, bayut_text: str) -> str:
        c_txt = clean(comp_text or "")
        b_txt = clean(bayut_text or "")
        if len(c_txt) < 160:
            return ""
        # only trigger when competitor meaningfully longer OR has extra themes
        if len(c_txt) < (1.25 * max(len(b_txt), 1)):
            # still allow if competitor introduces new theme signals
            if len(theme_flags(c_txt) - theme_flags(b_txt)) == 0:
                return ""
        return summarize_content_gap_action(c_txt, b_txt)

    bayut_secs = section_nodes(bayut_nodes, levels=(2, 3))
    comp_secs = section_nodes(comp_nodes, levels=(2, 3))

    bayut_h2 = [s for s in bayut_secs if s["level"] == 2]
    bayut_h3 = [s for s in bayut_secs if s["level"] == 3]
    comp_h2 = [s for s in comp_secs if s["level"] == 2]
    comp_h3 = [s for s in comp_secs if s["level"] == 3]

    bayut_children_map = children_map(bayut_h3)
    comp_children_map = children_map(comp_h3)

    # H2: missing headers OR missing content inside existing headers
    for cs in comp_h2:
        comp_header = cs.get("header", "")
        comp_children = child_headers(comp_children_map, comp_header)
        comp_text = combined_h2_content(comp_header, comp_h2, comp_children_map) or cs.get("content", "")

        m = find_best_bayut_match(comp_header, bayut_h2, min_score=0.73)
        if not m:
            # Missing header on Bayut => row must exist
            desc = summarize_missing_section_action(comp_header, comp_children, comp_text)
            add_row(comp_header, [desc])
            continue

        bayut_header = m["bayut_section"]["header"]
        bayut_children = child_headers(bayut_children_map, bayut_header)
        missing_sub = missing_children(comp_children, bayut_children)

        parts = []
        if missing_sub:
            sub_list = format_gap_list(missing_sub, limit=6)
            if sub_list:
                parts.append(f"Competitor covers subtopics like {sub_list}.")

        bayut_text = combined_h2_content(bayut_header, bayut_h2, bayut_children_map)
        depth_note = depth_gap_summary(comp_text, bayut_text)
        if depth_note:
            parts.append(depth_note)

        # only output when there is missing content inside the existing/similar section
        if parts:
            add_row(comp_header, parts)

    # orphan H3s: only if competitor has H3 that doesn't sit under a meaningful H2, and Bayut lacks an equivalent
    comp_h2_norms = {norm_header(h.get("header", "")) for h in comp_h2}
    for cs in comp_h3:
        parent = cs.get("parent_h2") or ""
        if parent and norm_header(parent) in comp_h2_norms:
            continue
        m = find_best_bayut_match(cs["header"], bayut_h3 + bayut_h2, min_score=0.73)
        if m:
            continue
        desc = summarize_missing_section_action(cs["header"], None, cs.get("content", ""))
        add_row(cs["header"], [desc])

    rows = []
    for r in rows_map.values():
        desc = " ".join(r.get("DescriptionParts", [])).strip()
        rows.append({"Headers": r.get("Headers", ""), "Description": desc, "Source": r.get("Source", "")})

    if max_missing_headers and len(rows) > max_missing_headers:
        rows = rows[:max_missing_headers]

    # FAQs: always “Competitor covers … topics: X”
    faq_row = missing_faqs_row(bayut_nodes, bayut_fr, comp_nodes, comp_fr, comp_url)
    if faq_row:
        rows.append(faq_row)

    return dedupe_rows(rows)
    # app.py (PART 2/2)
# (Continue directly after PART 1/2 — do NOT change anything above)

# =====================================================
# TABLE RENDERING (keeps links + safe text)
# =====================================================
def _safe_html(s: str) -> str:
    return html_lib.escape((s or "").strip())

def render_table(df: pd.DataFrame):
    if df is None or df.empty:
        st.info("No gaps detected for the selected URLs.")
        return

    # Escape user/page text, keep Source as clickable link (already formed by source_link()).
    safe_df = df.copy()
    if "Headers" in safe_df.columns:
        safe_df["Headers"] = safe_df["Headers"].apply(_safe_html)
    if "Description" in safe_df.columns:
        safe_df["Description"] = safe_df["Description"].apply(_safe_html)
    # Source contains <a href=...> from source_link(), keep as-is
    if "Source" in safe_df.columns:
        safe_df["Source"] = safe_df["Source"].astype(str)

    html = safe_df.to_html(index=False, escape=False)
    st.markdown(html, unsafe_allow_html=True)


# =====================================================
# LIGHT URL NORMALIZATION
# =====================================================
def normalize_url(u: str) -> str:
    u = (u or "").strip()
    if not u:
        return ""
    if u.startswith("http://") or u.startswith("https://"):
        return u
    return "https://" + u

def split_urls(text: str) -> List[str]:
    urls = []
    for line in (text or "").splitlines():
        line = line.strip()
        if not line:
            continue
        # allow comma separated too
        parts = [p.strip() for p in re.split(r"[,\s]+", line) if p.strip()]
        for p in parts:
            pu = normalize_url(p)
            if pu:
                urls.append(pu)

    # keep order, de-dupe
    out = []
    seen = set()
    for u in urls:
        k = u.strip().lower()
        if k in seen:
            continue
        seen.add(k)
        out.append(u)
    return out


# =====================================================
# UI
# =====================================================
# Mode (single mode now, design kept)
st.markdown('<div class="mode-wrap">', unsafe_allow_html=True)
colA, colB, colC = st.columns([1, 1, 1])
with colB:
    st.button("Update Mode", disabled=True)  # visual only
st.markdown("</div>", unsafe_allow_html=True)
st.markdown('<div class="mode-note">Analyzes competitor articles to surface missing and under-covered sections.</div>', unsafe_allow_html=True)

st.markdown('<div class="section-pill section-pill-tight">Inputs</div>', unsafe_allow_html=True)

bayut_url = st.text_input("Bayut URL", placeholder="Paste Bayut article URL…", key="bayut_url_input")

competitors_text = st.text_area(
    "Competitor URLs (one per line)",
    placeholder="Paste competitor URLs (one per line)…",
    height=140,
    key="competitors_urls_input",
)

optional_fkw = st.text_input(
    "Optional: Focus Keyword",
    placeholder="Optional — for future keyword usage / relevance checks",
    key="optional_fkw_input",
)

run = st.button("Content Gaps Table", type="primary")


# =====================================================
# RUN
# =====================================================
if run:
    bayut_url_n = normalize_url(bayut_url)
    comp_urls = split_urls(competitors_text)

    if not bayut_url_n:
        st.error("Please provide a Bayut URL.")
        st.stop()

    if not comp_urls:
        st.error("Please provide at least one competitor URL.")
        st.stop()

    # =====================================================
    # FETCH ALL (hard gate: no missing URLs)
    # =====================================================
    st.markdown('<div class="section-pill section-pill-tight">Fetching</div>', unsafe_allow_html=True)

    all_urls = [bayut_url_n] + comp_urls
    fr_map = resolve_all_or_require_manual(agent, all_urls, st_key_prefix="fetch_all")

    # =====================================================
    # HEADINGS EXTRACT (hard gate: no missing headings)
    # =====================================================
    st.markdown('<div class="section-pill section-pill-tight">Extracting headings</div>', unsafe_allow_html=True)

    tree_map = ensure_headings_or_require_repaste(all_urls, fr_map, st_key_prefix="trees_all")

    bayut_fr = fr_map[bayut_url_n]
    bayut_tree = tree_map[bayut_url_n]
    bayut_nodes = bayut_tree.get("nodes", [])

    # =====================================================
    # BUILD CONTENT GAPS TABLE
    # =====================================================
    st.markdown('<div class="section-pill section-pill-tight">Content Gaps Table</div>', unsafe_allow_html=True)

    all_rows: List[dict] = []

    for cu in comp_urls:
        comp_fr = fr_map[cu]
        comp_tree = tree_map[cu]
        comp_nodes = comp_tree.get("nodes", [])

        rows = update_mode_rows_header_first(
            bayut_nodes=bayut_nodes,
            bayut_fr=bayut_fr,
            comp_nodes=comp_nodes,
            comp_fr=comp_fr,
            comp_url=cu,
            max_missing_headers=None
        )
        all_rows.extend(rows)

    # de-dupe across competitors
    all_rows = dedupe_rows(all_rows)

    # final ordering
    df = pd.DataFrame(all_rows)
    if not df.empty:
        # Ensure columns exist in order
        for c in ["Headers", "Description", "Source"]:
            if c not in df.columns:
                df[c] = ""
        df = df[["Headers", "Description", "Source"]]

        # Sort: FAQs last, then alphabetical (stable)
        df["_faq"] = df["Headers"].apply(lambda x: 1 if norm_header(str(x)) == "faqs" else 0)
        df["_h"] = df["Headers"].apply(lambda x: norm_header(str(x)))
        df = df.sort_values(by=["_faq", "_h"], ascending=[True, True]).drop(columns=["_faq", "_h"])

    render_table(df)

    # =====================================================
    # OPTIONAL DEBUG (safe + helpful, collapsed)
    # =====================================================
    with st.expander("Debug (optional)", expanded=False):
        st.markdown("**Bayut**")
        st.write(f"Fetch source: {bayut_fr.source} | status: {bayut_fr.status}")
        st.write(f"H1/H2 preview: {get_first_h1(bayut_nodes)}")
        st.write(f"Real FAQ detected: {page_has_real_faq(bayut_fr, bayut_nodes)}")
        if page_has_real_faq(bayut_fr, bayut_nodes):
            bqs = extract_faq_questions(bayut_fr, bayut_nodes)
            st.write(f"FAQ questions parsed: {len(bqs)}")
            if bqs:
                st.write("FAQ topics (brief): " + ", ".join(faq_topics_from_questions_brief(bqs, limit=10)))

        st.markdown("---")
        st.markdown("**Competitors**")
        for cu in comp_urls:
            cfr = fr_map[cu]
            cnodes = tree_map[cu].get("nodes", [])
            st.write(f"- {cu}")
            st.write(f"  Fetch source: {cfr.source} | status: {cfr.status}")
            st.write(f"  H1/H2 preview: {get_first_h1(cnodes)}")
            st.write(f"  Real FAQ detected: {page_has_real_faq(cfr, cnodes)}")
            if page_has_real_faq(cfr, cnodes):
                cqs = extract_faq_questions(cfr, cnodes)
                st.write(f"  FAQ questions parsed: {len(cqs)}")
                if cqs:
                    st.write("  FAQ topics (brief): " + ", ".join(faq_topics_from_questions_brief(cqs, limit=10)))
            st.markdown("")


