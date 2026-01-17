# app.py
import streamlit as st
import requests
import re
from bs4 import BeautifulSoup
from urllib.parse import quote_plus, urlparse
import pandas as pd
import time, random, hashlib, json
from dataclasses import dataclass
from typing import Optional, Dict, Tuple, List
from difflib import SequenceMatcher

# Optional (recommended): JS rendering tool
# pip install playwright
# playwright install chromium
try:
    from playwright.sync_api import sync_playwright
    PLAYWRIGHT_OK = True
except Exception:
    PLAYWRIGHT_OK = False

# Optional: Google Search Console for ranking
try:
    from google.oauth2 import service_account
    from googleapiclient.discovery import build as gbuild
    GSC_OK = True
except Exception:
    GSC_OK = False


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
        padding-top: 1.2rem !important;
        padding-bottom: 2.2rem !important;
      }}
      .hero {{
        text-align:center;
        margin-top: 0.6rem;
        margin-bottom: 1.0rem;
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
        margin: 6px 0 6px 0;
      }}
      .tight {{
        margin-top: 4px !important;
        margin-bottom: 6px !important;
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
      }}
      thead th {{
        background: {LIGHT_GREEN} !important;
        text-align: center !important;
        font-weight: 900 !important;
        color: {TEXT_DARK} !important;
        padding: 12px 10px !important;
        border-bottom: 1px solid #E5E7EB !important;
      }}
      tbody td {{
        vertical-align: top !important;
        padding: 12px 10px !important;
        border-bottom: 1px solid #F1F5F9 !important;
        color: {TEXT_DARK} !important;
        font-size: 14px !important;
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
      .ai-summary {{
        background: white;
        border: 1px solid #E5E7EB;
        border-radius: 14px;
        padding: 14px 14px;
        margin: 8px 0 10px 0;
      }}
      .muted {{
        color:#6B7280;
        font-size: 13px;
      }}
      .small-note {{
        color:#6B7280;
        font-size: 12px;
        margin-top: -6px;
      }}
      .btn-row {{
        margin-top: 0px !important;
        margin-bottom: 0px !important;
      }}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    f"""
    <div class="hero">
      <h1><span class="bayut">Bayut</span> Competitor Gap Analysis</h1>
      <p><b>Header-first</b> gap logic — Missing headers first, then <b>(missing parts)</b> under matching headers. FAQs are <b>one row</b> (topics only).</p>
    </div>
    """,
    unsafe_allow_html=True,
)


# =====================================================
# SMALL UI HELPERS (HEADER + BUTTON SAME LINE)
# =====================================================
def section_title_with_button(title: str, button_label: str, btn_key: str, show_key: str) -> bool:
    if show_key not in st.session_state:
        st.session_state[show_key] = False

    left, right = st.columns([4.2, 1.3], gap="small")
    with left:
        st.markdown(f"<div class='section-pill tight'>{title}</div>", unsafe_allow_html=True)
    with right:
        pressed = st.button(button_label, type="secondary", use_container_width=True, key=btn_key)
        if pressed:
            st.session_state[show_key] = True
    return bool(st.session_state.get(show_key))


def render_table(df: pd.DataFrame):
    if df is None or df.empty:
        st.info("No results to show.")
        return
    html = df.to_html(index=False, escape=False)
    st.markdown(html, unsafe_allow_html=True)


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

GENERIC_SECTION_HEADERS = {
    "introduction", "overview",
    # user wants conclusion visible, so don't add them here
}

STOP = {
    "the","and","for","with","that","this","from","you","your","are","was","were","will","have","has","had",
    "but","not","can","may","more","most","into","than","then","they","them","their","our","out","about",
    "also","over","under","between","within","near","where","when","what","why","how","who","which",
    "a","an","to","of","in","on","at","as","is","it","be","or","by","we","i","us"
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

    # If fr.html exists, always use it
    if fr.html:
        nodes = build_tree_from_html(fr.html)
    # If manual paste looks like HTML, use HTML parser
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

def domain_of(url: str) -> str:
    try:
        host = urlparse(url).netloc.lower().replace("www.", "")
        return host.split(":")[0]
    except Exception:
        return ""

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


# =====================================================
# STRICT FAQ DETECTION (only when competitor truly has FAQs)
# =====================================================
FAQ_TITLES = {
    "faq",
    "faqs",
    "frequently asked questions",
    "frequently asked question",
}

def header_is_faq(header: str) -> bool:
    nh = norm_header(header)
    return nh in FAQ_TITLES

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

def extract_questions_from_node(node: dict) -> List[str]:
    qs: List[str] = []

    def add_from_text_block(txt: str):
        txt = clean(txt or "")
        if not txt:
            return
        chunks = re.split(r"[\n\r]+|(?<=[\.\?\!])\s+", txt)
        for ch in chunks[:120]:
            ch = clean(ch)
            if not ch or len(ch) > 160:
                continue
            if _looks_like_question(ch):
                qs.append(normalize_question(ch))

    add_from_text_block(node.get("content", ""))

    def walk(n: dict):
        for c in n.get("children", []):
            hdr = clean(c.get("header", ""))
            if hdr and not is_noise_header(hdr) and _looks_like_question(hdr):
                qs.append(normalize_question(hdr))
            add_from_text_block(c.get("content", ""))
            walk(c)

    walk(node)

    seen = set()
    out = []
    for q in qs:
        k = norm_header(q)
        if not k or k in seen:
            continue
        seen.add(k)
        out.append(q)
    return out[:25]

def find_faq_nodes(nodes: List[dict]) -> List[dict]:
    faq = []
    for x in flatten(nodes):
        if x.get("level") not in (2, 3):
            continue
        if not header_is_faq(x.get("header", "")):
            continue
        qs = extract_questions_from_node(x)
        if len(qs) >= 3:
            faq.append(x)
    return faq

def extract_all_questions(nodes: List[dict]) -> List[str]:
    qs = []
    for x in flatten(nodes):
        h = clean(x.get("header", ""))
        if not h or is_noise_header(h):
            continue
        if _looks_like_question(h):
            qs.append(normalize_question(h))
    seen = set()
    out = []
    for q in qs:
        k = norm_header(q)
        if k in seen:
            continue
        seen.add(k)
        out.append(q)
    return out[:30]

def faq_subject(q: str) -> str:
    s = norm_header(normalize_question(q))
    if any(k in s for k in ["how much", "cost", "price", "pricing", "fees", "budget"]):
        return "Pricing / cost"
    if any(k in s for k in ["where", "located", "location", "distance", "near", "how to get", "map"]):
        return "Location / nearby"
    if any(k in s for k in ["who is it for", "who should", "is it for", "suitable", "best for"]):
        return "Who it suits"
    if any(k in s for k in ["pros", "cons", "advantages", "disadvantages", "worth it"]):
        return "Decision help (pros/cons)"
    if any(k in s for k in ["safe", "safety", "secure"]):
        return "Safety"
    if any(k in s for k in ["school", "education", "kids", "family"]):
        return "Family / education"
    if any(k in s for k in ["transport", "metro", "bus", "commute", "traffic", "parking"]):
        return "Transport / traffic / parking"
    if any(k in s for k in ["restaurants", "cafes", "nightlife", "things to do", "attractions", "lifestyle"]):
        return "Lifestyle / things to do"
    return "Other FAQ topics"

def faq_subjects_from_questions(questions: List[str], limit: int = 10) -> List[str]:
    out: List[str] = []
    seen = set()
    for q in questions:
        subj = faq_subject(q)
        k = norm_header(subj)
        if k in seen:
            continue
        seen.add(k)
        out.append(subj)
        if len(out) >= limit:
            break
    return out

def missing_faqs_row(bayut_nodes: List[dict], comp_nodes: List[dict], comp_url: str) -> Optional[dict]:
    comp_faq_nodes = find_faq_nodes(comp_nodes)
    if not comp_faq_nodes:
        return None

    comp_qs = []
    for fn in comp_faq_nodes:
        comp_qs.extend(extract_questions_from_node(fn))
    if not comp_qs:
        comp_qs = extract_all_questions(comp_nodes)

    comp_qs = [q for q in comp_qs if q and len(q) > 5]
    if len(comp_qs) < 3:
        return None

    bayut_faq_nodes = find_faq_nodes(bayut_nodes)
    bayut_qs = []
    for fn in bayut_faq_nodes:
        bayut_qs.extend(extract_questions_from_node(fn))
    bayut_qs = [q for q in bayut_qs if q and len(q) > 5]

    def q_key(q: str) -> str:
        q2 = normalize_question(q)
        q2 = re.sub(r"[^a-z0-9\s]", "", q2.lower())
        q2 = re.sub(r"\s+", " ", q2).strip()
        return q2

    bayut_set = {q_key(q) for q in bayut_qs if q}

    if not bayut_qs:
        topics = faq_subjects_from_questions(comp_qs, limit=10)
        return {
            "Headers": "FAQs",
            "Description": "Competitor has an FAQ section covering topics such as: " + ", ".join(topics) + ".",
            "Source": source_link(comp_url),
        }

    missing_qs = [q for q in comp_qs if q_key(q) not in bayut_set]
    if not missing_qs:
        return None

    topics = faq_subjects_from_questions(missing_qs, limit=10)
    return {
        "Headers": "FAQs",
        "Description": "Missing FAQ topics: " + ", ".join(topics) + ".",
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
# CONTENT GAP THEMES (NEUTRAL DESCRIPTION)
# =====================================================
def theme_flags(text: str) -> set:
    t = (text or "").lower()
    flags = set()

    def has_any(words: List[str]) -> bool:
        return any(w in t for w in words)

    if has_any(["metro", "public transport", "commute", "connectivity", "access", "highway", "roads", "bus", "train"]):
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

def summarize_missing_section_action(header: str, subheaders: Optional[List[str]], comp_content: str) -> str:
    hn = norm_header(header)

    if ("importance" in hn and "pros" in hn and "cons" in hn) or ("consider" in hn and "pros" in hn and "cons" in hn):
        return "Competitor includes decision framing on how to weigh pros vs cons before concluding."

    if "comparison" in hn or "compare" in hn or "vs" in hn or "versus" in hn:
        if subheaders:
            hint = ", ".join(subheaders[:3])
            return f"Competitor includes a comparison section and breaks it into alternatives such as: {hint}."
        return "Competitor includes a comparison section explaining alternatives and how they differ."

    themes = list(theme_flags(comp_content))
    human_map = {
        "transport": "commute & connectivity",
        "traffic_parking": "traffic/parking realities",
        "cost": "cost considerations",
        "lifestyle": "lifestyle & vibe",
        "daily_life": "day-to-day convenience",
        "safety": "safety angle",
        "decision_frame": "decision framing",
        "comparison": "comparison context",
    }
    picks = [human_map.get(x, x) for x in themes][:3]
    if picks:
        return f"Competitor covers this as a dedicated section with practical details (e.g., {', '.join(picks)})."
    return "Competitor covers this as a dedicated section with extra context and practical specifics."

def summarize_content_gap_action(header: str, comp_content: str, bayut_content: str) -> str:
    comp_flags = theme_flags(comp_content)
    bayut_flags = theme_flags(bayut_content)
    missing = list(comp_flags - bayut_flags)

    human_map = {
        "transport": "commute & connectivity",
        "traffic_parking": "traffic/parking realities",
        "cost": "cost considerations",
        "lifestyle": "lifestyle & vibe",
        "daily_life": "day-to-day convenience",
        "safety": "safety angle",
        "decision_frame": "decision framing",
        "comparison": "comparison context",
    }
    missing_human = [human_map.get(x, x) for x in missing][:3]
    if missing_human:
        return "Competitor goes deeper on: " + ", ".join(missing_human) + "."
    return "Competitor provides more depth and practical specifics than Bayut under the same header."


# =====================================================
# UPDATE MODE ENGINE (Headers | Description | Source)
# =====================================================
def update_mode_rows_header_first(
    bayut_nodes: List[dict],
    comp_nodes: List[dict],
    comp_url: str,
    max_missing_headers: int = 7,
    max_missing_parts: int = 5
) -> List[dict]:
    rows: List[dict] = []

    bayut_secs = section_nodes(bayut_nodes, levels=(2, 3))
    comp_secs = section_nodes(comp_nodes, levels=(2, 3))

    bayut_h2 = [s for s in bayut_secs if s["level"] == 2]
    bayut_h3 = [s for s in bayut_secs if s["level"] == 3]
    comp_h2 = [s for s in comp_secs if s["level"] == 2]
    comp_h3 = [s for s in comp_secs if s["level"] == 3]

    # Rank competitor H2 by importance (content length + #children)
    comp_h2_ranked = []
    for h2 in comp_h2:
        child_count = sum(
            1 for h3 in comp_h3
            if norm_header(h3.get("parent_h2") or "") == norm_header(h2["header"])
        )
        score = len(clean(h2.get("content", ""))) + (child_count * 120)
        comp_h2_ranked.append((score, h2))
    comp_h2_ranked.sort(key=lambda x: x[0], reverse=True)

    missing_h2_norms = set()

    # 1) Missing H2
    missing_rows = []
    for _, cs in comp_h2_ranked:
        m = find_best_bayut_match(cs["header"], bayut_h2, min_score=0.73)
        if m:
            continue

        missing_h2_norms.add(norm_header(cs["header"]))

        children = []
        for h3 in comp_h3:
            if norm_header(h3.get("parent_h2") or "") == norm_header(cs["header"]):
                children.append(h3["header"])

        comp_text = (cs.get("content", "") or "")
        desc = summarize_missing_section_action(cs["header"], children, comp_text)

        if children:
            hint = ", ".join(children[:3])
            desc = desc + f" (Breakdown: {hint}.)"

        missing_rows.append({
            "Headers": cs["header"],
            "Description": desc,
            "Source": source_link(comp_url),
        })

        if len(missing_rows) >= max_missing_headers:
            break

    rows.extend(missing_rows)

    # 2) Missing H3 only if parent H2 is NOT missing
    if len(rows) < max_missing_headers:
        for cs in comp_h3:
            parent = cs.get("parent_h2") or ""
            if parent and norm_header(parent) in missing_h2_norms:
                continue

            if parent:
                parent_match = find_best_bayut_match(parent, bayut_h2, min_score=0.73)
                if not parent_match:
                    continue

            m = find_best_bayut_match(cs["header"], bayut_h3, min_score=0.73)
            if m:
                continue

            label = f"{parent} → {cs['header']}" if parent else cs["header"]
            rows.append({
                "Headers": label,
                "Description": summarize_missing_section_action(cs["header"], None, cs.get("content", "")),
                "Source": source_link(comp_url),
            })

            if len(rows) >= max_missing_headers:
                break

    # 3) Matching headers where competitor is stronger => (missing parts)
    missing_parts_rows = []
    for cs in comp_secs:
        m = find_best_bayut_match(cs["header"], bayut_secs, min_score=0.73)
        if not m:
            continue

        bs = m["bayut_section"]
        c_txt = clean(cs.get("content", ""))
        b_txt = clean(bs.get("content", ""))

        if len(c_txt) < 140:
            continue
        if len(c_txt) < (1.30 * max(len(b_txt), 1)):
            continue

        comp_flags = theme_flags(c_txt)
        bayut_flags = theme_flags(b_txt)
        if len(comp_flags - bayut_flags) < 1 and len(c_txt) < 650:
            continue

        missing_parts_rows.append({
            "Headers": f"{bs['header']} (missing parts)",
            "Description": summarize_content_gap_action(bs["header"], c_txt, b_txt),
            "Source": source_link(comp_url),
        })

        if len(missing_parts_rows) >= max_missing_parts:
            break

    rows.extend(missing_parts_rows)

    # 4) FAQs — ONE row only (only when competitor truly has FAQ heading)
    faq_row = missing_faqs_row(bayut_nodes, comp_nodes, comp_url)
    if faq_row:
        rows.append(faq_row)

    return dedupe_rows(rows)


# =====================================================
# SEO HELPERS (TRUE header counts from HTML tags only)
# =====================================================
def url_slug(url: str) -> str:
    try:
        p = urlparse(url).path.strip("/")
        return "/" + p if p else "/"
    except Exception:
        return "/"

def extract_head_seo(html: str) -> Tuple[str, str]:
    if not html:
        return ("Not available", "Not available")
    soup = BeautifulSoup(html, "html.parser")

    title = ""
    t = soup.find("title")
    if t:
        title = clean(t.get_text(" "))

    desc = ""
    md = soup.find("meta", attrs={"name": re.compile("^description$", re.I)})
    if md and md.get("content"):
        desc = clean(md.get("content"))

    return (title or "Not available", desc or "Not available")

def extract_media_used(html: str) -> str:
    if not html:
        return "Not available"
    soup = BeautifulSoup(html, "html.parser")
    for t in soup.find_all(list(IGNORE_TAGS)):
        t.decompose()
    root = soup.find("article") or soup

    imgs = len(root.find_all("img"))
    videos = len(root.find_all("video"))
    iframes = len(root.find_all("iframe"))
    audios = len(root.find_all("audio"))
    tables = len(root.find_all("table"))

    parts = []
    if imgs: parts.append(f"Images: {imgs}")
    if videos: parts.append(f"Videos: {videos}")
    if iframes: parts.append(f"Iframes: {iframes}")
    if audios: parts.append(f"Audio: {audios}")
    if tables: parts.append(f"Tables: {tables}")

    return " | ".join(parts) if parts else "None detected"

def count_headers_from_html(html: str) -> Dict[str, int]:
    """
    TRUE counts (only if HTML exists). Uses real H1/H2/H3/H4 tags inside <article> if present.
    """
    counts = {"H1": 0, "H2": 0, "H3": 0, "H4": 0}
    if not html:
        return counts
    soup = BeautifulSoup(html, "html.parser")
    for t in soup.find_all(list(IGNORE_TAGS)):
        t.decompose()
    root = soup.find("article") or soup

    # Only count headings that look non-noise (so the number stays meaningful)
    for tag, key in [("h1","H1"),("h2","H2"),("h3","H3"),("h4","H4")]:
        hs = root.find_all(tag)
        kept = 0
        for h in hs:
            txt = clean(h.get_text(" "))
            if txt and not is_noise_header(txt):
                kept += 1
        counts[key] = kept
    return counts

def get_first_h1(nodes: List[dict]) -> str:
    for x in flatten(nodes):
        if x.get("level") == 1:
            h = clean(x.get("header",""))
            if h:
                return h
    return ""

def headings_blob(nodes: List[dict]) -> str:
    hs = []
    for x in flatten(nodes):
        if x.get("level") in (1,2,3,4):
            h = clean(x.get("header",""))
            if h and not is_noise_header(h):
                hs.append(h)
    return " ".join(hs[:80])

def kw_in_text_count(text: str, phrase: str) -> int:
    if not text or not phrase:
        return 0
    t = " " + re.sub(r"\s+", " ", (text or "").lower()) + " "
    p = " " + re.sub(r"\s+", " ", (phrase or "").lower()) + " "
    return t.count(p)

def kw_present(text: str, phrase: str) -> str:
    if not text or not phrase:
        return "No"
    return "Yes" if kw_in_text_count(text, phrase) > 0 else "No"


# =====================================================
# GOOGLE SEARCH CONSOLE: UAE ranking (desktop + mobile)
# =====================================================
def init_gsc_service() -> Tuple[Optional[object], str]:
    """
    Uses env vars:
      - GSC_SERVICE_ACCOUNT_JSON: full service account json as a string
      - GSC_SITE_URL: e.g. "sc-domain:bayut.com" OR "https://www.bayut.com/"
    """
    sa_json = (st.secrets.get("GSC_SERVICE_ACCOUNT_JSON", None) if hasattr(st, "secrets") else None)  # optional
    site_url = (st.secrets.get("GSC_SITE_URL", None) if hasattr(st, "secrets") else None)

    # allow env too
    import os
    if not sa_json:
        sa_json = os.environ.get("GSC_SERVICE_ACCOUNT_JSON", "")
    if not site_url:
        site_url = os.environ.get("GSC_SITE_URL", "")

    if not GSC_OK:
        return (None, "")
    if not sa_json or not site_url:
        return (None, "")

    try:
        info = json.loads(sa_json) if isinstance(sa_json, str) else sa_json
        creds = service_account.Credentials.from_service_account_info(
            info,
            scopes=["https://www.googleapis.com/auth/webmasters.readonly"],
        )
        service = gbuild("searchconsole", "v1", credentials=creds, cache_discovery=False)
        return (service, site_url)
    except Exception:
        return (None, "")

def gsc_avg_position_for_page(
    service,
    site_url: str,
    page_url: str,
    days: int,
    country_iso3: str,
    device: str,
    query_filter: Optional[str] = None
) -> str:
    """
    Returns average position (float) for the page in a country+device over last N days.
    If query_filter provided, attempts to filter to that query too (best effort).
    """
    if not service or not site_url or not page_url:
        return "Not available"

    # dates
    import datetime as dt
    end = dt.date.today()
    start = end - dt.timedelta(days=max(days, 7))

    # Build request
    req = {
        "startDate": start.isoformat(),
        "endDate": end.isoformat(),
        "dimensions": ["page"],
        "rowLimit": 50,
        "dimensionFilterGroups": [{
            "filters": [
                {"dimension": "page", "operator": "equals", "expression": page_url},
                {"dimension": "country", "operator": "equals", "expression": country_iso3},
                {"dimension": "device", "operator": "equals", "expression": device},
            ]
        }]
    }

    # best-effort query filter (API may accept it even if dimension not included)
    if query_filter and clean(query_filter):
        req["dimensionFilterGroups"][0]["filters"].append(
            {"dimension": "query", "operator": "equals", "expression": clean(query_filter)}
        )

    try:
        resp = service.searchanalytics().query(siteUrl=site_url, body=req).execute()
        rows = resp.get("rows", [])
        if not rows:
            return "Not available"
        # For a single page, aggregate: use weighted avg position if clicks/impressions present
        impressions = 0
        weighted = 0.0
        for r in rows:
            imp = int(r.get("impressions", 0) or 0)
            pos = float(r.get("position", 0.0) or 0.0)
            impressions += imp
            weighted += pos * imp
        if impressions > 0:
            return f"{(weighted / impressions):.2f}"
        # fallback: plain avg of positions
        poss = [float(r.get("position", 0.0) or 0.0) for r in rows]
        poss = [p for p in poss if p > 0]
        return f"{(sum(poss)/len(poss)):.2f}" if poss else "Not available"
    except Exception:
        return "Not available"


# =====================================================
# AI VISIBILITY (SERP API - optional)
# =====================================================
def serpapi_ai_overview_status(query: str, device: str, gl: str = "ae", hl: str = "en") -> dict:
    """
    Optional: fills AI Overview presence + citations using SerpApi.
    Requires env/secret: SERPAPI_KEY
    """
    import os
    key = (st.secrets.get("SERPAPI_KEY", None) if hasattr(st, "secrets") else None)
    if not key:
        key = os.environ.get("SERPAPI_KEY", "")

    if not key or not query:
        return {"ok": False, "reason": "missing_serpapi_key_or_query"}

    params = {
        "engine": "google",
        "q": query,
        "gl": gl,
        "hl": hl,
        "device": "mobile" if device.upper() == "MOBILE" else "desktop",
        "api_key": key,
    }

    try:
        r = requests.get("https://serpapi.com/search.json", params=params, timeout=25)
        data = r.json() if r.ok else {}
    except Exception:
        return {"ok": False, "reason": "serpapi_request_failed"}

    # best-effort parse across possible shapes
    ai_block = None
    for k in ["ai_overview", "ai_overviews", "ai_overview_results", "ai_overview_block", "ai_overview_response"]:
        if isinstance(data.get(k), dict):
            ai_block = data.get(k)
            break
        if isinstance(data.get(k), list) and data.get(k):
            ai_block = {"items": data.get(k)}
            break

    has_ai = bool(ai_block)

    citations = []
    if ai_block:
        # common places for sources
        possible_lists = []
        for k in ["sources", "citations", "links", "references", "source_links", "items"]:
            v = ai_block.get(k)
            if isinstance(v, list):
                possible_lists.extend(v)

        # also check nested
        if isinstance(ai_block.get("sections"), list):
            for s in ai_block["sections"]:
                if isinstance(s, dict):
                    for k in ["sources", "citations", "links"]:
                        v = s.get(k)
                        if isinstance(v, list):
                            possible_lists.extend(v)

        for item in possible_lists[:60]:
            if isinstance(item, str):
                citations.append(item)
            elif isinstance(item, dict):
                u = item.get("link") or item.get("url") or item.get("source") or item.get("href")
                if u and isinstance(u, str):
                    citations.append(u)

    # clean citations to urls
    out_urls = []
    seen = set()
    for u in citations:
        u = clean(u)
        if not u or not u.startswith("http"):
            continue
        if u in seen:
            continue
        seen.add(u)
        out_urls.append(u)

    return {"ok": True, "has_ai": has_ai, "citations": out_urls}


def build_ai_visibility_table(target_query: str, sites: List[Tuple[str, str]]) -> pd.DataFrame:
    """
    sites: list of tuples (label, domain)
    """
    rows = []
    for device in ["DESKTOP", "MOBILE"]:
        status = serpapi_ai_overview_status(target_query, device=device, gl="ae", hl="en")
        if not status.get("ok"):
            # still show table, but mark not available
            for label, dom in sites:
                rows.append({
                    "Site": label,
                    "Device": "Desktop" if device == "DESKTOP" else "Mobile",
                    "UAE AI Overview present": "Not available",
                    "Cited in AI Overview": "Not available",
                    "Example cited URL": "Not available",
                })
            continue

        has_ai = status.get("has_ai", False)
        cites = status.get("citations", []) or []

        for label, dom in sites:
            cited_urls = [u for u in cites if dom and dom in domain_of(u)]
            rows.append({
                "Site": label,
                "Device": "Desktop" if device == "DESKTOP" else "Mobile",
                "UAE AI Overview present": "Yes" if has_ai else "No",
                "Cited in AI Overview": "Yes" if cited_urls else "No",
                "Example cited URL": cited_urls[0] if cited_urls else ("—" if has_ai else "—"),
            })
    return pd.DataFrame(rows)


# =====================================================
# SEO ROW (uses TARGET FKW input; long-tails removed)
# =====================================================
def seo_row_for_page(
    label: str,
    url: str,
    fr: FetchResult,
    nodes: List[dict],
    target_fkw: str,
    gsc_service=None,
    gsc_site_url: str = "",
    gsc_days: int = 28,
) -> dict:
    seo_title, meta_desc = extract_head_seo(fr.html or "")

    # Fallback for SEO title if head not available: use H1
    h1 = get_first_h1(nodes)
    if seo_title == "Not available" and h1:
        seo_title = h1

    media = extract_media_used(fr.html or "")
    slug = url_slug(url)

    # TRUE header counts only when HTML exists
    if fr.html:
        hc = count_headers_from_html(fr.html)
        headers_summary = f"H1:{hc['H1']} | H2:{hc['H2']} | H3:{hc['H3']} | H4:{hc['H4']}"
    else:
        headers_summary = "Not available (no HTML)"

    # Target keyword signals (user-provided)
    tfkw = clean(target_fkw)
    headings_text = headings_blob(nodes)
    body_text = fr.text or ""

    fkw_in_title = kw_present(seo_title, tfkw) if tfkw else "Not provided"
    fkw_in_h1 = kw_present(h1, tfkw) if tfkw else "Not provided"
    fkw_in_headings_n = str(kw_in_text_count(headings_text, tfkw)) if tfkw else "Not provided"
    fkw_repeats_body = str(kw_in_text_count(body_text, tfkw)) if tfkw else "Not provided"

    # Google ranking (UAE desktop + mobile) via GSC
    rank_uae_desktop = "Not available"
    rank_uae_mobile = "Not available"
    if gsc_service and gsc_site_url and url.startswith("http"):
        rank_uae_desktop = gsc_avg_position_for_page(
            service=gsc_service,
            site_url=gsc_site_url,
            page_url=url,
            days=gsc_days,
            country_iso3="ARE",
            device="DESKTOP",
            query_filter=tfkw if tfkw else None,
        )
        rank_uae_mobile = gsc_avg_position_for_page(
            service=gsc_service,
            site_url=gsc_site_url,
            page_url=url,
            days=gsc_days,
            country_iso3="ARE",
            device="MOBILE",
            query_filter=tfkw if tfkw else None,
        )

    return {
        "Page": label,
        "SEO title": seo_title,
        "Meta description": meta_desc,
        "Slug": slug,
        "Media used": media,
        "Headers count": headers_summary,
        "Target FKW": tfkw if tfkw else "Not provided",
        "FKW in title": fkw_in_title,
        "FKW in H1": fkw_in_h1,
        "FKW in headings (#)": fkw_in_headings_n,
        "FKW repeats (body)": fkw_repeats_body,
        "Rank UAE (Desktop)": rank_uae_desktop,
        "Rank UAE (Mobile)": rank_uae_mobile,
    }

def build_seo_analysis_update(
    bayut_url: str,
    bayut_fr: FetchResult,
    bayut_nodes: List[dict],
    competitors: List[str],
    comp_fr_map: Dict[str, FetchResult],
    comp_tree_map: Dict[str, dict],
    target_fkw: str,
    gsc_service=None,
    gsc_site_url: str = "",
) -> pd.DataFrame:
    rows = []
    rows.append(seo_row_for_page("Bayut", bayut_url, bayut_fr, bayut_nodes, target_fkw, gsc_service, gsc_site_url))
    for u in competitors:
        fr = comp_fr_map[u]
        nodes = comp_tree_map[u]["nodes"]
        rows.append(seo_row_for_page(site_name(u), u, fr, nodes, target_fkw, gsc_service, gsc_site_url))
    return pd.DataFrame(rows)

def build_seo_analysis_newpost(
    new_title: str,
    competitors: List[str],
    comp_fr_map: Dict[str, FetchResult],
    comp_tree_map: Dict[str, dict],
    target_fkw: str,
    gsc_service=None,
    gsc_site_url: str = "",
) -> pd.DataFrame:
    rows = []
    for u in competitors:
        fr = comp_fr_map[u]
        nodes = comp_tree_map[u]["nodes"]
        rows.append(seo_row_for_page(site_name(u), u, fr, nodes, target_fkw, gsc_service, gsc_site_url))

    # Target row (New post) - keep clean + no fake ranking
    fake_nodes = [{"level": 1, "header": new_title, "content": "", "children": []}]
    fake_fr = FetchResult(True, "synthetic", 200, "", new_title, None)
    row = seo_row_for_page("Target (New Post)", "Not applicable", fake_fr, fake_nodes, target_fkw, None, "")
    row["Slug"] = "Suggested: /" + re.sub(r"[^a-z0-9]+", "-", new_title.lower()).strip("-") + "/"
    row["Meta description"] = "Suggested: write a 140–160 char meta based on the intro angle."
    row["Media used"] = "Suggested: 1 hero image + 1 map/graphic (optional)"
    row["Headers count"] = "Not applicable"
    row["FKW in headings (#)"] = "Not applicable"
    row["FKW repeats (body)"] = "Not applicable"
    row["Rank UAE (Desktop)"] = "Not applicable"
    row["Rank UAE (Mobile)"] = "Not applicable"
    rows.insert(0, row)

    return pd.DataFrame(rows)

def ai_summary_from_gaps_df(df: pd.DataFrame) -> str:
    if df is None or df.empty:
        return "No gaps to summarize."
    # rows per competitor
    src_counts = {}
    header_counts = {}
    for _, r in df.iterrows():
        src = re.sub(r"<[^>]+>", "", str(r.get("Source", ""))).strip()
        # source is an <a> tag in html; fall back to plain
        if not src or src == "nan":
            src = "Source"
        # try to parse anchor text
        m = re.search(r'>([^<]+)<', str(r.get("Source", "")))
        if m:
            src = m.group(1).strip()

        src_counts[src] = src_counts.get(src, 0) + 1
        h = str(r.get("Headers", "")).strip()
        if h:
            header_counts[h] = header_counts.get(h, 0) + 1

    total = int(len(df))
    src_lines = ", ".join([f"{k}: {v}" for k, v in sorted(src_counts.items(), key=lambda x: x[1], reverse=True)])

    top_headers = sorted(header_counts.items(), key=lambda x: x[1], reverse=True)[:6]
    top_headers_line = ", ".join([f"{h} ({c})" for h, c in top_headers])

    return (
        f"Total gap rows: {total}\n"
        f"Rows by competitor: {src_lines}\n"
        f"Most repeated gap headers: {top_headers_line}"
    )

def ai_summary_from_seo_df(df: pd.DataFrame) -> str:
    if df is None or df.empty:
        return "No SEO data to summarize."
    # Focus on Target FKW + structure + ranks when present
    lines = []
    tfkw = ""
    try:
        tfkw = str(df.iloc[0].get("Target FKW", "")).strip()
    except Exception:
        tfkw = ""

    if tfkw and tfkw != "Not provided":
        lines.append(f"Target FKW: “{tfkw}”")

    # Bayut vs competitors if exists
    bayut = None
    for _, r in df.iterrows():
        if str(r.get("Page","")).lower() == "bayut":
            bayut = r
            break

    def parse_h2(s: str) -> int:
        m = re.search(r"H2:(\d+)", str(s))
        return int(m.group(1)) if m else 0

    if bayut is not None:
        b_h2 = parse_h2(bayut.get("Headers count",""))
        b_rank_d = str(bayut.get("Rank UAE (Desktop)", ""))
        b_rank_m = str(bayut.get("Rank UAE (Mobile)", ""))
        lines.append(f"Bayut H2 count: {b_h2}")
        if b_rank_d != "Not available" or b_rank_m != "Not available":
            lines.append(f"Bayut rank UAE: Desktop={b_rank_d} | Mobile={b_rank_m}")

        for _, r in df.iterrows():
            page = str(r.get("Page",""))
            if page.lower() == "bayut":
                continue
            c_h2 = parse_h2(r.get("Headers count",""))
            if c_h2 and b_h2 and c_h2 > b_h2:
                lines.append(f"{page} has more H2 sections than Bayut.")
            if tfkw and tfkw != "Not provided":
                tit = str(r.get("FKW in title",""))
                h1 = str(r.get("FKW in H1",""))
                if tit == "Yes" and h1 == "Yes":
                    lines.append(f"{page} places FKW in both title + H1 (strong on intent).")

            rd = str(r.get("Rank UAE (Desktop)", ""))
            rm = str(r.get("Rank UAE (Mobile)", ""))
            if rd != "Not available" or rm != "Not available":
                lines.append(f"{page} rank UAE: Desktop={rd} | Mobile={rm}")
    else:
        # New Post Mode: list competitor signals
        for _, r in df.iterrows():
            page = str(r.get("Page",""))
            if page.lower().startswith("target"):
                continue
            if tfkw and tfkw != "Not provided":
                if str(r.get("FKW in title","")) == "Yes":
                    lines.append(f"{page} includes FKW in title.")
                if str(r.get("FKW in H1","")) == "Yes":
                    lines.append(f"{page} includes FKW in H1.")
    if not lines:
        return "SEO summary: Not enough data to summarize."
    return "\n".join(lines[:10])


# =====================================================
# NEW POST MODE (kept simple)
# =====================================================
def list_headers(nodes: List[dict], level: int) -> List[str]:
    return [x["header"] for x in flatten(nodes) if x["level"] == level and not is_noise_header(x["header"])]

def detect_main_angle(comp_nodes: List[dict]) -> str:
    h2s = [norm_header(h) for h in list_headers(comp_nodes, 2)]
    blob = " ".join(h2s)
    if ("pros" in blob and "cons" in blob) or ("advantages" in blob and "disadvantages" in blob):
        return "pros-and-cons decision guide"
    if "payment plan" in blob:
        return "buyer decision / payment-plan-led guide"
    if "amenities" in blob and "location" in blob:
        return "community overview for buyers"
    return "decision-led overview"

def new_post_coverage_rows(comp_nodes: List[dict], comp_url: str) -> List[dict]:
    h1s = list_headers(comp_nodes, 1)
    h1_title = strip_label(h1s[0]) if h1s else ""
    angle = detect_main_angle(comp_nodes)
    h1_text = f"{h1_title} — The competitor frames the page as a {angle}." if h1_title else f"The competitor frames the page as a {angle}."

    h2s = [strip_label(h) for h in list_headers(comp_nodes, 2)]
    h2_main = [h for h in h2s if h and not header_is_faq(h)]
    h2_main = h2_main[:6]
    has_faq = any(header_is_faq(h) for h in h2s)

    if h2_main:
        h2_text = "Major sections include: " + " → ".join(h2_main) + "."
    else:
        h2_text = "Major sections introduce the topic, break down key points, and end with wrap-up context."
    if has_faq:
        h2_text += " It also includes a separate FAQ section."

    h3s = [strip_label(h) for h in list_headers(comp_nodes, 3)]
    themes = []
    seen = set()
    for h in h3s:
        if not h or is_noise_header(h) or header_is_faq(h):
            continue
        k = norm_header(h)
        if k in seen:
            continue
        seen.add(k)
        themes.append(h)
        if len(themes) >= 7:
            break

    if themes:
        h3_text = "Subsections break sections into practical themes such as: " + ", ".join(themes) + "."
    else:
        h3_text = "Subsections add practical depth inside each major section."
    if has_faq:
        h3_text += " FAQs appear as question-style items."

    return [
        {"Headers covered": "H1 (main angle)", "Content covered": h1_text, "Source": site_name(comp_url)},
        {"Headers covered": "H2 (sections covered)", "Content covered": h2_text, "Source": site_name(comp_url)},
        {"Headers covered": "H3 (subsections covered)", "Content covered": h3_text, "Source": site_name(comp_url)},
    ]


# =====================================================
# MODE SELECTOR (CENTERED BUTTONS)
# =====================================================
if "mode" not in st.session_state:
    st.session_state.mode = "update"  # "update" or "new"

st.markdown("<div class='mode-wrap'>", unsafe_allow_html=True)
outer_l, outer_m, outer_r = st.columns([1, 2.2, 1])
with outer_m:
    b1, b2 = st.columns(2)
    with b1:
        if st.button(
            "Update Mode",
            type="primary" if st.session_state.mode == "update" else "secondary",
            use_container_width=True,
            key="mode_update_btn",
        ):
            st.session_state.mode = "update"
    with b2:
        if st.button(
            "New Post Mode",
            type="primary" if st.session_state.mode == "new" else "secondary",
            use_container_width=True,
            key="mode_new_btn",
        ):
            st.session_state.mode = "new"
st.markdown("</div>", unsafe_allow_html=True)
st.markdown("<div class='mode-note'>Tip: competitors one per line. If any page blocks the server, you will be forced to paste it — so nothing is missing.</div>", unsafe_allow_html=True)

show_internal_fetch = st.sidebar.checkbox("Admin: show internal fetch log", value=False)

# Optional: GSC status (no UI config, purely env/secrets)
gsc_service, gsc_site_url = init_gsc_service()
gsc_days = 28
if show_internal_fetch:
    st.sidebar.markdown("### Integrations")
    st.sidebar.write(f"Playwright enabled: {PLAYWRIGHT_OK}")
    st.sidebar.write(f"GSC available: {bool(gsc_service and gsc_site_url)} (GSC lib: {GSC_OK})")

# Keep last results visible
if "update_df" not in st.session_state:
    st.session_state.update_df = pd.DataFrame()
if "update_fetch" not in st.session_state:
    st.session_state.update_fetch = []
if "seo_update_df" not in st.session_state:
    st.session_state.seo_update_df = pd.DataFrame()
if "ai_vis_update_df" not in st.session_state:
    st.session_state.ai_vis_update_df = pd.DataFrame()

if "new_df" not in st.session_state:
    st.session_state.new_df = pd.DataFrame()
if "new_fetch" not in st.session_state:
    st.session_state.new_fetch = []
if "seo_new_df" not in st.session_state:
    st.session_state.seo_new_df = pd.DataFrame()
if "ai_vis_new_df" not in st.session_state:
    st.session_state.ai_vis_new_df = pd.DataFrame()


# =====================================================
# UI - UPDATE MODE
# =====================================================
if st.session_state.mode == "update":
    st.markdown("<div class='section-pill'>Update Mode (Header-first gaps)</div>", unsafe_allow_html=True)

    bayut_url = st.text_input("Bayut article URL", placeholder="https://www.bayut.com/mybayut/...")
    competitors_text = st.text_area(
        "Competitor URLs (one per line)",
        height=120,
        placeholder="https://example.com/article\nhttps://example.com/another"
    )
    competitors = [c.strip() for c in competitors_text.splitlines() if c.strip()]

    # FKW OPTION (acts on SEO + ranking + AI visibility)
    target_fkw = st.text_input(
        "Target FKW (optional, but recommended)",
        placeholder="e.g. living in business bay",
        help="Used for keyword placement signals, UAE ranking (GSC if connected), and AI Visibility table."
    )

    run = st.button("Run analysis", type="primary")

    if run:
        if not bayut_url.strip():
            st.error("Bayut article URL is required.")
            st.stop()
        if not competitors:
            st.error("Add at least one competitor URL.")
            st.stop()

        # 1) HARD: Bayut must be available (or paste)
        with st.spinner("Fetching Bayut (no exceptions)…"):
            bayut_fr_map = resolve_all_or_require_manual(agent, [bayut_url.strip()], st_key_prefix="bayut")
            bayut_tree_map = ensure_headings_or_require_repaste([bayut_url.strip()], bayut_fr_map, st_key_prefix="bayut_tree")
        bayut_fr = bayut_fr_map[bayut_url.strip()]
        bayut_nodes = bayut_tree_map[bayut_url.strip()]["nodes"]

        # 2) HARD: ALL competitors must be available (or paste)
        with st.spinner("Fetching ALL competitors (no exceptions)…"):
            comp_fr_map = resolve_all_or_require_manual(agent, competitors, st_key_prefix="comp_update")
            comp_tree_map = ensure_headings_or_require_repaste(competitors, comp_fr_map, st_key_prefix="comp_update_tree")

        # 3) Build content-gap rows
        all_rows = []
        internal_fetch = []

        for comp_url in competitors:
            src = comp_fr_map[comp_url].source
            internal_fetch.append((comp_url, f"ok ({src})"))
            comp_nodes = comp_tree_map[comp_url]["nodes"]

            all_rows.extend(update_mode_rows_header_first(
                bayut_nodes=bayut_nodes,
                comp_nodes=comp_nodes,
                comp_url=comp_url,
                max_missing_headers=7,
                max_missing_parts=5,
            ))

        st.session_state.update_fetch = internal_fetch
        st.session_state.update_df = (
            pd.DataFrame(all_rows)[["Headers", "Description", "Source"]]
            if all_rows
            else pd.DataFrame(columns=["Headers", "Description", "Source"])
        )

        # 4) SEO Analysis table (Target FKW + TRUE header counts + UAE ranking if GSC connected)
        st.session_state.seo_update_df = build_seo_analysis_update(
            bayut_url=bayut_url.strip(),
            bayut_fr=bayut_fr,
            bayut_nodes=bayut_nodes,
            competitors=competitors,
            comp_fr_map=comp_fr_map,
            comp_tree_map=comp_tree_map,
            target_fkw=target_fkw.strip(),
            gsc_service=gsc_service,
            gsc_site_url=gsc_site_url,
        )

        # 5) AI Visibility table (UAE desktop + mobile) for Bayut + competitor domains (SERP API optional)
        sites = [("Bayut", domain_of(bayut_url.strip()))] + [(site_name(u), domain_of(u)) for u in competitors]
        st.session_state.ai_vis_update_df = build_ai_visibility_table(
            target_query=clean(target_fkw) if clean(target_fkw) else "",
            sites=sites
        )

        # reset summary boxes when re-run
        st.session_state["show_gaps_summary_update"] = False
        st.session_state["show_seo_summary_update"] = False
        st.session_state["show_ai_vis_summary_update"] = False
        st.session_state.pop("gaps_update_summary_text", None)
        st.session_state.pop("seo_update_summary_text", None)
        st.session_state.pop("ai_vis_update_summary_text", None)

    if show_internal_fetch and st.session_state.update_fetch:
        st.sidebar.markdown("### Internal fetch log (Update Mode)")
        for u, s in st.session_state.update_fetch:
            st.sidebar.write(u, "—", s)

    # -------------------------
    # GAPS TABLE (header + button same level; summary appears only after click)
    # -------------------------
    show_gaps_summary = section_title_with_button(
        title="Gaps Table",
        button_label="Summarize with AI",
        btn_key="btn_gaps_sum_update",
        show_key="show_gaps_summary_update"
    )

    if st.session_state.update_df is None or st.session_state.update_df.empty:
        st.info("Run analysis to see results.")
    else:
        if show_gaps_summary:
            st.session_state["gaps_update_summary_text"] = ai_summary_from_gaps_df(st.session_state.update_df)
            st.markdown(
                f"<div class='ai-summary'><b>AI Summary — Gaps</b><div class='muted'>Generated on demand.</div>"
                f"<pre style='white-space:pre-wrap;margin:8px 0 0 0;'>{st.session_state['gaps_update_summary_text']}</pre></div>",
                unsafe_allow_html=True
            )
        render_table(st.session_state.update_df)

    # -------------------------
    # SEO TABLE
    # -------------------------
    show_seo_summary = section_title_with_button(
        title="SEO Analysis",
        button_label="Summarize with AI",
        btn_key="btn_seo_sum_update",
        show_key="show_seo_summary_update"
    )

    if st.session_state.seo_update_df is None or st.session_state.seo_update_df.empty:
        st.info("Run analysis to see SEO comparison.")
    else:
        if show_seo_summary:
            st.session_state["seo_update_summary_text"] = ai_summary_from_seo_df(st.session_state.seo_update_df)
            st.markdown(
                f"<div class='ai-summary'><b>AI Summary — SEO</b><div class='muted'>Target-FKW signals + structure + UAE rank (if GSC connected).</div>"
                f"<pre style='white-space:pre-wrap;margin:8px 0 0 0;'>{st.session_state['seo_update_summary_text']}</pre></div>",
                unsafe_allow_html=True
            )
        render_table(st.session_state.seo_update_df)

    # -------------------------
    # AI VISIBILITY TABLE
    # -------------------------
    show_ai_vis_summary = section_title_with_button(
        title="AI Visibility (UAE)",
        button_label="Summarize with AI",
        btn_key="btn_ai_vis_sum_update",
        show_key="show_ai_vis_summary_update"
    )

    if st.session_state.ai_vis_update_df is None or st.session_state.ai_vis_update_df.empty:
        st.info("Provide Target FKW and run analysis to see AI Visibility. (Requires SERP API key to be fully automatic.)")
    else:
        if show_ai_vis_summary:
            dfv = st.session_state.ai_vis_update_df
            total_yes = int((dfv["Cited in AI Overview"] == "Yes").sum()) if "Cited in AI Overview" in dfv.columns else 0
            present = dfv["UAE AI Overview present"].iloc[0] if "UAE AI Overview present" in dfv.columns and len(dfv) else "Not available"
            st.session_state["ai_vis_update_summary_text"] = (
                f"AI Overview present (UAE): {present}\n"
                f"Total site/device rows cited: {total_yes}"
            )
            st.markdown(
                f"<div class='ai-summary'><b>AI Summary — Visibility</b><div class='muted'>UAE desktop + mobile.</div>"
                f"<pre style='white-space:pre-wrap;margin:8px 0 0 0;'>{st.session_state['ai_vis_update_summary_text']}</pre></div>",
                unsafe_allow_html=True
            )
        render_table(st.session_state.ai_vis_update_df)


# =====================================================
# UI - NEW POST MODE
# =====================================================
else:
    st.markdown("<div class='section-pill'>New Post Mode</div>", unsafe_allow_html=True)

    new_title = st.text_input("New post title", placeholder="Pros & Cons of Living in Business Bay (2026)")
    competitors_text = st.text_area(
        "Competitor URLs (one per line)",
        height=120,
        placeholder="https://example.com/article\nhttps://example.com/another"
    )
    competitors = [c.strip() for c in competitors_text.splitlines() if c.strip()]

    # FKW BOX (requested)
    target_fkw = st.text_input(
        "Target FKW (optional, but recommended)",
        placeholder="e.g. living in business bay",
        help="Used for keyword placement signals (SEO table) and AI Visibility table."
    )

    run = st.button("Generate competitor coverage", type="primary")

    if run:
        if not new_title.strip():
            st.error("New post title is required.")
            st.stop()
        if not competitors:
            st.error("Add at least one competitor URL.")
            st.stop()

        with st.spinner("Fetching ALL competitors (no exceptions)…"):
            comp_fr_map = resolve_all_or_require_manual(agent, competitors, st_key_prefix="comp_new")
            comp_tree_map = ensure_headings_or_require_repaste(competitors, comp_fr_map, st_key_prefix="comp_new_tree")

        rows = []
        internal_fetch = []

        for comp_url in competitors:
            src = comp_fr_map[comp_url].source
            internal_fetch.append((comp_url, f"ok ({src})"))
            comp_nodes = comp_tree_map[comp_url]["nodes"]
            rows.extend(new_post_coverage_rows(comp_nodes, comp_url))

        st.session_state.new_fetch = internal_fetch
        st.session_state.new_df = (
            pd.DataFrame(rows)[["Headers covered", "Content covered", "Source"]]
            if rows
            else pd.DataFrame(columns=["Headers covered", "Content covered", "Source"])
        )

        # SEO Analysis (New Post Mode) — no long-tail, uses Target FKW
        st.session_state.seo_new_df = build_seo_analysis_newpost(
            new_title=new_title.strip(),
            competitors=competitors,
            comp_fr_map=comp_fr_map,
            comp_tree_map=comp_tree_map,
            target_fkw=target_fkw.strip(),
            gsc_service=gsc_service,
            gsc_site_url=gsc_site_url,
        )

        # AI Visibility (UAE) — competitor domains only (SERP API optional)
        sites = [(site_name(u), domain_of(u)) for u in competitors]
        st.session_state.ai_vis_new_df = build_ai_visibility_table(
            target_query=clean(target_fkw) if clean(target_fkw) else "",
            sites=sites
        )

        # reset summary boxes when re-run
        st.session_state["show_coverage_summary_new"] = False
        st.session_state["show_seo_summary_new"] = False
        st.session_state["show_ai_vis_summary_new"] = False
        st.session_state.pop("coverage_new_summary_text", None)
        st.session_state.pop("seo_new_summary_text", None)
        st.session_state.pop("ai_vis_new_summary_text", None)

    if show_internal_fetch and st.session_state.new_fetch:
        st.sidebar.markdown("### Internal fetch log (New Post Mode)")
        for u, s in st.session_state.new_fetch:
            st.sidebar.write(u, "—", s)

    # Competitor Coverage
    show_cov_summary = section_title_with_button(
        title="Competitor Coverage",
        button_label="Summarize with AI",
        btn_key="btn_cov_sum_new",
        show_key="show_coverage_summary_new"
    )

    if st.session_state.new_df is None or st.session_state.new_df.empty:
        st.info("Generate competitor coverage to see results.")
    else:
        if show_cov_summary:
            # deterministic summary
            dfc = st.session_state.new_df
            src_counts = dfc["Source"].value_counts().to_dict() if "Source" in dfc.columns else {}
            st.session_state["coverage_new_summary_text"] = (
                "Rows by competitor: " + ", ".join([f"{k}: {v}" for k, v in src_counts.items()])
            )
            st.markdown(
                f"<div class='ai-summary'><b>AI Summary — Coverage</b><div class='muted'>Generated on demand.</div>"
                f"<pre style='white-space:pre-wrap;margin:8px 0 0 0;'>{st.session_state['coverage_new_summary_text']}</pre></div>",
                unsafe_allow_html=True
            )
        render_table(st.session_state.new_df)

    # SEO Analysis
    show_seo_summary = section_title_with_button(
        title="SEO Analysis",
        button_label="Summarize with AI",
        btn_key="btn_seo_sum_new",
        show_key="show_seo_summary_new"
    )

    if st.session_state.seo_new_df is None or st.session_state.seo_new_df.empty:
        st.info("Generate competitor coverage to see SEO comparison.")
    else:
        if show_seo_summary:
            st.session_state["seo_new_summary_text"] = ai_summary_from_seo_df(st.session_state.seo_new_df)
            st.markdown(
                f"<div class='ai-summary'><b>AI Summary — SEO</b><div class='muted'>Target-FKW signals + structure.</div>"
                f"<pre style='white-space:pre-wrap;margin:8px 0 0 0;'>{st.session_state['seo_new_summary_text']}</pre></div>",
                unsafe_allow_html=True
            )
        render_table(st.session_state.seo_new_df)

    # AI Visibility
    show_ai_vis_summary = section_title_with_button(
        title="AI Visibility (UAE)",
        button_label="Summarize with AI",
        btn_key="btn_ai_vis_sum_new",
        show_key="show_ai_vis_summary_new"
    )

    if st.session_state.ai_vis_new_df is None or st.session_state.ai_vis_new_df.empty:
        st.info("Provide Target FKW and run analysis to see AI Visibility. (Requires SERP API key to be fully automatic.)")
    else:
        if show_ai_vis_summary:
            dfv = st.session_state.ai_vis_new_df
            total_yes = int((dfv["Cited in AI Overview"] == "Yes").sum()) if "Cited in AI Overview" in dfv.columns else 0
            present = dfv["UAE AI Overview present"].iloc[0] if "UAE AI Overview present" in dfv.columns and len(dfv) else "Not available"
            st.session_state["ai_vis_new_summary_text"] = (
                f"AI Overview present (UAE): {present}\n"
                f"Total competitor site/device rows cited: {total_yes}"
            )
            st.markdown(
                f"<div class='ai-summary'><b>AI Summary — Visibility</b><div class='muted'>UAE desktop + mobile.</div>"
                f"<pre style='white-space:pre-wrap;margin:8px 0 0 0;'>{st.session_state['ai_vis_new_summary_text']}</pre></div>",
                unsafe_allow_html=True
            )
        render_table(st.session_state.ai_vis_new_df)
