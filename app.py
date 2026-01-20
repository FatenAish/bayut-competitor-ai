# app.py
import base64
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
      /* tighter + closer to table (your request) */
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

GENERIC_SECTION_HEADERS = {
    "introduction", "overview",
}

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

def _faq_heading_nodes(nodes: List[dict]) -> List[dict]:
    out = []
    for x in flatten(nodes):
        if x.get("level") in (2, 3) and header_is_faq(x.get("header", "")):
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
    """
    FINAL RULE:
    - Must have an explicit FAQ heading (FAQ/FAQs/Frequently Asked Questions)
    AND
      - If HTML exists: either FAQPage schema OR >=3 question-style child headings under FAQ section
      - If no HTML: require >=3 question-style child headings under FAQ section
    """
    faq_nodes = _faq_heading_nodes(nodes)
    if not faq_nodes:
        return False

    # If HTML exists, allow schema or question headings
    if fr and fr.html:
        if _has_faq_schema(fr.html):
            return True
        for fn in faq_nodes:
            if len(_question_heading_children(fn)) >= 3:
                return True
        return False

    # No HTML: must be visible question headings
    for fn in faq_nodes:
        if len(_question_heading_children(fn)) >= 3:
            return True
    return False

def extract_questions_from_node(node: dict) -> List[str]:
    qs: List[str] = []
    # We prefer question HEADINGS (strict)
    qh = _question_heading_children(node)
    qs.extend(qh)

    # fallback to text if still not enough (but still strict: keep short)
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

def missing_faqs_row(
    bayut_nodes: List[dict],
    bayut_fr: FetchResult,
    comp_nodes: List[dict],
    comp_fr: FetchResult,
    comp_url: str
) -> Optional[dict]:
    # Competitor must have REAL FAQ (no false positives)
    if not page_has_real_faq(comp_fr, comp_nodes):
        return None

    comp_faq_nodes = _faq_heading_nodes(comp_nodes)
    comp_qs = []
    for fn in comp_faq_nodes:
        comp_qs.extend(extract_questions_from_node(fn))
    comp_qs = [q for q in comp_qs if q and len(q) > 5]
    if len(comp_qs) < 3:
        return None

    # Bayut must be detected correctly (no false negatives)
    bayut_has = page_has_real_faq(bayut_fr, bayut_nodes)
    bayut_qs = []
    if bayut_has:
        for fn in _faq_heading_nodes(bayut_nodes):
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
            "Description": "Competitor has a real FAQ section covering topics such as: " + ", ".join(topics) + ".",
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
    bayut_fr: FetchResult,
    comp_nodes: List[dict],
    comp_fr: FetchResult,
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

    # 4) FAQs — ONE row only (ONLY if competitor has REAL FAQ)
    faq_row = missing_faqs_row(bayut_nodes, bayut_fr, comp_nodes, comp_fr, comp_url)
    if faq_row:
        rows.append(faq_row)

    return dedupe_rows(rows)


# =====================================================
# SEO ANALYSIS (SKWs REMOVED) + GOOGLE UAE RANKING + AI VISIBILITY
# =====================================================
def _secrets_get(key: str, default=None):
    try:
        if hasattr(st, "secrets") and key in st.secrets:
            return st.secrets[key]
    except Exception:
        pass
    return default

SERPAPI_API_KEY = _secrets_get("SERPAPI_API_KEY", None)
OPENAI_API_KEY = _secrets_get("OPENAI_API_KEY", None)
OPENAI_MODEL = _secrets_get("OPENAI_MODEL", "gpt-4o-mini")

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

def count_headers_true_from_html(html: str) -> Dict[str, int]:
    counts = {"H1": 0, "H2": 0, "H3": 0, "H4": 0}
    if not html:
        return counts
    soup = BeautifulSoup(html, "html.parser")
    for t in soup.find_all(list(IGNORE_TAGS)):
        t.decompose()
    root = soup.find("article") or soup

    counts["H1"] = len(root.find_all("h1"))
    counts["H2"] = len(root.find_all("h2"))
    counts["H3"] = len(root.find_all("h3"))
    counts["H4"] = len(root.find_all("h4"))
    return counts

def count_headers_from_nodes(nodes: List[dict]) -> Dict[str, int]:
    counts = {"H1": 0, "H2": 0, "H3": 0, "H4": 0}
    for x in flatten(nodes):
        lvl = x.get("level", 0)
        if lvl == 1: counts["H1"] += 1
        if lvl == 2: counts["H2"] += 1
        if lvl == 3: counts["H3"] += 1
        if lvl == 4: counts["H4"] += 1
    return counts

def tokenize(text: str) -> List[str]:
    text = (text or "").lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    toks = [t for t in text.split() if t and len(t) >= 3]
    return toks

def phrase_candidates(text: str, n_min=2, n_max=4) -> Dict[str, int]:
    toks = tokenize(text)
    freq: Dict[str, int] = {}
    for n in range(n_min, n_max + 1):
        for i in range(0, max(len(toks) - n + 1, 0)):
            chunk = toks[i:i+n]
            if not chunk:
                continue
            if chunk[0] in STOP or chunk[-1] in STOP:
                continue
            if all(w in STOP or w in GENERIC_STOP for w in chunk):
                continue
            phrase = " ".join(chunk)
            if len(phrase) < 8:
                continue
            freq[phrase] = freq.get(phrase, 0) + 1
    return freq

def pick_fkw_only(seo_title: str, h1: str, headings_blob_text: str, body_text: str, manual_fkw: str = "") -> str:
    manual_fkw = clean(manual_fkw)
    if manual_fkw:
        return manual_fkw.lower()

    base = " ".join([seo_title or "", h1 or "", headings_blob_text or "", body_text or ""])
    freq = phrase_candidates(base, n_min=2, n_max=4)
    if not freq:
        return "Not available"

    title_low = (seo_title or "").lower()
    h1_low = (h1 or "").lower()

    scored = []
    for ph, c in freq.items():
        boost = 1.0
        if ph in title_low:
            boost += 0.9
        if ph in h1_low:
            boost += 0.6
        scored.append((c * boost, ph))
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[0][1] if scored else "Not available"

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

def compute_kw_repetition(text: str, phrase: str) -> str:
    if not text or not phrase or phrase == "Not available":
        return "Not available"
    t = " " + re.sub(r"\s+", " ", (text or "").lower()) + " "
    p = " " + re.sub(r"\s+", " ", (phrase or "").lower()) + " "
    return str(t.count(p))

def normalize_url_for_match(u: str) -> str:
    try:
        p = urlparse(u)
        host = p.netloc.lower().replace("www.", "")
        path = (p.path or "").rstrip("/")
        return host + path
    except Exception:
        return (u or "").strip().lower().replace("www.", "").rstrip("/")

@st.cache_data(show_spinner=False, ttl=1800)
def serpapi_serp_cached(query: str, device: str) -> dict:
    if not SERPAPI_API_KEY:
        return {"_error": "missing_serpapi_key"}

    params = {
        "engine": "google",
        "q": query,
        "google_domain": "google.ae",
        "gl": "ae",
        "hl": "en",
        "api_key": SERPAPI_API_KEY,
        "num": 20,
        "device": device,
    }
    try:
        r = requests.get("https://serpapi.com/search.json", params=params, timeout=35)
        if r.status_code != 200:
            return {"_error": f"serpapi_http_{r.status_code}", "_text": r.text[:400]}
        return r.json()
    except Exception as e:
        return {"_error": str(e)}

def serp_rank_for_url(query: str, page_url: str, device: str) -> Tuple[Optional[int], str]:
    data = serpapi_serp_cached(query, device)
    if not data or data.get("_error"):
        return (None, f"Not available ({data.get('_error')})" if isinstance(data, dict) else "Not available")

    organic = data.get("organic_results") or []
    target = normalize_url_for_match(page_url)

    for item in organic:
        link = item.get("link") or ""
        if not link:
            continue
        if normalize_url_for_match(link) == target or target in normalize_url_for_match(link):
            pos = item.get("position")
            try:
                return (int(pos), "OK")
            except Exception:
                return (None, "Not available")

    return (None, "Not in top results")

def serp_ai_visibility(query: str, page_url: str, device: str) -> Dict[str, str]:
    data = serpapi_serp_cached(query, device)
    if not data or data.get("_error"):
        return {
            "AI Overview present": "Not available",
            "Cited in AI Overview": "Not available",
            "AI Notes": f"{data.get('_error')}" if isinstance(data, dict) else "Not available"
        }

    aio = data.get("ai_overview") or data.get("ai_overview_results") or None
    if not aio:
        ab = data.get("answer_box") or {}
        if isinstance(ab, dict) and ("ai" in str(ab.get("type","")).lower() or "overview" in str(ab.get("type","")).lower()):
            aio = ab

    present = "Yes" if aio else "No"
    cited = "No"
    notes = ""

    if aio:
        sources = []
        if isinstance(aio, dict):
            for k in ["sources", "citations", "references", "links"]:
                v = aio.get(k)
                if isinstance(v, list):
                    sources = v
                    break

        target = normalize_url_for_match(page_url)
        for s in sources:
            link = ""
            if isinstance(s, dict):
                link = s.get("link") or s.get("url") or ""
            elif isinstance(s, str):
                link = s
            if link and (target in normalize_url_for_match(link) or normalize_url_for_match(link) in target):
                cited = "Yes"
                break

        if sources and cited == "No":
            notes = "AI overview detected, but this page was not found in the exposed citations."
        if cited == "Yes":
            notes = "Page appears in AI overview citations (based on SERP API fields)."

    return {
        "AI Overview present": present,
        "Cited in AI Overview": cited,
        "AI Notes": notes or ("No AI overview detected in returned SERP fields." if present == "No" else "")
    }

def seo_row_for_page(label: str, url: str, fr: FetchResult, nodes: List[dict], manual_fkw: str = "") -> dict:
    seo_title, meta_desc = extract_head_seo(fr.html or "")
    h1 = get_first_h1(nodes)
    if seo_title == "Not available" and h1:
        seo_title = h1

    media = extract_media_used(fr.html or "")
    slug = url_slug(url)

    if fr.html:
        hc = count_headers_true_from_html(fr.html)
    else:
        hc = count_headers_from_nodes(nodes)
    headers_summary = f"H1:{hc['H1']} | H2:{hc['H2']} | H3:{hc['H3']} | H4:{hc['H4']}"

    blob = headings_blob(nodes)
    body_text = fr.text or ""

    fkw = pick_fkw_only(seo_title, h1, blob, body_text, manual_fkw=manual_fkw)
    fkw_count = compute_kw_repetition(body_text, fkw)

    return {
        "Page": label,
        "SEO title": seo_title,
        "Meta description": meta_desc,
        "Slug": slug,
        "Media used": media,
        "Headers count": headers_summary,
        "FKW": fkw,
        "FKW repeats (body)": fkw_count,
        "Google rank UAE (Desktop)": "Not run",
        "Google rank UAE (Mobile)": "Not run",
    }

def enrich_seo_df_with_rank_and_ai(df: pd.DataFrame, manual_query: str = "") -> Tuple[pd.DataFrame, pd.DataFrame]:
    if df is None or df.empty:
        return df, pd.DataFrame(columns=["Page", "Query", "AI Overview present", "Cited in AI Overview", "AI Notes"])

    rows_ai = []
    df2 = df.copy()

    for i, r in df2.iterrows():
        page = str(r.get("Page", ""))
        page_url = str(r.get("__url", ""))

        if page.lower().startswith("target"):
            df2.at[i, "Google rank UAE (Desktop)"] = "Not applicable"
            df2.at[i, "Google rank UAE (Mobile)"] = "Not applicable"
            continue

        query = clean(manual_query) if clean(manual_query) else str(r.get("FKW", ""))
        if not query or query == "Not available":
            df2.at[i, "Google rank UAE (Desktop)"] = "Not available"
            df2.at[i, "Google rank UAE (Mobile)"] = "Not available"
            rows_ai.append({
                "Page": page,
                "Query": "Not available",
                "AI Overview present": "Not available",
                "Cited in AI Overview": "Not available",
                "AI Notes": "No FKW available to query Google ranking."
            })
            continue

        if not page_url:
            df2.at[i, "Google rank UAE (Desktop)"] = "Not available"
            df2.at[i, "Google rank UAE (Mobile)"] = "Not available"
            rows_ai.append({
                "Page": page,
                "Query": query,
                "AI Overview present": "Not available",
                "Cited in AI Overview": "Not available",
                "AI Notes": "Missing URL mapping."
            })
            continue

        rank_d, note_d = serp_rank_for_url(query, page_url, device="desktop")
        rank_m, note_m = serp_rank_for_url(query, page_url, device="mobile")

        df2.at[i, "Google rank UAE (Desktop)"] = str(rank_d) if rank_d else note_d
        df2.at[i, "Google rank UAE (Mobile)"] = str(rank_m) if rank_m else note_m

        ai = serp_ai_visibility(query, page_url, device="desktop")
        rows_ai.append({
            "Page": page,
            "Query": query,
            "AI Overview present": ai.get("AI Overview present", "Not available"),
            "Cited in AI Overview": ai.get("Cited in AI Overview", "Not available"),
            "AI Notes": ai.get("AI Notes", ""),
        })

    ai_df = pd.DataFrame(rows_ai)
    return df2, ai_df

def build_seo_analysis_update(
    bayut_url: str,
    bayut_fr: FetchResult,
    bayut_nodes: List[dict],
    competitors: List[str],
    comp_fr_map: Dict[str, FetchResult],
    comp_tree_map: Dict[str, dict],
    manual_fkw: str = ""
) -> pd.DataFrame:
    rows = []
    row_b = seo_row_for_page("Bayut", bayut_url, bayut_fr, bayut_nodes, manual_fkw=manual_fkw)
    row_b["__url"] = bayut_url
    rows.append(row_b)

    for u in competitors:
        fr = comp_fr_map[u]
        nodes = comp_tree_map[u]["nodes"]
        rr = seo_row_for_page(site_name(u), u, fr, nodes, manual_fkw=manual_fkw)
        rr["__url"] = u
        rows.append(rr)

    return pd.DataFrame(rows)

def build_seo_analysis_newpost(
    new_title: str,
    competitors: List[str],
    comp_fr_map: Dict[str, FetchResult],
    comp_tree_map: Dict[str, dict],
    manual_fkw: str = ""
) -> pd.DataFrame:
    rows = []

    for u in competitors:
        fr = comp_fr_map[u]
        nodes = comp_tree_map[u]["nodes"]
        rr = seo_row_for_page(site_name(u), u, fr, nodes, manual_fkw=manual_fkw)
        rr["__url"] = u
        rows.append(rr)

    fake_nodes = [{"level": 1, "header": new_title, "content": "", "children": []}]
    fake_fr = FetchResult(True, "synthetic", 200, "", new_title, None)
    row = seo_row_for_page("Target (New Post)", "Not applicable", fake_fr, fake_nodes, manual_fkw=manual_fkw)
    row["Slug"] = "Suggested: /" + re.sub(r"[^a-z0-9]+", "-", new_title.lower()).strip("-") + "/"
    row["Meta description"] = "Suggested: write a 140–160 char meta based on the intro angle."
    row["Media used"] = "Suggested: 1 hero image + 1 map/graphic (optional)"
    row["FKW repeats (body)"] = "Not applicable"
    row["Google rank UAE (Desktop)"] = "Not applicable"
    row["Google rank UAE (Mobile)"] = "Not applicable"
    row["__url"] = ""
    rows.insert(0, row)

    return pd.DataFrame(rows)


# =====================================================
# CONTENT QUALITY TABLE (FAQs fixed + your rows)
# =====================================================
def domain_of(url: str) -> str:
    try:
        host = urlparse(url).netloc.lower().replace("www.", "")
        return host.split(":")[0]
    except Exception:
        return ""

@st.cache_data(show_spinner=False, ttl=86400)
def _head_last_modified(url: str) -> str:
    try:
        r = requests.head(url, headers=DEFAULT_HEADERS, allow_redirects=True, timeout=18)
        return r.headers.get("Last-Modified", "") or ""
    except Exception:
        return ""

def _extract_last_modified_from_html(html: str) -> str:
    if not html:
        return ""
    soup = BeautifulSoup(html, "html.parser")

    meta_candidates = [
        ("meta", {"property": "article:modified_time"}, "content"),
        ("meta", {"property": "article:published_time"}, "content"),
        ("meta", {"name": "lastmod"}, "content"),
        ("meta", {"name": "last-modified"}, "content"),
        ("meta", {"name": "date"}, "content"),
        ("meta", {"itemprop": "dateModified"}, "content"),
        ("meta", {"itemprop": "datePublished"}, "content"),
    ]
    for tag, attrs, key in meta_candidates:
        t = soup.find(tag, attrs=attrs)
        if t and t.get(key):
            v = clean(t.get(key))
            if v:
                return v

    tm = soup.find("time", attrs={"datetime": True})
    if tm and tm.get("datetime"):
        v = clean(tm.get("datetime"))
        if v:
            return v

    return ""

def get_last_modified(url: str, html: str) -> str:
    v = _extract_last_modified_from_html(html or "")
    if v:
        return v
    h = _head_last_modified(url)
    return h if h else "Not available"

def word_count_from_text(text: str) -> int:
    t = clean(text or "")
    if not t:
        return 0
    return len(re.findall(r"\b\w+\b", t))

def _kw_stuffing_label(word_count: int, repeats: str) -> str:
    try:
        rep = int(str(repeats).strip())
    except Exception:
        return "Not available"
    if word_count <= 0:
        return "Not available"
    per_1k = (rep / max(word_count, 1)) * 1000.0
    if per_1k >= 18:
        return f"High ({rep} repeats, {per_1k:.1f}/1k words)"
    if per_1k >= 10:
        return f"Moderate ({rep} repeats, {per_1k:.1f}/1k words)"
    return f"Low ({rep} repeats, {per_1k:.1f}/1k words)"

def _latest_year_mentioned(text: str) -> int:
    years = re.findall(r"\b(19\d{2}|20\d{2})\b", (text or ""))
    ys = []
    for y in years:
        try:
            ys.append(int(y))
        except Exception:
            pass
    return max(ys) if ys else 0

def _has_brief_summary(nodes: List[dict], text: str) -> str:
    blob = (headings_blob(nodes) or "").lower()
    t = (text or "").lower()
    cues = ["tl;dr", "tldr", "key takeaways", "in summary", "summary", "quick summary", "at a glance"]
    if any(c in blob for c in cues) or any(c in t[:1400] for c in cues):
        return "Yes"
    return "No"

def _count_tables_videos(html: str) -> Tuple[int, int]:
    if not html:
        return (0, 0)
    soup = BeautifulSoup(html, "html.parser")
    for t in soup.find_all(list(IGNORE_TAGS)):
        t.decompose()
    root = soup.find("article") or soup
    tables = len(root.find_all("table"))
    videos = len(root.find_all("video"))
    ifr = root.find_all("iframe")
    for x in ifr:
        src = (x.get("src") or "").lower()
        if any(k in src for k in ["youtube", "youtu.be", "vimeo", "dailymotion"]):
            videos += 1
    return (tables, videos)

def _styling_layout_label(html: str, nodes: List[dict], text: str) -> str:
    wc = word_count_from_text(text)
    h2_count = sum(1 for x in flatten(nodes) if x.get("level") == 2)
    has_toc = "table of contents" in (text or "").lower()
    tables, videos = _count_tables_videos(html or "")

    score = 0
    if wc >= 1200: score += 1
    if h2_count >= 6: score += 1
    if has_toc: score += 1
    if tables > 0: score += 1
    if videos > 0: score += 1

    if score >= 4: return "Strong"
    if score >= 2: return "OK"
    return "Weak"

def _latest_information_label(last_modified: str, text: str) -> str:
    lm = (last_modified or "").lower()
    y = _latest_year_mentioned(text or "")
    if ("2026" in lm) or ("2025" in lm) or y >= 2025:
        return "Likely up-to-date"
    if y >= 2024:
        return "Somewhat recent"
    return "Unclear/Older"

def _outdated_label(last_modified: str, text: str) -> str:
    lm = (last_modified or "").lower()
    y = _latest_year_mentioned(text or "")
    if ("2026" in lm) or ("2025" in lm) or y >= 2025:
        return "No obvious outdated signal"
    if y and y <= 2022:
        return "Potentially outdated (mentions older years)"
    if y and y <= 2023:
        return "Possibly outdated"
    return "Unclear"

def _topic_cannibalization_label(query: str, page_url: str) -> str:
    if not SERPAPI_API_KEY:
        return "Not available (no SERPAPI_API_KEY)"
    dom = domain_of(page_url)
    if not dom or not query or query == "Not available":
        return "Not available"
    site_q = f"site:{dom} {query}"
    data = serpapi_serp_cached(site_q, device="desktop")
    if not data or data.get("_error"):
        return f"Not available ({data.get('_error')})" if isinstance(data, dict) else "Not available"

    organic = data.get("organic_results") or []
    target = normalize_url_for_match(page_url)
    others = []
    for it in organic:
        link = it.get("link") or ""
        if not link:
            continue
        nm = normalize_url_for_match(link)
        if dom in nm and nm != target:
            others.append(link)

    cnt = len(set(others))
    if cnt >= 3:
        return f"High risk (≈{cnt} other pages on same domain)"
    if cnt >= 1:
        return f"Medium risk (≈{cnt} other page(s) on same domain)"
    return "Low risk"

def build_content_quality_table_from_seo(
    seo_df: pd.DataFrame,
    fr_map_by_url: Dict[str, FetchResult],
    tree_map_by_url: Dict[str, dict],
    manual_query: str = ""
) -> pd.DataFrame:
    if seo_df is None or seo_df.empty:
        return pd.DataFrame()

    metrics = [
        "Topic Cannibalization",
        "Word Count",
        "Last Modified",
        "Keyword Stuffing",
        "Latest Information",
        "Outdated/Misleading Information",
        "Provides a brief summary",
        "FAQs",
        "Tables",
        "Video",
        "Styling / Overall Layout",
    ]

    out = {"Content Quality": metrics}

    for _, row in seo_df.iterrows():
        page = str(row.get("Page", "")).strip()
        page_url = str(row.get("__url", "")).strip()

        if page.lower().startswith("target"):
            out[page] = [
                "Not applicable",
                "Not applicable",
                "Not applicable",
                "Not applicable",
                "Suggested: include 2026 updates + fresh stats",
                "Not applicable",
                "Suggested: add TL;DR / key takeaways",
                "Suggested: add FAQs (6–10)",
                "Suggested: add 1–2 tables",
                "Suggested: add 1 short video (optional)",
                "Suggested: TOC + clean spacing + visuals",
            ]
            continue

        fr = fr_map_by_url.get(page_url)
        tr = tree_map_by_url.get(page_url) or {}
        nodes = tr.get("nodes", []) if isinstance(tr, dict) else []

        html = (fr.html if fr else "") or ""
        text = (fr.text if fr else "") or ""

        wc = word_count_from_text(text)
        lm = get_last_modified(page_url, html)
        fkw = clean(manual_query) if clean(manual_query) else str(row.get("FKW", ""))
        repeats = row.get("FKW repeats (body)", "Not available")

        tables, videos = _count_tables_videos(html)

        # ✅ FIX: FAQ must use the STRICT real-FAQ detector (no false negatives)
        faq_value = "Yes" if (fr and page_has_real_faq(fr, nodes)) else "No"

        out[page] = [
            _topic_cannibalization_label(fkw, page_url),
            str(wc) if wc else "Not available",
            lm,
            _kw_stuffing_label(wc, repeats),
            _latest_information_label(lm, text),
            _outdated_label(lm, text),
            _has_brief_summary(nodes, text),
            faq_value,
            str(tables),
            str(videos),
            _styling_layout_label(html, nodes, text),
        ]

    return pd.DataFrame(out)


# =====================================================
# AI SUMMARY (BULLETS ONLY – ORDERED & CLEAR)
# =====================================================
def openai_summarize_block(title: str, payload_text: str, max_bullets: int = 8) -> str:
    payload_text = clean(payload_text)

    # If no OpenAI key, return the already-bulleted payload
    if not OPENAI_API_KEY:
        return payload_text

    try:
        headers = {
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json",
        }

        system = (
            "You are a senior SEO/editorial analyst.\n"
            "Output ONLY bullet points.\n"
            "Rules:\n"
            f"- Use '• ' at the start of each bullet\n"
            f"- {max_bullets-2}–{max_bullets} bullets maximum\n"
            "- One clear action per bullet\n"
            "- Short, direct, no explanations\n"
            "- No paragraphs, no numbering, no grouping text\n"
            "- Never merge multiple actions into one bullet"
        )

        body = {
            "model": OPENAI_MODEL,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": f"{title}\n\n{payload_text}\n\nReturn bullets only."}
            ],
            "temperature": 0.2,
        }

        r = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            data=json.dumps(body),
            timeout=35
        )

        if r.status_code != 200:
            return payload_text

        out = clean(r.json()["choices"][0]["message"]["content"])

        # hard safety: force bullets if model misbehaves
        lines = [l.strip() for l in out.splitlines() if l.strip()]
        bullets = [l for l in lines if l.startswith("•")]

        if not bullets:
            chunks = re.split(r"(?<=[.!?])\s+", out)
            chunks = [c.strip() for c in chunks if c.strip()]
            bullets = [f"• {c}" for c in chunks]

        # Ensure each bullet is ONE line (no multi-actions with • inside)
        cleaned = []
        for b in bullets:
            b = b.replace("•", "").strip()
            if not b:
                continue
            cleaned.append("• " + b)

        return "\n".join(cleaned[:max_bullets])

    except Exception:
        return payload_text


def _ignore_summary_header(h: str) -> bool:
    return norm_header(h) in {
        "conclusion", "summary", "final thoughts",
        "wrap up", "closing thoughts", "key takeaways"
    }


def concise_gaps_summary(df: pd.DataFrame, min_bullets: int = 4, max_bullets: int = 8) -> str:
    """
    Guarantees 4–8 bullets:
    - Add missing sections
    - Expand missing parts
    - Optional FAQ
    - If still too few, add safe editorial fallbacks
    """
    if df is None or df.empty:
        return "• No content gaps detected."

    add_sections = []
    expand_sections = []
    faq_needed = False

    seen_add = set()
    seen_expand = set()

    for _, r in df.iterrows():
        h = str(r.get("Headers", "")).strip()
        if not h:
            continue

        if _ignore_summary_header(h):
            continue

        # FAQs (only once)
        if norm_header(h) == "faqs":
            faq_needed = True
            continue

        # ignore question-style headers
        if _looks_like_question(h):
            continue

        # missing parts
        if "(missing parts)" in h.lower():
            base = clean(re.sub(r"\(missing parts\)", "", h, flags=re.I))
            k = norm_header(base)
            if k and k not in seen_expand:
                seen_expand.add(k)
                expand_sections.append(base)
            continue

        # normal missing sections
        k = norm_header(h)
        if k and k not in seen_add:
            seen_add.add(k)
            add_sections.append(h)

    bullets = []

    # 1) Add missing sections
    for h in add_sections:
        bullets.append(f"• Add section: {h}")
        if len(bullets) >= max_bullets:
            return "\n".join(bullets[:max_bullets])

    # 2) Expand existing sections
    for h in expand_sections:
        bullets.append(f"• Expand section: {h}")
        if len(bullets) >= max_bullets:
            return "\n".join(bullets[:max_bullets])

    # 3) FAQs
    if faq_needed and len(bullets) < max_bullets:
        bullets.append("• Add or expand FAQs based on competitor coverage")

    # 4) Force minimum (SAFE editorial fallbacks)
    fallbacks = [
        "• Strengthen decision framing (why/when this area fits different lifestyles)",
        "• Add a short pros/cons recap with clear takeaways for readers",
        "• Add freshness signals (updated year, latest prices/transport changes if applicable)",
        "• Improve scannability (shorter paragraphs, more subheadings, 1 table where relevant)",
    ]
    for fb in fallbacks:
        if len(bullets) >= min_bullets:
            break
        bullets.append(fb)

    return "\n".join(bullets[:max_bullets]) if bullets else "• No meaningful gaps detected."


def concise_seo_summary(df: pd.DataFrame, min_bullets: int = 4, max_bullets: int = 8) -> str:
    if df is None or df.empty:
        return "• No SEO data available."

    bullets = []
    for _, r in df.iterrows():
        page = str(r.get("Page", "")).strip()
        if page.lower().startswith("target"):
            continue

        fkw = str(r.get("FKW", "")).strip()
        rd = str(r.get("Google rank UAE (Desktop)", "")).strip()
        rm = str(r.get("Google rank UAE (Mobile)", "")).strip()

        if not fkw or fkw == "Not available":
            bullets.append(f"• {page}: define a clear Focus Keyword (FKW) before optimizing rankings")
        else:
            bullets.append(f"• {page}: focus on '{fkw}' (UAE rank D/M: {rd}/{rm})")

        if len(bullets) >= max_bullets:
            break

    # Force minimum (SEO-safe fallbacks)
    seo_fallbacks = [
        "• Ensure title + H1 contain the exact Focus Keyword naturally",
        "• Reduce keyword repetition if it looks stuffed; prioritize readability",
        "• Add internal links to related Bayut guides to improve relevance",
    ]
    for fb in seo_fallbacks:
        if len(bullets) >= min_bullets:
            break
        bullets.append(fb)

    return "\n".join(bullets[:max_bullets])


def ai_summary_from_df(kind: str, df: pd.DataFrame) -> str:
    if kind == "gaps":
        base = concise_gaps_summary(df, min_bullets=4, max_bullets=8)
        return openai_summarize_block("Turn this into a clear writer checklist.", base, max_bullets=8)

    if kind == "seo":
        base = concise_seo_summary(df, min_bullets=4, max_bullets=8)
        return openai_summarize_block("Turn this into a clear SEO checklist.", base, max_bullets=8)

    return "• No summary available."


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

    if h2_main:
        h2_text = "Major sections include: " + " → ".join(h2_main) + "."
    else:
        h2_text = "Major sections introduce the topic, break down key points, and end with wrap-up context."

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

    return [
        {"Headers covered": "H1 (main angle)", "Content covered": h1_text, "Source": site_name(comp_url)},
        {"Headers covered": "H2 (sections covered)", "Content covered": h2_text, "Source": site_name(comp_url)},
        {"Headers covered": "H3 (subsections covered)", "Content covered": h3_text, "Source": site_name(comp_url)},
    ]


# =====================================================
# HTML TABLE RENDER (with hyperlinks)
# =====================================================
def render_table(df: pd.DataFrame, drop_internal_url: bool = True):
    if df is None or df.empty:
        st.info("No results to show.")
        return
    if drop_internal_url and "__url" in df.columns:
        df = df.drop(columns=["__url"])
    html = df.to_html(index=False, escape=False)
    st.markdown(html, unsafe_allow_html=True)

def section_header_with_ai_button(title: str, button_label: str, button_key: str) -> bool:
    c1, c2 = st.columns([4.2, 1.3])
    with c1:
        st.markdown(f"<div class='section-pill section-pill-tight'>{title}</div>", unsafe_allow_html=True)
    with c2:
        clicked = st.button(button_label, type="secondary", use_container_width=True, key=button_key)
    return clicked


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

# Keep last results visible
if "update_df" not in st.session_state:
    st.session_state.update_df = pd.DataFrame()
if "update_fetch" not in st.session_state:
    st.session_state.update_fetch = []
if "seo_update_df" not in st.session_state:
    st.session_state.seo_update_df = pd.DataFrame()
if "ai_update_df" not in st.session_state:
    st.session_state.ai_update_df = pd.DataFrame()
if "cq_update_df" not in st.session_state:
    st.session_state.cq_update_df = pd.DataFrame()

if "new_df" not in st.session_state:
    st.session_state.new_df = pd.DataFrame()
if "new_fetch" not in st.session_state:
    st.session_state.new_fetch = []
if "seo_new_df" not in st.session_state:
    st.session_state.seo_new_df = pd.DataFrame()
if "ai_new_df" not in st.session_state:
    st.session_state.ai_new_df = pd.DataFrame()
if "cq_new_df" not in st.session_state:
    st.session_state.cq_new_df = pd.DataFrame()


# =====================================================
# UI - UPDATE MODE
# =====================================================
if st.session_state.mode == "update":
    st.markdown("<div class='section-pill section-pill-tight'>Update Mode</div>", unsafe_allow_html=True)

    bayut_url = st.text_input("Bayut article URL", placeholder="https://www.bayut.com/mybayut/...")
    competitors_text = st.text_area(
        "Competitor URLs (one per line)",
        height=120,
        placeholder="https://example.com/article\nhttps://example.com/another"
    )
    competitors = [c.strip() for c in competitors_text.splitlines() if c.strip()]

    manual_fkw_update = st.text_input("Optional: Focus Keyword (FKW) for analysis + UAE ranking", placeholder="e.g., pros and cons business bay")

    run = st.button("Run analysis", type="primary")

    if run:
        if not bayut_url.strip():
            st.error("Bayut article URL is required.")
            st.stop()
        if not competitors:
            st.error("Add at least one competitor URL.")
            st.stop()

        with st.spinner("Fetching Bayut (no exceptions)…"):
            bayut_fr_map = resolve_all_or_require_manual(agent, [bayut_url.strip()], st_key_prefix="bayut")
            bayut_tree_map = ensure_headings_or_require_repaste([bayut_url.strip()], bayut_fr_map, st_key_prefix="bayut_tree")
        bayut_fr = bayut_fr_map[bayut_url.strip()]
        bayut_nodes = bayut_tree_map[bayut_url.strip()]["nodes"]

        with st.spinner("Fetching ALL competitors (no exceptions)…"):
            comp_fr_map = resolve_all_or_require_manual(agent, competitors, st_key_prefix="comp_update")
            comp_tree_map = ensure_headings_or_require_repaste(competitors, comp_fr_map, st_key_prefix="comp_update_tree")

        all_rows = []
        internal_fetch = []

        for comp_url in competitors:
            src = comp_fr_map[comp_url].source
            internal_fetch.append((comp_url, f"ok ({src})"))
            comp_nodes = comp_tree_map[comp_url]["nodes"]

            all_rows.extend(update_mode_rows_header_first(
                bayut_nodes=bayut_nodes,
                bayut_fr=bayut_fr,
                comp_nodes=comp_nodes,
                comp_fr=comp_fr_map[comp_url],
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

        st.session_state.seo_update_df = build_seo_analysis_update(
            bayut_url=bayut_url.strip(),
            bayut_fr=bayut_fr,
            bayut_nodes=bayut_nodes,
            competitors=competitors,
            comp_fr_map=comp_fr_map,
            comp_tree_map=comp_tree_map,
            manual_fkw=manual_fkw_update.strip()
        )

        with st.spinner("Fetching Google UAE ranking (desktop + mobile) + AI visibility…"):
            seo_enriched, ai_df = enrich_seo_df_with_rank_and_ai(
                st.session_state.seo_update_df,
                manual_query=manual_fkw_update.strip()
            )
            st.session_state.seo_update_df = seo_enriched
            st.session_state.ai_update_df = ai_df

        # Content Quality table (uses STRICT FAQ detector)
        st.session_state.cq_update_df = build_content_quality_table_from_seo(
            seo_df=st.session_state.seo_update_df,
            fr_map_by_url={bayut_url.strip(): bayut_fr, **comp_fr_map},
            tree_map_by_url={bayut_url.strip(): {"nodes": bayut_nodes}, **{u: comp_tree_map[u] for u in competitors}},
            manual_query=manual_fkw_update.strip()
        )

    if show_internal_fetch and st.session_state.update_fetch:
        st.sidebar.markdown("### Internal fetch log (Update Mode)")
        st.sidebar.write(f"Playwright enabled: {PLAYWRIGHT_OK}")
        for u, s in st.session_state.update_fetch:
            st.sidebar.write(u, "—", s)

    gaps_clicked = section_header_with_ai_button("Gaps Table", "Summarize by AI", "btn_gaps_summary_update")
    if gaps_clicked:
        st.session_state["gaps_update_summary_text"] = ai_summary_from_df("gaps", st.session_state.update_df)

    if st.session_state.get("gaps_update_summary_text"):
        st.markdown(
            f"<div class='ai-summary'><b>AI Summary</b><div class='muted'>6–8 bullets only (real summary).</div><pre style='white-space:pre-wrap;margin:8px 0 0 0;'>{st.session_state['gaps_update_summary_text']}</pre></div>",
            unsafe_allow_html=True
        )

    if st.session_state.update_df is None or st.session_state.update_df.empty:
        st.info("Run analysis to see results.")
    else:
        render_table(st.session_state.update_df)

    seo_clicked = section_header_with_ai_button("SEO Analysis", "Summarize by AI", "btn_seo_summary_update")
    if seo_clicked:
        st.session_state["seo_update_summary_text"] = ai_summary_from_df("seo", st.session_state.seo_update_df)

    if st.session_state.get("seo_update_summary_text"):
        st.markdown(
            f"<div class='ai-summary'><b>AI Summary</b><div class='muted'>6–8 bullets only.</div><pre style='white-space:pre-wrap;margin:8px 0 0 0;'>{st.session_state['seo_update_summary_text']}</pre></div>",
            unsafe_allow_html=True
        )

    if st.session_state.seo_update_df is None or st.session_state.seo_update_df.empty:
        st.info("Run analysis to see SEO comparison.")
    else:
        render_table(st.session_state.seo_update_df, drop_internal_url=True)

    st.markdown("<div class='section-pill section-pill-tight'>AI Visibility (Google AI Overview)</div>", unsafe_allow_html=True)
    if st.session_state.ai_update_df is None or st.session_state.ai_update_df.empty:
        st.info("Run analysis to see AI visibility signals.")
    else:
        render_table(st.session_state.ai_update_df, drop_internal_url=True)

    st.markdown("<div class='section-pill section-pill-tight'>Content Quality</div>", unsafe_allow_html=True)
    if st.session_state.cq_update_df is None or st.session_state.cq_update_df.empty:
        st.info("Run analysis to see Content Quality signals.")
    else:
        render_table(st.session_state.cq_update_df, drop_internal_url=True)


# =====================================================
# UI - NEW POST MODE
# =====================================================
else:
    st.markdown("<div class='section-pill section-pill-tight'>New Post Mode</div>", unsafe_allow_html=True)

    new_title = st.text_input("New post title", placeholder="Pros & Cons of Living in Business Bay (2026)")
    competitors_text = st.text_area(
        "Competitor URLs (one per line)",
        height=120,
        placeholder="https://example.com/article\nhttps://example.com/another"
    )
    competitors = [c.strip() for c in competitors_text.splitlines() if c.strip()]

    manual_fkw_new = st.text_input("Optional: Focus Keyword (FKW) for SEO + UAE ranking", placeholder="e.g., pros and cons business bay")

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

        st.session_state.seo_new_df = build_seo_analysis_newpost(
            new_title=new_title.strip(),
            competitors=competitors,
            comp_fr_map=comp_fr_map,
            comp_tree_map=comp_tree_map,
            manual_fkw=manual_fkw_new.strip()
        )

        with st.spinner("Fetching Google UAE ranking (desktop + mobile) + AI visibility…"):
            seo_enriched, ai_df = enrich_seo_df_with_rank_and_ai(
                st.session_state.seo_new_df,
                manual_query=manual_fkw_new.strip()
            )
            st.session_state.seo_new_df = seo_enriched
            st.session_state.ai_new_df = ai_df

        st.session_state.cq_new_df = build_content_quality_table_from_seo(
            seo_df=st.session_state.seo_new_df,
            fr_map_by_url={**comp_fr_map},
            tree_map_by_url={u: comp_tree_map[u] for u in competitors},
            manual_query=manual_fkw_new.strip()
        )

    if show_internal_fetch and st.session_state.new_fetch:
        st.sidebar.markdown("### Internal fetch log (New Post Mode)")
        st.sidebar.write(f"Playwright enabled: {PLAYWRIGHT_OK}")
        for u, s in st.session_state.new_fetch:
            st.sidebar.write(u, "—", s)

    cov_clicked = section_header_with_ai_button("Competitor Coverage", "Summarize by AI", "btn_cov_summary_new")
    if cov_clicked:
        # compact summary for coverage: turn rows into bullets
        if st.session_state.new_df is None or st.session_state.new_df.empty:
            st.session_state["cov_new_summary_text"] = "No data."
        else:
            bullets = []
            for _, r in st.session_state.new_df.head(6).iterrows():
                bullets.append(f"• {r.get('Source','')}: {r.get('Headers covered','')}")
            st.session_state["cov_new_summary_text"] = openai_summarize_block(
                "Summarize competitor coverage (6–8 bullets).",
                "\n".join(bullets) if bullets else "No data."
            )

    if st.session_state.get("cov_new_summary_text"):
        st.markdown(
            f"<div class='ai-summary'><b>AI Summary</b><div class='muted'>6–8 bullets only.</div><pre style='white-space:pre-wrap;margin:8px 0 0 0;'>{st.session_state['cov_new_summary_text']}</pre></div>",
            unsafe_allow_html=True
        )

    if st.session_state.new_df is None or st.session_state.new_df.empty:
        st.info("Generate competitor coverage to see results.")
    else:
        render_table(st.session_state.new_df)

    seo_clicked = section_header_with_ai_button("SEO Analysis", "Summarize by AI", "btn_seo_summary_new")
    if seo_clicked:
        st.session_state["seo_new_summary_text"] = ai_summary_from_df("seo", st.session_state.seo_new_df)

    if st.session_state.get("seo_new_summary_text"):
        st.markdown(
            f"<div class='ai-summary'><b>AI Summary</b><div class='muted'>6–8 bullets only.</div><pre style='white-space:pre-wrap;margin:8px 0 0 0;'>{st.session_state['seo_new_summary_text']}</pre></div>",
            unsafe_allow_html=True
        )

    if st.session_state.seo_new_df is None or st.session_state.seo_new_df.empty:
        st.info("Generate competitor coverage to see SEO comparison.")
    else:
        render_table(st.session_state.seo_new_df, drop_internal_url=True)

    st.markdown("<div class='section-pill section-pill-tight'>AI Visibility (Google AI Overview)</div>", unsafe_allow_html=True)
    if st.session_state.ai_new_df is None or st.session_state.ai_new_df.empty:
        st.info("Generate competitor coverage to see AI visibility signals.")
    else:
        render_table(st.session_state.ai_new_df, drop_internal_url=True)

    st.markdown("<div class='section-pill section-pill-tight'>Content Quality</div>", unsafe_allow_html=True)
    if st.session_state.cq_new_df is None or st.session_state.cq_new_df.empty:
        st.info("Generate competitor coverage to see Content Quality signals.")
    else:
        render_table(st.session_state.cq_new_df, drop_internal_url=True)

# Helpful note (only if missing SERPAPI key)
if (st.session_state.seo_update_df is not None and not st.session_state.seo_update_df.empty) or \
   (st.session_state.seo_new_df is not None and not st.session_state.seo_new_df.empty):
    if not SERPAPI_API_KEY:
        st.warning("Google UAE ranking + AI visibility requires SERPAPI_API_KEY in Streamlit secrets.")



 
