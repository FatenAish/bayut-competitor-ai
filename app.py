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

# Optional: JS rendering tool
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
LIGHT_GREEN_2 = "#DFF3EA"
TEXT_DARK = "#1F2937"
PAGE_BG = "#F3FBF7"

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
      <p><b>Header-first</b> gap logic — Missing headers first, then <b>(missing parts)</b> under matching headers. FAQs are <b>one row</b> (ONLY if the page has a REAL FAQ).</p>
    </div>
    """,
    unsafe_allow_html=True,
)


# =====================================================
# FETCH
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
    Resolver:
    - direct HTML
    - optional JS render (Playwright)
    - Jina reader
    - Textise
    If all fail => force manual paste (hard gate).
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
        article = soup.find("article") or soup.find("main") or soup
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
                page.wait_for_timeout(1600)
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


# =====================================================
# FORCE “NO STALE RESULTS”
# =====================================================
def signature(mode: str, a: str, b: str, c: str) -> str:
    raw = f"{mode}||{(a or '').strip()}||{(b or '').strip()}||{(c or '').strip()}"
    return hashlib.md5(raw.encode("utf-8")).hexdigest()

def clear_update_results():
    st.session_state.update_df = pd.DataFrame()
    st.session_state.seo_update_df = pd.DataFrame()
    st.session_state.ai_update_df = pd.DataFrame()
    st.session_state.cq_update_df = pd.DataFrame()
    st.session_state.pop("gaps_update_summary_text", None)
    st.session_state.pop("seo_update_summary_text", None)

def clear_new_results():
    st.session_state.new_df = pd.DataFrame()
    st.session_state.seo_new_df = pd.DataFrame()
    st.session_state.ai_new_df = pd.DataFrame()
    st.session_state.cq_new_df = pd.DataFrame()
    st.session_state.pop("cov_new_summary_text", None)
    st.session_state.pop("seo_new_summary_text", None)


# =====================================================
# HTML/TEXT gating helpers (FORCE correct FAQ/video/table)
# =====================================================
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

def ensure_html_for_quality(urls: List[str], fr_map: Dict[str, FetchResult], st_key_prefix: str) -> Dict[str, FetchResult]:
    """
    For FAQ/video/table accuracy we need HTML.
    If we only have reader text (jina/textise), force manual HTML paste.
    """
    need = []
    for u in urls:
        fr = fr_map[u]
        if (not fr.html) or (fr.source in ("jina", "textise")):
            need.append(u)

    if not need:
        return fr_map

    st.warning("For accurate FAQs/Videos/Tables we need real HTML. Paste HTML for each URL below (no missing).")

    for u in need:
        with st.expander(f"HTML required for accurate Content Quality: {u}", expanded=True):
            pasted = st.text_area(
                "Paste FULL HTML (preferred) — this fixes FAQ/video/table detection:",
                key=_safe_key(st_key_prefix + "__need_html", u),
                height=240,
            )
            if pasted and "<" in pasted and len(pasted.strip()) > 800:
                fr_map[u] = FetchResult(
                    True,
                    "manual_html",
                    200,
                    pasted.strip(),
                    clean(BeautifulSoup(pasted, "html.parser").get_text(" ")),
                    None
                )

    still = [u for u in need if not fr_map[u].html]
    if still:
        st.stop()

    return fr_map


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
    "dubai","uae","community","area","living","pros","cons",
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

    root = soup.find("article") or soup.find("main") or soup
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
    elif fr.source in ("manual", "manual_html") and maybe_html:
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
# STRICT FAQ DETECTION
# =====================================================
FAQ_TITLES = {
    "faq",
    "faqs",
    "frequently asked questions",
    "frequently asked question",
}

def header_is_faq(header: str) -> bool:
    nh = norm_header(header)
    if any(nh == t for t in FAQ_TITLES):
        return True
    if nh.startswith("faq ") or nh.startswith("faqs "):
        return True
    if nh.startswith("frequently asked questions"):
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
    if fr and fr.html and _has_faq_schema(fr.html):
        return True

    faq_nodes = _faq_heading_nodes(nodes)
    if not faq_nodes:
        return False

    for fn in faq_nodes:
        if len(_question_heading_children(fn)) >= 3:
            return True
    return False


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
# ✅ CONTENT GAPS ENGINE (FIXED + NO DEPENDENCY ON SEO FUNCTIONS)
# =====================================================
def _simple_tokens(text: str) -> List[str]:
    t = (text or "").lower()
    t = re.sub(r"[^a-z0-9\s]", " ", t)
    toks = [x for x in t.split() if x and len(x) >= 3 and x not in STOP]
    return toks

def _ngram_phrases(text: str, n_min: int = 2, n_max: int = 4, top_k: int = 18) -> List[str]:
    toks = _simple_tokens(text)
    freq: Dict[str, int] = {}
    for n in range(n_min, n_max + 1):
        for i in range(0, max(len(toks) - n + 1, 0)):
            chunk = toks[i:i+n]
            # drop super-generic chunks
            if all(w in GENERIC_STOP for w in chunk):
                continue
            phrase = " ".join(chunk)
            if len(phrase) < 8:
                continue
            freq[phrase] = freq.get(phrase, 0) + 1

    if not freq:
        return []
    items = sorted(freq.items(), key=lambda x: x[1], reverse=True)
    out = []
    for ph, _ in items:
        if ph not in out:
            out.append(ph)
        if len(out) >= top_k:
            break
    return out

def _missing_parts_snippet(comp_content: str, bayut_content: str, max_phrases: int = 7) -> str:
    c = clean(comp_content or "").lower()
    b = clean(bayut_content or "").lower()
    if len(c) < 80:
        return ""

    cand = _ngram_phrases(c, n_min=2, n_max=4, top_k=18)
    missing = []
    for ph in cand:
        if ph.lower() not in b:
            missing.append(ph)
        if len(missing) >= max_phrases:
            break

    if not missing:
        return ""
    return "Add details about: " + ", ".join(missing) + "."

def compute_gaps_header_first(
    bayut_url: str,
    bayut_fr: FetchResult,
    bayut_nodes: List[dict],
    competitors: List[str],
    comp_fr_map: Dict[str, FetchResult],
    comp_tree_map: Dict[str, dict],
) -> pd.DataFrame:
    bayut_secs = section_nodes(bayut_nodes, levels=(2, 3))
    bayut_has_faq = page_has_real_faq(bayut_fr, bayut_nodes)

    rows: List[dict] = []

    for cu in competitors:
        tr = comp_tree_map.get(cu) or {}
        comp_nodes = tr.get("nodes", [])
        comp_fr = comp_fr_map.get(cu)

        if not comp_nodes:
            continue

        comp_secs = section_nodes(comp_nodes, levels=(2, 3))

        # 1) Missing headers + 2) Missing parts
        for cs in comp_secs:
            comp_h = cs.get("header", "")
            comp_content = cs.get("content", "")
            if not comp_h:
                continue

            match = find_best_bayut_match(comp_h, bayut_secs, min_score=0.73)

            if not match:
                rows.append({
                    "Headers": comp_h,
                    "What to add": f"Competitor covers this section but Bayut doesn’t. Add a dedicated section titled '{comp_h}' and cover the key points discussed there.",
                    "Source": source_link(cu),
                })
            else:
                bsec = match["bayut_section"]
                bayut_content = bsec.get("content", "")
                snippet = _missing_parts_snippet(comp_content, bayut_content, max_phrases=7)
                if snippet:
                    rows.append({
                        "Headers": f"{comp_h} (missing parts)",
                        "What to add": snippet,
                        "Source": source_link(cu),
                    })

        # 3) FAQ one row ONLY if REAL FAQ exists
        comp_has_faq = page_has_real_faq(comp_fr, comp_nodes) if comp_fr else False
        if comp_has_faq and not bayut_has_faq:
            rows.append({
                "Headers": "FAQs",
                "What to add": "Competitor has a REAL FAQ section. Add an FAQ block (4–6 questions) matching the competitor intent.",
                "Source": source_link(cu),
            })

    rows = dedupe_rows(rows)
    df = pd.DataFrame(rows, columns=["Headers", "What to add", "Source"])
    return df


# =====================================================
# GAPS SUMMARY (bullets)
# =====================================================
def _secrets_get(key: str, default=None):
    try:
        if hasattr(st, "secrets") and key in st.secrets:
            return st.secrets[key]
    except Exception:
        pass
    return default

def openai_summarize_block(title: str, payload_text: str, max_bullets: int = 8) -> str:
    payload_text = clean(payload_text)

    OPENAI_API_KEY = _secrets_get("OPENAI_API_KEY", None)
    OPENAI_MODEL = _secrets_get("OPENAI_MODEL", "gpt-4o-mini")

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
            "- Short, direct\n"
            "- No paragraphs\n"
            "- No numbering\n"
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
        lines = [l.strip() for l in out.splitlines() if l.strip()]
        bullets = [l for l in lines if l.startswith("•")]
        if not bullets:
            bullets = [f"• {x}" for x in re.split(r"(?<=[.!?])\s+", out) if x.strip()]
        return "\n".join(bullets[:max_bullets])
    except Exception:
        return payload_text

def concise_gaps_summary(df: pd.DataFrame) -> str:
    if df is None or df.empty:
        return "• No content gaps detected."

    add_sections = []
    expand_sections = []
    faq_needed = False

    for _, r in df.iterrows():
        h = str(r.get("Headers", "")).strip()
        if not h:
            continue
        if norm_header(h) == "faqs":
            faq_needed = True
            continue
        if "(missing parts)" in h.lower():
            base = clean(re.sub(r"\(missing parts\)", "", h, flags=re.I))
            expand_sections.append(base)
        else:
            add_sections.append(h)

    bullets = []
    for h in add_sections[:6]:
        bullets.append(f"• Add section: {h}")
    for h in expand_sections[:2]:
        bullets.append(f"• Expand section: {h}")
    if faq_needed:
        bullets.append("• Add or expand FAQs based on competitor coverage")

    bullets = bullets[:8]
    while len(bullets) < 4:
        bullets.append("• Improve scannability (more subheadings + 1 table where relevant)")

    return "\n".join(bullets[:8])

def ai_summary_from_df(kind: str, df: pd.DataFrame) -> str:
    if kind == "gaps":
        base = concise_gaps_summary(df)
        return openai_summarize_block("Turn this into a clear writer checklist.", base, max_bullets=8)
    return "• No summary available."


# =====================================================
# DATAFORSEO
# =====================================================
DATAFORSEO_LOGIN = _secrets_get("DATAFORSEO_LOGIN", None)
DATAFORSEO_PASSWORD = _secrets_get("DATAFORSEO_PASSWORD", None)

DFS_BASE = "https://api.dataforseo.com"
DFS_SERP_PATH = "/v3/serp/google/organic/live/advanced"

DEFAULT_LOCATION_NAME = "United Arab Emirates"
DEFAULT_LANGUAGE_CODE = "en"
DEFAULT_SE_DOMAIN = "google.ae"

def normalize_url_for_match(u: str) -> str:
    try:
        p = urlparse(u)
        host = p.netloc.lower().replace("www.", "")
        path = (p.path or "").rstrip("/")
        return host + path
    except Exception:
        return (u or "").strip().lower().replace("www.", "").rstrip("/")

@st.cache_data(show_spinner=False, ttl=1800)
def dfs_serp_cached(keyword: str, device: str, location_name: str) -> dict:
    if not DATAFORSEO_LOGIN or not DATAFORSEO_PASSWORD:
        return {"_error": "missing_dataforseo_credentials"}

    task = {
        "keyword": keyword,
        "language_code": DEFAULT_LANGUAGE_CODE,
        "location_name": location_name,
        "device": device,
        "os": "windows" if device == "desktop" else "android",
        "depth": 20,
        "se_domain": DEFAULT_SE_DOMAIN,
        "load_async_ai_overview": True,
    }

    try:
        r = requests.post(
            DFS_BASE + DFS_SERP_PATH,
            auth=(DATAFORSEO_LOGIN, DATAFORSEO_PASSWORD),
            headers={"Content-Type": "application/json"},
            json=[task],
            timeout=40
        )
        if r.status_code != 200:
            return {"_error": f"dfs_http_{r.status_code}", "_text": r.text[:400]}
        data = r.json()
        if data.get("status_code") != 20000:
            return {"_error": f"dfs_status_{data.get('status_code')}", "_text": str(data)[:400]}
        return data
    except Exception as e:
        return {"_error": str(e)}

def dfs_rank_for_url(query: str, page_url: str, device: str, location_name: str) -> Tuple[Optional[int], str]:
    data = dfs_serp_cached(query, device=device, location_name=location_name)
    if not data or data.get("_error"):
        return (None, f"Not available ({data.get('_error')})")

    try:
        tasks = data.get("tasks") or []
        if not tasks:
            return (None, "Not available")
        res = (tasks[0].get("result") or [])
        if not res:
            return (None, "Not available")

        items = res[0].get("items") or []
        target = normalize_url_for_match(page_url)

        for it in items:
            if str(it.get("type","")) != "organic":
                continue
            url = it.get("url") or ""
            if not url:
                continue
            if normalize_url_for_match(url) == target or target in normalize_url_for_match(url):
                pos = it.get("rank_absolute") or it.get("position") or it.get("rank_group")
                try:
                    return (int(pos), "OK")
                except Exception:
                    return (None, "Not available")
        return (None, "Not in top results")
    except Exception:
        return (None, "Not available")

def dfs_ai_visibility(query: str, page_url: str, device: str, location_name: str) -> Dict[str, str]:
    data = dfs_serp_cached(query, device=device, location_name=location_name)
    if not data or data.get("_error"):
        return {"AI Overview present": "Not available", "Cited in AI Overview": "Not available", "AI Notes": str(data.get("_error"))}

    try:
        tasks = data.get("tasks") or []
        res = (tasks[0].get("result") or [])
        items = res[0].get("items") or []

        aio_items = [it for it in items if "ai_overview" in str(it.get("type","")).lower()]
        present = "Yes" if aio_items else "No"

        cited = "No"
        notes = ""

        if aio_items:
            refs = []
            for block in aio_items:
                for sub in (block.get("items") or []):
                    for ref in (sub.get("references") or []):
                        url = ref.get("url") if isinstance(ref, dict) else ""
                        if url:
                            refs.append(url)

            target = normalize_url_for_match(page_url)
            for u in refs:
                if target in normalize_url_for_match(u) or normalize_url_for_match(u) in target:
                    cited = "Yes"
                    break

            if cited == "Yes":
                notes = "Page found in AI Overview references (DataForSEO)."
            else:
                notes = "AI Overview present, but this page not found in returned references."
        else:
            notes = "No AI Overview detected in SERP items."

        return {"AI Overview present": present, "Cited in AI Overview": cited, "AI Notes": notes}
    except Exception:
        return {"AI Overview present": "Not available", "Cited in AI Overview": "Not available", "AI Notes": "Parse error"}
# =====================================================
# SEO extraction + MEDIA
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

def _main_root_for_counts(soup: BeautifulSoup):
    for t in soup.find_all(list(IGNORE_TAGS)):
        t.decompose()
    return soup.find("main") or soup

def extract_media_used(html: str) -> str:
    if not html:
        return "Not available"
    soup = BeautifulSoup(html, "html.parser")
    root = _main_root_for_counts(soup)

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
    root = _main_root_for_counts(soup)
    counts["H1"] = len(root.find_all("h1"))
    counts["H2"] = len(root.find_all("h2"))
    counts["H3"] = len(root.find_all("h3"))
    counts["H4"] = len(root.find_all("h4"))
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

def pick_fkw_only(seo_title: str, h1: str, headings_text: str, body_text: str, manual_fkw: str = "") -> str:
    manual_fkw = clean(manual_fkw)
    if manual_fkw:
        return manual_fkw.lower()

    base = " ".join([seo_title or "", h1 or "", headings_text or "", body_text or ""])
    base = base.lower()
    # fallback simple phrase: pick most common non-stop token bigram
    toks = [t for t in re.sub(r"[^a-z0-9\s]", " ", base).split() if len(t) >= 3 and t not in STOP]
    if len(toks) < 4:
        return "Not available"
    freq = {}
    for i in range(len(toks)-1):
        a, b = toks[i], toks[i+1]
        if a in GENERIC_STOP and b in GENERIC_STOP:
            continue
        ph = f"{a} {b}"
        freq[ph] = freq.get(ph, 0) + 1
    if not freq:
        return "Not available"
    return sorted(freq.items(), key=lambda x: x[1], reverse=True)[0][0]

def compute_kw_repetition(text: str, phrase: str) -> str:
    if not text or not phrase or phrase == "Not available":
        return "Not available"
    t = " " + re.sub(r"\s+", " ", (text or "").lower()) + " "
    p = " " + re.sub(r"\s+", " ", (phrase or "").lower()) + " "
    return str(t.count(p))

def seo_row_for_page(label: str, url: str, fr: FetchResult, nodes: List[dict], manual_fkw: str = "") -> dict:
    seo_title, meta_desc = extract_head_seo(fr.html or "")
    h1 = get_first_h1(nodes)
    if seo_title == "Not available" and h1:
        seo_title = h1

    media = extract_media_used(fr.html or "")
    slug = url_slug(url)

    hc = count_headers_true_from_html(fr.html) if fr.html else {"H1":0,"H2":0,"H3":0,"H4":0}
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

def enrich_seo_df_with_rank_and_ai(df: pd.DataFrame, manual_query: str = "", location_name: str = DEFAULT_LOCATION_NAME) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if df is None or df.empty:
        return df, pd.DataFrame(columns=["Page", "Query", "AI Overview present", "Cited in AI Overview", "AI Notes"])

    rows_ai = []
    df2 = df.copy()

    for i, r in df2.iterrows():
        page = str(r.get("Page", ""))
        page_url = str(r.get("__url", ""))

        query = clean(manual_query) if clean(manual_query) else str(r.get("FKW", ""))
        if not query or query == "Not available":
            df2.at[i, "Google rank UAE (Desktop)"] = "Not available"
            df2.at[i, "Google rank UAE (Mobile)"] = "Not available"
            rows_ai.append({
                "Page": page,
                "Query": "Not available",
                "AI Overview present": "Not available",
                "Cited in AI Overview": "Not available",
                "AI Notes": "No FKW available."
            })
            continue

        rank_d, note_d = dfs_rank_for_url(query, page_url, device="desktop", location_name=location_name)
        rank_m, note_m = dfs_rank_for_url(query, page_url, device="mobile", location_name=location_name)

        df2.at[i, "Google rank UAE (Desktop)"] = str(rank_d) if rank_d else note_d
        df2.at[i, "Google rank UAE (Mobile)"] = str(rank_m) if rank_m else note_m

        ai = dfs_ai_visibility(query, page_url, device="desktop", location_name=location_name)
        rows_ai.append({
            "Page": page,
            "Query": query,
            "AI Overview present": ai.get("AI Overview present", "Not available"),
            "Cited in AI Overview": ai.get("Cited in AI Overview", "Not available"),
            "AI Notes": ai.get("AI Notes", ""),
        })

    return df2, pd.DataFrame(rows_ai)


# =====================================================
# CONTENT QUALITY
# =====================================================
def domain_of(url: str) -> str:
    try:
        host = urlparse(url).netloc.lower().replace("www.", "")
        return host.split(":")[0]
    except Exception:
        return ""

def word_count_from_text(text: str) -> int:
    t = clean(text or "")
    if not t:
        return 0
    return len(re.findall(r"\b\w+\b", t))

def _count_tables_videos(html: str) -> Tuple[int, int]:
    if not html:
        return (0, 0)
    soup = BeautifulSoup(html, "html.parser")
    root = _main_root_for_counts(soup)

    tables = len(root.find_all("table"))
    videos = len(root.find_all("video"))
    for x in root.find_all("iframe"):
        src = (x.get("src") or "").lower()
        if any(k in src for k in ["youtube", "youtu.be", "vimeo", "dailymotion"]):
            videos += 1
    return (tables, videos)

def _topic_cannibalization_label(query: str, page_url: str, location_name: str) -> str:
    dom = domain_of(page_url)
    if not dom or not query or query == "Not available":
        return "Not available"

    site_q = f"site:{dom} {query}"
    data = dfs_serp_cached(site_q, device="desktop", location_name=location_name)
    if not data or data.get("_error"):
        return f"Not available ({data.get('_error')})"

    try:
        items = (((data.get("tasks") or [])[0].get("result") or [])[0].get("items") or [])
        target = normalize_url_for_match(page_url)
        others = []
        for it in items:
            if str(it.get("type","")) != "organic":
                continue
            link = it.get("url") or ""
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
    except Exception:
        return "Not available"

def build_content_quality_table_from_seo(
    seo_df: pd.DataFrame,
    fr_map_by_url: Dict[str, FetchResult],
    tree_map_by_url: Dict[str, dict],
    manual_query: str = "",
    location_name: str = DEFAULT_LOCATION_NAME
) -> pd.DataFrame:
    if seo_df is None or seo_df.empty:
        return pd.DataFrame()

    metrics = [
        "Topic Cannibalization",
        "Word Count",
        "FAQs",
        "Tables",
        "Video",
    ]

    out = {"Content Quality": metrics}

    for _, row in seo_df.iterrows():
        page = str(row.get("Page", "")).strip()
        page_url = str(row.get("__url", "")).strip()

        fr = fr_map_by_url.get(page_url)
        tr = tree_map_by_url.get(page_url) or {}
        nodes = tr.get("nodes", []) if isinstance(tr, dict) else []

        html = (fr.html if fr else "") or ""
        text = (fr.text if fr else "") or ""

        wc = word_count_from_text(text)
        tables, videos = _count_tables_videos(html)

        fkw = clean(manual_query) if clean(manual_query) else str(row.get("FKW", ""))
        cann = _topic_cannibalization_label(fkw, page_url, location_name)

        faq_value = "Yes" if (fr and page_has_real_faq(fr, nodes)) else "No"

        out[page] = [
            cann,
            str(wc) if wc else "Not available",
            faq_value,
            str(tables),
            str(videos),
        ]

    return pd.DataFrame(out)


# =====================================================
# HTML TABLE RENDER
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
# MODE SELECTOR
# =====================================================
if "mode" not in st.session_state:
    st.session_state.mode = "update"

st.markdown("<div class='mode-wrap'>", unsafe_allow_html=True)
outer_l, outer_m, outer_r = st.columns([1, 2.2, 1])
with outer_m:
    b1, b2 = st.columns(2)
    with b1:
        if st.button("Update Mode", type="primary" if st.session_state.mode == "update" else "secondary", use_container_width=True):
            st.session_state.mode = "update"
    with b2:
        if st.button("New Post Mode", type="primary" if st.session_state.mode == "new" else "secondary", use_container_width=True):
            st.session_state.mode = "new"
st.markdown("</div>", unsafe_allow_html=True)
st.markdown("<div class='mode-note'>Tip: competitors one per line. If any page blocks the server, you will be forced to paste it — so nothing is missing.</div>", unsafe_allow_html=True)

show_internal_fetch = st.sidebar.checkbox("Admin: show internal fetch log", value=False)

# Session defaults
st.session_state.setdefault("update_df", pd.DataFrame())
st.session_state.setdefault("seo_update_df", pd.DataFrame())
st.session_state.setdefault("ai_update_df", pd.DataFrame())
st.session_state.setdefault("cq_update_df", pd.DataFrame())
st.session_state.setdefault("update_fetch", [])
st.session_state.setdefault("last_sig_update", "")

st.session_state.setdefault("new_df", pd.DataFrame())
st.session_state.setdefault("seo_new_df", pd.DataFrame())
st.session_state.setdefault("ai_new_df", pd.DataFrame())
st.session_state.setdefault("cq_new_df", pd.DataFrame())
st.session_state.setdefault("new_fetch", [])
st.session_state.setdefault("last_sig_new", "")


# =====================================================
# ✅ UPDATE MODE UI (CONTENT GAPS FIXED)
# =====================================================
if st.session_state.mode == "update":
    st.markdown("<div class='section-pill section-pill-tight'>Update Mode (Header-first gaps)</div>", unsafe_allow_html=True)

    bayut_url = st.text_input("Bayut article URL", placeholder="https://www.bayut.com/mybayut/...")
    competitors_text = st.text_area("Competitor URLs (one per line)", height=120)
    competitors = [c.strip() for c in competitors_text.splitlines() if c.strip()]
    manual_fkw_update = st.text_input("Optional: Focus Keyword (FKW) for analysis + UAE ranking", placeholder="e.g., living in Dubai Marina")

    current_sig = signature("update", bayut_url, competitors_text, manual_fkw_update)
    if st.session_state.last_sig_update and st.session_state.last_sig_update != current_sig:
        clear_update_results()
        st.session_state.update_fetch = []

    run = st.button("Run analysis", type="primary")

    if run:
        if not bayut_url.strip():
            st.error("Bayut article URL is required.")
            st.stop()
        if not competitors:
            st.error("Add at least one competitor URL.")
            st.stop()

        clear_update_results()
        st.session_state.update_fetch = []
        st.session_state.last_sig_update = current_sig

        with st.spinner("Fetching Bayut (no exceptions)…"):
            bayut_fr_map = resolve_all_or_require_manual(agent, [bayut_url.strip()], st_key_prefix="bayut")
            bayut_fr_map = ensure_html_for_quality([bayut_url.strip()], bayut_fr_map, st_key_prefix="bayut_html")
            bayut_tree_map = ensure_headings_or_require_repaste([bayut_url.strip()], bayut_fr_map, st_key_prefix="bayut_tree")

        bayut_fr = bayut_fr_map[bayut_url.strip()]
        bayut_nodes = bayut_tree_map[bayut_url.strip()]["nodes"]

        with st.spinner("Fetching ALL competitors (no exceptions)…"):
            comp_fr_map = resolve_all_or_require_manual(agent, competitors, st_key_prefix="comp_update")
            comp_fr_map = ensure_html_for_quality(competitors, comp_fr_map, st_key_prefix="comp_update_html")
            comp_tree_map = ensure_headings_or_require_repaste(competitors, comp_fr_map, st_key_prefix="comp_update_tree")

        st.session_state.update_fetch = [(u, f"ok ({comp_fr_map[u].source})") for u in competitors]

        # ✅ CONTENT GAPS TABLE (ACTUAL COMPUTE)
        st.session_state.update_df = compute_gaps_header_first(
            bayut_url=bayut_url.strip(),
            bayut_fr=bayut_fr,
            bayut_nodes=bayut_nodes,
            competitors=competitors,
            comp_fr_map=comp_fr_map,
            comp_tree_map=comp_tree_map,
        )

        # --- SEO table
        rows = []
        rb = seo_row_for_page("Bayut", bayut_url.strip(), bayut_fr, bayut_nodes, manual_fkw=manual_fkw_update.strip())
        rb["__url"] = bayut_url.strip()
        rows.append(rb)
        for u in competitors:
            rr = seo_row_for_page(site_name(u), u, comp_fr_map[u], comp_tree_map[u]["nodes"], manual_fkw=manual_fkw_update.strip())
            rr["__url"] = u
            rows.append(rr)

        st.session_state.seo_update_df = pd.DataFrame(rows)

        with st.spinner("Fetching Google UAE ranking (desktop + mobile) + AI visibility via DataForSEO…"):
            seo_enriched, ai_df = enrich_seo_df_with_rank_and_ai(
                st.session_state.seo_update_df,
                manual_query=manual_fkw_update.strip(),
                location_name=DEFAULT_LOCATION_NAME
            )
            st.session_state.seo_update_df = seo_enriched
            st.session_state.ai_update_df = ai_df

        st.session_state.cq_update_df = build_content_quality_table_from_seo(
            seo_df=st.session_state.seo_update_df,
            fr_map_by_url={bayut_url.strip(): bayut_fr, **comp_fr_map},
            tree_map_by_url={bayut_url.strip(): {"nodes": bayut_nodes}, **{u: comp_tree_map[u] for u in competitors}},
            manual_query=manual_fkw_update.strip(),
            location_name=DEFAULT_LOCATION_NAME
        )

    if show_internal_fetch and st.session_state.update_fetch:
        st.sidebar.markdown("### Internal fetch log (Update Mode)")
        st.sidebar.write(f"Playwright enabled: {PLAYWRIGHT_OK}")
        for u, s in st.session_state.update_fetch:
            st.sidebar.write(u, "—", s)

    gaps_clicked = section_header_with_ai_button("Gaps Table", "Summarize by AI", "btn_gaps_summary_update")
    if gaps_clicked:
        if st.session_state.update_df is None or st.session_state.update_df.empty:
            st.error("Run analysis first (current URLs).")
        else:
            st.session_state["gaps_update_summary_text"] = ai_summary_from_df("gaps", st.session_state.update_df)

    if st.session_state.get("gaps_update_summary_text"):
        st.markdown(
            f"<div class='ai-summary'><b>AI Summary</b><div class='muted'>6–8 bullets only.</div>"
            f"<pre style='white-space:pre-wrap;margin:8px 0 0 0;'>{st.session_state['gaps_update_summary_text']}</pre></div>",
            unsafe_allow_html=True
        )

    if st.session_state.update_df is None or st.session_state.update_df.empty:
        st.info("Run analysis to see results.")
    else:
        render_table(st.session_state.update_df)

    st.markdown("<div class='section-pill section-pill-tight'>SEO Analysis</div>", unsafe_allow_html=True)
    render_table(st.session_state.seo_update_df, drop_internal_url=True)

    st.markdown("<div class='section-pill section-pill-tight'>AI Visibility (Google AI Overview)</div>", unsafe_allow_html=True)
    render_table(st.session_state.ai_update_df, drop_internal_url=True)

    st.markdown("<div class='section-pill section-pill-tight'>Content Quality</div>", unsafe_allow_html=True)
    render_table(st.session_state.cq_update_df, drop_internal_url=True)

else:
    st.markdown("<div class='section-pill section-pill-tight'>New Post Mode</div>", unsafe_allow_html=True)
    st.info("New Post Mode can be re-added the same way (DataForSEO + signature reset).")
