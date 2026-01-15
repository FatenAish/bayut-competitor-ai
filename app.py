# app.py
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
        font-weight: 800;
        color: {TEXT_DARK};
        display: inline-block;
        margin: 16px 0 10px 0;
      }}
      .stTextInput input, .stTextArea textarea {{
        background: {LIGHT_GREEN} !important;
        border: 1px solid {LIGHT_GREEN_2} !important;
        border-radius: 12px !important;
      }}
      .stButton button {{
        border-radius: 14px !important;
        padding: 0.65rem 1rem !important;
        font-weight: 800 !important;
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
        font-weight: 800 !important;
        text-decoration: underline !important;
      }}
      code {{
        background: rgba(0,0,0,0.04);
        padding: 2px 6px;
        border-radius: 8px;
      }}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    f"""
    <div class="hero">
      <h1><span class="bayut">Bayut</span> Competitor Gap Analysis</h1>
      <p><b>Header-first</b> gap logic — Missing headers first, then content gaps under matching headers. FAQ subjects-only.</p>
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
                results[u] = FetchResult(True, "manual", 200, "", pasted.strip(), None)

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

# extra “generic headings” that should NOT create missing-header rows
GENERIC_SECTION_HEADERS = {
    "introduction", "overview", "conclusion", "final thoughts", "summary",
    "closing thoughts", "wrap up", "in summary", "takeaway", "key takeaways",
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
                fr_map[u] = FetchResult(True, "manual", 200, "", repaste.strip(), None)

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
            "header": n["header"],
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

def header_is_faq(header: str) -> bool:
    nh = norm_header(header)
    return ("faq" in nh) or ("frequently asked" in nh)

def find_faq_nodes(nodes: List[dict]) -> List[dict]:
    faq = []
    for x in flatten(nodes):
        if x["level"] in (2, 3) and header_is_faq(x["header"]):
            faq.append(x)
    return faq

def normalize_question(q: str) -> str:
    q = clean(q or "")
    q = re.sub(r"^\s*\d+[\.\)]\s*", "", q)
    q = q.strip()
    return q

def extract_questions_from_node(node: dict) -> List[str]:
    qs = []
    def walk(n: dict):
        for c in n.get("children", []):
            hdr = clean(c.get("header", ""))
            if not hdr or is_noise_header(hdr):
                walk(c)
                continue
            if "?" in hdr or re.match(r"^\s*\d+[\.\)]\s+.*", hdr):
                qs.append(normalize_question(hdr))
            walk(c)
    walk(node)

    seen = set()
    out = []
    for q in qs:
        k = norm_header(q).replace(" ", "")
        if k in seen:
            continue
        seen.add(k)
        out.append(q)
    return out[:25]


# =====================================================
# KEYWORDS / THEME EXTRACTION (NO QUOTES)
# (keywords are OK for internal detection, but NOT shown to user)
# =====================================================
STOP = {
    "the","and","for","with","that","this","from","you","your","are","was","were","will","have","has","had",
    "but","not","can","may","more","most","into","than","then","they","them","their","our","out","about",
    "also","over","under","between","within","near","where","when","what","why","how","who","which",
    "a","an","to","of","in","on","at","as","is","it","be","or","by","we","i","us"
}

GENERIC_STOP = {
    "dubai","uae","business","bay","community","area","living","pros","cons",
    "property","properties","rent","sale","apartments","villas"
}

def top_keywords(text: str, n: int = 7) -> List[str]:
    words = re.findall(r"[a-zA-Z]{4,}", (text or "").lower())
    freq = {}
    for w in words:
        if w in STOP or w in GENERIC_STOP:
            continue
        freq[w] = freq.get(w, 0) + 1
    return [w for w, _ in sorted(freq.items(), key=lambda x: (-x[1], x[0]))[:n]]


# =====================================================
# HUMAN THEME SUMMARIES (SHOWN TO USER)
# =====================================================
def human_themes_from_text(text: str) -> List[str]:
    """
    Turn competitor content into human themes (no keyword dumps).
    Deterministic, lightweight.
    """
    t = (text or "").lower()
    themes: List[str] = []

    def has_any(words: List[str]) -> bool:
        return any(w in t for w in words)

    if has_any(["metro", "public transport", "bus", "commute", "roads", "highway", "access", "connectivity"]):
        themes.append("connectivity & commute")

    if has_any(["parking", "traffic", "congestion", "rush", "busy roads"]):
        themes.append("traffic & parking realities")

    if has_any(["cost", "expensive", "budget", "afford", "pricing", "rent", "rental"]):
        themes.append("cost considerations")

    if has_any(["restaurants", "cafes", "nightlife", "vibe", "atmosphere", "lifestyle"]):
        themes.append("lifestyle & vibe")

    if has_any(["burj", "downtown", "dubai mall", "landmarks", "attractions"]):
        themes.append("nearby landmarks & attractions")

    if has_any(["marina", "beach", "waterfront", "walk", "promenade"]):
        themes.append("waterfront lifestyle angle")

    if has_any(["schools", "clinic", "hospital", "supermarket", "family", "kids"]):
        themes.append("day-to-day convenience")

    if has_any(["pros", "cons", "advantages", "disadvantages", "weigh", "consider"]):
        themes.append("decision framing (pros vs cons)")

    # de-dupe keep order
    out, seen = [], set()
    for x in themes:
        k = norm_header(x)
        if k in seen:
            continue
        seen.add(k)
        out.append(x)
    return out[:5]


def summarize_missing_section_action(header: str, subheaders: Optional[List[str]], comp_content: str) -> str:
    hn = norm_header(header)

    # Pros/cons importance intro
    if ("importance" in hn and "pros" in hn and "cons" in hn) or ("consider" in hn and "pros" in hn and "cons" in hn):
        return (
            "Add a short intro that sets expectations: explain why readers should weigh the benefits vs drawbacks before deciding "
            "to move, so the rest of the article feels like a clear decision guide."
        )

    # Comparison section grouping
    if "comparison" in hn or "compare" in hn:
        msg = (
            "Add a comparison section with other Dubai neighborhoods, explaining how they differ from Business Bay "
            "(vibe, access, lifestyle fit, and who each area suits best)."
        )
        if subheaders:
            bullets = "<br>".join([f"• {h}" for h in subheaders[:8]])
            msg += f"<br><br>Competitor breaks this down into subsections like:<br>{bullets}"
        return msg

    themes = human_themes_from_text(comp_content)
    msg = f"Add a dedicated section for <b>{header}</b> with practical, decision-led details."
    if themes:
        msg += " Cover points like: " + ", ".join(themes) + "."
    return msg


def summarize_content_gap_action(header: str, comp_content: str, bayut_content: str) -> str:
    themes = human_themes_from_text(comp_content)
    msg = f"Expand <b>{header}</b> — Bayut has the header, but the competitor goes deeper."
    if themes:
        msg += " Add more detail around: " + ", ".join(themes) + "."
    else:
        msg += " Add more decision-led detail and practical specifics."
    return msg


# =====================================================
# SECTION EXTRACTION (HEADER-FIRST COMPARISON)
# =====================================================
def section_nodes(nodes: List[dict], levels=(2,3)) -> List[dict]:
    """
    Returns list of section dicts:
    {level, header, content, parent_h2}
    Only H2/H3 by default.
    """
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

    # de-dupe by normalized header (keep first)
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
    """
    Stable header match score using:
    - token overlap
    - sequence similarity
    """
    a_n = norm_header(a)
    b_n = norm_header(b)
    if not a_n or not b_n:
        return 0.0

    a_set = set(a_n.split())
    b_set = set(b_n.split())
    if not a_set or not b_set:
        jacc = 0.0
    else:
        jacc = len(a_set & b_set) / max(len(a_set | b_set), 1)

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


def content_gap(comp_section: dict, bayut_section: dict) -> Tuple[bool, str]:
    """
    Decide if there is a content gap under same header.
    Returns: (gap?, short_reason)
    Rules:
    - competitor section must have meaningful length
    - competitor must be significantly longer AND introduce missing themes
    """
    c_txt = clean(comp_section.get("content",""))
    b_txt = clean(bayut_section.get("content",""))

    if len(c_txt) < 120:
        return False, ""

    if len(b_txt) < 60 and len(c_txt) >= 160:
        return True, "Bayut header exists but has minimal content compared to competitor."

    if len(c_txt) < (1.35 * max(len(b_txt), 1)):
        return False, ""

    c_kws = top_keywords(c_txt, n=9)
    b_low = b_txt.lower()
    missing = [k for k in c_kws if k not in b_low]

    if len(missing) >= 3:
        return True, "Competitor expands the section with additional decision themes."
    return False, ""


def dedupe_rows(rows: List[dict]) -> List[dict]:
    """
    Remove repeated rows caused by overlaps.
    Key = normalized Header(Gap) + normalized source name.
    """
    out = []
    seen = set()
    for r in rows:
        hk = norm_header(r.get("Header (Gap)",""))
        sk = norm_header(re.sub(r"<[^>]+>", "", r.get("Source","")))
        k = hk + "||" + sk
        if k in seen:
            continue
        seen.add(k)
        out.append(r)
    return out


# =======================
# FAQ SUBJECT LOGIC (KEEP)
# =======================
def faq_subject(q: str) -> str:
    s = norm_header(normalize_question(q))

    if any(k in s for k in ["where is", "located", "location", "map", "how to get", "nearest", "distance"]):
        return "Location"
    if any(k in s for k in ["price", "starting price", "cost", "how much", "prices"]):
        return "Pricing"
    if any(k in s for k in ["types of properties", "property types", "villa", "townhouse", "apartment", "layouts", "floor plan", "floor plans", "bedroom", "unit types"]):
        return "Property types & layouts"
    if any(k in s for k in ["amenities", "facilities", "features", "pool", "gym", "park"]):
        return "Amenities"
    if any(k in s for k in ["developer", "developed by", "who is the developer"]):
        return "Developer"
    if any(k in s for k in ["how big", "size", "area", "master plan", "community size", "sqm", "sq ft"]):
        return "Community size & master plan"
    if any(k in s for k in ["payment plan", "payment", "installment", "down payment", "cash", "crypto", "cryptocurrency", "mortgage"]):
        return "Payment & purchase methods"
    if any(k in s for k in ["foreign", "investors", "freehold", "eligib", "can i buy", "ownership"]):
        return "Eligibility & ownership"
    if any(k in s for k in ["handover", "completion", "ready", "move in", "construction", "progress", "delivery", "available now"]):
        return "Handover & availability"
    if any(k in s for k in ["brochure", "download", "pdf"]):
        return "Brochure & downloads"
    if any(k in s for k in ["expression of interest", "register", "enquire", "inquire", "contact", "how can i submit", "submit", "interest"]):
        return "How to register interest"
    if any(k in s for k in ["what is", "about", "overview"]):
        return "Project overview"

    return "Other FAQs"

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
    bayut_faq_nodes = find_faq_nodes(bayut_nodes)
    comp_faq_nodes = find_faq_nodes(comp_nodes)
    if not comp_faq_nodes:
        return None

    comp_qs = []
    for fn in comp_faq_nodes:
        comp_qs.extend(extract_questions_from_node(fn))
    comp_qs = [q for q in comp_qs if q and len(q) > 5]
    if not comp_qs:
        return None

    def q_key(q: str) -> str:
        q2 = normalize_question(q)
        q2 = re.sub(r"[^a-z0-9\s]", "", q2.lower())
        q2 = re.sub(r"\s+", " ", q2).strip()
        return q2

    if bayut_faq_nodes:
        bayut_qs = []
        for fn in bayut_faq_nodes:
            bayut_qs.extend(extract_questions_from_node(fn))
        bayut_set = {q_key(q) for q in bayut_qs if q}

        missing_qs = [q for q in comp_qs if q_key(q) not in bayut_set]
        if not missing_qs:
            return None

        subjects = faq_subjects_from_questions(missing_qs, limit=10)
        msg = "Add missing FAQ topics (subjects only), such as: " + ", ".join(subjects) + "."
        return {
            "Header (Gap)": "Missing FAQs",
            "What to add (Bayut action)": msg,
            "Source": source_link(comp_url),
        }

    subjects = faq_subjects_from_questions(comp_qs, limit=10)
    msg = "Add an FAQ block on Bayut. Suggested topics include: " + ", ".join(subjects) + "."
    return {
        "Header (Gap)": "FAQ Block",
        "What to add (Bayut action)": msg,
        "Source": source_link(comp_url),
    }


# =====================================================
# UPDATE MODE ENGINE (HEADER-FIRST, THEN CONTENT GAPS)
# (FIXED: groups H3 under missing H2; human wording)
# =====================================================
def update_mode_rows_header_first(bayut_nodes: List[dict], comp_nodes: List[dict], comp_url: str,
                                 max_missing_headers: int = 10,
                                 max_content_gaps: int = 8) -> List[dict]:
    """
    Rules:
    1) Check missing headers first (competitor H2/H3 not in Bayut)
    2) If Bayut has header, only add a gap if content is weaker
    3) FAQ row stays
    EXTRA:
    - If an H2 is missing, do NOT output its H3 children as separate missing rows.
      They are grouped INSIDE the H2 action text (your request).
    - Output wording is human (no keyword lists).
    """
    rows: List[dict] = []

    bayut_secs = section_nodes(bayut_nodes, levels=(2,3))
    comp_secs = section_nodes(comp_nodes, levels=(2,3))

    bayut_h2 = [s for s in bayut_secs if s["level"] == 2]
    bayut_h3 = [s for s in bayut_secs if s["level"] == 3]
    comp_h2 = [s for s in comp_secs if s["level"] == 2]
    comp_h3 = [s for s in comp_secs if s["level"] == 3]

    missing_h2_norms: set = set()

    # --- 1) Missing H2 headers (group their H3 inside the same row) ---
    missing_rows: List[dict] = []
    for cs in comp_h2:
        m = find_best_bayut_match(cs["header"], bayut_h2, min_score=0.73)
        if not m:
            missing_h2_norms.add(norm_header(cs["header"]))

            # collect H3 children for this H2 (if any)
            children = []
            child_text_parts = []
            for h3 in comp_h3:
                if norm_header(h3.get("parent_h2") or "") == norm_header(cs["header"]):
                    children.append(h3["header"])
                    if h3.get("content"):
                        child_text_parts.append(h3["content"])

            comp_text = (cs.get("content","") or "") + " " + " ".join(child_text_parts)

            missing_rows.append({
                "Header (Gap)": cs["header"],
                "What to add (Bayut action)": summarize_missing_section_action(cs["header"], children, comp_text),
                "Source": source_link(comp_url),
            })

    rows.extend(missing_rows[:max_missing_headers])

    # --- 1b) Missing H3 headers ONLY if their parent H2 is NOT missing ---
    # (So Downtown/JLT/Marina won't appear as separate rows if "Comparison..." H2 is missing.)
    remaining_slots = max(0, max_missing_headers - len(rows))
    missing_h3_rows: List[dict] = []

    for cs in comp_h3:
        parent = cs.get("parent_h2") or ""
        if parent and norm_header(parent) in missing_h2_norms:
            continue  # grouped under missing parent H2

        # parent should exist in Bayut (or be matched)
        if parent:
            parent_match = find_best_bayut_match(parent, bayut_h2, min_score=0.73)
            if not parent_match:
                continue

        m = find_best_bayut_match(cs["header"], bayut_h3, min_score=0.73)
        if not m:
            label = f"{parent} → {cs['header']}" if parent else cs["header"]
            missing_h3_rows.append({
                "Header (Gap)": label,
                "What to add (Bayut action)": summarize_missing_section_action(cs["header"], None, cs.get("content","")),
                "Source": source_link(comp_url),
            })

    if remaining_slots > 0:
        rows.extend(missing_h3_rows[:remaining_slots])

    # --- 2) Content gaps under matching headers (human) ---
    content_gap_rows: List[dict] = []
    for cs in comp_secs:
        # skip if missing H2 already claimed
        if cs["level"] == 2 and norm_header(cs["header"]) in missing_h2_norms:
            continue

        m = find_best_bayut_match(cs["header"], bayut_secs, min_score=0.73)
        if not m:
            continue

        bs = m["bayut_section"]
        gap, _reason = content_gap(cs, bs)
        if gap:
            content_gap_rows.append({
                "Header (Gap)": f"{bs['header']} (Content gap)",
                "What to add (Bayut action)": summarize_content_gap_action(bs["header"], cs.get("content",""), bs.get("content","")),
                "Source": source_link(comp_url),
            })

    rows.extend(content_gap_rows[:max_content_gaps])

    # --- 3) FAQ row (subjects-only) ---
    faq_row = missing_faqs_row(bayut_nodes, comp_nodes, comp_url)
    if faq_row:
        rows.append(faq_row)

    return dedupe_rows(rows)


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
# HTML TABLE RENDER (with hyperlinks)
# =====================================================
def render_table(df: pd.DataFrame):
    if df.empty:
        st.info("No results to show.")
        return
    html = df.to_html(index=False, escape=False)
    st.markdown(html, unsafe_allow_html=True)


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

if "new_df" not in st.session_state:
    st.session_state.new_df = pd.DataFrame()
if "new_fetch" not in st.session_state:
    st.session_state.new_fetch = []


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
        bayut_nodes = bayut_tree_map[bayut_url.strip()]["nodes"]

        # 2) HARD: ALL competitors must be available (or paste)
        with st.spinner("Fetching ALL competitors (no exceptions)…"):
            comp_fr_map = resolve_all_or_require_manual(agent, competitors, st_key_prefix="comp_update")
            comp_tree_map = ensure_headings_or_require_repaste(competitors, comp_fr_map, st_key_prefix="comp_update_tree")

        # 3) Build rows (missing headers first, then content gaps)
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
                max_missing_headers=10,
                max_content_gaps=8
            ))

        st.session_state.update_fetch = internal_fetch
        st.session_state.update_df = (
            pd.DataFrame(all_rows)[["Header (Gap)", "What to add (Bayut action)", "Source"]]
            if all_rows
            else pd.DataFrame(columns=["Header (Gap)", "What to add (Bayut action)", "Source"])
        )

    if show_internal_fetch and st.session_state.update_fetch:
        st.sidebar.markdown("### Internal fetch log (Update Mode)")
        st.sidebar.write(f"Playwright enabled: {PLAYWRIGHT_OK}")
        for u, s in st.session_state.update_fetch:
            st.sidebar.write(u, "—", s)

    st.markdown("<div class='section-pill'>Gaps Table</div>", unsafe_allow_html=True)

    if st.session_state.update_df is None or st.session_state.update_df.empty:
        st.info("Run analysis to see results.")
    else:
        render_table(st.session_state.update_df)


# =====================================================
# UI - NEW POST MODE
# =====================================================
else:
    st.markdown("<div class='section-pill'>New Post Mode</div>", unsafe_allow_html=True)

    new_title = st.text_input("New post title", placeholder="Arabian Ranches vs Mudon")
    competitors_text = st.text_area(
        "Competitor URLs (one per line)",
        height=120,
        placeholder="https://example.com/article\nhttps://example.com/another"
    )
    competitors = [c.strip() for c in competitors_text.splitlines() if c.strip()]

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

    if show_internal_fetch and st.session_state.new_fetch:
        st.sidebar.markdown("### Internal fetch log (New Post Mode)")
        st.sidebar.write(f"Playwright enabled: {PLAYWRIGHT_OK}")
        for u, s in st.session_state.new_fetch:
            st.sidebar.write(u, "—", s)

    st.markdown("<div class='section-pill'>Competitor Coverage</div>", unsafe_allow_html=True)

    if st.session_state.new_df is None or st.session_state.new_df.empty:
        st.info("Generate competitor coverage to see results.")
    else:
        render_table(st.session_state.new_df)
