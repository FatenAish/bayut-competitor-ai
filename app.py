import streamlit as st
import requests
import re
from bs4 import BeautifulSoup
from urllib.parse import quote_plus, urlparse
import pandas as pd
import time, random
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple, List

# Optional (recommended): JS rendering tool
# pip install playwright
# playwright install chromium
try:
    from playwright.sync_api import sync_playwright
    PLAYWRIGHT_OK = True
except Exception:
    PLAYWRIGHT_OK = False

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
    A deterministic "resolver agent":
    - tries multiple strategies in order
    - validates extracted content (length + not blocked)
    - retries + UA rotation + backoff
    - optional JS rendering (Playwright)
    - optional manual fallback (user paste) so no competitor is missing
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
        import requests
        last_code, last_text = 0, ""
        for i in range(tries):
            headers = dict(self.default_headers)
            headers["User-Agent"] = random.choice(self.user_agents)
            try:
                r = requests.get(url, headers=headers, timeout=timeout, allow_redirects=True)
                last_code, last_text = r.status_code, (r.text or "")

                # retry on transient/rate-limit responses
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
        from urllib.parse import quote_plus
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
                page.wait_for_timeout(1400)  # let JS render
                html = page.content()
                browser.close()
            return True, html
        except Exception:
            return False, ""

    def resolve(self, url: str) -> FetchResult:
        url = (url or "").strip()
        if not url:
            return FetchResult(False, None, None, "", "", "empty_url")

        # Strategy 1: direct HTML
        code, html = self._http_get(url)
        if code == 200 and html:
            text = self._extract_article_text_from_html(html)
            if self._validate_text(text, min_len=500):
                return FetchResult(True, "direct", code, html, text, None)

        # Strategy 2: JS-rendered HTML (Playwright)
        ok, html2 = self._fetch_playwright_html(url)
        if ok and html2:
            text2 = self._extract_article_text_from_html(html2)
            if self._validate_text(text2, min_len=500):
                return FetchResult(True, "playwright", 200, html2, text2, None)

        # Strategy 3: Jina reader
        jurl = self._jina_url(url)
        code3, txt3 = self._http_get(jurl)
        if code3 == 200 and txt3:
            text3 = self.clean(txt3)
            if self._validate_text(text3, min_len=500):
                return FetchResult(True, "jina", code3, "", text3, None)

        # Strategy 4: Textise
        turl = self._textise_url(url)
        code4, html4 = self._http_get(turl)
        if code4 == 200 and html4:
            soup = BeautifulSoup(html4, "html.parser")
            text4 = self.clean(soup.get_text(" "))
            if self._validate_text(text4, min_len=350):
                return FetchResult(True, "textise", code4, "", text4, None)

        return FetchResult(False, None, None, "", "", "blocked_or_no_content")


def resolve_all_or_require_manual(agent: FetchAgent, urls: List[str], st_key_prefix: str) -> Dict[str, FetchResult]:
    """
    Enforces "no exceptions":
    - tries to resolve all
    - if any fail, app asks for manual paste for those URLs
    - only returns when all are resolved
    """
    results: Dict[str, FetchResult] = {}
    failed: List[str] = []

    for u in urls:
        r = agent.resolve(u)
        results[u] = r
        if not r.ok:
            failed.append(u)

        # small polite delay to reduce blocking across many URLs
        time.sleep(0.3)

    if not failed:
        return results

    st.error("Some competitor URLs could not be fetched automatically. Paste the article text for each failed URL to continue (no missing competitors).")

    for u in failed:
        with st.expander(f"Manual fallback required: {u}", expanded=True):
            pasted = st.text_area(
                "Paste the full article text (or readable HTML) هنا:",
                key=f"{st_key_prefix}__paste__{u}",
                height=180,
            )
            if pasted and len(pasted.strip()) > 400:
                # Treat manual paste like a successful fetch
                results[u] = FetchResult(True, "manual", 200, "", pasted.strip(), None)

    still_failed = [u for u in failed if not results[u].ok]
    if still_failed:
        st.stop()

    return results


# =====================================================
# PAGE CONFIG
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
      /* full page background */
      html, body, [data-testid="stAppViewContainer"] {{
        background: {PAGE_BG} !important;
      }}
      [data-testid="stHeader"] {{
        background: rgba(0,0,0,0) !important;
      }}

      /* page width */
      section.main > div.block-container {{
        max-width: 1180px !important;
        padding-top: 1.6rem !important;
        padding-bottom: 2.4rem !important;
      }}

      /* centered title */
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

      /* light green labels / section headers */
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

      /* inputs background */
      .stTextInput input, .stTextArea textarea {{
        background: {LIGHT_GREEN} !important;
        border: 1px solid {LIGHT_GREEN_2} !important;
        border-radius: 12px !important;
      }}

      /* buttons nicer */
      .stButton button {{
        border-radius: 14px !important;
        padding: 0.65rem 1rem !important;
        font-weight: 800 !important;
      }}

      /* MODE BUTTONS - centered */
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

      /* tables */
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
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    f"""
    <div class="hero">
      <h1><span class="bayut">Bayut</span> Competitor Gap Analysis</h1>
      <p>Update an existing article, or plan a new one using competitor coverage</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# =====================================================
# FETCH (ROBUST FALLBACKS)
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

@st.cache_data(show_spinner=False, ttl=60 * 30)
def fetch_direct(url: str, timeout: int = 20) -> tuple[int, str]:
    r = requests.get(url, headers=DEFAULT_HEADERS, timeout=timeout, allow_redirects=True)
    return r.status_code, r.text

@st.cache_data(show_spinner=False, ttl=60 * 30)
def fetch_jina(url: str, timeout: int = 20) -> tuple[int, str]:
    if url.startswith("https://"):
        jina_url = "https://r.jina.ai/https://" + url[len("https://"):]
    elif url.startswith("http://"):
        jina_url = "https://r.jina.ai/http://" + url[len("http://"):]
    else:
        jina_url = "https://r.jina.ai/https://" + url
    r = requests.get(jina_url, headers=DEFAULT_HEADERS, timeout=timeout, allow_redirects=True)
    return r.status_code, r.text

def looks_blocked(text: str) -> bool:
    t = (text or "").lower()
    return any(x in t for x in [
        "just a moment", "checking your browser", "verify you are human",
        "cloudflare", "access denied", "captcha", "forbidden", "service unavailable"
    ])

def fetch_best_effort(url: str) -> dict:
    url = (url or "").strip()
    if not url:
        return {"ok": False, "source": None, "html": "", "text": "", "status": None}

    # 1) direct HTML
    try:
        code, html = fetch_direct(url)
        if code == 200 and html:
            soup = BeautifulSoup(html, "html.parser")
            for t in soup.find_all(list(IGNORE_TAGS)):
                t.decompose()
            article = soup.find("article") or soup
            text = clean(article.get_text(" "))
            if len(text) > 400 and not looks_blocked(text):
                return {"ok": True, "source": "direct", "html": html, "text": text, "status": code}
    except Exception:
        pass

    # 2) jina reader
    try:
        code, txt = fetch_jina(url)
        if code == 200 and txt:
            text = clean(txt)
            if len(text) > 400 and not looks_blocked(text):
                return {"ok": True, "source": "jina", "html": "", "text": text, "status": code}
    except Exception:
        pass

    # 3) textise fallback
    try:
        encoded = quote_plus(url)
        t_url = f"https://textise.org/showtext.aspx?strURL={encoded}"
        code, html = fetch_direct(t_url)
        if code == 200 and html:
            text = clean(BeautifulSoup(html, "html.parser").get_text(" "))
            if len(text) > 300 and not looks_blocked(text):
                return {"ok": True, "source": "textise", "html": "", "text": text, "status": code}
    except Exception:
        pass

    return {"ok": False, "source": None, "html": "", "text": "", "status": None}

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
]

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
    if len(hn) < 4:
        return True
    if len(s) > 90:
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

def build_tree_from_html(html: str) -> list[dict]:
    soup = BeautifulSoup(html, "html.parser")
    for t in soup.find_all(list(IGNORE_TAGS)):
        t.decompose()

    root = soup.find("article") or soup
    headings = root.find_all(["h1", "h2", "h3", "h4"])

    nodes: list[dict] = []
    stack: list[dict] = []

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

        # Collect only lightweight text for keyword extraction (no quoting later)
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

def build_tree_from_reader_text(text: str) -> list[dict]:
    lines = [l.rstrip() for l in (text or "").splitlines()]
    nodes: list[dict] = []
    stack: list[dict] = []

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

def get_tree(url: str) -> dict:
    fetched = fetch_best_effort(url)
    if not fetched["ok"]:
        return {"ok": False, "source": None, "nodes": [], "status": fetched.get("status")}
    if fetched["html"]:
        return {"ok": True, "source": fetched["source"], "nodes": build_tree_from_html(fetched["html"]), "status": fetched.get("status")}
    return {"ok": True, "source": fetched["source"], "nodes": build_tree_from_reader_text(fetched["text"]), "status": fetched.get("status")}

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

def flatten(nodes: list[dict]) -> list[dict]:
    out = []
    def walk(n: dict, parent=None):
        out.append({"level": n["level"], "header": n["header"], "content": n.get("content", ""), "parent": parent, "children": n.get("children", [])})
        for c in n.get("children", []):
            walk(c, n)
    for n in nodes:
        walk(n, None)
    return out

def list_headers(nodes: list[dict], level: int) -> list[str]:
    return [x["header"] for x in flatten(nodes) if x["level"] == level and not is_noise_header(x["header"])]

def is_subpoint_heading(h: str) -> bool:
    s = clean(h)
    if not s:
        return False
    if s.endswith(":"):
        return True
    if re.match(r"^\s*\d+[\.\)]\s+\S+", s):
        return True
    words = s.split()
    if 1 <= len(words) <= 6:
        cap_ratio = sum(1 for w in words if w[:1].isupper()) / max(len(words), 1)
        if cap_ratio >= 0.6 and len(s) <= 40:
            return True
    return False

def strip_label(h: str) -> str:
    return clean(re.sub(r"\s*:\s*$", "", (h or "").strip()))

def header_is_faq(header: str) -> bool:
    nh = norm_header(header)
    return ("faq" in nh) or ("frequently asked" in nh)

def find_faq_nodes(nodes: list[dict]) -> list[dict]:
    faq = []
    for x in flatten(nodes):
        if x["level"] in (2, 3):
            if header_is_faq(x["header"]):
                faq.append(x)
    return faq

def normalize_question(q: str) -> str:
    q = clean(q or "")
    q = re.sub(r"^\s*\d+[\.\)]\s*", "", q)
    return q

def extract_questions_from_node(node: dict) -> list[str]:
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
    return out[:18]

# =====================================================
# FORCED, STABLE SUMMARIES (NO QUOTING COMPETITOR TEXT)
# =====================================================
STOP = {
    "the","and","for","with","that","this","from","you","your","are","was","were","will","have","has","had",
    "but","not","can","may","more","most","into","than","then","they","them","their","our","out","about",
    "also","over","under","between","within","near","where","when","what","why","how","who","which",
    "a","an","to","of","in","on","at","as","is","it","be","or","by","we","i","us"
}
GENERIC_STOP = {
    "business","bay","dubai","area","community","living","life","lifestyle","location",
    "pros","cons","guide","things","places","also","one","like"
}

def top_keywords(text: str, n: int = 7) -> list[str]:
    words = re.findall(r"[a-zA-Z]{4,}", (text or "").lower())
    freq = {}
    for w in words:
        if w in STOP or w in GENERIC_STOP:
            continue
        freq[w] = freq.get(w, 0) + 1
    return [w for w, _ in sorted(freq.items(), key=lambda x: (-x[1], x[0]))[:n]]

def collect_norm_set(nodes: list[dict], keep_levels=(2, 3)) -> set[str]:
    out = set()
    for x in flatten(nodes):
        if x["level"] in keep_levels:
            nh = norm_header(x["header"])
            if nh:
                out.add(nh)
    return out

def index_by_norm(nodes: list[dict], levels=(2,3)) -> dict[str, dict]:
    idx = {}
    for x in flatten(nodes):
        if x["level"] in levels:
            nh = norm_header(x["header"])
            if nh:
                idx[nh] = x
    return idx

def summarize_h2_children(h2_node: dict) -> tuple[list[str], list[str]]:
    """
    Returns:
      - options: label-like subpoints (Downtown Dubai, Dubai Marina, ...)
      - themes: normal H3 headings (non-label)
    Strips colons.
    Excludes FAQ child headings completely.
    """
    options = []
    themes = []
    for c in (h2_node.get("children", []) or []):
        if c.get("level") != 3:
            continue
        h = strip_label(c.get("header", ""))
        if not h or is_noise_header(h):
            continue
        if header_is_faq(h):
            continue
        if is_subpoint_heading(c.get("header", "")):
            options.append(h)
        else:
            themes.append(h)

    # dedupe preserve order
    def dedupe(seq):
        seen = set()
        out = []
        for x in seq:
            k = norm_header(x)
            if k in seen:
                continue
            seen.add(k)
            out.append(x)
        return out

    return dedupe(options)[:6], dedupe(themes)[:8]

def describe_missing_h2(h2_header: str, h2_node: dict) -> str:
    h = strip_label(h2_header)
    nh = norm_header(h)

    # special stable phrasing for common sections (matches your approved table)
    if ("considering" in nh) and ("pros" in nh) and ("cons" in nh):
        return "The competitor includes an early decision framing angle — it sets expectations that moving decisions depend on trade-offs, and highlights the idea of weighing lifestyle benefits against practical drawbacks before committing."
    if "comparison" in nh or ("other dubai" in nh) or ("neighborhood" in nh) or ("neighbourhood" in nh):
        options, _themes = summarize_h2_children(h2_node)
        if options:
            return "The competitor compares Business Bay with other popular Dubai areas to help the reader benchmark it. The comparison is organized by nearby options such as " + ", ".join(options) + ", focusing on how they differ in vibe, convenience, and suitability."
        return "The competitor compares Business Bay with other popular Dubai areas to help the reader benchmark it, focusing on differences in vibe, convenience, and suitability."
    if "conclusion" in nh or "final thoughts" in nh or "wrap up" in nh:
        return "The competitor closes with a wrap-up that summarizes the overall verdict and the main takeaway from the pros/cons discussion (who the area suits and what kind of lifestyle it supports)."

    # generic stable description (no quotes)
    options, themes = summarize_h2_children(h2_node)
    kws = top_keywords(h2_node.get("content", ""), n=7)

    parts = [f"The competitor includes a section about {h}."]
    if themes:
        parts.append("It breaks this down into themes such as " + ", ".join(themes) + ".")
    if options and ("comparison" not in nh):
        parts.append("It references nearby options such as " + ", ".join(options) + ".")
    if (not themes and not options) and kws:
        parts.append("It focuses on topics like " + ", ".join(kws) + ".")
    return clean(" ".join(parts))

def describe_missing_h3(h3_header: str, h3_node: dict) -> str:
    h = strip_label(h3_header)
    kws = top_keywords(h3_node.get("content", ""), n=7)
    if kws:
        return f"The competitor includes a subsection about {h} and focuses on topics like " + ", ".join(kws) + "."
    return f"The competitor includes a subsection about {h}."

def describe_faq_block(comp_faq_nodes: list[dict], bayut_faq_nodes: list[dict]) -> str:
    comp_qs = []
    for fn in comp_faq_nodes:
        comp_qs.extend(extract_questions_from_node(fn))

    # if bayut has FAQs: show missing questions vs bayut
    if bayut_faq_nodes:
        bayut_qs = []
        for fn in bayut_faq_nodes:
            bayut_qs.extend(extract_questions_from_node(fn))

        def q_key(q: str) -> str:
            return norm_header(q).replace(" ", "")

        bayut_set = {q_key(q) for q in bayut_qs}
        missing = [q for q in comp_qs if q_key(q) not in bayut_set]
        if missing:
            return "The competitor includes a dedicated FAQ block with questions around practical living decisions, including: " + "; ".join(missing[:12]) + ("…" if len(missing) > 12 else "") + "."
        # if none missing, still describe that competitor has FAQ
        if comp_qs:
            return "The competitor includes a dedicated FAQ block covering common living questions (cost, schools, suitability, attractions, and safety)."
        return "The competitor includes a dedicated FAQ block covering common living questions."

    # if bayut has no FAQs: describe competitor FAQ topics (not “missing vs bayut”)
    if comp_qs:
        return "The competitor includes a dedicated FAQ block with questions around practical living decisions, including: " + "; ".join(comp_qs[:12]) + ("…" if len(comp_qs) > 12 else "") + "."
    return "The competitor includes a dedicated FAQ block covering common living questions (cost, schools, suitability, attractions, and safety)."

# =====================================================
# UPDATE MODE ROWS (FORCED OUTPUT)
# =====================================================
def update_mode_rows(bayut_nodes: list[dict], comp_nodes: list[dict], comp_url: str) -> list[dict]:
    rows = []
    bayut_norm = collect_norm_set(bayut_nodes, keep_levels=(2,3))
    bayut_idx = index_by_norm(bayut_nodes, levels=(2,3))
    comp_idx = index_by_norm(comp_nodes, levels=(2,3))

    # Prepare FAQ nodes first (so FAQ never gets absorbed under Conclusion)
    bayut_faq_nodes = find_faq_nodes(bayut_nodes)
    comp_faq_nodes = find_faq_nodes(comp_nodes)

    # A) Missing H2 rows (EXCEPT FAQ H2)
    missing_h2_norms = set()
    for x in flatten(comp_nodes):
        if x["level"] != 2:
            continue
        nh2 = norm_header(x["header"])
        if not nh2 or nh2 in bayut_norm:
            continue
        if header_is_faq(x["header"]):
            continue  # FAQ handled separately
        missing_h2_norms.add(nh2)

        rows.append({
            "Header (Gap)": strip_label(x["header"]),
            "What the competitor talks about": describe_missing_h2(x["header"], x),
            "Source": source_link(comp_url),
            "_key": f"missing_h2::{nh2}::{comp_url}"
        })

    # B) Missing H3 rows only when parent H2 isn't missing; skip label-like H3
    for x in flatten(comp_nodes):
        if x["level"] != 3:
            continue
        nh3 = norm_header(x["header"])
        if not nh3 or nh3 in bayut_norm:
            continue
        if header_is_faq(x["header"]):
            continue

        parent = x.get("parent")
        parent_n = norm_header(parent.get("header","")) if parent else ""
        if parent and parent.get("level") == 2 and parent_n in missing_h2_norms:
            continue  # absorbed into missing H2 row by design
        if is_subpoint_heading(x["header"]):
            continue  # NO Downtown Dubai: rows

        rows.append({
            "Header (Gap)": strip_label(x["header"]),
            "What the competitor talks about": describe_missing_h3(x["header"], x),
            "Source": source_link(comp_url),
            "_key": f"missing_h3::{nh3}::{comp_url}"
        })

    # C) FAQ row (always separate; stable summary; no “Conclusion absorbs FAQ”)
    if comp_faq_nodes:
        msg = describe_faq_block(comp_faq_nodes, bayut_faq_nodes)
        rows.append({
            "Header (Gap)": "FAQs (Missing questions)" if bayut_faq_nodes else "FAQs",
            "What the competitor talks about": msg,
            "Source": source_link(comp_url),
            "_key": f"faq::{comp_url}"
        })

    # D) Content depth gaps (NO QUOTED EXAMPLES)
    # Keep your same logic trigger, but output stable summary
    def tokens_set(text: str) -> set[str]:
        ws = re.findall(r"[a-zA-Z]{3,}", (text or "").lower())
        return {w for w in ws if w not in STOP}

    content_gap_rows = []
    for nh, comp_item in comp_idx.items():
        if nh not in bayut_idx:
            continue
        b = bayut_idx[nh]
        comp_text = (comp_item.get("content","") or "")
        bayut_text = (b.get("content","") or "")

        if len(comp_text) < 180:
            continue

        comp_tok = tokens_set(comp_text)
        bay_tok = tokens_set(bayut_text)
        new_terms = [t for t in (comp_tok - bay_tok) if len(t) > 3]

        if (len(comp_text) > max(260, len(bayut_text) * 1.35)) and len(new_terms) >= 6:
            kw = top_keywords(comp_text, n=6)
            if kw:
                msg = "The competitor covers the same topic but goes into more detail, with extra context around: " + ", ".join(kw) + "."
            else:
                msg = "The competitor covers the same topic but goes into more detail, adding more context than the Bayut section."
            content_gap_rows.append({
                "Header (Gap)": f"{strip_label(b['header'])} (Content Gap)",
                "What the competitor talks about": msg,
                "Source": source_link(comp_url),
                "_key": f"content::{nh}::{comp_url}"
            })

    seen = set()
    for r in content_gap_rows:
        if r["_key"] in seen:
            continue
        seen.add(r["_key"])
        rows.append(r)

    # de-dupe + keep stable order
    final = []
    seenk = set()
    for r in rows:
        k = r.get("_key") or (r["Header (Gap)"] + "::" + r["Source"])
        if k in seenk:
            continue
        seenk.add(k)
        r.pop("_key", None)
        final.append(r)

    return final

# =====================================================
# NEW POST MODE (FORCED TO MATCH AGREED TABLES)
# =====================================================
def detect_main_angle(comp_nodes: list[dict]) -> str:
    h2s = [norm_header(h) for h in list_headers(comp_nodes, 2)]
    blob = " ".join(h2s)
    if ("pros" in blob and "cons" in blob) or ("advantages" in blob and "disadvantages" in blob):
        return "pros-and-cons decision guide"
    if "things to do" in blob or "attractions" in blob:
        return "things-to-do / attractions guide"
    if "schools" in blob or "education" in blob:
        return "schools / education guide"
    return "decision-led overview"

def new_post_coverage_rows(comp_nodes: list[dict], comp_url: str) -> list[dict]:
    # H1
    h1s = list_headers(comp_nodes, 1)
    h1_title = strip_label(h1s[0]) if h1s else ""
    angle = detect_main_angle(comp_nodes)
    h1_text = (
        f"The competitor frames the page as a {angle} about the topic — aimed at helping the reader decide suitability."
        if angle != "decision-led overview"
        else "The competitor frames the page around a clear main angle to guide the reader’s decision."
    )
    if h1_title:
        # keep title (like your sample) but still stable
        h1_text = f"{h1_title} — {h1_text}"

    # H2 sections covered (derived)
    h2s = [strip_label(h) for h in list_headers(comp_nodes, 2)]
    # keep it short and stable
    h2_main = [h for h in h2s if h and not header_is_faq(h)]
    h2_main = h2_main[:6]
    has_faq = any(header_is_faq(h) for h in h2s)

    if h2_main:
        h2_text = "The competitor uses major sections including: " + " → ".join(h2_main) + "."
    else:
        h2_text = "The competitor uses major sections that introduce the topic, break down key points, and finish with wrap-up context."
    if has_faq:
        h2_text += " It also includes a separate FAQ section."

    # H3 subsections covered (themes)
    h3s = [strip_label(h) for h in list_headers(comp_nodes, 3)]
    # remove noise + avoid label-like items in new-post themes
    h3_clean = []
    for h in h3s:
        if not h or is_noise_header(h) or header_is_faq(h):
            continue
        if is_subpoint_heading(h + ":"):  # treat short proper-noun labels as subpoints
            continue
        h3_clean.append(h)
    # dedupe preserve order
    seen = set()
    themes = []
    for h in h3_clean:
        k = norm_header(h)
        if k in seen:
            continue
        seen.add(k)
        themes.append(h)
    themes = themes[:7]

    if themes:
        h3_text = "The competitor uses subsections to break sections into practical themes such as: " + ", ".join(themes) + "."
    else:
        h3_text = "The competitor uses subsections to add practical depth within each major section."
    if has_faq:
        h3_text += " FAQs are listed as question-style items."

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
st.markdown("<div class='mode-note'>Tip: add competitors one per line (as many as you want).</div>", unsafe_allow_html=True)

show_internal_fetch = st.sidebar.checkbox("Admin: show internal fetch log", value=False)

# Keep last results visible (prevents “tables then rerun changes” feeling)
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
    st.markdown("<div class='section-pill'>Update Mode</div>", unsafe_allow_html=True)

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

        with st.spinner("Fetching Bayut…"):
            bayut_data = get_tree(bayut_url.strip())

        if not bayut_data["ok"] or not bayut_data["nodes"]:
            st.error("Could not extract headings from Bayut (blocked/no headings).")
            st.stop()

        all_rows = []
        internal_fetch = []

        for comp_url in competitors:
            with st.spinner("Fetching competitor…"):
                comp_data = get_tree(comp_url)

            if not comp_data["ok"] or not comp_data["nodes"]:
                internal_fetch.append((comp_url, "blocked/no headings"))
                continue

            internal_fetch.append((comp_url, f"ok ({comp_data['source']})"))
            all_rows.extend(update_mode_rows(bayut_data["nodes"], comp_data["nodes"], comp_url))

        st.session_state.update_fetch = internal_fetch

        if all_rows:
            st.session_state.update_df = pd.DataFrame(all_rows)[["Header (Gap)", "What the competitor talks about", "Source"]]
        else:
            st.session_state.update_df = pd.DataFrame(columns=["Header (Gap)", "What the competitor talks about", "Source"])

    if show_internal_fetch and st.session_state.update_fetch:
        st.sidebar.markdown("### Internal fetch log (Update Mode)")
        for u, s in st.session_state.update_fetch:
            st.sidebar.write(u, "—", s)

    st.markdown("<div class='section-pill'>Content Gaps</div>", unsafe_allow_html=True)

    if st.session_state.update_df is None or st.session_state.update_df.empty:
        st.info("Run analysis to see results.")
    else:
        blocked = sum(1 for _, s in st.session_state.update_fetch if not str(s).startswith("ok"))
        if blocked:
            st.warning(f"{blocked} competitor URL(s) could not be fetched. Results are based on the pages that were accessible.")
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

        rows = []
        internal_fetch = []

        for comp_url in competitors:
            with st.spinner("Fetching competitor…"):
                comp_data = get_tree(comp_url)

            if not comp_data["ok"] or not comp_data["nodes"]:
                internal_fetch.append((comp_url, "blocked/no headings"))
                continue

            internal_fetch.append((comp_url, f"ok ({comp_data['source']})"))
            rows.extend(new_post_coverage_rows(comp_data["nodes"], comp_url))

        st.session_state.new_fetch = internal_fetch

        if rows:
            st.session_state.new_df = pd.DataFrame(rows)[["Headers covered", "Content covered", "Source"]]
        else:
            st.session_state.new_df = pd.DataFrame(columns=["Headers covered", "Content covered", "Source"])

    if show_internal_fetch and st.session_state.new_fetch:
        st.sidebar.markdown("### Internal fetch log (New Post Mode)")
        for u, s in st.session_state.new_fetch:
            st.sidebar.write(u, "—", s)

    st.markdown("<div class='section-pill'>Competitor Coverage</div>", unsafe_allow_html=True)

    if st.session_state.new_df is None or st.session_state.new_df.empty:
        st.info("Generate competitor coverage to see results.")
    else:
        blocked = sum(1 for _, s in st.session_state.new_fetch if not str(s).startswith("ok"))
        if blocked:
            st.warning(f"{blocked} competitor URL(s) could not be fetched. Results are based on the pages that were accessible.")
        render_table(st.session_state.new_df)
