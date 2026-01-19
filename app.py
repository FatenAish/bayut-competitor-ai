# =====================================================
# PART 1/2 — Core extract + gap logic (UPDATED)
# =====================================================
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
  margin: 10px 0 6px 0;
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
.muted {{
  color:#6B7280;
  font-size: 13px;
}}
.small {{
  font-size: 12px;
  color:#6B7280;
}}
</style>
""",
    unsafe_allow_html=True,
)

# UPDATED hero subheader (per your request)
st.markdown(
    f"""
<div class="hero">
  <h1><span class="bayut">Bayut</span> Competitor Gap Analysis</h1>
  <p><b>Analyzes competitor articles</b> to surface missing and under-covered sections.</p>
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
    Resolver order:
    1) direct HTML
    2) JS rendered HTML (Playwright)
    3) Jina reader (text-only)
    4) Textise (text-only)

    HARD RULE:
    - If extraction is weak or blocked -> force manual paste (no partial analysis).
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
                page.wait_for_timeout(1200)

                # light scrolling to trigger lazy-loaded content
                for _ in range(6):
                    try:
                        page.mouse.wheel(0, 2200)
                        page.wait_for_timeout(650)
                    except Exception:
                        break

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
            if self._validate_text(text, min_len=650):
                return FetchResult(True, "direct", code, html, text, None)

        # 2) JS-rendered HTML
        ok, html2 = self._fetch_playwright_html(url)
        if ok and html2:
            text2 = self._extract_article_text_from_html(html2)
            if self._validate_text(text2, min_len=650):
                return FetchResult(True, "playwright", 200, html2, text2, None)

        # 3) Jina reader (text only)
        jurl = self._jina_url(url)
        code3, txt3 = self._http_get(jurl)
        if code3 == 200 and txt3:
            text3 = self.clean(txt3)
            if self._validate_text(text3, min_len=650):
                return FetchResult(True, "jina", code3, "", text3, None)

        # 4) Textise (text only)
        turl = self._textise_url(url)
        code4, html4 = self._http_get(turl)
        if code4 == 200 and html4:
            soup = BeautifulSoup(html4, "html.parser")
            text4 = self.clean(soup.get_text(" "))
            if self._validate_text(text4, min_len=500):
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

def clear_results():
    st.session_state.gaps_df = pd.DataFrame()
    st.session_state.cq_df = pd.DataFrame()
    st.session_state.seo_df = pd.DataFrame()
    st.session_state.rank_df = pd.DataFrame()


# =====================================================
# HARD GATING (no partial analysis allowed)
# =====================================================
def resolve_all_or_require_manual(agent: FetchAgent, urls: List[str], st_key_prefix: str) -> Dict[str, FetchResult]:
    results: Dict[str, FetchResult] = {}
    failed: List[str] = []

    for u in urls:
        r = agent.resolve(u)
        results[u] = r
        if not r.ok:
            failed.append(u)
        time.sleep(0.20)

    if not failed:
        return results

    st.error("Some URLs could not be fetched automatically. Paste the article HTML/text for EACH failed URL to continue (no missing URLs).")
    for u in failed:
        with st.expander(f"Manual fallback required: {u}", expanded=True):
            pasted = st.text_area(
                "Paste the full article HTML OR readable article text:",
                key=_safe_key(st_key_prefix + "__paste", u),
                height=220,
            )
            if pasted and len(pasted.strip()) > 500:
                results[u] = FetchResult(True, "manual", 200, pasted.strip(), pasted.strip(), None)

    still_failed = [u for u in failed if not results[u].ok]
    if still_failed:
        st.stop()

    return results

def ensure_html_for_quality(urls: List[str], fr_map: Dict[str, FetchResult], st_key_prefix: str) -> Dict[str, FetchResult]:
    """
    For accurate FAQs/Videos/Tables we need real HTML.
    If we only have reader text (jina/textise), force manual HTML paste.
    """
    need = []
    for u in urls:
        fr = fr_map[u]
        if (not fr.html) or (fr.source in ("jina", "textise")):
            need.append(u)

    if not need:
        return fr_map

    st.warning("For accurate Content Quality (FAQs/Videos/Tables), paste real HTML for EACH URL below (no missing).")
    for u in need:
        with st.expander(f"HTML required: {u}", expanded=True):
            pasted = st.text_area(
                "Paste FULL HTML (preferred):",
                key=_safe_key(st_key_prefix + "__need_html", u),
                height=240,
            )
            if pasted and "<" in pasted and len(pasted.strip()) > 900:
                html = pasted.strip()
                txt = clean(BeautifulSoup(html, "html.parser").get_text(" "))
                fr_map[u] = FetchResult(True, "manual_html", 200, html, txt, None)

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
        n["content"] = clean(n.get("content", ""))
        n["children"] = [walk(c) for c in n.get("children", [])]
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
            if repaste and len(repaste.strip()) > 500:
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
# FAQ DETECTION (REAL FAQ ONLY)
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

def flatten(nodes: List[dict]) -> List[dict]:
    out = []
    def walk(n: dict, parent=None):
        out.append({
            "level": n["level"],
            "header": n.get("header",""),
            "content": n.get("content", ""),
            "parent": parent,
            "children": n.get("children", []),
        })
        for c in n.get("children", []):
            walk(c, n)
    for n in nodes:
        walk(n, None)
    return out

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
    REAL FAQ if:
    - FAQPage schema exists, OR
    - explicit FAQ heading + >=3 question headings underneath
    """
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
# SECTION EXTRACTION (header-first comparison)
# =====================================================
def strip_label(h: str) -> str:
    return clean(re.sub(r"\s*:\s*$", "", (h or "").strip()))

def section_nodes(nodes: List[dict], levels=(2, 3)) -> List[dict]:
    secs = []
    current_h2 = None
    for x in flatten(nodes):
        lvl = x["level"]
        h = strip_label(x.get("header", ""))
        if not h or is_noise_header(h) or header_is_faq(h):
            continue
        if lvl == 2:
            current_h2 = h
        if lvl in levels:
            c = clean(x.get("content", ""))
            secs.append({"level": lvl, "header": h, "content": c, "parent_h2": current_h2})

    # de-dupe by normalized header
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

def find_best_match(comp_header: str, target_sections: List[dict], min_score: float = 0.73) -> Optional[dict]:
    best = None
    best_score = 0.0
    for b in target_sections:
        sc = header_similarity(comp_header, b["header"])
        if sc > best_score:
            best_score = sc
            best = b
    if best and best_score >= min_score:
        return {"target_section": best, "score": best_score}
    return None


# =====================================================
# LINK HELPERS
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


# =====================================================
# MISSING PARTS LOGIC (under-covered detection)
# - Not about repeats; it’s about relevance + completeness.
# =====================================================
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

def _tokens(s: str) -> List[str]:
    s = clean(s).lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    parts = [p for p in s.split() if p and len(p) >= 3]
    parts = [p for p in parts if p not in STOP and p not in GENERIC_STOP]
    return parts

def extract_missing_terms(comp_text: str, target_text: str, max_terms: int = 10) -> List[str]:
    """
    Returns competitor terms that are absent from target.
    This is NOT a keyword stuffing counter — it’s a completeness hint.
    """
    c = set(_tokens(comp_text))
    t = set(_tokens(target_text))
    missing = [x for x in c if x not in t]
    # simple priority: longer terms first (often more specific)
    missing.sort(key=lambda x: (-len(x), x))
    return missing[:max_terms]

def is_undercovered(comp_section: dict, target_section: dict) -> bool:
    """
    Under-covered if competitor has materially more content AND introduces unique terms.
    """
    cw = len(_tokens(comp_section.get("content", "")))
    tw = len(_tokens(target_section.get("content", "")))
    if cw < 70:
        return False
    if tw == 0:
        return True
    if cw >= int(tw * 1.35):
        missing = extract_missing_terms(comp_section.get("content",""), target_section.get("content",""), max_terms=6)
        return len(missing) >= 3
    return False


# =====================================================
# CONTENT GAPS TABLE (header-first + missing parts)
# =====================================================
def compute_content_gaps(
    target_url: str,
    target_sections: List[dict],
    competitor_sections_map: Dict[str, List[dict]],
    target_has_faq: bool,
    competitor_has_faq_map: Dict[str, bool],
) -> pd.DataFrame:
    rows: List[dict] = []

    for comp_url, comp_sections in competitor_sections_map.items():
        for cs in comp_sections:
            match = find_best_match(cs["header"], target_sections)
            if not match:
                rows.append({
                    "Headers": cs["header"],
                    "Gap": "Missing section (competitor covers this).",
                    "Source": source_link(comp_url),
                })
            else:
                ts = match["target_section"]
                if is_undercovered(cs, ts):
                    missing_terms = extract_missing_terms(cs.get("content",""), ts.get("content",""), max_terms=10)
                    hint = "Add missing details to match competitor depth."
                    if missing_terms:
                        hint += " Missing concepts: " + ", ".join(missing_terms[:10])
                    rows.append({
                        "Headers": f"{ts['header']} (Missing parts)",
                        "Gap": hint,
                        "Source": source_link(comp_url),
                    })

    # FAQ: one row only, and only if competitor has REAL FAQ and target doesn't
    if (not target_has_faq) and any(competitor_has_faq_map.values()):
        first = next((u for u, v in competitor_has_faq_map.items() if v), None)
        if first:
            rows.append({
                "Headers": "FAQs",
                "Gap": "Competitor includes a real FAQ section; add a focused FAQ if it genuinely fits the topic.",
                "Source": source_link(first),
            })

    if not rows:
        return pd.DataFrame(columns=["Headers", "Gap", "Source"])

    df = pd.DataFrame(rows)

    # De-dupe (header + source)
    def _dedupe_key(r):
        return norm_header(str(r.get("Headers",""))) + "||" + re.sub(r"<[^>]+>", "", str(r.get("Source","")))
    df["_k"] = df.apply(_dedupe_key, axis=1)
    df = df.drop_duplicates("_k").drop(columns=["_k"])

    # Sort: Missing headers first, then "(Missing parts)"
    df["_is_missing_parts"] = df["Headers"].astype(str).str.lower().str.contains(r"\(missing parts\)")
    df = df.sort_values(by=["_is_missing_parts", "Headers"], ascending=[True, True]).drop(columns=["_is_missing_parts"])

    return df.reset_index(drop=True)


# =====================================================
# CONTENT QUALITY TABLE (2nd table requirement)
# =====================================================
def count_tables(html: str) -> int:
    if not html:
        return 0
    soup = BeautifulSoup(html, "html.parser")
    return len(soup.find_all("table"))

def count_videos(html: str) -> int:
    if not html:
        return 0
    soup = BeautifulSoup(html, "html.parser")
    vids = 0
    vids += len(soup.find_all("video"))
    # common embeds
    for iframe in soup.find_all("iframe"):
        src = (iframe.get("src") or "").lower()
        if any(x in src for x in ["youtube", "youtu.be", "vimeo"]):
            vids += 1
    return vids

def count_lists(html: str) -> int:
    if not html:
        return 0
    soup = BeautifulSoup(html, "html.parser")
    return len(soup.find_all(["ul", "ol"]))

def content_quality_rows(urls: List[str], fr_map: Dict[str, FetchResult], tree_map: Dict[str, dict]) -> pd.DataFrame:
    rows = []
    for u in urls:
        fr = fr_map[u]
        nodes = (tree_map[u].get("nodes") or [])
        secs = section_nodes(nodes)
        has_faq = page_has_real_faq(fr, nodes)

        word_count = len(clean(fr.text).split())
        rows.append({
            "Page": source_link(u),
            "Words": word_count,
            "Sections": len(secs),
            "Tables": count_tables(fr.html),
            "Videos": count_videos(fr.html),
            "Lists": count_lists(fr.html),
            "Real FAQ": "Yes" if has_faq else "No",
        })
    return pd.DataFrame(rows)


# =====================================================
# SEO SNAPSHOT (NO header counts, NO repeats; relevance-focused)
# =====================================================
def extract_title_and_meta(html: str) -> Tuple[str, str]:
    if not html:
        return "", ""
    soup = BeautifulSoup(html, "html.parser")
    title = ""
    meta_desc = ""
    t = soup.find("title")
    if t:
        title = clean(t.get_text(" "))
    md = soup.find("meta", attrs={"name": re.compile(r"^description$", re.I)})
    if md and md.get("content"):
        meta_desc = clean(md.get("content"))
    return title, meta_desc

def extract_h1(nodes: List[dict]) -> str:
    for x in flatten(nodes):
        if x.get("level") == 1:
            return clean(x.get("header",""))
    # fallback: first H2 as proxy
    for x in flatten(nodes):
        if x.get("level") == 2:
            return clean(x.get("header",""))
    return ""

def keyword_present(text: str, kw: str) -> bool:
    text_n = clean(text).lower()
    kw_n = clean(kw).lower()
    if not kw_n:
        return False
    return kw_n in text_n

def keyword_usage_notes(kw: str, title: str, h1: str, body_text: str) -> str:
    """
    Relevance-first guidance (NOT repetition):
    - It's about proper placement + natural usage.
    - Avoid stuffing / unnecessary repetition.
    """
    if not kw:
        return "—"

    notes = []
    if not keyword_present(title, kw):
        notes.append("Add keyword (or a close variant) in the title if it fits naturally.")
    if not keyword_present(h1, kw):
        notes.append("Use keyword (or a close variant) in the H1 only if it matches intent.")
    # intro check: first ~60 words
    intro = " ".join(clean(body_text).split()[:60])
    if not keyword_present(intro, kw):
        notes.append("Mention it early (intro) once, if natural.")
    notes.append("Focus on relevance and clarity; avoid stuffing or repeating the exact phrase unnecessarily.")

    # keep it short
    out = " ".join(notes)
    if len(out) > 220:
        out = out[:217].rstrip() + "..."
    return out

def seo_snapshot(urls: List[str], fr_map: Dict[str, FetchResult], tree_map: Dict[str, dict], focus_kw: str) -> pd.DataFrame:
    rows = []
    for u in urls:
        fr = fr_map[u]
        nodes = tree_map[u].get("nodes") or []
        title, meta_desc = extract_title_and_meta(fr.html)
        h1 = extract_h1(nodes)

        rows.append({
            "Page": source_link(u),
            "Title present": "Yes" if bool(title) else "No",
            "Meta description present": "Yes" if bool(meta_desc) else "No",
            "KW in Title": "Yes" if keyword_present(title, focus_kw) else "No",
            "KW in H1": "Yes" if keyword_present(h1, focus_kw) else "No",
            "Keyword usage notes": keyword_usage_notes(focus_kw, title, h1, fr.text),
        })
    return pd.DataFrame(rows)


# =====================================================
# SIMPLE (NON-AI) WRITER CHECKLIST SUMMARY
# =====================================================
def gaps_checklist(df: pd.DataFrame) -> str:
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
    for h in add_sections[:7]:
        bullets.append(f"• Add section: {h}")
    for h in expand_sections[:3]:
        bullets.append(f"• Expand section: {h}")
    if faq_needed:
        bullets.append("• Add FAQs only if it genuinely fits the article (real user questions).")

    return "\n".join(bullets[:10])
# =====================================================
# PART 2/2 — DataForSEO + Streamlit UI (UPDATED)
# =====================================================

# =====================================================
# DATAFORSEO (UAE ranking)
# =====================================================
def _secrets_get(key: str, default=None):
    try:
        if hasattr(st, "secrets") and key in st.secrets:
            return st.secrets[key]
    except Exception:
        pass
    return default

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
            timeout=40,
        )
        if r.status_code != 200:
            return {"_error": f"dfs_http_{r.status_code}", "_text": (r.text or "")[:400]}

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
            if str(it.get("type", "")) != "organic":
                continue
            u = it.get("url") or ""
            if normalize_url_for_match(u) == target:
                return (int(it.get("rank_absolute")), "Found")
        return (None, "Not in top 20")
    except Exception:
        return (None, "Not available")

def build_rank_df(focus_kw: str, urls: List[str], location_name: str = DEFAULT_LOCATION_NAME) -> pd.DataFrame:
    rows = []
    for u in urls:
        dpos, dnote = dfs_rank_for_url(focus_kw, u, device="desktop", location_name=location_name)
        mpos, mnote = dfs_rank_for_url(focus_kw, u, device="mobile", location_name=location_name)
        rows.append({
            "Page": source_link(u),
            "Desktop rank (UAE)": dpos if dpos is not None else "—",
            "Mobile rank (UAE)": mpos if mpos is not None else "—",
            "Notes": f"D:{dnote} | M:{mnote}",
        })
    return pd.DataFrame(rows)


# =====================================================
# UI
# =====================================================
if "last_sig" not in st.session_state:
    st.session_state.last_sig = ""
if "gaps_df" not in st.session_state:
    clear_results()

with st.container():
    st.markdown('<div class="section-pill section-pill-tight">Inputs</div>', unsafe_allow_html=True)

    colA, colB = st.columns([1, 1])
    with colA:
        target_url = st.text_input("Bayut URL (target page)", value="", placeholder="https://www.bayut.com/...")
    with colB:
        mode = st.radio("Mode", ["New Analysis", "Update Mode"], horizontal=True)

    competitors_blob = st.text_area(
        "Competitor URLs (one per line)",
        value="",
        height=140,
        placeholder="https://...\nhttps://...\nhttps://...",
    )

    focus_kw = st.text_input("Optional: Focus Keyword", value="", placeholder="e.g., dubai marina area guide")

    # signature: if inputs change, wipe results so nothing stale ever shows
    sig = signature(mode, target_url, competitors_blob, focus_kw)
    if sig != st.session_state.last_sig:
        clear_results()
        st.session_state.last_sig = sig

    run = st.button("Run analysis", use_container_width=True)


def _render_table(df: pd.DataFrame):
    if df is None or df.empty:
        st.info("No data to display.")
        return
    st.markdown(df.to_html(index=False, escape=False), unsafe_allow_html=True)


if run:
    # ---------- validate inputs ----------
    target_url = (target_url or "").strip()
    competitor_urls = [u.strip() for u in (competitors_blob or "").splitlines() if u.strip()]
    competitor_urls = list(dict.fromkeys(competitor_urls))  # de-dupe keep order

    if not target_url or not competitor_urls:
        st.error("Please add a target URL and at least 1 competitor URL.")
        st.stop()

    all_urls = [target_url] + competitor_urls

    # ---------- fetch with hard gating ----------
    fr_map = resolve_all_or_require_manual(agent, all_urls, st_key_prefix="fetch")

    # ---------- headings gating ----------
    tree_map = ensure_headings_or_require_repaste(all_urls, fr_map, st_key_prefix="headings")

    # ---------- HTML gating (for Content Quality accuracy) ----------
    fr_map = ensure_html_for_quality(all_urls, fr_map, st_key_prefix="quality_html")

    # re-build tree_map after possible manual_html replacements
    tree_map = ensure_headings_or_require_repaste(all_urls, fr_map, st_key_prefix="headings_after_html")

    # ---------- build sections ----------
    target_nodes = tree_map[target_url]["nodes"]
    target_sections = section_nodes(target_nodes)

    competitor_sections_map: Dict[str, List[dict]] = {}
    competitor_has_faq_map: Dict[str, bool] = {}

    target_has_faq = page_has_real_faq(fr_map[target_url], target_nodes)

    for cu in competitor_urls:
        nodes = tree_map[cu]["nodes"]
        competitor_sections_map[cu] = section_nodes(nodes)
        competitor_has_faq_map[cu] = page_has_real_faq(fr_map[cu], nodes)

    # ---------- compute tables ----------
    gaps_df = compute_content_gaps(
        target_url=target_url,
        target_sections=target_sections,
        competitor_sections_map=competitor_sections_map,
        target_has_faq=target_has_faq,
        competitor_has_faq_map=competitor_has_faq_map,
    )

    cq_df = content_quality_rows(all_urls, fr_map, tree_map)

    seo_df = pd.DataFrame()
    rank_df = pd.DataFrame()

    if clean(focus_kw):
        seo_df = seo_snapshot(all_urls, fr_map, tree_map, focus_kw=clean(focus_kw))

        # UAE ranking only if DataForSEO creds exist
        if DATAFORSEO_LOGIN and DATAFORSEO_PASSWORD:
            rank_df = build_rank_df(clean(focus_kw), all_urls, location_name=DEFAULT_LOCATION_NAME)

    # store
    st.session_state.gaps_df = gaps_df
    st.session_state.cq_df = cq_df
    st.session_state.seo_df = seo_df
    st.session_state.rank_df = rank_df


# =====================================================
# OUTPUT (order enforced)
# 1) Content Gaps Table
# 2) Content Quality
# 3) SEO Snapshot (optional)
# 4) UAE Ranking (optional)
# =====================================================
if st.session_state.gaps_df is not None and not st.session_state.gaps_df.empty:
    st.markdown('<div class="section-pill section-pill-tight">1) Content Gaps Table</div>', unsafe_allow_html=True)

    # Button label updated per your request
    st.download_button(
        "Content Gaps Table",
        data=st.session_state.gaps_df.to_csv(index=False).encode("utf-8"),
        file_name="content_gaps_table.csv",
        mime="text/csv",
        use_container_width=True,
    )

    _render_table(st.session_state.gaps_df)

    st.markdown("**Writer checklist (deterministic, no AI):**")
    st.code(gaps_checklist(st.session_state.gaps_df), language="markdown")


if st.session_state.cq_df is not None and not st.session_state.cq_df.empty:
    # Content Quality must be 2nd table (per your request)
    st.markdown('<div class="section-pill section-pill-tight">2) Content Quality</div>', unsafe_allow_html=True)
    st.download_button(
        "Download Content Quality (CSV)",
        data=st.session_state.cq_df.to_csv(index=False).encode("utf-8"),
        file_name="content_quality.csv",
        mime="text/csv",
        use_container_width=True,
    )
    _render_table(st.session_state.cq_df)


if st.session_state.seo_df is not None and not st.session_state.seo_df.empty:
    st.markdown('<div class="section-pill section-pill-tight">3) SEO Snapshot</div>', unsafe_allow_html=True)
    st.download_button(
        "Download SEO Snapshot (CSV)",
        data=st.session_state.seo_df.to_csv(index=False).encode("utf-8"),
        file_name="seo_snapshot.csv",
        mime="text/csv",
        use_container_width=True,
    )
    _render_table(st.session_state.seo_df)

    st.markdown(
        '<div class="small">Keyword evaluation is relevance-focused (natural placement + intent alignment). '
        'It does not encourage stuffing or unnecessary repetition.</div>',
        unsafe_allow_html=True,
    )


if st.session_state.rank_df is not None and not st.session_state.rank_df.empty:
    st.markdown('<div class="section-pill section-pill-tight">4) UAE SERP Ranking</div>', unsafe_allow_html=True)
    st.download_button(
        "Download UAE Ranking (CSV)",
        data=st.session_state.rank_df.to_csv(index=False).encode("utf-8"),
        file_name="uae_ranking.csv",
        mime="text/csv",
        use_container_width=True,
    )
    _render_table(st.session_state.rank_df)
elif clean(focus_kw) and (not (DATAFORSEO_LOGIN and DATAFORSEO_PASSWORD)):
    st.info("UAE SERP ranking is unavailable because DataForSEO credentials are not set in Streamlit secrets.")
