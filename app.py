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

# Optional JS rendering (recommended)
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
        padding-top: 1.2rem !important;
        padding-bottom: 2.2rem !important;
      }}
      .hero {{
        text-align:center;
        margin-top: 0.4rem;
        margin-bottom: 1.0rem;
      }}
      .hero h1 {{
        font-size: 48px;
        line-height: 1.08;
        margin: 0;
        color: {TEXT_DARK};
        font-weight: 850;
        letter-spacing: -0.02em;
      }}
      .hero .bayut {{
        color: {BAYUT_GREEN};
      }}
      .hero p {{
        margin: 10px 0 0 0;
        color: #6B7280;
        font-size: 15px;
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
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    f"""
    <div class="hero">
      <h1><span class="bayut">Bayut</span> Competitor Gap Analysis</h1>
      <p>Analyzes competitor articles to surface missing and under-covered sections.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# =====================================================
# FETCH (robust + forced manual paste if blocked)
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
    def __init__(self, default_headers: dict, ignore_tags: set):
        self.default_headers = default_headers
        self.ignore_tags = ignore_tags
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

    def _extract_article_text_from_html(self, html: str) -> str:
        soup = BeautifulSoup(html, "html.parser")
        for t in soup.find_all(list(self.ignore_tags)):
            t.decompose()
        article = soup.find("article") or soup
        return clean(article.get_text(" "))

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

    def _validate_text(self, text: str, min_len: int) -> bool:
        t = clean(text)
        if len(t) < min_len:
            return False
        if looks_blocked(t):
            return False
        return True

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

        # 2) JS render (Playwright)
        ok, html2 = self._fetch_playwright_html(url)
        if ok and html2:
            text2 = self._extract_article_text_from_html(html2)
            if self._validate_text(text2, min_len=500):
                return FetchResult(True, "playwright", 200, html2, text2, None)

        # 3) Jina reader
        jurl = self._jina_url(url)
        code3, txt3 = self._http_get(jurl)
        if code3 == 200 and txt3:
            t3 = clean(txt3)
            if self._validate_text(t3, min_len=500):
                return FetchResult(True, "jina", code3, "", t3, None)

        # 4) Textise
        turl = self._textise_url(url)
        code4, html4 = self._http_get(turl)
        if code4 == 200 and html4:
            soup = BeautifulSoup(html4, "html.parser")
            t4 = clean(soup.get_text(" "))
            if self._validate_text(t4, min_len=350):
                return FetchResult(True, "textise", code4, "", t4, None)

        return FetchResult(False, None, code or None, "", "", "blocked_or_no_content")

agent = FetchAgent(DEFAULT_HEADERS, IGNORE_TAGS)

def _safe_key(prefix: str, url: str) -> str:
    h = hashlib.md5((url or "").encode("utf-8")).hexdigest()
    return f"{prefix}__{h}"

def resolve_all_or_require_manual(urls: List[str], st_key_prefix: str) -> Dict[str, FetchResult]:
    results: Dict[str, FetchResult] = {}
    failed: List[str] = []
    for u in urls:
        r = agent.resolve(u)
        results[u] = r
        if not r.ok:
            failed.append(u)
        time.sleep(0.2)

    if not failed:
        return results

    st.error("Some URLs could not be fetched automatically. Paste the full article HTML/text for EACH failed URL to continue.")
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
# HEADINGS + TREE
# =====================================================
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
    # Support markdown-ish headings from r.jina.ai
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
            pop_to_level(lvl)
            node = {"level": lvl, "header": header, "content": "", "children": []}
            add_node(node)
            current = node
        else:
            if current:
                current["content"] = clean(current["content"] + " " + s)

    return nodes

def build_tree_from_plain_text(text: str) -> List[dict]:
    raw = (text or "").replace("\r", "")
    lines = [clean(l) for l in raw.split("\n")]
    lines = [l for l in lines if l]

    def looks_like_heading(line: str) -> bool:
        if len(line) < 5 or len(line) > 90:
            return False
        if line.endswith("."):
            return False
        words = line.split()
        if len(words) < 2 or len(words) > 14:
            return False
        caps_ratio = sum(1 for w in words if w[:1].isupper()) / max(len(words), 1)
        return caps_ratio >= 0.6

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

def get_tree(fr: FetchResult) -> List[dict]:
    if fr.html:
        return build_tree_from_html(fr.html)
    # manual could be html
    txt = fr.text or ""
    if "<h1" in txt.lower() or "<h2" in txt.lower() or "<article" in txt.lower() or "<html" in txt.lower():
        return build_tree_from_html(txt)
    nodes = build_tree_from_reader_text(txt)
    if nodes:
        return nodes
    return build_tree_from_plain_text(txt)

def flatten(nodes: List[dict]) -> List[dict]:
    out = []
    def walk(n: dict, parent=None):
        out.append({
            "level": n.get("level", 0),
            "header": n.get("header",""),
            "content": n.get("content",""),
            "parent": parent,
            "children": n.get("children", [])
        })
        for c in n.get("children", []) or []:
            walk(c, n)
    for n in nodes:
        walk(n, None)
    return out

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
# FAQ + VIDEO detection (fix false "No")
# =====================================================
FAQ_TITLES = {"faq", "faqs", "frequently asked questions", "frequently asked question"}

def header_is_faq(header: str) -> bool:
    return norm_header(header) in FAQ_TITLES

def _looks_like_question(s: str) -> bool:
    s = clean(s)
    if not s or len(s) < 6:
        return False
    if "?" in s:
        return True
    s_low = s.lower()
    return bool(re.match(r"^(what|where|when|why|how|who|which|can|is|are|do|does|did|should)\b", s_low))

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

def page_has_real_faq(fr: FetchResult, nodes: List[dict]) -> bool:
    # "REAL FAQ": schema OR explicit FAQ heading with >=3 question-like child headings OR >=3 question lines near it
    if fr and fr.html and _has_faq_schema(fr.html):
        return True

    # Find FAQ heading node (h2/h3)
    faq_nodes = []
    for x in flatten(nodes):
        if x.get("level") in (2,3,4) and header_is_faq(x.get("header","")):
            faq_nodes.append(x)

    if not faq_nodes:
        # fallback: detect "FAQ" section by raw html text label
        if fr and fr.html and re.search(r"\b(faqs?|frequently asked questions)\b", fr.html, flags=re.I):
            # still need question pattern count
            txt = clean(BeautifulSoup(fr.html, "html.parser").get_text(" "))
            q_count = len(re.findall(r"\?\s", txt))
            return q_count >= 3
        return False

    for fn in faq_nodes:
        # children headings questions
        child_q = 0
        for c in fn.get("children", []) or []:
            if _looks_like_question(c.get("header","")):
                child_q += 1
        if child_q >= 3:
            return True

        # fallback: question marks inside FAQ content
        content = fn.get("content","") or ""
        if len(re.findall(r"\?", content)) >= 3:
            return True

    return False

def count_tables_and_videos(html: str) -> Tuple[int, int]:
    if not html:
        return (0, 0)
    soup = BeautifulSoup(html, "html.parser")
    for t in soup.find_all(list(IGNORE_TAGS)):
        t.decompose()

    root = soup.find("article") or soup
    tables = len(root.find_all("table"))

    videos = len(root.find_all("video"))
    # count embedded videos (youtube/vimeo) as videos too
    for x in root.find_all("iframe"):
        src = (x.get("src") or "").lower()
        if any(k in src for k in ["youtube.com", "youtu.be", "vimeo.com", "dailymotion.com"]):
            videos += 1

    return (tables, videos)

def word_count(text: str) -> int:
    t = clean(text or "")
    if not t:
        return 0
    return len(re.findall(r"\b\w+\b", t))

# =====================================================
# HEADER-FIRST GAP LOGIC
# =====================================================
def section_nodes(nodes: List[dict], levels=(2,3)) -> List[dict]:
    secs = []
    current_h2 = None
    for x in flatten(nodes):
        lvl = x.get("level", 0)
        hdr = clean(x.get("header",""))
        if not hdr:
            continue
        if header_is_faq(hdr):
            continue
        if lvl == 2:
            current_h2 = hdr
        if lvl in levels:
            secs.append({
                "level": lvl,
                "header": hdr,
                "content": clean(x.get("content","")),
                "parent_h2": current_h2
            })

    # dedupe by normalized header
    seen = set()
    out = []
    for s in secs:
        k = norm_header(s["header"])
        if not k or k in seen:
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
    jacc = len(a_set & b_set) / max(len(a_set | b_set), 1)
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
        return {"section": best, "score": best_score}
    return None

def theme_flags(text: str) -> set:
    t = (text or "").lower()
    flags = set()
    def has_any(words: List[str]) -> bool:
        return any(w in t for w in words)
    if has_any(["metro", "public transport", "commute", "connectivity", "highway", "roads", "bus"]):
        flags.add("transport")
    if has_any(["parking", "traffic", "congestion", "rush hour"]):
        flags.add("traffic_parking")
    if has_any(["cost", "price", "pricing", "expensive", "afford", "budget", "rent", "fees"]):
        flags.add("cost")
    if has_any(["restaurants", "cafes", "nightlife", "vibe", "atmosphere"]):
        flags.add("lifestyle")
    if has_any(["schools", "nursery", "kids", "family", "clinic", "hospital", "supermarket"]):
        flags.add("daily_life")
    if has_any(["safe", "safety", "security", "crime"]):
        flags.add("safety")
    if has_any(["compare", "comparison", "vs ", "versus", "alternative"]):
        flags.add("comparison")
    return flags

def summarize_missing_section(header: str, comp_content: str) -> str:
    themes = list(theme_flags(comp_content))
    human = {
        "transport": "commute & connectivity",
        "traffic_parking": "traffic/parking realities",
        "cost": "cost considerations",
        "lifestyle": "lifestyle & vibe",
        "daily_life": "day-to-day convenience",
        "safety": "safety angle",
        "comparison": "comparison context",
    }
    picks = [human.get(x, x) for x in themes][:3]
    if picks:
        return f"Missing section. Competitor treats this as a dedicated section with practical details (e.g., {', '.join(picks)})."
    return "Missing section. Competitor covers this as a dedicated section with extra context and specifics."

def summarize_undercovered(comp_content: str, target_content: str) -> str:
    comp_flags = theme_flags(comp_content)
    tar_flags = theme_flags(target_content)
    missing = list(comp_flags - tar_flags)
    human = {
        "transport": "commute & connectivity",
        "traffic_parking": "traffic/parking realities",
        "cost": "cost considerations",
        "lifestyle": "lifestyle & vibe",
        "daily_life": "day-to-day convenience",
        "safety": "safety angle",
        "comparison": "comparison context",
    }
    missing_h = [human.get(x, x) for x in missing][:3]
    if missing_h:
        return "Under-covered. Competitor goes deeper on: " + ", ".join(missing_h) + "."
    return "Under-covered. Competitor provides more depth and practical specifics."

def build_content_gaps(target_nodes: List[dict], comp_nodes: List[dict], comp_url: str,
                      max_missing_headers: int = 7, max_missing_parts: int = 5) -> List[dict]:
    rows: List[dict] = []
    target_secs = section_nodes(target_nodes, levels=(2,3))
    comp_secs = section_nodes(comp_nodes, levels=(2,3))

    target_h2 = [s for s in target_secs if s["level"] == 2]
    target_h3 = [s for s in target_secs if s["level"] == 3]
    comp_h2 = [s for s in comp_secs if s["level"] == 2]
    comp_h3 = [s for s in comp_secs if s["level"] == 3]

    # Missing H2 first (ranked by content + children)
    ranked = []
    for h2 in comp_h2:
        child_count = sum(1 for h3 in comp_h3 if norm_header(h3.get("parent_h2") or "") == norm_header(h2["header"]))
        score = len(clean(h2.get("content",""))) + child_count * 120
        ranked.append((score, h2))
    ranked.sort(key=lambda x: x[0], reverse=True)

    missing_h2 = set()
    for _, cs in ranked:
        if find_best_match(cs["header"], target_h2, min_score=0.73):
            continue
        missing_h2.add(norm_header(cs["header"]))
        rows.append({
            "Headers": cs["header"],
            "Description": summarize_missing_section(cs["header"], cs.get("content","")),
            "Source": source_link(comp_url),
        })
        if len(rows) >= max_missing_headers:
            break

    # Missing H3 (only when parent exists in target)
    if len(rows) < max_missing_headers:
        for cs in comp_h3:
            parent = cs.get("parent_h2") or ""
            if parent and norm_header(parent) in missing_h2:
                continue
            if parent and not find_best_match(parent, target_h2, min_score=0.73):
                continue
            if find_best_match(cs["header"], target_h3, min_score=0.73):
                continue

            label = f"{parent} → {cs['header']}" if parent else cs["header"]
            rows.append({
                "Headers": label,
                "Description": summarize_missing_section(cs["header"], cs.get("content","")),
                "Source": source_link(comp_url),
            })
            if len(rows) >= max_missing_headers:
                break

    # Under-covered (missing parts)
    under = []
    for cs in comp_secs:
        m = find_best_match(cs["header"], target_secs, min_score=0.73)
        if not m:
            continue
        ts = m["section"]
        c_txt = clean(cs.get("content",""))
        t_txt = clean(ts.get("content",""))
        if len(c_txt) < 140:
            continue
        if len(c_txt) < 1.30 * max(len(t_txt), 1):
            continue
        under.append({
            "Headers": f"{ts['header']} (missing parts)",
            "Description": summarize_undercovered(c_txt, t_txt),
            "Source": source_link(comp_url),
        })
        if len(under) >= max_missing_parts:
            break

    rows.extend(under)

    # FAQs as ONE row ONLY if competitor has real FAQ AND target does not
    # (keeps your rule; fixes false "No" by using improved detector)
    # NOTE: this row is helpful for "missing" only. If target already has FAQ => no gap row.
    return rows

# =====================================================
# DATAFORSEO (UAE Rank + AI Overview signals if available)
# =====================================================
def _secrets_get(key: str, default=None):
    try:
        if hasattr(st, "secrets") and key in st.secrets:
            return st.secrets[key]
    except Exception:
        pass
    return default

DATAFORSEO_LOGIN = _secrets_get("DATAFORSEO_LOGIN")
DATAFORSEO_PASSWORD = _secrets_get("DATAFORSEO_PASSWORD")
DATAFORSEO_LOCATION_CODE = str(_secrets_get("DATAFORSEO_LOCATION_CODE", "0"))  # 0 = "All locations" in DataForSEO

def _d4s_auth_header() -> Dict[str, str]:
    if not DATAFORSEO_LOGIN or not DATAFORSEO_PASSWORD:
        return {}
    token = base64.b64encode(f"{DATAFORSEO_LOGIN}:{DATAFORSEO_PASSWORD}".encode()).decode()
    return {"Authorization": f"Basic {token}"}

@st.cache_data(show_spinner=False, ttl=1800)
def d4s_google_serp(keyword: str, device: str, depth: int = 50) -> dict:
    # device: "desktop" or "mobile"
    if not DATAFORSEO_LOGIN or not DATAFORSEO_PASSWORD:
        return {"_error": "missing_dataforseo_credentials"}

    endpoint = "https://api.dataforseo.com/v3/serp/google/organic/live/advanced"
    headers = {"Content-Type": "application/json"}
    headers.update(_d4s_auth_header())

    payload = [{
        "keyword": keyword,
        "location_code": int(DATAFORSEO_LOCATION_CODE) if str(DATAFORSEO_LOCATION_CODE).isdigit() else 0,
        "language_code": "en",
        "device": device,
        "os": "windows" if device == "desktop" else "android",
        "depth": depth
    }]

    try:
        r = requests.post(endpoint, headers=headers, data=json.dumps(payload), timeout=45)
        if r.status_code != 200:
            return {"_error": f"d4s_http_{r.status_code}", "_text": (r.text or "")[:500]}
        return r.json()
    except Exception as e:
        return {"_error": str(e)}

def normalize_url_for_match(u: str) -> str:
    try:
        p = urlparse(u)
        host = p.netloc.lower().replace("www.", "")
        path = (p.path or "").rstrip("/")
        return host + path
    except Exception:
        return (u or "").strip().lower().replace("www.", "").rstrip("/")

def d4s_rank_for_url(keyword: str, page_url: str, device: str) -> Tuple[str, str]:
    data = d4s_google_serp(keyword, device=device, depth=50)
    if not data or data.get("_error"):
        return (f"Not available", str(data.get("_error")) if isinstance(data, dict) else "Not available")

    try:
        tasks = data.get("tasks") or []
        if not tasks:
            return ("Not available", "No tasks returned")
        result = (tasks[0].get("result") or [])
        if not result:
            return ("Not available", "No result returned")
        items = result[0].get("items") or []
    except Exception:
        return ("Not available", "Unexpected response shape")

    target = normalize_url_for_match(page_url)

    best_pos = None
    for it in items:
        it_url = it.get("url") or it.get("link") or ""
        if not it_url:
            continue
        nm = normalize_url_for_match(it_url)
        if nm == target or target in nm or nm in target:
            pos = it.get("rank_group") or it.get("rank_absolute") or it.get("position")
            try:
                best_pos = int(pos)
                break
            except Exception:
                best_pos = None

    if best_pos is None:
        return ("Not in top 50", "OK")
    return (str(best_pos), "OK")

def d4s_ai_overview_signals(keyword: str, page_url: str, device: str) -> Dict[str, str]:
    # DataForSEO may or may not expose AI overview as a distinct item depending on engine output.
    data = d4s_google_serp(keyword, device=device, depth=50)
    if not data or data.get("_error"):
        return {"AI Overview Present": "Not available", "AI Overview Cited?": "Not available"}

    try:
        items = (data.get("tasks")[0].get("result")[0].get("items")) or []
    except Exception:
        return {"AI Overview Present": "Not available", "AI Overview Cited?": "Not available"}

    present = "No"
    cited = "Not available"  # only set to Yes/No if we can confidently detect citation links
    target = normalize_url_for_match(page_url)

    # Heuristic: any item type mentions ai_overview
    for it in items:
        t = str(it.get("type") or "").lower()
        if "ai_overview" in t or "ai-overview" in t or "ai overview" in t:
            present = "Yes"
            # attempt citation extraction if present
            links = it.get("references") or it.get("links") or it.get("citations") or []
            if isinstance(links, list) and links:
                cited = "No"
                for lk in links:
                    u = ""
                    if isinstance(lk, dict):
                        u = lk.get("url") or lk.get("link") or ""
                    elif isinstance(lk, str):
                        u = lk
                    if u and (target in normalize_url_for_match(u) or normalize_url_for_match(u) in target):
                        cited = "Yes"
                        break
            break

    return {"AI Overview Present": present, "AI Overview Cited?": cited}

# =====================================================
# SEO (no Headers count, no FKW columns shown)
# + "KW Usage Quality" instead
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

def get_first_h1(nodes: List[dict]) -> str:
    for x in flatten(nodes):
        if x.get("level") == 1:
            h = clean(x.get("header",""))
            if h:
                return h
    return ""

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
            phrase = " ".join(chunk)
            if len(phrase) < 8:
                continue
            freq[phrase] = freq.get(phrase, 0) + 1
    return freq

def pick_primary_keyword(seo_title: str, h1: str, body_text: str, manual_kw: str) -> str:
    manual_kw = clean(manual_kw)
    if manual_kw:
        return manual_kw.lower()
    base = " ".join([seo_title or "", h1 or "", body_text or ""])
    freq = phrase_candidates(base, 2, 4)
    if not freq:
        return ""
    title_low = (seo_title or "").lower()
    h1_low = (h1 or "").lower()
    scored = []
    for ph, c in freq.items():
        boost = 1.0
        if ph in title_low:
            boost += 0.8
        if ph in h1_low:
            boost += 0.5
        scored.append((c * boost, ph))
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[0][1] if scored else ""

def kw_usage_quality(body_text: str, seo_title: str, h1: str, keyword: str) -> str:
    keyword = clean(keyword).lower()
    if not keyword:
        return "Not available (no keyword)"

    t = " " + re.sub(r"\s+", " ", (body_text or "").lower()) + " "
    p = " " + re.sub(r"\s+", " ", keyword) + " "
    repeats = t.count(p)
    wc = word_count(body_text)
    per_1k = (repeats / max(wc, 1)) * 1000.0 if wc else 0.0

    in_title = keyword in (seo_title or "").lower()
    in_h1 = keyword in (h1 or "").lower()

    # quality bands (practical)
    if wc and per_1k >= 18:
        return f"Stuffing risk (≈{repeats} repeats, {per_1k:.1f}/1k). Reduce repetition + vary phrasing."
    if wc and per_1k >= 10:
        return f"Needs smoothing (≈{repeats} repeats, {per_1k:.1f}/1k). Keep natural, avoid forced repeats."
    # low repetition: check placement
    if not in_title and not in_h1:
        return f"Under-used. Add keyword naturally to title or H1 (repeats ≈{repeats})."
    return f"Good / natural (repeats ≈{repeats}, {per_1k:.1f}/1k)."

def build_seo_table(pages: List[Tuple[str, str, FetchResult, List[dict]]], manual_kw: str) -> pd.DataFrame:
    rows = []
    for label, url, fr, nodes in pages:
        seo_title, meta_desc = extract_head_seo(fr.html or "")
        h1 = get_first_h1(nodes)
        if seo_title == "Not available" and h1:
            seo_title = h1
        body_text = fr.text or ""
        keyword = pick_primary_keyword(seo_title, h1, body_text, manual_kw)

        rank_d, _ = d4s_rank_for_url(keyword, url, device="desktop") if keyword else ("Not available", "")
        rank_m, _ = d4s_rank_for_url(keyword, url, device="mobile") if keyword else ("Not available", "")
        ai_sig = d4s_ai_overview_signals(keyword, url, device="desktop") if keyword else {"AI Overview Present":"Not available","AI Overview Cited?":"Not available"}

        rows.append({
            "Source": label,
            "Slug": url_slug(url),
            "SEO Title": seo_title,
            "Meta Description": meta_desc,
            "KW Usage Quality": kw_usage_quality(body_text, seo_title, h1, keyword),
            "UAE Rank (Desktop)": rank_d,
            "UAE Rank (Mobile)": rank_m,
            "AI Overview Present": ai_sig.get("AI Overview Present","Not available"),
            "AI Overview Cited?": ai_sig.get("AI Overview Cited?","Not available"),
        })
    return pd.DataFrame(rows)

# =====================================================
# CONTENT QUALITY TABLE (2nd table)
# =====================================================
def build_content_quality(pages: List[Tuple[str, str, FetchResult, List[dict]]]) -> pd.DataFrame:
    rows = []
    for label, url, fr, nodes in pages:
        wc = word_count(fr.text or "")
        tables, videos = count_tables_and_videos(fr.html or "")
        faq = "Yes" if page_has_real_faq(fr, nodes) else "No"
        rows.append({
            "Source": label,
            "Word Count": wc,
            "FAQs": faq,
            "Tables": tables,
            "Video": videos,
        })
    return pd.DataFrame(rows)

# =====================================================
# HTML TABLE RENDER
# =====================================================
def render_table(df: pd.DataFrame):
    if df is None or df.empty:
        st.info("No results to show.")
        return
    html = df.to_html(index=False, escape=False)
    st.markdown(html, unsafe_allow_html=True)

# =====================================================
# MODE SELECTOR
# =====================================================
if "mode" not in st.session_state:
    st.session_state.mode = "update"

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

# =====================================================
# SESSION STATE (fix "old summary / old results")
# =====================================================
def reset_results():
    st.session_state.analysis = None
    st.session_state.analysis_err = None

if "analysis" not in st.session_state:
    st.session_state.analysis = None
if "analysis_err" not in st.session_state:
    st.session_state.analysis_err = None

# =====================================================
# UI - UPDATE MODE
# =====================================================
if st.session_state.mode == "update":
    st.markdown("<div class='section-pill'>Just Update Mode</div>", unsafe_allow_html=True)

    target_url = st.text_input("Bayut article URL", placeholder="https://www.bayut.com/mybayut/...")
    competitors_text = st.text_area("Competitor URLs (one per line)", height=120)
    competitors = [c.strip() for c in competitors_text.splitlines() if c.strip()]

    manual_kw = st.text_input("Optional: Focus Keyword", placeholder="e.g., pros and cons business bay")

    run_clicked = st.button("Run analysis", type="primary")

    if run_clicked:
        reset_results()

        if not target_url.strip():
            st.session_state.analysis_err = "Bayut article URL is required."
        elif not competitors:
            st.session_state.analysis_err = "Add at least one competitor URL."
        else:
            debug = []
            try:
                with st.spinner("Fetching target + competitors (forced to complete)…"):
                    fr_map = resolve_all_or_require_manual([target_url.strip()] + competitors, st_key_prefix="all_update")
                target_fr = fr_map[target_url.strip()]
                comp_fr_map = {u: fr_map[u] for u in competitors}

                target_nodes = get_tree(target_fr)
                comp_nodes_map = {u: get_tree(comp_fr_map[u]) for u in competitors}

                debug.append(f"Target fetch source: {target_fr.source}")
                for u in competitors:
                    debug.append(f"{site_name(u)} fetch source: {comp_fr_map[u].source}")

                # Content gaps
                all_rows = []
                for u in competitors:
                    all_rows.extend(build_content_gaps(target_nodes, comp_nodes_map[u], u))

                    # FAQs gap row (ONE row only if competitor has REAL FAQ and target does not)
                    comp_has_faq = page_has_real_faq(comp_fr_map[u], comp_nodes_map[u])
                    target_has_faq = page_has_real_faq(target_fr, target_nodes)
                    if comp_has_faq and not target_has_faq:
                        all_rows.append({
                            "Headers": "FAQs",
                            "Description": "Competitor includes a real FAQ section; consider adding FAQs if missing.",
                            "Source": source_link(u),
                        })

                gaps_df = pd.DataFrame(all_rows) if all_rows else pd.DataFrame(columns=["Headers","Description","Source"])
                if not gaps_df.empty:
                    gaps_df = gaps_df[["Headers","Description","Source"]]

                # Pages list for tables
                pages = [("Bayut", target_url.strip(), target_fr, target_nodes)]
                for u in competitors:
                    pages.append((site_name(u), u, comp_fr_map[u], comp_nodes_map[u]))

                cq_df = build_content_quality(pages)

                # SEO table (DataForSEO)
                seo_df = build_seo_table(pages, manual_kw=manual_kw.strip())

                st.session_state.analysis = {
                    "gaps_df": gaps_df,
                    "content_quality_df": cq_df,
                    "seo_df": seo_df,
                    "debug": debug,
                }

            except Exception as e:
                st.session_state.analysis_err = f"Unexpected error: {e}"

# =====================================================
# UI - NEW POST MODE
# =====================================================
else:
    st.markdown("<div class='section-pill'>New Post Mode</div>", unsafe_allow_html=True)

    new_title = st.text_input("New post title", placeholder="Pros & Cons of Living in Business Bay (2026)")
    competitors_text = st.text_area("Competitor URLs (one per line)", height=120)
    competitors = [c.strip() for c in competitors_text.splitlines() if c.strip()]

    manual_kw = st.text_input("Optional: Focus Keyword", placeholder="e.g., pros and cons business bay")

    run_clicked = st.button("Generate competitor coverage", type="primary")

    if run_clicked:
        reset_results()

        if not new_title.strip():
            st.session_state.analysis_err = "New post title is required."
        elif not competitors:
            st.session_state.analysis_err = "Add at least one competitor URL."
        else:
            debug = []
            try:
                with st.spinner("Fetching competitors (forced to complete)…"):
                    fr_map = resolve_all_or_require_manual(competitors, st_key_prefix="all_new")

                comp_fr_map = {u: fr_map[u] for u in competitors}
                comp_nodes_map = {u: get_tree(comp_fr_map[u]) for u in competitors}

                for u in competitors:
                    debug.append(f"{site_name(u)} fetch source: {comp_fr_map[u].source}")

                # In New Post Mode we show a simple coverage table (not gaps vs bayut)
                rows = []
                for u in competitors:
                    nodes = comp_nodes_map[u]
                    h2s = [clean(x["header"]) for x in flatten(nodes) if x.get("level")==2 and x.get("header")]
                    h2s = h2s[:7]
                    rows.append({
                        "Competitor": source_link(u),
                        "Main Sections (H2)": " → ".join(h2s) if h2s else "Not detected",
                    })
                coverage_df = pd.DataFrame(rows)

                pages = []
                for u in competitors:
                    pages.append((site_name(u), u, comp_fr_map[u], comp_nodes_map[u]))

                cq_df = build_content_quality(pages)
                seo_df = build_seo_table(pages, manual_kw=manual_kw.strip())

                st.session_state.analysis = {
                    "gaps_df": coverage_df,  # reuse renderer slot
                    "content_quality_df": cq_df,
                    "seo_df": seo_df,
                    "debug": debug,
                }

            except Exception as e:
                st.session_state.analysis_err = f"Unexpected error: {e}"

# =====================================================
# RESULTS (NO AI SUMMARY BUTTONS)
# =====================================================
analysis = st.session_state.get("analysis")
err = st.session_state.get("analysis_err")

st.write("")
st.subheader("Content Gaps Table")

if err:
    st.error(err)

if analysis:
    gaps_df = analysis.get("gaps_df")
    cq_df = analysis.get("content_quality_df")
    seo_df = analysis.get("seo_df")
    debug = analysis.get("debug", [])

    # 1) Content gaps / coverage
    if gaps_df is None or gaps_df.empty:
        st.info("No results.")
    else:
        render_table(gaps_df)

    st.divider()

    # 2) Content Quality (2nd table)
    st.subheader("Content Quality")
    if cq_df is None or cq_df.empty:
        st.info("No content quality data.")
    else:
        render_table(cq_df)

    st.divider()

    # 3) SEO Analysis
    st.subheader("SEO Analysis")
    if seo_df is None or seo_df.empty:
        st.info("No SEO data.")
    else:
        render_table(seo_df)

    # Secrets warning (ONLY DataForSEO)
    if not DATAFORSEO_LOGIN or not DATAFORSEO_PASSWORD:
        st.warning("DataForSEO ranking requires DATAFORSEO_LOGIN and DATAFORSEO_PASSWORD in Streamlit secrets.")

    if debug:
        with st.expander("Debug (only if needed)"):
            for d in debug:
                st.write(f"• {d}")
else:
    st.info("Run analysis to see results.")
