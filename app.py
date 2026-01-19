# app.py (PART 1/2)
import streamlit as st
import requests
import re
import time, random, hashlib, json
from dataclasses import dataclass
from typing import Optional, Dict, Tuple, List
from urllib.parse import quote_plus, urlparse
from difflib import SequenceMatcher

import pandas as pd
from bs4 import BeautifulSoup

# Optional JS rendering
try:
    from playwright.sync_api import sync_playwright
    PLAYWRIGHT_OK = True
except Exception:
    PLAYWRIGHT_OK = False


# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(page_title="Bayut Competitor Gap Analysis", layout="wide")


# =====================================================
# STYLE (LIGHT GREEN)
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
    """
    <div class="hero">
      <h1><span class="bayut">Bayut</span> Competitor Gap Analysis</h1>
      <p>Analyzes competitor articles to surface missing and under-covered sections.</p>
    </div>
    """,
    unsafe_allow_html=True,
)


# =====================================================
# BASIC HELPERS
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

STOP = {
    "the","and","for","with","that","this","from","you","your","are","was","were","will","have","has","had",
    "but","not","can","may","more","most","into","than","then","they","them","their","our","out","about",
    "also","over","under","between","within","near","where","when","what","why","how","who","which",
    "a","an","to","of","in","on","at","as","is","it","be","or","by","we","i","us"
}

def clean(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip()

def norm_header(h: str) -> str:
    h = clean(h).lower()
    h = re.sub(r"[^a-z0-9\s]", "", h)
    h = re.sub(r"\s+", " ", h).strip()
    return h

def site_name(url: str) -> str:
    try:
        host = urlparse(url).netloc.lower().replace("www.", "")
        base = host.split(":")[0]
        name = base.split(".")[0]
        return name[:1].upper() + name[1:]
    except Exception:
        return "Source"

def url_slug(url: str) -> str:
    try:
        p = urlparse(url).path.strip("/")
        return "/" + p if p else "/"
    except Exception:
        return "/"

def looks_blocked(text: str) -> bool:
    t = (text or "").lower()
    return any(x in t for x in [
        "just a moment", "checking your browser", "verify you are human",
        "cloudflare", "access denied", "captcha", "forbidden", "service unavailable"
    ])

def _safe_key(prefix: str, url: str) -> str:
    h = hashlib.md5((url or "").encode("utf-8")).hexdigest()
    return f"{prefix}__{h}"


# =====================================================
# FETCH AGENT (HARD GATING + HTML REQUIRED FOR FAQ/VIDEO/TABLE)
# =====================================================
@dataclass
class FetchResult:
    ok: bool
    source: Optional[str]          # direct | playwright | jina | textise | manual
    status: Optional[int]
    html: str                      # real html if available
    text: str                      # extracted readable text
    reason: Optional[str]
    has_real_html: bool            # True only for direct/playwright/manual-html


class FetchAgent:
    def __init__(self):
        self.user_agents = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_6) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.3 Safari/605.1.15",
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0 Safari/537.36",
        ]

    def _http_get(self, url: str, timeout: int = 25, tries: int = 3) -> Tuple[int, str]:
        last_code, last_text = 0, ""
        for i in range(tries):
            headers = dict(DEFAULT_HEADERS)
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

    def _extract_text_from_html(self, html: str) -> str:
        soup = BeautifulSoup(html, "html.parser")
        for t in soup.find_all(list(IGNORE_TAGS)):
            t.decompose()
        main = soup.find("article") or soup.find("main") or soup
        return clean(main.get_text(" "))

    def _jina_url(self, url: str) -> str:
        if url.startswith("https://"):
            return "https://r.jina.ai/https://" + url[len("https://"):]
        if url.startswith("http://"):
            return "https://r.jina.ai/http://" + url[len("http://"):]
        return "https://r.jina.ai/https://" + url

    def _textise_url(self, url: str) -> str:
        # Textise direct URL format (text-only)
        return f"https://www.textise.net/showtext.aspx?strurl={quote_plus(url)}"

    def _fetch_playwright_html(self, url: str, timeout_ms: int = 25000) -> Tuple[bool, str]:
        if not PLAYWRIGHT_OK:
            return False, ""
        try:
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True, args=["--no-sandbox"])
                ctx = browser.new_context(user_agent=random.choice(self.user_agents))
                page = ctx.new_page()
                page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)
                page.wait_for_timeout(1500)
                html = page.content()
                browser.close()
            return True, html
        except Exception:
            return False, ""

    def resolve(self, url: str) -> FetchResult:
        url = (url or "").strip()
        if not url:
            return FetchResult(False, None, None, "", "", "empty_url", False)

        # 1) direct HTML
        code, html = self._http_get(url)
        if code == 200 and html and not looks_blocked(html):
            txt = self._extract_text_from_html(html)
            if len(txt) >= 450 and not looks_blocked(txt):
                return FetchResult(True, "direct", code, html, txt, None, True)

        # 2) Playwright HTML
        ok, html2 = self._fetch_playwright_html(url)
        if ok and html2 and not looks_blocked(html2):
            txt2 = self._extract_text_from_html(html2)
            if len(txt2) >= 450 and not looks_blocked(txt2):
                return FetchResult(True, "playwright", 200, html2, txt2, None, True)

        # 3) Jina reader (TEXT ONLY → not real HTML)
        jurl = self._jina_url(url)
        code3, txt3 = self._http_get(jurl)
        if code3 == 200 and txt3:
            t3 = clean(txt3)
            if len(t3) >= 450 and not looks_blocked(t3):
                return FetchResult(True, "jina", code3, "", t3, None, False)

        # 4) Textise (TEXT ONLY → not real HTML)
        turl = self._textise_url(url)
        code4, html4 = self._http_get(turl)
        if code4 == 200 and html4:
            soup = BeautifulSoup(html4, "html.parser")
            t4 = clean(soup.get_text(" "))
            if len(t4) >= 350 and not looks_blocked(t4):
                return FetchResult(True, "textise", code4, "", t4, None, False)

        return FetchResult(False, None, code or None, "", "", "blocked_or_no_content", False)


agent = FetchAgent()


def resolve_all_or_require_manual(urls: List[str], st_key_prefix: str) -> Dict[str, FetchResult]:
    """
    HARD GATE:
    - If any URL fails -> must paste (no missing URLs).
    """
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
                # If user pasted HTML, treat as real HTML
                maybe_html = "<html" in pasted.lower() or "<article" in pasted.lower() or "<h1" in pasted.lower() or "<h2" in pasted.lower()
                html = pasted.strip() if maybe_html else ""
                txt = clean(BeautifulSoup(pasted, "html.parser").get_text(" ")) if maybe_html else pasted.strip()
                results[u] = FetchResult(True, "manual", 200, html, txt, None, bool(html))

    still_failed = [u for u in failed if not results[u].ok]
    if still_failed:
        st.stop()

    return results


def require_html_for_accuracy(fr_map: Dict[str, FetchResult], urls: List[str], st_key_prefix: str) -> Dict[str, FetchResult]:
    """
    IMPORTANT RULE YOU SET:
    For FAQ / videos / tables accuracy -> must have REAL HTML.
    If fetched via jina/textise -> force manual HTML paste.
    """
    need_html = [u for u in urls if fr_map[u].ok and not fr_map[u].has_real_html]

    if not need_html:
        return fr_map

    st.warning("For accurate FAQs / tables / videos detection, these pages must provide REAL HTML. Please paste HTML for each one below (no skipping).")

    for u in need_html:
        with st.expander(f"HTML required for accuracy: {u}", expanded=True):
            pasted = st.text_area(
                "Paste the FULL page HTML (recommended):",
                key=_safe_key(st_key_prefix + "__html", u),
                height=240,
            )
            if pasted and len(pasted.strip()) > 600:
                txt = clean(BeautifulSoup(pasted, "html.parser").get_text(" "))
                fr_map[u] = FetchResult(True, "manual", 200, pasted.strip(), txt, None, True)

    still = [u for u in need_html if not fr_map[u].has_real_html]
    if still:
        st.stop()

    return fr_map


# =====================================================
# HEADING TREE (clean, header-first)
# =====================================================
NOISE_PATTERNS = [
    r"\btable of contents\b", r"\bcontents\b", r"\bback to top\b",
    r"\bshare\b", r"\bsubscribe\b", r"\bnewsletter\b",
    r"\brelated posts\b", r"\byou may also like\b", r"\brecommended\b",
    r"\bcomments\b", r"\bnext\b", r"\bprevious\b",
    r"\blogin\b", r"\bregister\b", r"\bsign up\b",
    r"\bcontact\b", r"\bget in touch\b", r"\bwhatsapp\b",
]

FAQ_TITLES = {"faq", "faqs", "frequently asked questions", "frequently asked question"}

def is_noise_header(h: str) -> bool:
    s = clean(h)
    if not s:
        return True
    hn = norm_header(s)
    if len(hn) < 4:
        return True
    if len(s) > 95:
        return True
    for pat in NOISE_PATTERNS:
        if re.search(pat, hn):
            return True
    return False

def header_is_faq(h: str) -> bool:
    return norm_header(h) in FAQ_TITLES

def build_tree_from_html(html: str) -> List[dict]:
    soup = BeautifulSoup(html, "html.parser")
    for t in soup.find_all(list(IGNORE_TAGS)):
        t.decompose()
    root = soup.find("article") or soup.find("main") or soup

    headings = root.find_all(["h1", "h2", "h3", "h4"])
    nodes: List[dict] = []
    stack: List[dict] = []

    def level_of(tag: str) -> int:
        try:
            return int(tag[1])
        except Exception:
            return 9

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
        if not header or is_noise_header(header):
            continue

        lvl = level_of(h.name)
        pop_to_level(lvl)

        node = {"level": lvl, "header": header, "content": "", "children": []}
        add_node(node)

        # collect text until next heading
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

    # dedupe headers (keep first)
    seen = set()
    out = []
    for n in nodes:
        k = norm_header(n["header"])
        if k in seen:
            continue
        seen.add(k)
        out.append(n)
    return out

def flatten(nodes: List[dict]) -> List[dict]:
    out = []
    def walk(n: dict, parent=None):
        out.append({
            "level": n["level"],
            "header": n.get("header",""),
            "content": n.get("content",""),
            "parent": parent,
            "children": n.get("children",[])
        })
        for c in n.get("children",[]):
            walk(c, n)
    for n in nodes:
        walk(n, None)
    return out

def get_first_h1(nodes: List[dict]) -> str:
    for x in flatten(nodes):
        if x.get("level") == 1:
            h = clean(x.get("header",""))
            if h:
                return h
    return ""
# app.py (PART 2/2)

# =====================================================
# STRICT FAQ RULE (ONE ROW ONLY)
# =====================================================
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

def _has_faq_schema(html: str) -> bool:
    if not html:
        return False
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
    return False

def _faq_nodes(nodes: List[dict]) -> List[dict]:
    out = []
    for x in flatten(nodes):
        if x.get("level") in (2, 3) and header_is_faq(x.get("header","")):
            out.append(x)
    return out

def _question_children(node: dict) -> List[str]:
    qs = []
    for c in node.get("children", []) or []:
        h = clean(c.get("header",""))
        if h and _looks_like_question(h):
            qs.append(h)
    return qs

def page_has_real_faq(fr: FetchResult, nodes: List[dict]) -> bool:
    faq_nodes = _faq_nodes(nodes)
    if not faq_nodes:
        return False
    # If real HTML exists: allow schema OR >=3 question headings under FAQ section
    if fr and fr.has_real_html and fr.html:
        if _has_faq_schema(fr.html):
            return True
        for fn in faq_nodes:
            if len(_question_children(fn)) >= 3:
                return True
        return False
    # Without HTML: require >=3 visible question headings under FAQ section
    for fn in faq_nodes:
        if len(_question_children(fn)) >= 3:
            return True
    return False

def faq_topics_from_questions(questions: List[str], limit: int = 8) -> List[str]:
    # quick grouping: return top unique "topic buckets"
    out = []
    seen = set()
    for q in questions:
        qn = norm_header(q)
        if any(k in qn for k in ["price","cost","rent","fees"]):
            t = "Pricing / cost"
        elif any(k in qn for k in ["metro","transport","parking","traffic"]):
            t = "Transport / traffic / parking"
        elif any(k in qn for k in ["safe","safety","security"]):
            t = "Safety"
        elif any(k in qn for k in ["school","kids","family"]):
            t = "Family / schools"
        elif any(k in qn for k in ["things to do","restaurants","cafes","nightlife"]):
            t = "Lifestyle / things to do"
        else:
            t = "Other key FAQs"
        k = norm_header(t)
        if k not in seen:
            seen.add(k)
            out.append(t)
        if len(out) >= limit:
            break
    return out

def missing_faqs_row(bayut_nodes: List[dict], bayut_fr: FetchResult, comp_nodes: List[dict], comp_fr: FetchResult, comp_url: str) -> Optional[dict]:
    if not page_has_real_faq(comp_fr, comp_nodes):
        return None

    comp_qs = []
    for fn in _faq_nodes(comp_nodes):
        comp_qs.extend(_question_children(fn))
    comp_qs = [q for q in comp_qs if q]

    if len(comp_qs) < 3:
        return None

    bayut_has = page_has_real_faq(bayut_fr, bayut_nodes)
    bayut_qs = []
    if bayut_has:
        for fn in _faq_nodes(bayut_nodes):
            bayut_qs.extend(_question_children(fn))
    bayut_qs = [q for q in bayut_qs if q]

    def key_q(q: str) -> str:
        q = norm_header(q)
        q = re.sub(r"[^a-z0-9\s]", "", q)
        q = re.sub(r"\s+", " ", q).strip()
        return q

    bayut_set = {key_q(q) for q in bayut_qs}

    if not bayut_qs:
        topics = faq_topics_from_questions(comp_qs, limit=8)
        return {
            "Headers": "FAQs",
            "Description": "Competitor has a real FAQ section covering topics such as: " + ", ".join(topics) + ".",
            "Source": site_name(comp_url),
        }

    missing = [q for q in comp_qs if key_q(q) not in bayut_set]
    if not missing:
        return None
    topics = faq_topics_from_questions(missing, limit=8)
    return {
        "Headers": "FAQs",
        "Description": "Missing FAQ topics: " + ", ".join(topics) + ".",
        "Source": site_name(comp_url),
    }


# =====================================================
# CONTENT GAP LOGIC (AGREED BEHAVIOR)
# Missing headers FIRST (H2), then (missing parts), then FAQ row (if valid)
# =====================================================
def section_nodes(nodes: List[dict], levels=(2,3)) -> List[dict]:
    secs = []
    current_h2 = None
    for x in flatten(nodes):
        lvl = x.get("level")
        h = clean(x.get("header",""))
        if not h or is_noise_header(h) or header_is_faq(h):
            continue
        if lvl == 2:
            current_h2 = h
        if lvl in levels:
            secs.append({
                "level": lvl,
                "header": h,
                "content": clean(x.get("content","")),
                "parent_h2": current_h2
            })

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

def find_best_match(comp_header: str, bayut_sections: List[dict], min_score: float = 0.73) -> Optional[dict]:
    best = None
    best_score = 0.0
    for b in bayut_sections:
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
    if has_any(["metro","public transport","commute","connectivity","highway","roads","bus","train"]):
        flags.add("transport")
    if has_any(["parking","traffic","congestion","rush hour"]):
        flags.add("traffic_parking")
    if has_any(["cost","price","pricing","expensive","afford","budget","rent","fees"]):
        flags.add("cost")
    if has_any(["restaurants","cafes","nightlife","vibe","atmosphere","entertainment"]):
        flags.add("lifestyle")
    if has_any(["schools","nursery","kids","family","clinic","hospital","supermarket","pharmacy"]):
        flags.add("daily_life")
    if has_any(["safe","safety","security","crime"]):
        flags.add("safety")
    if has_any(["compare","comparison","vs ","versus","alternative"]):
        flags.add("comparison")
    if has_any(["pros","cons","advantages","disadvantages","worth it","should you"]):
        flags.add("decision_frame")
    return flags

def summarize_missing_header(header: str, comp_content: str) -> str:
    flags = list(theme_flags(comp_content))
    human_map = {
        "transport": "commute & connectivity",
        "traffic_parking": "traffic/parking realities",
        "cost": "cost considerations",
        "lifestyle": "lifestyle & vibe",
        "daily_life": "day-to-day convenience",
        "safety": "safety angle",
        "comparison": "comparison context",
        "decision_frame": "decision framing",
    }
    picks = [human_map.get(x, x) for x in flags][:3]
    if picks:
        return f"Competitor includes this section with practical details (e.g., {', '.join(picks)})."
    return "Competitor includes this section with additional context and practical details."

def summarize_missing_parts(b_header: str, comp_text: str, bayut_text: str) -> str:
    comp_flags = theme_flags(comp_text)
    bayut_flags = theme_flags(bayut_text)
    missing = list(comp_flags - bayut_flags)
    human_map = {
        "transport": "commute & connectivity",
        "traffic_parking": "traffic/parking realities",
        "cost": "cost considerations",
        "lifestyle": "lifestyle & vibe",
        "daily_life": "day-to-day convenience",
        "safety": "safety angle",
        "comparison": "comparison context",
        "decision_frame": "decision framing",
    }
    missing_h = [human_map.get(x, x) for x in missing][:3]
    if missing_h:
        return "Competitor goes deeper on: " + ", ".join(missing_h) + "."
    return "Competitor provides more depth and practical specifics than Bayut under the same header."

def dedupe_rows(rows: List[dict]) -> List[dict]:
    out = []
    seen = set()
    for r in rows:
        hk = norm_header(r.get("Headers",""))
        sk = norm_header(r.get("Source",""))
        k = hk + "||" + sk
        if k in seen:
            continue
        seen.add(k)
        out.append(r)
    return out

def content_gaps_header_first(
    bayut_nodes: List[dict],
    bayut_fr: FetchResult,
    comp_nodes: List[dict],
    comp_fr: FetchResult,
    comp_url: str,
    max_missing_headers: int = 7,
    max_missing_parts: int = 5,
) -> List[dict]:
    rows: List[dict] = []

    bayut_secs = section_nodes(bayut_nodes, levels=(2,3))
    comp_secs = section_nodes(comp_nodes, levels=(2,3))

    bayut_h2 = [s for s in bayut_secs if s["level"] == 2]
    comp_h2 = [s for s in comp_secs if s["level"] == 2]

    # A) Missing headers FIRST (competitor H2 not in Bayut H2)
    missing_h2_norms = set()
    missing_count = 0
    for cs in comp_h2:
        m = find_best_match(cs["header"], bayut_h2, min_score=0.73)
        if m:
            continue
        missing_h2_norms.add(norm_header(cs["header"]))
        rows.append({
            "Headers": cs["header"],
            "Description": summarize_missing_header(cs["header"], cs.get("content","")),
            "Source": site_name(comp_url),
        })
        missing_count += 1
        if missing_count >= max_missing_headers:
            break

    # B) (missing parts) only for matching headers where competitor is stronger
    parts_count = 0
    for cs in comp_h2:
        m = find_best_match(cs["header"], bayut_h2, min_score=0.73)
        if not m:
            continue
        bs = m["section"]
        c_txt = clean(cs.get("content",""))
        b_txt = clean(bs.get("content",""))

        # if competitor content is materially richer
        if len(c_txt) < 160:
            continue
        if len(c_txt) < (1.30 * max(len(b_txt), 1)):
            continue

        comp_flags = theme_flags(c_txt)
        bayut_flags = theme_flags(b_txt)
        # require meaningful missing themes or big content
        if len(comp_flags - bayut_flags) < 1 and len(c_txt) < 650:
            continue

        rows.append({
            "Headers": f"{bs['header']} (missing parts)",
            "Description": summarize_missing_parts(bs["header"], c_txt, b_txt),
            "Source": site_name(comp_url),
        })
        parts_count += 1
        if parts_count >= max_missing_parts:
            break

    # C) FAQ ONE ROW ONLY (ONLY if competitor has real FAQ)
    faq_row = missing_faqs_row(bayut_nodes, bayut_fr, comp_nodes, comp_fr, comp_url)
    if faq_row:
        rows.append(faq_row)

    return dedupe_rows(rows)


# =====================================================
# SEO EXTRACT (NO HEADERS COUNT, NEW KW USAGE LOGIC)
# =====================================================
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

def extract_media_used_counts(html: str) -> Dict[str, int]:
    counts = {"tables": 0, "videos": 0}
    if not html:
        return counts
    soup = BeautifulSoup(html, "html.parser")
    for t in soup.find_all(list(IGNORE_TAGS)):
        t.decompose()
    root = soup.find("article") or soup.find("main") or soup

    counts["tables"] = len(root.find_all("table"))

    vids = len(root.find_all("video"))
    # count youtube/vimeo iframes as video
    for x in root.find_all("iframe"):
        src = (x.get("src") or "").lower()
        if any(k in src for k in ["youtube", "youtu.be", "vimeo", "dailymotion"]):
            vids += 1
    counts["videos"] = vids
    return counts

def word_count_from_text(text: str) -> int:
    t = clean(text or "")
    if not t:
        return 0
    return len(re.findall(r"\b\w+\b", t))

def tokenize(text: str) -> List[str]:
    text = (text or "").lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return [t for t in text.split() if t and len(t) >= 3]

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

def pick_fkw(seo_title: str, h1: str, body_text: str, manual_fkw: str = "") -> str:
    manual_fkw = clean(manual_fkw)
    if manual_fkw:
        return manual_fkw
    base = " ".join([seo_title or "", h1 or "", body_text or ""])
    freq = phrase_candidates(base, 2, 4)
    if not freq:
        return "Not available"
    title_low = (seo_title or "").lower()
    h1_low = (h1 or "").lower()

    scored = []
    for ph, c in freq.items():
        boost = 1.0
        if ph in title_low:
            boost += 0.8
        if ph in h1_low:
            boost += 0.6
        scored.append((c * boost, ph))
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[0][1] if scored else "Not available"

def kw_usage_quality(body_text: str, fkw: str, seo_title: str, h1: str) -> str:
    """
    Replaces "FKW repeats":
    - checks relevance placement
    - checks stuffing-ish repetition per 1k words
    """
    if not body_text or not fkw or fkw == "Not available":
        return "Not available"

    wc = word_count_from_text(body_text)
    if wc <= 0:
        return "Not available"

    t = " " + re.sub(r"\s+", " ", body_text.lower()) + " "
    p = " " + re.sub(r"\s+", " ", fkw.lower()) + " "
    repeats = t.count(p)

    per_1k = (repeats / max(wc, 1)) * 1000.0

    in_title = fkw.lower() in (seo_title or "").lower()
    in_h1 = fkw.lower() in (h1 or "").lower()

    placement = []
    if in_title: placement.append("title")
    if in_h1: placement.append("H1")
    placement_txt = " + ".join(placement) if placement else "not in title/H1"

    if per_1k >= 18:
        return f"Overused (≈{per_1k:.1f}/1k words, {placement_txt})"
    if per_1k >= 10:
        return f"Needs review (≈{per_1k:.1f}/1k words, {placement_txt})"
    return f"Good (≈{per_1k:.1f}/1k words, {placement_txt})"


# =====================================================
# DATAFORSEO: UAE RANKING + AI OVERVIEW VISIBILITY
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

def d4s_post(path: str, payload: list) -> dict:
    if not DATAFORSEO_LOGIN or not DATAFORSEO_PASSWORD:
        return {"_error": "missing_dataforseo_credentials"}
    try:
        url = f"https://api.dataforseo.com/v3/{path}"
        r = requests.post(
            url,
            auth=(DATAFORSEO_LOGIN, DATAFORSEO_PASSWORD),
            json=payload,
            timeout=45
        )
        if r.status_code != 200:
            return {"_error": f"http_{r.status_code}", "_text": r.text[:500]}
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

@st.cache_data(show_spinner=False, ttl=1200)
def d4s_serp_cached(keyword: str, device: str) -> dict:
    if not keyword:
        return {"_error": "missing_keyword"}

    payload = [{
        "keyword": keyword,
        "location_name": "United Arab Emirates",
        "language_code": "en",
        "se_domain": "google.ae",
        "device": device,                  # "desktop" | "mobile"
        "os": "windows" if device == "desktop" else "android",
        "load_async_ai_overview": True,
        "depth": 20
    }]
    return d4s_post("serp/google/organic/live/advanced", payload)

def d4s_rank_for_url(keyword: str, page_url: str, device: str) -> Tuple[Optional[int], str]:
    data = d4s_serp_cached(keyword, device=device)
    if not data or data.get("_error"):
        return None, f"Not available ({data.get('_error')})"

    try:
        tasks = data.get("tasks") or []
        if not tasks or not tasks[0].get("result"):
            return None, "Not available (no_result)"
        items = tasks[0]["result"][0].get("items") or []
    except Exception:
        return None, "Not available (bad_response)"

    target = normalize_url_for_match(page_url)
    for it in items:
        it_url = it.get("url") or it.get("destination_url") or ""
        if not it_url:
            continue
        if normalize_url_for_match(it_url) == target or target in normalize_url_for_match(it_url):
            pos = it.get("rank_group") or it.get("rank_absolute") or it.get("position")
            try:
                return int(pos), "OK"
            except Exception:
                return None, "Not available"
    return None, "Not in top results"

def d4s_ai_visibility(keyword: str, page_url: str, device: str) -> Dict[str, str]:
    data = d4s_serp_cached(keyword, device=device)
    if not data or data.get("_error"):
        return {"AI Overview present": "Not available", "Cited in AI Overview": "Not available", "AI Notes": str(data.get("_error"))}

    try:
        tasks = data.get("tasks") or []
        if not tasks or not tasks[0].get("result"):
            return {"AI Overview present": "Not available", "Cited in AI Overview": "Not available", "AI Notes": "no_result"}
        items = tasks[0]["result"][0].get("items") or []
    except Exception:
        return {"AI Overview present": "Not available", "Cited in AI Overview": "Not available", "AI Notes": "bad_response"}

    # Detect AI overview block
    aio_blocks = [x for x in items if "ai_overview" in str(x.get("type","")).lower()]
    present = "Yes" if aio_blocks else "No"

    cited = "No"
    notes = ""

    if aio_blocks:
        target = normalize_url_for_match(page_url)
        # Try to find cited links inside aio block payload
        for b in aio_blocks:
            # common fields: "references", "items", "links", etc.
            blob = json.dumps(b)[:20000].lower()
            if target and target in blob:
                cited = "Yes"
                break
        notes = "AI overview detected in SERP items." if cited == "No" else "Page appears referenced inside AI overview payload."

    return {"AI Overview present": present, "Cited in AI Overview": cited, "AI Notes": notes}


# =====================================================
# TABLE RENDER
# =====================================================
def render_table(df: pd.DataFrame):
    if df is None or df.empty:
        st.info("No results to show.")
        return
    html = df.to_html(index=False, escape=False)
    st.markdown(html, unsafe_allow_html=True)


# =====================================================
# SESSION RESET (fix old URLs / stale results issue)
# =====================================================
def compute_inputs_hash(*vals) -> str:
    raw = json.dumps(vals, ensure_ascii=False)
    return hashlib.md5(raw.encode("utf-8")).hexdigest()

def reset_results():
    st.session_state.update_df = pd.DataFrame()
    st.session_state.seo_df = pd.DataFrame()
    st.session_state.ai_df = pd.DataFrame()
    st.session_state.cq_df = pd.DataFrame()


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
        if st.button(
            "Update Mode",
            type="primary" if st.session_state.mode == "update" else "secondary",
            use_container_width=True,
            key="mode_update_btn",
        ):
            st.session_state.mode = "update"
            reset_results()
    with b2:
        if st.button(
            "New Post Mode",
            type="primary" if st.session_state.mode == "new" else "secondary",
            use_container_width=True,
            key="mode_new_btn",
        ):
            st.session_state.mode = "new"
            reset_results()
st.markdown("</div>", unsafe_allow_html=True)


# Initialize storage
if "update_df" not in st.session_state: st.session_state.update_df = pd.DataFrame()
if "seo_df" not in st.session_state: st.session_state.seo_df = pd.DataFrame()
if "ai_df" not in st.session_state: st.session_state.ai_df = pd.DataFrame()
if "cq_df" not in st.session_state: st.session_state.cq_df = pd.DataFrame()
if "last_inputs_hash" not in st.session_state: st.session_state.last_inputs_hash = ""


# =====================================================
# UPDATE MODE
# =====================================================
if st.session_state.mode == "update":
    st.markdown("<div class='section-pill section-pill-tight'>Just Update Mode</div>", unsafe_allow_html=True)

    bayut_url = st.text_input("Bayut article URL", placeholder="https://www.bayut.com/mybayut/...")
    competitors_text = st.text_area("Competitor URLs (one per line)", height=120)
    competitors = [c.strip() for c in competitors_text.splitlines() if c.strip()]
    manual_fkw = st.text_input("Optional: Focus Keyword", placeholder="e.g., living in Dubai Marina")

    # auto-reset if inputs changed (fix stale results)
    curr_hash = compute_inputs_hash("update", bayut_url.strip(), competitors, manual_fkw.strip())
    if st.session_state.last_inputs_hash and curr_hash != st.session_state.last_inputs_hash:
        reset_results()
    st.session_state.last_inputs_hash = curr_hash

    run = st.button("Run analysis", type="primary")

    if run:
        if not bayut_url.strip():
            st.error("Bayut article URL is required.")
            st.stop()
        if not competitors:
            st.error("Add at least one competitor URL.")
            st.stop()

        # 1) fetch + hard gate
        with st.spinner("Fetching Bayut (no missing)…"):
            bayut_fr_map = resolve_all_or_require_manual([bayut_url.strip()], "bayut")
            bayut_fr_map = require_html_for_accuracy(bayut_fr_map, [bayut_url.strip()], "bayut_html")

        with st.spinner("Fetching competitors (no missing)…"):
            comp_fr_map = resolve_all_or_require_manual(competitors, "comp")
            comp_fr_map = require_html_for_accuracy(comp_fr_map, competitors, "comp_html")

        bayut_fr = bayut_fr_map[bayut_url.strip()]
        bayut_nodes = build_tree_from_html(bayut_fr.html)

        comp_nodes_map = {}
        for u in competitors:
            comp_nodes_map[u] = build_tree_from_html(comp_fr_map[u].html)

        # 2) content gaps
        rows = []
        for u in competitors:
            rows.extend(
                content_gaps_header_first(
                    bayut_nodes=bayut_nodes,
                    bayut_fr=bayut_fr,
                    comp_nodes=comp_nodes_map[u],
                    comp_fr=comp_fr_map[u],
                    comp_url=u,
                    max_missing_headers=7,
                    max_missing_parts=5
                )
            )
        st.session_state.update_df = pd.DataFrame(rows, columns=["Headers", "Description", "Source"])

        # 3) SEO table (no headers count, no repeats)
        seo_rows = []
        # Bayut
        b_title, b_desc = extract_head_seo(bayut_fr.html)
        b_h1 = get_first_h1(bayut_nodes)
        b_fkw = pick_fkw(b_title, b_h1, bayut_fr.text, manual_fkw=manual_fkw.strip())
        seo_rows.append({
            "Page": "Bayut",
            "SEO title": b_title,
            "Meta description": b_desc,
            "Slug": url_slug(bayut_url.strip()),
            "FKW": b_fkw,
            "KW usage": kw_usage_quality(bayut_fr.text, b_fkw, b_title, b_h1),
            "Google rank UAE (Desktop)": "Not run",
            "Google rank UAE (Mobile)": "Not run",
            "__url": bayut_url.strip()
        })

        # competitors
        for u in competitors:
            fr = comp_fr_map[u]
            nodes = comp_nodes_map[u]
            title, desc = extract_head_seo(fr.html)
            h1 = get_first_h1(nodes)
            fkw = pick_fkw(title, h1, fr.text, manual_fkw=manual_fkw.strip())
            seo_rows.append({
                "Page": site_name(u),
                "SEO title": title,
                "Meta description": desc,
                "Slug": url_slug(u),
                "FKW": fkw,
                "KW usage": kw_usage_quality(fr.text, fkw, title, h1),
                "Google rank UAE (Desktop)": "Not run",
                "Google rank UAE (Mobile)": "Not run",
                "__url": u
            })

        seo_df = pd.DataFrame(seo_rows)
        st.session_state.seo_df = seo_df

        # 4) Ranking + AI Visibility via DataForSEO
        with st.spinner("Fetching Google UAE ranking (desktop + mobile) + AI visibility (DataForSEO)…"):
            ai_rows = []
            for i, r in st.session_state.seo_df.iterrows():
                page = r["Page"]
                page_url = r["__url"]
                query = manual_fkw.strip() if manual_fkw.strip() else r["FKW"]

                if not query or query == "Not available":
                    st.session_state.seo_df.at[i, "Google rank UAE (Desktop)"] = "Not available"
                    st.session_state.seo_df.at[i, "Google rank UAE (Mobile)"] = "Not available"
                    ai_rows.append({
                        "Page": page,
                        "Query": "Not available",
                        "AI Overview present": "Not available",
                        "Cited in AI Overview": "Not available",
                        "AI Notes": "No keyword available for SERP check."
                    })
                    continue

                rank_d, note_d = d4s_rank_for_url(query, page_url, device="desktop")
                rank_m, note_m = d4s_rank_for_url(query, page_url, device="mobile")

                st.session_state.seo_df.at[i, "Google rank UAE (Desktop)"] = str(rank_d) if rank_d else note_d
                st.session_state.seo_df.at[i, "Google rank UAE (Mobile)"] = str(rank_m) if rank_m else note_m

                ai = d4s_ai_visibility(query, page_url, device="desktop")
                ai_rows.append({
                    "Page": page,
                    "Query": query,
                    "AI Overview present": ai.get("AI Overview present", "Not available"),
                    "Cited in AI Overview": ai.get("Cited in AI Overview", "Not available"),
                    "AI Notes": ai.get("AI Notes", ""),
                })

            st.session_state.ai_df = pd.DataFrame(ai_rows)

        # 5) Content Quality (SECOND TABLE) — accurate FAQ/video/table using real HTML
        cq_cols = ["Content Quality", "Bayut"]
        for u in competitors:
            cq_cols.append(site_name(u))

        metrics = ["Word Count", "FAQs", "Tables", "Video"]
        out = {"Content Quality": metrics}

        # Bayut
        b_media = extract_media_used_counts(bayut_fr.html)
        out["Bayut"] = [
            str(word_count_from_text(bayut_fr.text)),
            "Yes" if page_has_real_faq(bayut_fr, bayut_nodes) else "No",
            str(b_media["tables"]),
            str(b_media["videos"]),
        ]

        # competitors
        for u in competitors:
            fr = comp_fr_map[u]
            nodes = comp_nodes_map[u]
            media = extract_media_used_counts(fr.html)
            out[site_name(u)] = [
                str(word_count_from_text(fr.text)),
                "Yes" if page_has_real_faq(fr, nodes) else "No",
                str(media["tables"]),
                str(media["videos"]),
            ]

        st.session_state.cq_df = pd.DataFrame(out)

    # ===== OUTPUT ORDER (YOUR REQUEST) =====
    st.markdown("<div class='section-pill section-pill-tight'>Content Gaps Table</div>", unsafe_allow_html=True)
    render_table(st.session_state.update_df)

    st.markdown("<div class='section-pill section-pill-tight'>Content Quality</div>", unsafe_allow_html=True)
    render_table(st.session_state.cq_df)

    st.markdown("<div class='section-pill section-pill-tight'>SEO Analysis</div>", unsafe_allow_html=True)
    if st.session_state.seo_df is not None and not st.session_state.seo_df.empty:
        render_table(st.session_state.seo_df.drop(columns=["__url"]))
    else:
        st.info("Run analysis to see SEO comparison.")

    st.markdown("<div class='section-pill section-pill-tight'>AI Visibility (Google AI Overview)</div>", unsafe_allow_html=True)
    render_table(st.session_state.ai_df)


# =====================================================
# NEW POST MODE (kept minimal)
# =====================================================
else:
    st.markdown("<div class='section-pill section-pill-tight'>New Post Mode</div>", unsafe_allow_html=True)

    new_title = st.text_input("New post title", placeholder="Pros & Cons of Living in Dubai Marina (2026)")
    competitors_text = st.text_area("Competitor URLs (one per line)", height=120)
    competitors = [c.strip() for c in competitors_text.splitlines() if c.strip()]
    manual_fkw = st.text_input("Optional: Focus Keyword", placeholder="e.g., living in Dubai Marina")

    curr_hash = compute_inputs_hash("new", new_title.strip(), competitors, manual_fkw.strip())
    if st.session_state.last_inputs_hash and curr_hash != st.session_state.last_inputs_hash:
        reset_results()
    st.session_state.last_inputs_hash = curr_hash

    run = st.button("Generate competitor coverage", type="primary")

    if run:
        if not new_title.strip():
            st.error("New post title is required.")
            st.stop()
        if not competitors:
            st.error("Add at least one competitor URL.")
            st.stop()

        with st.spinner("Fetching competitors (no missing)…"):
            comp_fr_map = resolve_all_or_require_manual(competitors, "comp_new")
            comp_fr_map = require_html_for_accuracy(comp_fr_map, competitors, "comp_new_html")

        comp_nodes_map = {u: build_tree_from_html(comp_fr_map[u].html) for u in competitors}

        # (simple coverage table)
        rows = []
        for u in competitors:
            h1 = get_first_h1(comp_nodes_map[u])
            h2s = [x["header"] for x in flatten(comp_nodes_map[u]) if x["level"] == 2 and not is_noise_header(x["header"]) and not header_is_faq(x["header"])]
            rows.append({
                "Headers covered": "H1 (main angle)",
                "Content covered": h1 or "Not available",
                "Source": site_name(u)
            })
            rows.append({
                "Headers covered": "H2 (sections covered)",
                "Content covered": " → ".join(h2s[:6]) if h2s else "Not available",
                "Source": site_name(u)
            })

        st.session_state.update_df = pd.DataFrame(rows, columns=["Headers covered","Content covered","Source"])

        # SEO + ranks + AI visibility
        seo_rows = []
        for u in competitors:
            fr = comp_fr_map[u]
            nodes = comp_nodes_map[u]
            title, desc = extract_head_seo(fr.html)
            h1 = get_first_h1(nodes)
            fkw = pick_fkw(title, h1, fr.text, manual_fkw=manual_fkw.strip())
            seo_rows.append({
                "Page": site_name(u),
                "SEO title": title,
                "Meta description": desc,
                "Slug": url_slug(u),
                "FKW": fkw,
                "KW usage": kw_usage_quality(fr.text, fkw, title, h1),
                "Google rank UAE (Desktop)": "Not run",
                "Google rank UAE (Mobile)": "Not run",
                "__url": u
            })

        st.session_state.seo_df = pd.DataFrame(seo_rows)

        with st.spinner("Fetching Google UAE ranking (desktop + mobile) + AI visibility (DataForSEO)…"):
            ai_rows = []
            for i, r in st.session_state.seo_df.iterrows():
                page = r["Page"]
                page_url = r["__url"]
                query = manual_fkw.strip() if manual_fkw.strip() else r["FKW"]

                rank_d, note_d = d4s_rank_for_url(query, page_url, device="desktop") if query else (None, "Not available")
                rank_m, note_m = d4s_rank_for_url(query, page_url, device="mobile") if query else (None, "Not available")

                st.session_state.seo_df.at[i, "Google rank UAE (Desktop)"] = str(rank_d) if rank_d else note_d
                st.session_state.seo_df.at[i, "Google rank UAE (Mobile)"] = str(rank_m) if rank_m else note_m

                ai = d4s_ai_visibility(query, page_url, device="desktop") if query else {"AI Overview present":"Not available","Cited in AI Overview":"Not available","AI Notes":"No query"}
                ai_rows.append({
                    "Page": page,
                    "Query": query or "Not available",
                    "AI Overview present": ai.get("AI Overview present", "Not available"),
                    "Cited in AI Overview": ai.get("Cited in AI Overview", "Not available"),
                    "AI Notes": ai.get("AI Notes", ""),
                })
            st.session_state.ai_df = pd.DataFrame(ai_rows)

        # Content quality
        metrics = ["Word Count", "FAQs", "Tables", "Video"]
        out = {"Content Quality": metrics}
        for u in competitors:
            fr = comp_fr_map[u]
            nodes = comp_nodes_map[u]
            media = extract_media_used_counts(fr.html)
            out[site_name(u)] = [
                str(word_count_from_text(fr.text)),
                "Yes" if page_has_real_faq(fr, nodes) else "No",
                str(media["tables"]),
                str(media["videos"]),
            ]
        st.session_state.cq_df = pd.DataFrame(out)

    st.markdown("<div class='section-pill section-pill-tight'>Competitor Coverage</div>", unsafe_allow_html=True)
    render_table(st.session_state.update_df)

    st.markdown("<div class='section-pill section-pill-tight'>Content Quality</div>", unsafe_allow_html=True)
    render_table(st.session_state.cq_df)

    st.markdown("<div class='section-pill section-pill-tight'>SEO Analysis</div>", unsafe_allow_html=True)
    if st.session_state.seo_df is not None and not st.session_state.seo_df.empty:
        render_table(st.session_state.seo_df.drop(columns=["__url"]))
    else:
        st.info("Generate competitor coverage to see SEO comparison.")

    st.markdown("<div class='section-pill section-pill-tight'>AI Visibility (Google AI Overview)</div>", unsafe_allow_html=True)
    render_table(st.session_state.ai_df)
