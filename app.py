# ===============================
# app.py — PART 1 / 3
# Fetching + Parsing + Header Matching Normalization
# ===============================

import base64
import html as html_lib
import streamlit as st
import requests
import re
from bs4 import BeautifulSoup
from urllib.parse import quote_plus, urlparse, urljoin
import pandas as pd
import time, random, hashlib
from dataclasses import dataclass
from typing import Optional, Dict, Tuple, List
from difflib import SequenceMatcher
import json

# Optional JS rendering (Playwright)
try:
    from playwright.sync_api import sync_playwright
    PLAYWRIGHT_OK = True
except Exception:
    PLAYWRIGHT_OK = False

# ===============================
# PAGE CONFIG (MUST BE FIRST)
# ===============================
st.set_page_config(page_title="Bayut Competitor Gap Analysis", layout="wide")

# ===============================
# STYLE (KEEP YOUR CURRENT LOOK)
# ===============================
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
        padding: 6px 14px !important;
        border-bottom: 1px solid #E5E7EB !important;
      }}
      tbody td {{
        vertical-align: top !important;
        padding: 6px 6px !important;
        border-bottom: 1px solid #F1F5F9 !important;
        color: {TEXT_DARK} !important;
        font-size: 12px !important;
      }}
      tbody tr:last-child td {{
        border-bottom: 0 !important;
      }}
      a {{
        color: {BAYUT_GREEN} !important;
        font-weight: 900 !important;
        text-decoration: underline !important;
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

# ===============================
# FETCH (BEST EFFORT — NEVER STOP)
# ===============================
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
MEDIA_IGNORE_TAGS = {"nav", "footer", "script", "style", "noscript", "form", "aside"}

def clean(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip()

def looks_blocked(text: str) -> bool:
    t = (text or "").lower()
    return any(x in t for x in [
        "just a moment", "checking your browser", "verify you are human",
        "cloudflare", "access denied", "captcha", "forbidden", "service unavailable",
        "pardon our interruption", "made us think you were a bot", "enable cookies",
        "unusual traffic", "request blocked", "incapsula", "distil", "perimeterx",
        "datadome", "akamai", "sucuri", "radware", "bot detection"
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
    Best-effort resolver:
    - For dubizzle: Playwright FIRST (forced)
    - Otherwise: direct -> Playwright -> Jina -> Textise
    Never stops the app; caller decides to skip.
    """

    def __init__(self, default_headers: dict):
        self.default_headers = default_headers
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
        for t in soup.find_all(list(IGNORE_TAGS)):
            t.decompose()
        article = soup.find("article") or soup.find("main") or soup
        return clean(article.get_text(" "))

    def _validate_text(self, text: str, min_len: int) -> bool:
        if not text:
            return False
        if looks_blocked(text):
            return False
        t = clean(text)
        return len(t) >= min_len

    def _fetch_playwright_html(self, url: str, timeout_ms: int = 35000) -> Tuple[bool, str]:
        if not PLAYWRIGHT_OK:
            return False, ""
        try:
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True, args=["--no-sandbox"])
                ctx = browser.new_context(user_agent=random.choice(self.user_agents))
                page = ctx.new_page()
                page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)
                try:
                    page.wait_for_load_state("networkidle", timeout=timeout_ms)
                except Exception:
                    pass
                page.wait_for_timeout(1200)
                html = page.content()
                browser.close()
            return True, html
        except Exception:
            return False, ""

    def resolve(self, url: str) -> FetchResult:
        url = (url or "").strip()
        if not url:
            return FetchResult(False, None, None, "", "", "empty_url")

        host = (urlparse(url).netloc or "").lower()
        force_js = any(x in host for x in ["dubizzle.com"])

        # 0) Force JS FIRST for dubizzle
        if force_js:
            ok, html_js = self._fetch_playwright_html(url, timeout_ms=40000)
            if ok and html_js:
                txt = self._extract_article_text_from_html(html_js)
                if self._validate_text(txt, min_len=250):
                    return FetchResult(True, "playwright-forced", 200, html_js, txt, None)

        # 1) direct
        code, html = self._http_get(url)
        if code == 200 and html:
            txt = self._extract_article_text_from_html(html)
            if self._validate_text(txt, min_len=250):
                return FetchResult(True, "direct", code, html, txt, None)

        # 2) Playwright fallback
        ok, html_js = self._fetch_playwright_html(url, timeout_ms=35000)
        if ok and html_js:
            txt = self._extract_article_text_from_html(html_js)
            if self._validate_text(txt, min_len=250):
                return FetchResult(True, "playwright", 200, html_js, txt, None)

        # 3) Jina
        jurl = self._jina_url(url)
        code3, txt3 = self._http_get(jurl)
        if code3 == 200 and txt3 and self._validate_text(txt3, min_len=220):
            return FetchResult(True, "jina", code3, "", txt3, None)

        # 4) Textise
        turl = self._textise_url(url)
        code4, html4 = self._http_get(turl)
        if code4 == 200 and html4:
            soup = BeautifulSoup(html4, "html.parser")
            txt4 = soup.get_text("\n")
            if self._validate_text(txt4, min_len=220):
                return FetchResult(True, "textise", code4, "", txt4, None)

        return FetchResult(False, None, code or None, "", "", "blocked_or_no_content")

agent = FetchAgent(DEFAULT_HEADERS)

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

# ===============================
# HEADING TREE
# ===============================
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
    return ""

def tokenize(text: str) -> List[str]:
    text = (text or "").lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return [t for t in text.split() if t and len(t) >= 3]

STOP = {
    "the","and","for","with","that","this","from","you","your","are","was","were","will","have","has","had",
    "but","not","can","may","more","most","into","than","then","they","them","their","our","out","about",
    "also","over","under","between","within","near","where","when","what","why","how","who","which",
    "a","an","to","of","in","on","at","as","is","it","be","or","by","we","i","us","living","live"
}

def norm_header(h: str) -> str:
    h = clean(h).lower()
    h = re.sub(r"[^a-z0-9\s]", " ", h)
    h = re.sub(r"\s+", " ", h).strip()
    return h

def _context_terms_from_h1(h1: str) -> set:
    toks = tokenize(h1 or "")
    out = set()
    for t in toks:
        if t in STOP:
            continue
        if len(t) < 3:
            continue
        out.add(t)
    return out

def canonical_header_key_with_context(h: str, context_terms: set) -> str:
    t = norm_header(h)
    if not t:
        return ""

    # normalize synonyms
    t = re.sub(r"\badvantages\b|\bbenefits\b|\bpositives\b", "pros", t)
    t = re.sub(r"\bdisadvantages\b|\bdownsides\b|\bdrawbacks\b|\bnegatives\b", "cons", t)
    t = re.sub(r"\bpros and cons\b", "pros cons", t)

    # remove filler
    t = re.sub(r"\bpros of living in\b", "pros", t)
    t = re.sub(r"\bcons of living in\b", "cons", t)
    t = re.sub(r"\bof living in\b", "", t)
    t = re.sub(r"\bliving in\b", "", t)

    if context_terms:
        words = [w for w in t.split() if w not in context_terms]
        t = " ".join(words)

    t = re.sub(r"\s+", " ", t).strip()
    return t

def header_similarity(a: str, b: str, context_terms: set) -> float:
    a_n = canonical_header_key_with_context(a, context_terms)
    b_n = canonical_header_key_with_context(b, context_terms)
    if not a_n or not b_n:
        return 0.0
    a_set = set(a_n.split())
    b_set = set(b_n.split())
    jacc = len(a_set & b_set) / max(len(a_set | b_set), 1) if a_set and b_set else 0.0
    seq = SequenceMatcher(None, a_n, b_n).ratio()
    return (0.55 * seq) + (0.45 * jacc)

def section_nodes(nodes: List[dict], levels=(2,3,4)) -> List[dict]:
    secs = []
    current_h2 = None
    for x in flatten(nodes):
        lvl = x["level"]
        h = clean(x.get("header",""))
        if not h:
            continue
        if lvl == 2:
            current_h2 = h
        if lvl in levels:
            c = clean(x.get("content",""))
            secs.append({"level": lvl, "header": h, "content": c, "parent_h2": current_h2})
    return secs

def find_best_bayut_match(comp_header: str, bayut_sections: List[dict], context_terms: set, min_score: float = 0.70) -> Optional[dict]:
    best = None
    best_score = 0.0
    for b in bayut_sections:
        sc = header_similarity(comp_header, b["header"], context_terms=context_terms)
        if sc > best_score:
            best_score = sc
            best = b
    if best and best_score >= min_score:
        return {"bayut_section": best, "score": best_score}
    return None

def format_gap_points(points: List[str]) -> str:
    cleaned = []
    for p in points or []:
        p = clean(p).rstrip(".")
        if p and p not in cleaned:
            cleaned.append(p)
    if not cleaned:
        return ""
    if len(cleaned) == 1:
        return cleaned[0] + "."
    lis = "".join(f"<li>{html_lib.escape(p)}</li>" for p in cleaned)
    return f"<ul>{lis}</ul></ul>".replace("</ul></ul>", "</ul>")

def format_gap_list(items: List[str], limit: int = 8) -> str:
    out, seen = [], set()
    for it in (items or []):
        it2 = clean(it)
        k = norm_header(it2)
        if not it2 or not k or k in seen:
            continue
        seen.add(k)
        out.append(it2)
        if len(out) >= limit:
            break
    return ", ".join(out)

def resolve_all_best_effort(agent: FetchAgent, urls: List[str]) -> Tuple[Dict[str, FetchResult], List[str]]:
    results: Dict[str, FetchResult] = {}
    skipped: List[str] = []
    for u in urls:
        fr = agent.resolve(u)
        results[u] = fr
        if not fr.ok:
            skipped.append(u)
        time.sleep(0.2)
    return results, skipped

def trees_best_effort(fr_map: Dict[str, FetchResult], urls: List[str]) -> Tuple[Dict[str, List[dict]], List[str]]:
    tree_map: Dict[str, List[dict]] = {}
    bad: List[str] = []
    for u in urls:
        fr = fr_map.get(u)
        if not fr or not fr.ok:
            bad.append(u)
            continue
        html = fr.html or ""
        if not html and fr.source in ("jina", "textise"):
            # reader output may not be HTML; headings may be weak. We still try basic heading parse from HTML only.
            # If no HTML, we create a single pseudo node so the system never breaks.
            tree_map[u] = [{"level": 2, "header": "Overview", "content": clean(fr.text or ""), "children": []}]
            continue
        nodes = build_tree_from_html(html) if html else [{"level": 2, "header": "Overview", "content": clean(fr.text or ""), "children": []}]
        if not nodes:
            nodes = [{"level": 2, "header": "Overview", "content": clean(fr.text or ""), "children": []}]
        tree_map[u] = nodes
    return tree_map, bad
# ===============================
# app.py — PART 2 / 3
# FAQ detection + Content gap engine (NO fake "Pros missing")
# ===============================

FAQ_TITLES = {"faq", "faqs", "frequently asked questions", "frequently asked question"}

def header_is_faq(header: str) -> bool:
    nh = norm_header(header)
    if not nh:
        return False
    if nh in FAQ_TITLES:
        return True
    if "faq" in nh or "frequently asked" in nh:
        return True
    return False

def _looks_like_question(s: str) -> bool:
    s = clean(s)
    if not s or len(s) < 6:
        return False
    sl = s.lower()
    if "?" in s:
        return True
    if re.match(r"^(what|where|when|why|how|who|which|can|is|are|do|does|did|should|could|would|will)\b", sl):
        return True
    return False

def normalize_question(q: str) -> str:
    q = clean(q or "")
    q = re.sub(r"^\s*\d+[\.\)]\s*", "", q)
    q = re.sub(r"^\s*[-•]\s*", "", q)
    q = re.sub(r"^\s*(q|question|faq)s?\s*[:\-]\s*", "", q, flags=re.I)
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
                        if walk(v): return True
                elif isinstance(x, list):
                    for v in x:
                        if walk(v): return True
                return False
            if walk(j):
                return True
    except Exception:
        return False
    return False

def _faq_questions_from_schema(html: str) -> List[str]:
    if not html:
        return []
    qs = []
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
                t_list = []
                if isinstance(t, list): t_list = [str(z).lower() for z in t]
                elif isinstance(t, str): t_list = [t.lower()]
                if any("question" == z or z.endswith("question") for z in t_list):
                    name = x.get("name") or x.get("text") or ""
                    if name:
                        qn = normalize_question(name)
                        if qn and len(qn) <= 180:
                            qs.append(qn)
                for v in x.values():
                    walk(v)
            elif isinstance(x, list):
                for v in x:
                    walk(v)
        walk(j)

    out, seen = [], set()
    for q in qs:
        k = norm_header(q)
        if k and k not in seen:
            seen.add(k)
            out.append(q)
    return out

def _faq_questions_from_html(html: str) -> List[str]:
    if not html:
        return []
    soup = BeautifulSoup(html, "html.parser")
    for t in soup.find_all(list(IGNORE_TAGS)):
        t.decompose()

    qs = []
    qs.extend(_faq_questions_from_schema(html))

    # find likely FAQ containers
    candidates = []
    for tag in soup.find_all(True):
        id_attr = (tag.get("id") or "").lower()
        cls_attr = " ".join(tag.get("class", []) or []).lower()
        if re.search(r"\bfaq\b|\bfaqs\b|\baccordion\b|\bquestions\b", id_attr + " " + cls_attr):
            candidates.append(tag)

    for h in soup.find_all(["h1","h2","h3","h4"]):
        if header_is_faq(h.get_text(" ")):
            candidates.append(h.parent or h)

    for c in candidates[:10]:
        for el in c.find_all(["summary","button","h3","h4","h5","strong","p","li","dt"]):
            txt = clean(el.get_text(" "))
            if not txt or len(txt) < 6 or len(txt) > 180:
                continue
            if _looks_like_question(txt):
                qs.append(normalize_question(txt))

    out, seen = [], set()
    for q in qs:
        k = norm_header(q)
        if k and k not in seen:
            seen.add(k)
            out.append(q)
    return out

def page_has_real_faq(fr: FetchResult) -> bool:
    # strict: schema OR at least 2 questions extractable
    if fr and fr.html:
        if _has_faq_schema(fr.html):
            return True
        if len(_faq_questions_from_html(fr.html)) >= 2:
            return True
    return False

# ---------- FAQ TOPIC FIX (NO RAW NONSENSE) ----------
FAQ_TOPIC_PATTERNS = [
    (re.compile(r"\baverage\b.*\b(rent|price|cost)\b", re.I), "Average rent / pricing"),
    (re.compile(r"\b(rent|rental)\b.*\baverage\b", re.I), "Average rent / pricing"),
    (re.compile(r"\b(cost|expenses|budget)\b", re.I), "Cost of living"),
    (re.compile(r"\b(residential buildings|buildings|apartments|villas|homes|residential)\b", re.I), "Housing types"),
    (re.compile(r"\b(safe|safety|security|crime)\b", re.I), "Safety"),
    (re.compile(r"\bfreehold\b", re.I), "Freehold status"),
    (re.compile(r"\b(entry|access|enter)\b", re.I), "Entry / access"),
    (re.compile(r"\b(amenit(?:y|ies)|facilit(?:y|ies)|services)\b", re.I), "Amenities & services"),
    (re.compile(r"\b(schools|nurser(?:y|ies)|kids|family)\b", re.I), "Families & schools"),
    (re.compile(r"\b(transport|metro|bus|commute|traffic|parking)\b", re.I), "Commute / traffic / parking"),
    (re.compile(r"\b(vibe|lifestyle|atmosphere|nightlife|restaurants|cafes)\b", re.I), "Lifestyle & vibe"),
    (re.compile(r"\badvantages?\b|\bpros\b", re.I), "Pros"),
    (re.compile(r"\bdisadvantages?\b|\bcons\b|\bdownsides\b", re.I), "Cons"),
]

def faq_topic_from_question(q: str, context_terms: set) -> str:
    qn = normalize_question(q)
    if not qn:
        return ""
    ql = qn.lower()

    # map to clean topic buckets
    for rx, label in FAQ_TOPIC_PATTERNS:
        if rx.search(ql):
            return label

    # if it starts with community/location tokens, do NOT output the raw messy phrase
    first_words = tokenize(qn)[:4]
    if any(w in context_terms for w in first_words):
        return "About the community"

    # safe fallback: short cleaned topic
    qn2 = re.sub(r"[\?\.\!]+$", "", qn).strip()
    if len(qn2) > 60:
        qn2 = qn2[:60].rstrip()
    return qn2

def missing_faqs_row(bayut_fr: FetchResult, comp_fr: FetchResult, comp_url: str, context_terms: set) -> Optional[dict]:
    if not (comp_fr and comp_fr.ok and page_has_real_faq(comp_fr)):
        return None

    comp_qs = _faq_questions_from_html(comp_fr.html or "")
    if not comp_qs:
        return {
            "Headers": "FAQs",
            "Description": "FAQ section present, but topics could not be parsed for comparison.",
            "Source": source_link(comp_url),
        }

    bayut_has = (bayut_fr and bayut_fr.ok and page_has_real_faq(bayut_fr))
    bayut_qs = _faq_questions_from_html(bayut_fr.html or "") if bayut_has else []

    def q_key(x: str) -> str:
        x = normalize_question(x)
        x = re.sub(r"[^a-z0-9\s]", " ", x.lower())
        return re.sub(r"\s+", " ", x).strip()

    bayut_set = {q_key(x) for x in bayut_qs}

    missing_qs = [q for q in comp_qs if q_key(q) not in bayut_set] if bayut_set else comp_qs
    if not missing_qs:
        return None

    topics = []
    seen = set()
    for q in missing_qs:
        t = faq_topic_from_question(q, context_terms=context_terms)
        k = norm_header(t)
        if t and k and k not in seen:
            seen.add(k)
            topics.append(t)
        if len(topics) >= 10:
            break

    topic_list = format_gap_list(topics, limit=10)
    if not topic_list:
        return None

    return {
        "Headers": "FAQs",
        "Description": format_gap_points([f"Missing FAQ topics: {topic_list}"]),
        "Source": source_link(comp_url),
    }

# ---------- GAP ENGINE (NO FAKE "PROS MISSING") ----------
def theme_flags(text: str) -> set:
    t = (text or "").lower()
    flags = set()
    def has_any(ws): return any(w in t for w in ws)

    if has_any(["metro","public transport","commute","connectivity","access","highway","roads","bus"]):
        flags.add("transport")
    if has_any(["parking","traffic","congestion","rush hour"]):
        flags.add("traffic_parking")
    if has_any(["cost","price","pricing","expensive","afford","budget","rent","fees"]):
        flags.add("cost")
    if has_any(["restaurants","cafes","nightlife","vibe","atmosphere","entertainment"]):
        flags.add("lifestyle")
    if has_any(["schools","nursery","kids","family","clinic","hospital","supermarket","groceries","pharmacy"]):
        flags.add("daily_life")
    if has_any(["safe","safety","security","crime"]):
        flags.add("safety")
    if has_any(["compare","comparison","versus","vs "]):
        flags.add("comparison")
    return flags

def missing_key_phrases(comp_text: str, bayut_text: str, limit: int = 8) -> List[str]:
    comp_clean = clean(comp_text or "")
    bayut_norm = " " + norm_header(bayut_text or "") + " "
    if len(comp_clean) < 160:
        return []

    toks = tokenize(comp_clean)
    freq = {}
    for n in (2,3,4):
        for i in range(0, max(len(toks)-n+1, 0)):
            chunk = toks[i:i+n]
            if chunk[0] in STOP or chunk[-1] in STOP:
                continue
            phrase = " ".join(chunk)
            if len(phrase) < 8:
                continue
            freq[phrase] = freq.get(phrase, 0) + 1

    scored = []
    for ph, c in freq.items():
        ph_norm = " " + norm_header(ph) + " "
        if ph_norm in bayut_norm:
            continue
        scored.append((c, ph))
    scored.sort(key=lambda x: (-x[0], -len(x[1])))

    out, seen = [], set()
    for _, ph in scored:
        nice = ph.title()
        k = norm_header(nice)
        if k and k not in seen:
            seen.add(k)
            out.append(nice)
        if len(out) >= limit:
            break
    return out

def update_mode_rows(
    bayut_nodes: List[dict],
    bayut_fr: FetchResult,
    comp_nodes: List[dict],
    comp_fr: FetchResult,
    comp_url: str,
) -> List[dict]:
    rows = []
    source = source_link(comp_url)

    context_terms = set()
    context_terms |= _context_terms_from_h1(get_first_h1(comp_nodes))
    context_terms |= _context_terms_from_h1(get_first_h1(bayut_nodes))

    bayut_secs = section_nodes(bayut_nodes, levels=(2,3))
    comp_secs  = section_nodes(comp_nodes, levels=(2,3))

    bayut_h2 = [s for s in bayut_secs if s["level"] == 2]
    comp_h2  = [s for s in comp_secs  if s["level"] == 2]

    def child_headers(all_secs: List[dict], parent_h2: str) -> List[str]:
        ph = norm_header(parent_h2)
        out = []
        for s in all_secs:
            if s["level"] == 3 and norm_header(s.get("parent_h2") or "") == ph:
                out.append(s["header"])
        return out

    def combined_content(all_secs: List[dict], parent_h2: str) -> str:
        ph = norm_header(parent_h2)
        parts = []
        for s in all_secs:
            if s["level"] == 2 and norm_header(s["header"]) == ph:
                if s.get("content"): parts.append(s["content"])
            if s["level"] == 3 and norm_header(s.get("parent_h2") or "") == ph:
                if s.get("content"): parts.append(s["content"])
        return clean(" ".join(parts))

    for cs in comp_h2:
        comp_header = cs["header"]
        comp_text = combined_content(comp_secs, comp_header)

        m = find_best_bayut_match(comp_header, bayut_h2, context_terms=context_terms, min_score=0.70)

        # Truly missing section
        if not m:
            points = ["Section missing in Bayut coverage."]
            ch = child_headers(comp_secs, comp_header)
            if ch:
                points.append("Add subtopics: " + format_gap_list(ch, limit=10))
            kp = missing_key_phrases(comp_text, "", limit=8)
            if kp:
                points.append("Cover details like: " + format_gap_list(kp, limit=8))
            tf = list(theme_flags(comp_text))
            if tf:
                human = {
                    "transport":"commute & connectivity",
                    "traffic_parking":"traffic/parking realities",
                    "cost":"cost considerations",
                    "lifestyle":"lifestyle & vibe",
                    "daily_life":"day-to-day convenience",
                    "safety":"safety angle",
                    "comparison":"comparison context",
                }
                points.append("Include coverage on: " + format_gap_list([human.get(x,x) for x in tf], limit=6))
            rows.append({"Headers": comp_header, "Description": format_gap_points(points), "Source": source})
            continue

        # Matched section exists → ONLY add row if REAL gaps INSIDE content
        bayut_header = m["bayut_section"]["header"]
        bayut_text = combined_content(bayut_secs, bayut_header)

        comp_children = child_headers(comp_secs, comp_header)
        bayut_children = child_headers(bayut_secs, bayut_header)

        missing_sub = []
        for ch in comp_children:
            if not any(header_similarity(ch, bh, context_terms=context_terms) >= 0.70 for bh in bayut_children):
                missing_sub.append(ch)

        points = []
        if missing_sub:
            points.append("Missing subtopics: " + format_gap_list(missing_sub, limit=10))

        # depth gap by themes
        missing_flags = list(theme_flags(comp_text) - theme_flags(bayut_text))
        if missing_flags:
            human = {
                "transport":"commute & connectivity",
                "traffic_parking":"traffic/parking realities",
                "cost":"cost considerations",
                "lifestyle":"lifestyle & vibe",
                "daily_life":"day-to-day convenience",
                "safety":"safety angle",
                "comparison":"comparison context",
            }
            points.append("Missing depth on: " + format_gap_list([human.get(x,x) for x in missing_flags], limit=6))

        # missing key phrases
        kp = missing_key_phrases(comp_text, bayut_text, limit=8)
        if kp:
            points.append("Add detail on: " + format_gap_list(kp, limit=8))

        # ✅ IMPORTANT: only append row if points exist (prevents "Pros missing" when it's just a synonym)
        if points:
            rows.append({"Headers": comp_header, "Description": format_gap_points(points), "Source": source})

    # FAQ comparison (topics only)
    faq_row = missing_faqs_row(bayut_fr, comp_fr, comp_url, context_terms=context_terms)
    if faq_row:
        rows.append(faq_row)

    # Deduplicate (Header+Source)
    out, seen = [], set()
    for r in rows:
        k = norm_header(r.get("Headers","")) + "||" + norm_header(re.sub(r"<[^>]+>", "", r.get("Source","")))
        if k in seen: 
            continue
        seen.add(k)
        out.append(r)
    return out
# ===============================
# app.py — PART 3 / 3
# SEO + Content Quality + UI (CQ is ALWAYS 2nd table; skip bad URLs; never stop)
# ===============================

def section_header_pill(title: str):
    st.markdown(f"<div class='section-pill section-pill-tight'>{title}</div>", unsafe_allow_html=True)

def render_table(df: pd.DataFrame, drop_internal_url: bool = True):
    if df is None or df.empty:
        st.info("No results to show.")
        return
    if drop_internal_url:
        drop_cols = [c for c in df.columns if c.startswith("__")]
        if drop_cols:
            df = df.drop(columns=drop_cols)
    st.markdown(df.to_html(index=False, escape=False), unsafe_allow_html=True)

# -------------------------------
# SEO HELPERS (minimal, stable)
# -------------------------------
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
    title = clean(soup.find("title").get_text(" ")) if soup.find("title") else ""
    desc = ""
    md = soup.find("meta", attrs={"name": re.compile("^description$", re.I)})
    if md and md.get("content"):
        desc = clean(md.get("content"))
    return (title or "Not available", desc or "Not available")

def word_count(text: str) -> int:
    t = clean(text or "")
    if not t:
        return 0
    return len(re.findall(r"\b\w+\b", t))

def _extract_last_modified_from_html(html: str) -> str:
    if not html:
        return ""
    soup = BeautifulSoup(html, "html.parser")
    meta_candidates = [
        ("meta", {"property": "article:modified_time"}, "content"),
        ("meta", {"property": "article:published_time"}, "content"),
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
        return clean(tm.get("datetime"))
    return ""

def get_last_modified(url: str, html: str) -> str:
    v = _extract_last_modified_from_html(html or "")
    if v:
        return v
    try:
        r = requests.head(url, headers=DEFAULT_HEADERS, allow_redirects=True, timeout=18)
        return r.headers.get("Last-Modified", "") or "Not available"
    except Exception:
        return "Not available"

def _count_source_links(html: str) -> int:
    if not html:
        return 0
    soup = BeautifulSoup(html, "html.parser")
    return sum(1 for a in soup.find_all("a", href=True) if (a.get("href") or "").startswith("http"))

def _credible_sources_count(html: str, page_url: str) -> int:
    if not html:
        return 0
    base_dom = (urlparse(page_url).netloc or "").lower().replace("www.","")
    soup = BeautifulSoup(html, "html.parser")
    seen = set()
    for a in soup.find_all("a", href=True):
        href = a.get("href") or ""
        if not href.startswith("http"):
            continue
        dom = (urlparse(href).netloc or "").lower().replace("www.","")
        if not dom or dom == base_dom:
            continue
        if dom not in seen and (dom.endswith(".gov") or dom.endswith(".edu") or dom.endswith(".org")):
            seen.add(dom)
    return len(seen)

def _data_points_count(text: str) -> int:
    if not text:
        return 0
    matches = re.findall(r"\b\d{1,3}(?:[,\d]{0,})(?:\.\d+)?\b|\b\d+%|\b\d+\.\d+%", text)
    return len(matches)

def _data_backed_claims_count(text: str) -> int:
    if not text:
        return 0
    patterns = [r"according to", r"data from", r"study", r"survey", r"research", r"reported that", r"statistics"]
    return sum(len(re.findall(p, text, flags=re.I)) for p in patterns)

def _has_brief_summary(nodes: List[dict], text: str) -> str:
    blob = " ".join([clean(x.get("header","")).lower() for x in flatten(nodes)])
    t = (text or "").lower()
    cues = ["tl;dr", "tldr", "key takeaways", "in summary", "summary", "quick summary", "at a glance"]
    return "Yes" if any(c in blob for c in cues) or any(c in t[:1400] for c in cues) else "No"

def _references_section_present(nodes: List[dict], html: str) -> str:
    blob = " ".join([clean(x.get("header","")).lower() for x in flatten(nodes)])
    if any(k in blob for k in ["references", "sources", "further reading", "bibliography"]):
        return "Yes"
    if html:
        soup = BeautifulSoup(html, "html.parser")
        if "references" in soup.get_text(" ").lower() or "sources" in soup.get_text(" ").lower():
            return "Yes"
    return "No"

def _latest_year_mentioned(text: str) -> int:
    years = re.findall(r"\b(19\d{2}|20\d{2})\b", (text or ""))
    ys = []
    for y in years:
        try: ys.append(int(y))
        except Exception: pass
    return max(ys) if ys else 0

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
    if y and y <= 2023:
        return "Possibly outdated"
    return "Unclear"

def _styling_layout_label(html: str, nodes: List[dict], text: str) -> str:
    wc = word_count(text)
    h2_count = sum(1 for x in flatten(nodes) if x.get("level") == 2)
    score = 0
    if wc >= 1200: score += 1
    if h2_count >= 6: score += 1
    if "table of contents" in (text or "").lower(): score += 1
    if html and BeautifulSoup(html, "html.parser").find("table"): score += 1
    return "Strong" if score >= 3 else ("OK" if score >= 1 else "Weak")

# -------------------------------
# TABLE BUILDERS
# -------------------------------
def build_seo_table(urls: List[str], fr_map: Dict[str, FetchResult], tree_map: Dict[str, List[dict]]) -> pd.DataFrame:
    rows = []
    for u in urls:
        fr = fr_map.get(u)
        nodes = tree_map.get(u, [])
        if not fr or not fr.ok:
            rows.append({
                "Page": site_name(u),
                "SEO Title": "Skipped (unfetchable)",
                "Meta Description": "Skipped (unfetchable)",
                "URL Slug": url_slug(u),
            })
            continue
        title, desc = extract_head_seo(fr.html or "")
        rows.append({
            "Page": site_name(u),
            "SEO Title": title,
            "Meta Description": desc,
            "URL Slug": url_slug(u),
        })
    return pd.DataFrame(rows, columns=["Page","SEO Title","Meta Description","URL Slug"])

CONTENT_QUALITY_COLUMNS = [
    "Page",
    "Word Count",
    "Last Updated / Modified",
    "Brief Summary Present",
    "FAQs Present",
    "References Section Present",
    "Source Links Count",
    "Credible Sources Count",
    "Data Points Count (numbers/stats)",
    "Data-Backed Claims",
    "Latest Information Score",
    "Outdated / Misleading Info",
    "Styling / Layout",
]

def build_content_quality_table(urls: List[str], fr_map: Dict[str, FetchResult], tree_map: Dict[str, List[dict]]) -> pd.DataFrame:
    rows = []
    for u in urls:
        fr = fr_map.get(u)
        nodes = tree_map.get(u, [])
        page = site_name(u)

        if not fr or not fr.ok:
            rows.append({c: "Skipped" for c in CONTENT_QUALITY_COLUMNS})
            rows[-1]["Page"] = page
            continue

        html = fr.html or ""
        text = fr.text or ""
        wc = word_count(text)
        lm = get_last_modified(u, html)
        faqs_present = "Yes" if page_has_real_faq(fr) else "No"

        rows.append({
            "Page": page,
            "Word Count": str(wc) if wc else "Not available",
            "Last Updated / Modified": lm,
            "Brief Summary Present": _has_brief_summary(nodes, text),
            "FAQs Present": faqs_present,
            "References Section Present": _references_section_present(nodes, html),
            "Source Links Count": str(_count_source_links(html)),
            "Credible Sources Count": str(_credible_sources_count(html, u)),
            "Data Points Count (numbers/stats)": str(_data_points_count(text)),
            "Data-Backed Claims": str(_data_backed_claims_count(text)),
            "Latest Information Score": _latest_information_label(lm, text),
            "Outdated / Misleading Info": _outdated_label(lm, text),
            "Styling / Layout": _styling_layout_label(html, nodes, text),
        })

    return pd.DataFrame(rows, columns=CONTENT_QUALITY_COLUMNS)

# -------------------------------
# UI
# -------------------------------
section_header_pill("Update Mode")

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
    elif not competitors:
        st.error("Add at least one competitor URL.")
    else:
        all_urls = [bayut_url.strip()] + competitors

        with st.spinner("Fetching URLs (best effort — never stops)…"):
            fr_map, skipped_fetch = resolve_all_best_effort(agent, all_urls)
            tree_map, skipped_tree = trees_best_effort(fr_map, all_urls)

        # warnings (never stop)
        if skipped_fetch:
            st.warning("Skipped (unfetchable): " + ", ".join(skipped_fetch[:6]) + (" ..." if len(skipped_fetch) > 6 else ""))
        if skipped_tree:
            st.warning("Skipped (no headings): " + ", ".join(skipped_tree[:6]) + (" ..." if len(skipped_tree) > 6 else ""))

        bayut_fr = fr_map.get(bayut_url.strip())
        bayut_nodes = tree_map.get(bayut_url.strip(), [])

        # -------- GAP TABLE (skip any bad competitor, continue)
        gap_rows = []
        for cu in competitors:
            cfr = fr_map.get(cu)
            cnodes = tree_map.get(cu, [])
            if not cfr or not cfr.ok or not cnodes:
                continue
            gap_rows.extend(update_mode_rows(
                bayut_nodes=bayut_nodes,
                bayut_fr=bayut_fr,
                comp_nodes=cnodes,
                comp_fr=cfr,
                comp_url=cu,
            ))

        gaps_df = pd.DataFrame(gap_rows, columns=["Headers","Description","Source"]) if gap_rows else pd.DataFrame(columns=["Headers","Description","Source"])

        # -------- SEO + CONTENT QUALITY TABLES
        seo_df = build_seo_table(all_urls, fr_map, tree_map)
        cq_df  = build_content_quality_table(all_urls, fr_map, tree_map)

        # ===============================
        # TABLE 1 — GAPS
        # ===============================
        section_header_pill("Gaps Table")
        render_table(gaps_df, drop_internal_url=True)

        # ===============================
        # TABLE 2 — CONTENT QUALITY (ALWAYS SECOND)
        # ===============================
        section_header_pill("Content Quality (Table 2)")
        render_table(cq_df, drop_internal_url=True)

        # ===============================
        # TABLE 3 — SEO ANALYSIS
        # ===============================
        section_header_pill("SEO Analysis")
        render_table(seo_df, drop_internal_url=True)
