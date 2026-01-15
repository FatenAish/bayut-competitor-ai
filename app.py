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
      <p>Gaps only — no recommendations. No missing competitors (forced).</p>
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
    """
    HARD ENFORCEMENT:
    - tries to resolve all URLs
    - if any fail, forces manual paste for EACH failed URL
    - stops app until all are resolved
    """
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

    st.error("Some URLs could not be fetched automatically. Paste the article HTML/text for EACH failed URL to continue. (No missing competitors.)")

    for u in failed:
        with st.expander(f"Manual fallback required: {u}", expanded=True):
            pasted = st.text_area(
                "Paste the full article HTML OR the readable article text:",
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


def list_headers(nodes: List[dict], level: int) -> List[str]:
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


# ---------------------------
# FAQ detection (forced)
# ---------------------------
def header_is_faq(header: str) -> bool:
    nh = norm_header(header)
    # stricter but still flexible
    return (nh == "faq") or (nh == "faqs") or ("frequently asked" in nh) or re.search(r"\bfaq\b", nh) is not None


def find_faq_nodes(nodes: List[dict]) -> List[dict]:
    faq = []
    for x in flatten(nodes):
        if x["level"] in (2, 3) and header_is_faq(x["header"]):
            faq.append(x)
    return faq


def normalize_question(q: str) -> str:
    q = clean(q or "")
    q = re.sub(r"^\s*\d+[\.\)]\s*", "", q)
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
    return out[:30]


# =====================================================
# FAQ SUBJECTS (FORCED: never list full questions)
# =====================================================
def faq_subject(q: str) -> str:
    """
    Deterministic rules to map an FAQ question to a short subject label.
    This guarantees stable output and avoids dumping long question lists.
    """
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

    if any(k in s for k in ["brochure", "download", "pdf", "floorplan", "floor plan"]):
        return "Brochure & downloads"

    if any(k in s for k in ["expression of interest", "register", "enquire", "inquire", "contact", "how can i submit", "booking"]):
        return "How to register interest"

    return "Other FAQs"


def faq_subjects_from_questions(questions: List[str], limit: int = 10) -> List[str]:
    """
    Deduped, stable order list of subjects from questions.
    """
    out = []
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


# =====================================================
# FORCED, STABLE SUMMARIES (NO RECOMMENDATIONS, NO QUOTING)
# =====================================================
STOP = {
    "the", "and", "for", "with", "that", "this", "from", "you", "your", "are", "was", "were", "will", "have", "has", "had",
    "but", "not", "can", "may", "more", "most", "into", "than", "then", "they", "them", "their", "our", "out", "about",
    "also", "over", "under", "between", "within", "near", "where", "when", "what", "why", "how", "who", "which",
    "a", "an", "to", "of", "in", "on", "at", "as", "is", "it", "be", "or", "by", "we", "i", "us"
}
GENERIC_STOP = {
    "dubai", "damac", "islands", "island", "project", "development", "community", "properties", "property",
    "phase", "master", "plan", "plans", "floor", "layouts", "bedroom", "bedrooms"
}


def top_keywords(text: str, n: int = 7) -> List[str]:
    words = re.findall(r"[a-zA-Z]{4,}", (text or "").lower())
    freq = {}
    for w in words:
        if w in STOP or w in GENERIC_STOP:
            continue
        freq[w] = freq.get(w, 0) + 1
    return [w for w, _ in sorted(freq.items(), key=lambda x: (-x[1], x[0]))[:n]]


def collect_norm_set(nodes: List[dict], keep_levels=(2, 3)) -> set:
    out = set()
    for x in flatten(nodes):
        if x["level"] in keep_levels:
            nh = norm_header(x["header"])
            if nh:
                out.add(nh)
    return out


def index_by_norm(nodes: List[dict], levels=(2, 3)) -> Dict[str, dict]:
    idx = {}
    for x in flatten(nodes):
        if x["level"] in levels:
            nh = norm_header(x["header"])
            if nh:
                idx[nh] = x
    return idx


def summarize_h2_children(h2_node: dict) -> Tuple[List[str], List[str]]:
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
    options, themes = summarize_h2_children(h2_node)
    kws = top_keywords(h2_node.get("content", ""), n=7)

    parts = [f"The competitor includes a section titled “{h}”."]
    if themes:
        parts.append("It breaks this down into themes such as " + ", ".join(themes) + ".")
    if options:
        parts.append("It references sub-sections such as " + ", ".join(options) + ".")
    if (not themes and not options) and kws:
        parts.append("It focuses on topics like " + ", ".join(kws) + ".")
    return clean(" ".join(parts))


def describe_missing_h3(h3_header: str, h3_node: dict) -> str:
    h = strip_label(h3_header)
    kws = top_keywords(h3_node.get("content", ""), n=7)
    if kws:
        return f"The competitor includes a subsection titled “{h}” focusing on: " + ", ".join(kws) + "."
    return f"The competitor includes a subsection titled “{h}”."


# =====================================================
# FAQ ROW RULES (FORCED)
# =====================================================
def describe_faq_gap(comp_faq_nodes: List[dict], bayut_faq_nodes: List[dict]) -> Tuple[Optional[str], Optional[str]]:
    """
    FORCED RULES:
    - If Bayut has NO FAQ and competitor has FAQ => show row: "FAQ Block" (subjects only).
    - If Bayut HAS FAQ => show row: "Missing FAQs" ONLY if competitor has missing FAQ questions (subjects only).
    - Never list full questions.
    - If there is no gap => return (None, None) => no row.
    """
    comp_qs = []
    for fn in comp_faq_nodes:
        comp_qs.extend(extract_questions_from_node(fn))
    comp_qs = [q for q in comp_qs if q]

    if not comp_qs:
        return None, None

    # Bayut has no FAQ -> show competitor FAQ block (as gap: Bayut missing whole block)
    if not bayut_faq_nodes:
        subjects = faq_subjects_from_questions(comp_qs, limit=10)
        return "FAQ Block", "Bayut has no FAQ section. Competitor covers FAQ topics like: " + ", ".join(subjects) + "."

    # Bayut has FAQ -> only missing ones
    bayut_qs = []
    for fn in bayut_faq_nodes:
        bayut_qs.extend(extract_questions_from_node(fn))
    bayut_qs = [q for q in bayut_qs if q]

    def q_key(q: str) -> str:
        return norm_header(normalize_question(q)).replace(" ", "")

    bayut_set = {q_key(q) for q in bayut_qs}
    missing_qs = [q for q in comp_qs if q_key(q) not in bayut_set]

    if not missing_qs:
        return None, None  # forced: no gap row

    subjects = faq_subjects_from_questions(missing_qs, limit=10)
    return "Missing FAQs", "Competitor covers extra FAQ topics not on Bayut, such as: " + ", ".join(subjects) + "."


# =====================================================
# UPDATE MODE ROWS (FORCED: GAPS ONLY)
# =====================================================
def update_mode_rows(bayut_nodes: List[dict], comp_nodes: List[dict], comp_url: str) -> List[dict]:
    rows = []
    bayut_norm = collect_norm_set(bayut_nodes, keep_levels=(2, 3))
    bayut_idx = index_by_norm(bayut_nodes, levels=(2, 3))
    comp_idx = index_by_norm(comp_nodes, levels=(2, 3))

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

        # FORCED: do not treat FAQ H2 as a normal missing H2
        if header_is_faq(x["header"]):
            continue

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
        parent_n = norm_header(parent.get("header", "")) if parent else ""
        if parent and parent.get("level") == 2 and parent_n in missing_h2_norms:
            continue

        # FORCED: skip label-like subpoints
        if is_subpoint_heading(x["header"]):
            continue

        rows.append({
            "Header (Gap)": strip_label(x["header"]),
            "What the competitor talks about": describe_missing_h3(x["header"], x),
            "Source": source_link(comp_url),
            "_key": f"missing_h3::{nh3}::{comp_url}"
        })

    # C) FAQ row (FORCED RULES)
    if comp_faq_nodes:
        faq_header, faq_msg = describe_faq_gap(comp_faq_nodes, bayut_faq_nodes)
        if faq_header and faq_msg:
            rows.append({
                "Header (Gap)": faq_header,
                "What the competitor talks about": faq_msg,
                "Source": source_link(comp_url),
                "_key": f"faq::{comp_url}"
            })

    # D) Content depth gaps (optional, still "gap" if competitor clearly adds unique detail)
    def tokens_set(text: str) -> set:
        ws = re.findall(r"[a-zA-Z]{3,}", (text or "").lower())
        return {w for w in ws if w not in STOP}

    content_gap_rows = []
    for nh, comp_item in comp_idx.items():
        if nh not in bayut_idx:
            continue
        b = bayut_idx[nh]
        comp_text = (comp_item.get("content", "") or "")
        bayut_text = (b.get("content", "") or "")

        if len(comp_text) < 180:
            continue

        comp_tok = tokens_set(comp_text)
        bay_tok = tokens_set(bayut_text)
        new_terms = [t for t in (comp_tok - bay_tok) if len(t) > 3]

        if (len(comp_text) > max(260, len(bayut_text) * 1.35)) and len(new_terms) >= 6:
            kw = top_keywords(comp_text, n=6)
            msg = (
                "The competitor covers the same topic but adds extra detail around: " + ", ".join(kw) + "."
                if kw else
                "The competitor covers the same topic but adds more detail than the Bayut section."
            )
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

    # de-dupe stable
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
# NEW POST MODE (kept, not the focus)
# =====================================================
def detect_main_angle(comp_nodes: List[dict]) -> str:
    h2s = [norm_header(h) for h in list_headers(comp_nodes, 2)]
    blob = " ".join(h2s)
    if ("pros" in blob and "cons" in blob) or ("advantages" in blob and "disadvantages" in blob):
        return "pros-and-cons decision guide"
    if "payment plan" in blob:
        return "project purchase guide"
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

    h2_text = "Major sections include: " + " → ".join(h2_main) + "." if h2_main else "Major sections introduce the topic and break down key points."
    if has_faq:
        h2_text += " Includes a separate FAQ section."

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

    h3_text = "Subsections cover themes such as: " + ", ".join(themes) + "." if themes else "Subsections add practical depth within sections."
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
st.markdown("<div class='mode-note'>Competitors one per line. If any page blocks the server, you must paste it (forced) — so nothing is missing.</div>", unsafe_allow_html=True)

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

        # 1) HARD: Bayut must be available (or paste)
        with st.spinner("Fetching Bayut (no exceptions)…"):
            bayut_fr_map = resolve_all_or_require_manual(agent, [bayut_url.strip()], st_key_prefix="bayut")
            bayut_tree_map = ensure_headings_or_require_repaste([bayut_url.strip()], bayut_fr_map, st_key_prefix="bayut_tree")
        bayut_nodes = bayut_tree_map[bayut_url.strip()]["nodes"]

        # 2) HARD: ALL competitors must be available (or paste)
        with st.spinner("Fetching ALL competitors (no exceptions)…"):
            comp_fr_map = resolve_all_or_require_manual(agent, competitors, st_key_prefix="comp_update")
            comp_tree_map = ensure_headings_or_require_repaste(competitors, comp_fr_map, st_key_prefix="comp_update_tree")

        # 3) Build rows (no skipping)
        all_rows = []
        internal_fetch = []

        for comp_url in competitors:
            src = comp_fr_map[comp_url].source
            internal_fetch.append((comp_url, f"ok ({src})"))
            comp_nodes = comp_tree_map[comp_url]["nodes"]
            all_rows.extend(update_mode_rows(bayut_nodes, comp_nodes, comp_url))

        st.session_state.update_fetch = internal_fetch
        st.session_state.update_df = (
            pd.DataFrame(all_rows)[["Header (Gap)", "What the competitor talks about", "Source"]]
            if all_rows
            else pd.DataFrame(columns=["Header (Gap)", "What the competitor talks about", "Source"])
        )

    if show_internal_fetch and st.session_state.update_fetch:
        st.sidebar.markdown("### Internal fetch log (Update Mode)")
        st.sidebar.write(f"Playwright enabled: {PLAYWRIGHT_OK}")
        for u, s in st.session_state.update_fetch:
            st.sidebar.write(u, "—", s)

    st.markdown("<div class='section-pill'>Content Gaps</div>", unsafe_allow_html=True)

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
