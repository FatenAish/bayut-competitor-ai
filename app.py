import streamlit as st
import requests
import re
import pandas as pd
from bs4 import BeautifulSoup
from urllib.parse import quote_plus, urlparse

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="Bayut Competitor Gap Analysis", layout="wide")

# =========================
# GLOBAL UI STYLE
# =========================
BAYUT_GREEN = "#0E8A6D"
LIGHT_GREEN = "#E9F7F2"
LIGHT_GREEN_2 = "#DFF2EA"
TEXT_DARK = "#1f2937"
MUTED = "#6b7280"

st.markdown(
    f"""
    <style>
      section.main > div.block-container {{
        max-width: 1100px;
        padding-top: 2.2rem;
        padding-bottom: 2.2rem;
      }}

      .hero-title {{
        text-align: center;
        font-size: 52px;
        font-weight: 800;
        letter-spacing: -0.02em;
        margin-bottom: 6px;
        color: {TEXT_DARK};
      }}
      .hero-title .bayut {{ color: {BAYUT_GREEN}; }}

      .hero-sub {{
        text-align: center;
        font-size: 16px;
        color: {MUTED};
        margin-bottom: 18px;
      }}

      /* Mode buttons row */
      .mode-wrap {{
        display: flex;
        justify-content: center;
        gap: 14px;
        margin: 16px 0 22px 0;
      }}
      .mode-pill {{
        border: 1px solid {LIGHT_GREEN_2};
        background: white;
        border-radius: 14px;
        padding: 10px 18px;
        font-weight: 800;
        color: {TEXT_DARK};
        cursor: pointer;
        user-select: none;
        box-shadow: 0 1px 2px rgba(0,0,0,0.04);
      }}
      .mode-pill.active {{
        background: {LIGHT_GREEN_2};
        border-color: {BAYUT_GREEN};
      }}
      .mode-pill small {{
        display:block;
        font-weight: 600;
        color: {MUTED};
        margin-top: 2px;
      }}

      /* Bold labels */
      div[data-testid="stWidgetLabel"] > label,
      label {{
        font-weight: 800 !important;
        color: {TEXT_DARK} !important;
      }}

      /* Inputs in light green */
      div[data-testid="stTextInput"] input {{
        background: {LIGHT_GREEN} !important;
        border: 1px solid {LIGHT_GREEN_2} !important;
        border-radius: 12px !important;
        padding: 14px 14px !important;
      }}
      div[data-testid="stTextArea"] textarea {{
        background: {LIGHT_GREEN} !important;
        border: 1px solid {LIGHT_GREEN_2} !important;
        border-radius: 12px !important;
        padding: 14px 14px !important;
      }}

      /* Main button */
      div.stButton > button {{
        background: {BAYUT_GREEN};
        color: white;
        border: 0;
        border-radius: 12px;
        padding: 10px 18px;
        font-weight: 800;
      }}
      div.stButton > button:hover {{ filter: brightness(0.95); }}

      /* Table styling */
      .table-wrap {{ margin-top: 10px; }}

      table {{
        width: 100%;
        border-collapse: collapse;
        border: 1px solid #eef2f7;
        border-radius: 14px;
        overflow: hidden;
      }}

      th {{
        background: {LIGHT_GREEN_2};
        color: {TEXT_DARK};
        text-align: center !important;
        font-weight: 900;
        padding: 12px 10px;
        border-bottom: 1px solid #e5e7eb;
      }}

      td {{
        padding: 12px 10px;
        border-bottom: 1px solid #f1f5f9;
        vertical-align: top;
        font-size: 14px;
        color: {TEXT_DARK};
      }}

      tr:hover td {{ background: #fbfffd; }}

      a {{
        color: {BAYUT_GREEN};
        font-weight: 800;
        text-decoration: none;
      }}
      a:hover {{ text-decoration: underline; }}

      /* Hide Streamlit caption container if any */
      .stCaptionContainer {{ display:none; }}
    </style>
    """,
    unsafe_allow_html=True
)

# =========================
# FETCH (ROBUST FALLBACKS)
# =========================
DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

IGNORE = {"nav", "footer", "header", "aside", "script", "style", "noscript"}

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
        "cloudflare", "access denied", "captcha"
    ])

def fetch_best_effort(url: str) -> dict:
    url = (url or "").strip()
    if not url:
        return {"ok": False, "source": None, "html": "", "text": ""}

    # direct HTML
    try:
        code, html = fetch_direct(url)
        if code == 200 and html:
            soup = BeautifulSoup(html, "lxml")
            for t in soup.find_all(list(IGNORE)):
                t.decompose()
            article = soup.find("article") or soup
            text = clean(article.get_text(" "))
            if len(text) > 400 and not looks_blocked(text):
                return {"ok": True, "source": "direct", "html": html, "text": text}
    except Exception:
        pass

    # jina reader
    try:
        code, txt = fetch_jina(url)
        if code == 200 and txt:
            text = clean(txt)
            if len(text) > 400 and not looks_blocked(text):
                return {"ok": True, "source": "jina", "html": "", "text": text}
    except Exception:
        pass

    # textise
    try:
        encoded = quote_plus(url)
        t_url = f"https://textise.org/showtext.aspx?strURL={encoded}"
        code, html = fetch_direct(t_url)
        if code == 200 and html:
            text = clean(BeautifulSoup(html, "lxml").get_text(" "))
            if len(text) > 300 and not looks_blocked(text):
                return {"ok": True, "source": "textise", "html": "", "text": text}
    except Exception:
        pass

    return {"ok": False, "source": None, "html": "", "text": ""}

# =========================
# HEADING TREE + FILTERS
# =========================
def norm_header(h: str) -> str:
    h = clean(h).lower()
    h = re.sub(r"[^a-z0-9\s]", "", h)
    h = re.sub(r"\s+", " ", h).strip()
    return h

NOISE_PATTERNS = [
    r"\blooking to rent\b", r"\blooking to buy\b", r"\bexplore all available\b",
    r"\bview all\b", r"\bfind (a|an) (home|property|apartment|villa)\b",
    r"\bbrowse\b", r"\bsearch\b", r"\bproperties for (rent|sale)\b",
    r"\bavailable (rental|properties)\b", r"\bget in touch\b", r"\bcontact (us|agent)\b",
    r"\bcall (us|now)\b", r"\bwhatsapp\b", r"\benquire\b", r"\binquire\b", r"\bbook a viewing\b",

    r"\bshare\b", r"\bfollow us\b", r"\bsubscribe\b", r"\bnewsletter\b",
    r"\bsign up\b", r"\blogin\b", r"\bregister\b",

    r"\brelated (posts|articles)\b", r"\byou may also like\b", r"\brecommended\b",
    r"\bpopular posts\b", r"\bmore articles\b", r"\blatest (blogs|blog|podcasts|podcast|insights)\b",
    r"\breal estate insights\b",

    r"\btable of contents\b", r"\bcontents\b", r"\bback to top\b",
    r"\bread more\b", r"\bnext\b", r"\bprevious\b", r"\bcomments\b",
]

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
    soup = BeautifulSoup(html, "lxml")
    for t in soup.find_all(list(IGNORE)):
        t.decompose()

    root = soup.find("article") or soup

    headings = root.find_all(["h1", "h2", "h3", "h4"])
    nodes = []
    stack = []

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
            if sib.name in ["h1", "h2", "h3", "h4"]:
                break
            if sib.name in ["p", "li"]:
                txt = clean(sib.get_text(" "))
                if txt:
                    content_parts.append(txt)

        node["content"] = clean(" ".join(content_parts))

    return nodes

def build_tree_from_reader_text(text: str) -> list[dict]:
    # r.jina.ai often returns markdown-ish text with ### headers
    lines = [l.rstrip() for l in (text or "").splitlines()]
    nodes = []
    stack = []

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
        return {"ok": False, "source": None, "nodes": [], "page_title": ""}

    page_title = ""
    if fetched["html"]:
        try:
            soup = BeautifulSoup(fetched["html"], "lxml")
            if soup.title and soup.title.get_text():
                page_title = clean(soup.title.get_text())
        except Exception:
            page_title = ""

        return {
            "ok": True,
            "source": fetched["source"],
            "nodes": build_tree_from_html(fetched["html"]),
            "page_title": page_title
        }

    return {
        "ok": True,
        "source": fetched["source"],
        "nodes": build_tree_from_reader_text(fetched["text"]),
        "page_title": page_title
    }

# =========================
# UTIL: SOURCE NAME LINK
# =========================
def site_link(url: str) -> str:
    try:
        netloc = urlparse(url).netloc.lower().replace("www.", "")
        base = netloc.split(":")[0]
        name = base
        for suf in [".ae", ".com", ".net", ".org"]:
            name = name.replace(suf, "")
        name = name.replace("-", " ").strip()
        name = " ".join(w.capitalize() for w in name.split())
        if not name:
            name = base
    except Exception:
        name = "Source"
    return f'<a href="{url}" target="_blank">{name}</a>'

# =========================
# UPDATE MODE LOGIC (same as your current system)
# =========================
def is_subpoint_heading_text_only(h: str) -> bool:
    s = clean(h)
    if not s or is_noise_header(s):
        return False
    return s.endswith(":")

def flatten_sections(nodes: list[dict]) -> list[dict]:
    sections = []
    current = None

    def append_subpoint(h: str, txt: str):
        nonlocal current
        if not current:
            return
        label = clean(h).rstrip(":")
        if label:
            if label not in current["subpoints"]:
                current["subpoints"].append(label)
        if txt:
            current["text"] = clean((current["text"] + " " + txt).strip())

    def walk(n: dict):
        nonlocal current
        lvl = n.get("level", 9)
        h = clean(n.get("header", ""))
        txt = clean(n.get("content", ""))

        if lvl >= 4:
            append_subpoint(h, txt)
        else:
            if is_subpoint_heading_text_only(h):
                append_subpoint(h, txt)
            else:
                current = {"header": h, "norm": norm_header(h), "text": txt, "subpoints": []}
                sections.append(current)

        for c in n.get("children", []):
            walk(c)

    for n in nodes:
        walk(n)

    for s in sections:
        s["text"] = clean(s["text"])
        seen = set()
        sp = []
        for x in s["subpoints"]:
            nx = norm_header(x)
            if nx and nx not in seen:
                seen.add(nx)
                sp.append(x)
        s["subpoints"] = sp

    return sections

BUCKETS = {
    "overview": ["overview", "introduction", "about", "community overview", "neighborhood overview", "location overview"],
    "pros_cons": ["pros", "cons", "pros and cons", "pros & cons", "advantages", "disadvantages", "benefits", "drawbacks"],
    "transport": ["transport", "getting around", "metro", "connectivity", "commute", "public transport", "roads", "parking", "traffic"],
    "cost": ["cost", "prices", "rent", "rental", "sale prices", "affordability", "living expenses", "cost of living", "price range"],
    "lifestyle": ["lifestyle", "things to do", "attractions", "restaurants", "cafes", "nightlife", "shopping", "entertainment", "leisure", "amenities"],
    "family": ["family", "schools", "nurseries", "education", "kids", "family friendly"],
    "safety": ["safety", "safe", "security", "crime"],
    "nearby": ["nearby", "close to", "near", "around", "alternatives", "other areas", "similar areas", "compare", "comparison"],
    "faqs": ["faq", "faqs", "frequently asked questions"]
}

BUCKET_TITLES = {
    "overview": "Overview",
    "pros_cons": "Pros & Cons",
    "transport": "Transport & Connectivity",
    "cost": "Cost of Living & Prices",
    "lifestyle": "Lifestyle & Amenities",
    "family": "Family & Schools",
    "safety": "Safety",
    "nearby": "Nearby Areas & Alternatives",
    "faqs": "FAQs"
}

def bucket_for_header(header: str) -> str | None:
    h = norm_header(header)
    for b, keys in BUCKETS.items():
        for k in keys:
            if k in h:
                return b
    return None

STOPWORDS = {
    "the","and","for","with","that","this","from","you","your","are","was","were","will","have","has","had",
    "but","not","can","may","more","most","into","than","then","they","them","their","our","out","about",
    "also","over","under","between","within","near","where","when","what","why","how","who","which",
    "a","an","to","of","in","on","at","as","is","it","be","or","by","we","i","us",
    "dubai","business","bay"
}

def tokens(s: str) -> set:
    ws = re.findall(r"[a-z0-9]{3,}", norm_header(s))
    return {w for w in ws if w not in STOPWORDS}

def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 0.0
    return len(a & b) / max(1, len(a | b))

def best_match(comp_sec: dict, bayut_secs: list[dict]) -> dict:
    c_norm = comp_sec["norm"]
    c_bucket = bucket_for_header(comp_sec["header"])

    for b in bayut_secs:
        if b["norm"] == c_norm:
            return {"matched": True, "bayut_section": b, "bucket": c_bucket, "score": 1.0}

    if c_bucket:
        candidates = [b for b in bayut_secs if bucket_for_header(b["header"]) == c_bucket]
        if candidates:
            c_t = tokens(comp_sec["header"])
            best = None
            best_s = 0.0
            for b in candidates:
                s = jaccard(c_t, tokens(b["header"]))
                if s > best_s:
                    best_s = s
                    best = b
            return {"matched": True, "bayut_section": best or candidates[0], "bucket": c_bucket, "score": max(0.55, best_s)}

    c_t = tokens(comp_sec["header"])
    best = None
    best_s = 0.0
    for b in bayut_secs:
        s = jaccard(c_t, tokens(b["header"]))
        if s > best_s:
            best_s = s
            best = b
    if best and best_s >= 0.40:
        return {"matched": True, "bayut_section": best, "bucket": None, "score": best_s}

    return {"matched": False, "bayut_section": None, "bucket": c_bucket, "score": best_s}

def split_points(text: str) -> list[str]:
    text = clean(text)
    if not text:
        return []
    candidates = []
    raw = re.split(r"[•\n]|(?<=;)\s+", text)
    for r in raw:
        r = clean(r)
        if 50 <= len(r) <= 230:
            candidates.append(r)
    sents = re.split(r"(?<=[.!?])\s+", text)
    for s in sents:
        s = clean(s)
        if 55 <= len(s) <= 230:
            candidates.append(s)

    seen = set()
    out = []
    for c in candidates:
        fp = re.sub(r"[^a-z0-9]+", "", c.lower())[:160]
        if fp and fp not in seen:
            seen.add(fp)
            out.append(c)
    return out[:25]

def approx_in(text: str, point: str) -> bool:
    t = norm_header(text)
    p = norm_header(point)
    p_tokens = [w for w in re.findall(r"[a-z0-9]{4,}", p) if w not in STOPWORDS]
    if not p_tokens:
        return False
    hit = sum(1 for w in set(p_tokens) if w in t)
    return hit >= max(2, int(0.45 * len(set(p_tokens))))

def top_missing_points(bayut_text: str, comp_text: str, limit: int = 4) -> list[str]:
    points = split_points(comp_text)
    missing = []
    for p in points:
        if not approx_in(bayut_text, p):
            missing.append(p)
        if len(missing) >= limit:
            break
    return missing

FAQ_HEAD_RE = re.compile(r"\b(faq|faqs|frequently asked questions)\b", re.I)

def is_question(s: str) -> bool:
    s = clean(s)
    if not s:
        return False
    if s.endswith("?"):
        return True
    return bool(re.match(r"^(what|why|how|where|when|who|which|is|are|can|do|does|did|should|will)\b", s.lower()))

def normalize_q(s: str) -> str:
    s = clean(s).lower()
    s = re.sub(r"^\d+\s*[\.\)]\s*", "", s)
    s = re.sub(r"\s+", " ", s)
    s = s.strip(" .!?")
    return s

def prettify_question(q: str) -> str:
    q = clean(q)
    q = re.sub(r"^\d+\s*[\.\)]\s*", "", q).strip()
    return q

def find_faq_node(nodes: list[dict]) -> dict | None:
    def walk(n: dict):
        if FAQ_HEAD_RE.search(n.get("header", "")):
            return n
        for c in n.get("children", []):
            hit = walk(c)
            if hit:
                return hit
        return None
    for n in nodes:
        hit = walk(n)
        if hit:
            return hit
    return None

def extract_faq_questions(node: dict) -> list[str]:
    if not node:
        return []
    qs = []

    def walk(n: dict):
        h = clean(n.get("header", ""))
        if h and is_question(h):
            qs.append(prettify_question(h))

        content = n.get("content", "")
        if content:
            for part in re.split(r"[•\n]", content):
                p = clean(part)
                if is_question(p) and 6 <= len(p) <= 140:
                    qs.append(prettify_question(p))

        for c in n.get("children", []):
            walk(c)

    walk(node)

    seen = set()
    out = []
    for q in qs:
        nq = normalize_q(q)
        if nq and nq not in seen:
            seen.add(nq)
            out.append(q)
    return out

def format_points(points: list[str], limit: int = 4) -> str:
    pts = []
    for p in points[:limit]:
        p = clean(p)
        if len(p) > 160:
            p = p[:160].rstrip() + "..."
        pts.append(p)
    return " / ".join(pts)

def build_gap_brief(bucket_title: str, points: list[str], subpoints: list[str]) -> str:
    core = format_points(points, limit=4)
    if core:
        brief = f"Bayut covers {bucket_title}, but it misses: {core}. Add these points inside the existing section."
    else:
        brief = f"Bayut covers {bucket_title}, but the competitor adds useful detail. Add a short paragraph to match the depth."
    if subpoints:
        brief += f" Competitor also touches on: {', '.join(subpoints[:6])}."
        if len(subpoints) > 6:
            brief += f" (+{len(subpoints)-6} more)"
    return brief

def build_missing_brief(header: str, summary: str, subpoints: list[str]) -> str:
    summary = clean(summary)
    if summary:
        if len(summary) > 220:
            summary = summary[:220].rstrip() + "..."
        brief = f"Competitor includes this section and explains: {summary} Bayut doesn’t cover it yet, so adding it would help readers."
    else:
        brief = "Competitor includes this section. Bayut doesn’t cover it yet, so adding it would help readers."
    if subpoints:
        brief += f" It also mentions: {', '.join(subpoints[:6])}."
        if len(subpoints) > 6:
            brief += f" (+{len(subpoints)-6} more)"
    return brief

def build_rows_for_competitor(bayut_nodes, comp_nodes, comp_url):
    bayut_secs = flatten_sections(bayut_nodes)
    comp_secs = flatten_sections(comp_nodes)

    rows = []

    # FAQs (one row)
    bayut_faq = find_faq_node(bayut_nodes)
    comp_faq = find_faq_node(comp_nodes)
    if comp_faq:
        bayut_qs = extract_faq_questions(bayut_faq) if bayut_faq else []
        comp_qs = extract_faq_questions(comp_faq)
        bayut_norm_q = {normalize_q(q) for q in bayut_qs}
        missing_qs = [q for q in comp_qs if normalize_q(q) not in bayut_norm_q]

        if not bayut_faq:
            rows.append({
                "Header": "FAQs",
                "What to add": "Competitor has an FAQ block. Bayut doesn’t — add a short FAQs section answering common reader questions.",
                "Source": site_link(comp_url)
            })
        elif missing_qs:
            qs = "; ".join(missing_qs[:8])
            if len(missing_qs) > 8:
                qs += f" (+{len(missing_qs)-8} more)"
            rows.append({
                "Header": "FAQs (Content Gap)",
                "What to add": f"Bayut has FAQs, but it’s missing these questions the competitor answers: {qs}.",
                "Source": site_link(comp_url)
            })

    agg_gap = {}
    agg_missing = {}

    for c in comp_secs:
        if bucket_for_header(c["header"]) == "faqs" or FAQ_HEAD_RE.search(c["header"]):
            continue

        c_bucket = bucket_for_header(c["header"])
        m = best_match(c, bayut_secs)

        if m["matched"] and m["bayut_section"]:
            b = m["bayut_section"]
            bucket = m["bucket"] or c_bucket or bucket_for_header(b["header"])
            bucket_title = BUCKET_TITLES.get(bucket, b["header"])

            missing_points = top_missing_points(b.get("text", ""), c.get("text", ""), limit=4)
            if not missing_points:
                continue

            if bucket_title not in agg_gap:
                agg_gap[bucket_title] = {"points": set(), "subpoints": set()}
            for p in missing_points:
                agg_gap[bucket_title]["points"].add(clean(p))
            for sp in (c.get("subpoints") or []):
                agg_gap[bucket_title]["subpoints"].add(clean(sp))

        else:
            if c_bucket and c_bucket in BUCKET_TITLES:
                key = BUCKET_TITLES[c_bucket]
            else:
                key = c["header"]

            if key not in agg_missing:
                agg_missing[key] = {"summaries": [], "subpoints": set()}

            txt = clean(c.get("text", ""))
            if txt:
                sents = re.split(r"(?<=[.!?])\s+", txt)
                sents = [s.strip() for s in sents if len(s.strip()) > 45]
                if sents:
                    agg_missing[key]["summaries"].append(sents[0])
                else:
                    agg_missing[key]["summaries"].append(txt[:180] + ("..." if len(txt) > 180 else ""))

            for sp in (c.get("subpoints") or []):
                agg_missing[key]["subpoints"].add(clean(sp))

    for bucket_title, data in agg_gap.items():
        pts = sorted(list(data["points"]), key=lambda x: (len(x), x))[:6]
        subps = sorted(list(data["subpoints"]))[:10]
        rows.append({
            "Header": f"{bucket_title} (Content Gap)",
            "What to add": build_gap_brief(bucket_title, pts, subps),
            "Source": site_link(comp_url)
        })

    for key, data in agg_missing.items():
        parts = data["summaries"][:2]
        summary = " ".join(parts).strip()
        subps = sorted(list(data["subpoints"]))[:10]
        rows.append({
            "Header": key,
            "What to add": build_missing_brief(key, summary, subps),
            "Source": site_link(comp_url)
        })

    seen = set()
    out = []
    for r in rows:
        k = (r["Header"], r["Source"])
        if k not in seen:
            seen.add(k)
            out.append(r)
    return out

# =========================
# NEW POST MODE LOGIC (outline from competitors)
# =========================
def first_h1(nodes: list[dict], page_title: str = "") -> str:
    # Prefer real H1; fallback to page <title>; otherwise best effort.
    def walk_find_h1(n: dict):
        if n.get("level") == 1 and n.get("header"):
            return clean(n["header"])
        for c in n.get("children", []):
            hit = walk_find_h1(c)
            if hit:
                return hit
        return ""

    for n in nodes:
        hit = walk_find_h1(n)
        if hit:
            return hit

    if page_title:
        return page_title

    # fallback: first header
    for n in nodes:
        if n.get("header"):
            return clean(n["header"])
    return ""

def brief_from_content(txt: str, max_len: int = 220) -> str:
    txt = clean(txt)
    if not txt:
        return ""
    sents = re.split(r"(?<=[.!?])\s+", txt)
    sents = [s.strip() for s in sents if len(s.strip()) > 35]
    base = sents[0] if sents else txt
    base = clean(base)
    if len(base) > max_len:
        base = base[:max_len].rstrip() + "..."
    return base

def collect_subtree_headers(node: dict) -> list[str]:
    # Returns H3/H4 under an H2, as a compact list
    out = []
    def walk(n: dict):
        lvl = n.get("level", 9)
        h = clean(n.get("header",""))
        if lvl in (3, 4) and h and not is_noise_header(h):
            out.append(h)
        for c in n.get("children", []):
            walk(c)
    for c in node.get("children", []):
        walk(c)
    # de-dup preserve order
    seen = set()
    final = []
    for x in out:
        nx = norm_header(x)
        if nx and nx not in seen:
            seen.add(nx)
            final.append(x)
    return final

def outline_rows_for_competitor(nodes: list[dict], comp_url: str, page_title: str, desired_title: str) -> list[dict]:
    rows = []
    h1 = first_h1(nodes, page_title=page_title)
    # User-provided title is the "target" for new post; show it at top for context
    target = clean(desired_title)

    # We output one row per H2 (main section). If no H2, fallback to first available node as a "Section".
    def walk_h2(n: dict):
        if n.get("level") == 2:
            h2 = clean(n.get("header",""))
            subs = collect_subtree_headers(n)
            brief = brief_from_content(n.get("content",""))
            rows.append({
                "Target title (you write)": target,
                "Competitor H1": h1 or "(not detected)",
                "H2 section": h2,
                "Subsections (H3/H4)": "; ".join(subs[:12]) if subs else "",
                "What it covers (brief)": brief or "Competitor covers this section (add a short summary when drafting).",
                "Source": site_link(comp_url)
            })
        for c in n.get("children", []):
            walk_h2(c)

    for n in nodes:
        walk_h2(n)

    if not rows:
        # fallback: show a single row with best headers we can find
        first = ""
        first_content = ""
        for n in nodes:
            if n.get("header"):
                first = clean(n["header"])
                first_content = clean(n.get("content",""))
                break
        rows.append({
            "Target title (you write)": target,
            "Competitor H1": h1 or "(not detected)",
            "H2 section": first or "(no headings detected)",
            "Subsections (H3/H4)": "",
            "What it covers (brief)": brief_from_content(first_content) or "Competitor page extracted but headings are limited.",
            "Source": site_link(comp_url)
        })

    return rows

# =========================
# HERO HEADER
# =========================
st.markdown(
    f"""
    <div class="hero-title"><span class="bayut">Bayut</span> Competitor Gap Analysis</div>
    <div class="hero-sub">Choose a tool: Update an existing article, or plan a new one using competitor coverage</div>
    """,
    unsafe_allow_html=True
)

# =========================
# MODE SWITCH (2 TOOLS)
# =========================
if "mode" not in st.session_state:
    st.session_state.mode = "update"  # update | new

c1, c2, c3 = st.columns([1, 1, 1])
with c2:
    # centered-ish using columns
    left, mid, right = st.columns([1, 2, 1])
    with mid:
        a, b = st.columns(2)
        with a:
            if st.button("Update Mode", use_container_width=True):
                st.session_state.mode = "update"
        with b:
            if st.button("New Post Mode", use_container_width=True):
                st.session_state.mode = "new"

# =========================
# INPUTS (DIFFER BY MODE)
# =========================
competitors_text = st.text_area(
    "Competitor URLs (one per line)",
    placeholder="https://example.com/article\nhttps://example.com/another"
)
competitors = [c.strip() for c in competitors_text.splitlines() if c.strip()]

bayut_url = ""
new_title = ""

if st.session_state.mode == "update":
    bayut_url = st.text_input("Bayut article URL", placeholder="https://www.bayut.com/mybayut/...")
else:
    new_title = st.text_input("New post title", placeholder="e.g., Pros and Cons of Living in Business Bay")

# =========================
# RUN
# =========================
run_label = "Run analysis" if st.session_state.mode == "update" else "Build competitor outline"
if st.button(run_label):
    if not competitors:
        st.error("At least one competitor URL is required.")
        st.stop()

    # -------------------------
    # UPDATE MODE
    # -------------------------
    if st.session_state.mode == "update":
        if not bayut_url.strip():
            st.error("Bayut URL is required in Update Mode.")
            st.stop()

        with st.spinner("Fetching Bayut..."):
            bayut_data = get_tree(bayut_url.strip())

        if not bayut_data["ok"] or not bayut_data["nodes"]:
            st.error("Could not extract headings from Bayut page (blocked or no headings found).")
            st.stop()

        all_rows = []
        for comp_url in competitors:
            with st.spinner("Fetching competitor..."):
                comp_data = get_tree(comp_url)

            if not comp_data["ok"] or not comp_data["nodes"]:
                continue

            all_rows.extend(build_rows_for_competitor(
                bayut_nodes=bayut_data["nodes"],
                comp_nodes=comp_data["nodes"],
                comp_url=comp_url
            ))

        st.subheader("Content Gaps")

        if not all_rows:
            st.info("No meaningful gaps detected (or competitors blocked / not extractable).")
            st.stop()

        df = pd.DataFrame(all_rows)[["Header", "What to add", "Source"]]
        st.markdown('<div class="table-wrap">', unsafe_allow_html=True)
        st.markdown(df.to_html(escape=False, index=False), unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # -------------------------
    # NEW POST MODE
    # -------------------------
    else:
        if not new_title.strip():
            st.error("Title is required in New Post Mode.")
            st.stop()

        all_rows = []
        for comp_url in competitors:
            with st.spinner("Fetching competitor..."):
                comp_data = get_tree(comp_url)

            if not comp_data["ok"] or not comp_data["nodes"]:
                continue

            all_rows.extend(outline_rows_for_competitor(
                nodes=comp_data["nodes"],
                comp_url=comp_url,
                page_title=comp_data.get("page_title", ""),
                desired_title=new_title.strip()
            ))

        st.subheader("Competitor Coverage Outline")

        if not all_rows:
            st.info("Could not extract competitor headings (blocked / no headings). Try different competitor URLs.")
            st.stop()

        df = pd.DataFrame(all_rows)[
            ["Target title (you write)", "Competitor H1", "H2 section", "Subsections (H3/H4)", "What it covers (brief)", "Source"]
        ]

        st.markdown('<div class="table-wrap">', unsafe_allow_html=True)
        st.markdown(df.to_html(escape=False, index=False), unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
