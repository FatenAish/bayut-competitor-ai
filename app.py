import streamlit as st
import requests
import re
import pandas as pd
from bs4 import BeautifulSoup
from urllib.parse import quote_plus, urlparse
import hashlib

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="Bayut Competitor Gap Analysis", layout="wide")

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
    r"\blooking to rent\b",
    r"\blooking to buy\b",
    r"\bexplore all available\b",
    r"\bview all\b",
    r"\bfind (a|an) (home|property|apartment|villa)\b",
    r"\bbrowse\b",
    r"\bsearch\b",
    r"\bproperties for (rent|sale)\b",
    r"\bavailable (rental|properties)\b",
    r"\bget in touch\b",
    r"\bcontact (us|agent)\b",
    r"\bcall (us|now)\b",
    r"\bwhatsapp\b",
    r"\benquire\b",
    r"\binquire\b",
    r"\bbook a viewing\b",
    r"\bshare\b",
    r"\bfollow us\b",
    r"\bsubscribe\b",
    r"\bnewsletter\b",
    r"\bsign up\b",
    r"\blogin\b",
    r"\bregister\b",
    r"\brelated (posts|articles)\b",
    r"\byou may also like\b",
    r"\brecommended\b",
    r"\bpopular posts\b",
    r"\bmore articles\b",
    r"\blatest (blogs|blog|podcasts|podcast|insights)\b",
    r"\breal estate insights\b",
    r"\btable of contents\b",
    r"\bcontents\b",
    r"\bback to top\b",
    r"\bread more\b",
    r"\bnext\b",
    r"\bprevious\b",
    r"\bcomments\b",
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
    headings = root.find_all(["h2", "h3", "h4"])
    if not headings:
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
        return {"ok": False, "source": None, "nodes": []}

    if fetched["html"]:
        return {"ok": True, "source": fetched["source"], "nodes": build_tree_from_html(fetched["html"])}

    return {"ok": True, "source": fetched["source"], "nodes": build_tree_from_reader_text(fetched["text"])}

# =========================
# STRICT SUBPOINT RULES
# =========================
def is_subpoint_heading_text_only(h: str) -> bool:
    """
    STRICT: only treat as subpoint if it ends with ':'.
    We do NOT treat normal H3 like 'Pros' as subpoints.
    """
    s = clean(h)
    if not s or is_noise_header(s):
        return False
    return s.endswith(":")

# =========================
# FLATTEN SECTIONS (H2/H3 ONLY) + STRICT-FOLD H4 UNDER LATEST H2/H3
# =========================
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

        # H4 => ALWAYS subpoint under current H2/H3
        if lvl >= 4:
            append_subpoint(h, txt)
        else:
            # H2/H3: treat as real section unless it ends with ':' (rare)
            if is_subpoint_heading_text_only(h):
                append_subpoint(h, txt)
            else:
                current = {
                    "header": h,
                    "norm": norm_header(h),
                    "text": txt,
                    "subpoints": []
                }
                sections.append(current)

        for c in n.get("children", []):
            walk(c)

    for n in nodes:
        walk(n)

    # clean & dedupe subpoints
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

# =========================
# MATCHING (FIX PROS/CONS RELIABLY)
# =========================
BUCKETS = {
    "overview": ["overview", "introduction", "about", "community overview", "neighborhood overview", "location overview"],
    "pros_cons": [
        "pros", "cons", "pros and cons", "pros & cons", "pros cons",
        "advantages", "disadvantages", "benefits", "drawbacks"
    ],
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

    # exact normalized match
    for b in bayut_secs:
        if b["norm"] == c_norm:
            return {"matched": True, "bayut_section": b, "bucket": c_bucket, "score": 1.0}

    # bucket match (this is the important fix for Pros/Cons)
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

    # fuzzy
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

# =========================
# GAP EXTRACTION (short and focused)
# =========================
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

# =========================
# FAQs: compare missing questions
# =========================
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

# =========================
# HUMAN TONE (less repetitive)
# =========================
TEMPLATES_GAP = [
    "Bayut covers the topic, but it misses {missing}. Adding this would make the section more useful.",
    "Competitor goes deeper here — Bayut should add {missing} inside the existing section (no new header needed).",
    "The section exists on Bayut; what’s missing is {missing}. A short addition would close the gap.",
    "Bayut mentions it, but competitor includes {missing}. Worth adding these points to strengthen the section.",
]

TEMPLATES_MISSING = [
    "This section isn’t covered on Bayut. Competitor focuses on {summary}. Adding it would fill a clear reader need.",
    "Competitor covers {summary}. Since Bayut doesn’t address it yet, this is a worthwhile section to add.",
    "Bayut doesn’t have this section. Competitor explains {summary}, which would help round out the article.",
]

def pick_template(templates: list[str], key: str) -> str:
    h = hashlib.md5(key.encode("utf-8")).hexdigest()
    idx = int(h[:6], 16) % len(templates)
    return templates[idx]

def summarize_section(text: str) -> str:
    text = clean(text)
    if not text:
        return "the practical points readers usually look for"
    sents = re.split(r"(?<=[.!?])\s+", text)
    sents = [s.strip() for s in sents if len(s.strip()) > 45]
    s = sents[0] if sents else (text[:170] + ("..." if len(text) > 170 else ""))
    if len(s) > 210:
        s = s[:210].rstrip() + "..."
    return s

def format_missing_points(points: list[str]) -> str:
    if not points:
        return "a few practical angles the competitor includes"
    short = []
    for p in points[:4]:
        if len(p) > 140:
            p = p[:140].rstrip() + "..."
        short.append(p)
    return " / ".join(short)

# =========================
# SOURCE LINK (website name hyperlink)
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
# BUILD ROWS (ALL COMPETITORS -> ONE TABLE)
# =========================
def build_rows_for_competitor(bayut_nodes, comp_nodes, comp_url):
    bayut_secs = flatten_sections(bayut_nodes)
    comp_secs = flatten_sections(comp_nodes)

    rows = []

    # FAQs row logic
    bayut_faq = find_faq_node(bayut_nodes)
    comp_faq = find_faq_node(comp_nodes)

    if comp_faq:
        bayut_qs = extract_faq_questions(bayut_faq) if bayut_faq else []
        comp_qs = extract_faq_questions(comp_faq)
        bayut_norm_q = {normalize_q(q) for q in bayut_qs}

        missing_qs = [q for q in comp_qs if normalize_q(q) not in bayut_norm_q]

        if not bayut_faq:
            rows.append({
                "Header (Gap)": "FAQs",
                "What to add (human brief)": "Competitor has an FAQ block. Bayut doesn’t — add a short FAQs section answering common reader questions.",
                "Source": site_link(comp_url)
            })
        elif missing_qs:
            qs = "; ".join(missing_qs[:8])
            if len(missing_qs) > 8:
                qs += f" (+{len(missing_qs) - 8} more)"
            rows.append({
                "Header (Gap)": "FAQs (Content Gap)",
                "What to add (human brief)": f"Bayut has FAQs, but it’s missing these questions the competitor answers: {qs}.",
                "Source": site_link(comp_url)
            })

    # section matching
    for c in comp_secs:
        if bucket_for_header(c["header"]) == "faqs" or FAQ_HEAD_RE.search(c["header"]):
            continue

        m = best_match(c, bayut_secs)
        comp_bucket = bucket_for_header(c["header"])

        # IMPORTANT: if competitor header is "Pros" or "Cons",
        # and Bayut has a broader pros/cons section, treat it as a CONTENT GAP.
        force_bucket = comp_bucket

        if m["matched"] and m["bayut_section"]:
            b = m["bayut_section"]
            bucket = m["bucket"] or force_bucket or bucket_for_header(b["header"])
            display = BUCKET_TITLES.get(bucket, b["header"])
            display_header = f"{display} (Content Gap)"

            missing_points = top_missing_points(b.get("text", ""), c.get("text", ""), limit=4)
            if not missing_points:
                continue

            missing_str = format_missing_points(missing_points)
            tmpl = pick_template(TEMPLATES_GAP, key=f"{comp_url}|{display_header}|{c['header']}")
            brief = tmpl.format(missing=missing_str)

            # H4 subpoints only (strict)
            if c.get("subpoints"):
                brief += f" It also mentions: {', '.join(c['subpoints'][:6])}."
                if len(c["subpoints"]) > 6:
                    brief += f" (+{len(c['subpoints']) - 6} more)"

            rows.append({
                "Header (Gap)": display_header,
                "What to add (human brief)": brief,
                "Source": site_link(comp_url)
            })
        else:
            summary = summarize_section(c.get("text", ""))
            tmpl = pick_template(TEMPLATES_MISSING, key=f"{comp_url}|missing|{c['header']}")
            brief = tmpl.format(summary=summary)

            if c.get("subpoints"):
                brief += f" It also mentions: {', '.join(c['subpoints'][:6])}."
                if len(c["subpoints"]) > 6:
                    brief += f" (+{len(c['subpoints']) - 6} more)"

            rows.append({
                "Header (Gap)": c["header"],
                "What to add (human brief)": brief,
                "Source": site_link(comp_url)
            })

    # dedupe per competitor
    seen = set()
    out = []
    for r in rows:
        k = (r["Header (Gap)"], r["Source"], r["What to add (human brief)"][:80])
        if k not in seen:
            seen.add(k)
            out.append(r)
    return out

# =========================
# UI
# =========================
st.title("Bayut Competitor Gap Analysis")
st.caption("Rules enforced: Pros/Cons are bucket-matched (no fake 'missing'). H4 is always folded under its H3. All competitors appear in one combined table.")

bayut_url = st.text_input("Bayut article URL", placeholder="https://www.bayut.com/mybayut/...")
competitors_text = st.text_area(
    "Competitor URLs (one per line)",
    placeholder="https://example.com/article\nhttps://example.com/another"
)
competitors = [c.strip() for c in competitors_text.splitlines() if c.strip()]

if st.button("Run analysis"):
    if not bayut_url.strip():
        st.error("Bayut URL is required.")
        st.stop()
    if not competitors:
        st.error("At least one competitor URL is required.")
        st.stop()

    with st.spinner("Fetching Bayut..."):
        bayut_data = get_tree(bayut_url.strip())

    if not bayut_data["ok"] or not bayut_data["nodes"]:
        st.error("Could not extract headings from Bayut page (blocked or no headings found).")
        st.stop()

    fetch_report = []
    all_rows = []

    for comp_url in competitors:
        with st.spinner(f"Fetching competitor: {comp_url}"):
            comp_data = get_tree(comp_url)

        if not comp_data["ok"] or not comp_data["nodes"]:
            fetch_report.append((comp_url, "blocked/no headings"))
            continue

        fetch_report.append((comp_url, f"ok ({comp_data['source']})"))
        all_rows.extend(build_rows_for_competitor(
            bayut_nodes=bayut_data["nodes"],
            comp_nodes=comp_data["nodes"],
            comp_url=comp_url
        ))

    st.subheader("Fetch status")
    for u, status in fetch_report:
        if status.startswith("ok"):
            st.success(f"✔ {u} — {status}")
        else:
            st.warning(f"⚠ {u} — {status}")

    st.subheader("Combined gaps table (all competitors)")
    if not all_rows:
        st.info("No meaningful gaps detected (or competitors blocked / not extractable).")
        st.stop()

    df = pd.DataFrame(all_rows)

    # HTML table for clickable source links
    st.markdown(
        """
        <style>
          table { width: 100%; border-collapse: collapse; }
          th, td { border: 1px solid #eee; padding: 10px; vertical-align: top; }
          th { background: #fafafa; text-align: left; }
          td { font-size: 14px; }
          a { text-decoration: none; }
          a:hover { text-decoration: underline; }
        </style>
        """,
        unsafe_allow_html=True
    )
    st.markdown(df.to_html(escape=False, index=False), unsafe_allow_html=True)
