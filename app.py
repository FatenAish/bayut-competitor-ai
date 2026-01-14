import streamlit as st
import requests
import re
from bs4 import BeautifulSoup
from urllib.parse import quote_plus, urlparse
import pandas as pd

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(page_title="Bayut Competitor Gap Analysis", layout="wide")

# =====================================================
# STYLE (light green inputs + centered title + light green section headers)
# =====================================================
BAYUT_GREEN = "#0E8A6D"
LIGHT_GREEN = "#E9F7F1"
LIGHT_GREEN_2 = "#DFF3EA"
TEXT_DARK = "#1F2937"

st.markdown(
    f"""
    <style>
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
        font-weight: 700;
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
        border-radius: 12px !important;
        padding: 0.6rem 1rem !important;
        font-weight: 700 !important;
      }}

      /* MODE BUTTONS - centered + pill look */
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
      }}
      thead th {{
        background: {LIGHT_GREEN} !important;
        text-align: center !important;
        font-weight: 800 !important;
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
        font-weight: 700 !important;
        text-decoration: underline !important;
      }}

      /* small helper text */
      .helper {{
        color:#6B7280;
        font-size: 13px;
        margin-top: -6px;
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
    # r.jina.ai reader
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
    # CTA / property widgets
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

    # social/share/meta
    r"\bshare\b",
    r"\bshare this\b",
    r"\bfollow us\b",
    r"\blike\b",
    r"\bsubscribe\b",
    r"\bnewsletter\b",
    r"\bsign up\b",
    r"\blogin\b",
    r"\bregister\b",

    # site sections / related content
    r"\brelated (posts|articles)\b",
    r"\byou may also like\b",
    r"\brecommended\b",
    r"\bpopular posts\b",
    r"\bmore articles\b",
    r"\blatest (blogs|blog|podcasts|podcast|insights)\b",
    r"\breal estate insights\b",

    # navigation / widgets
    r"\btable of contents\b",
    r"\bcontents\b",
    r"\bback to top\b",
    r"\bread more\b",
    r"\bnext\b",
    r"\bprevious\b",
    r"\bcomments\b",
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

        # collect nearby paragraph/list text until next heading
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
    # markdown-ish reader output
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
        return {
            "ok": True,
            "source": fetched["source"],
            "nodes": build_tree_from_html(fetched["html"]),
            "status": fetched.get("status"),
        }

    return {
        "ok": True,
        "source": fetched["source"],
        "nodes": build_tree_from_reader_text(fetched["text"]),
        "status": fetched.get("status"),
    }

# =====================================================
# HELPERS
# =====================================================
def site_name(url: str) -> str:
    try:
        host = urlparse(url).netloc.lower()
        host = host.replace("www.", "")
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
        out.append({"level": n["level"], "header": n["header"], "content": n.get("content", ""), "parent": parent})
        for c in n.get("children", []):
            walk(c, n)
    for n in nodes:
        walk(n, None)
    return out

def list_headers(nodes: list[dict], level: int) -> list[str]:
    return [x["header"] for x in flatten(nodes) if x["level"] == level and not is_noise_header(x["header"])]

def group_h4_under_h3(nodes: list[dict]) -> dict[str, list[str]]:
    mapping = {}
    def walk(n: dict):
        if n["level"] == 3:
            h4s = [c["header"] for c in n.get("children", []) if c["level"] == 4 and not is_noise_header(c["header"])]
            mapping[n["header"]] = h4s
        for c in n.get("children", []):
            walk(c)
    for n in nodes:
        walk(n)
    return mapping

# =====================================================
# UPDATE MODE (existing article) - keep current logic
# =====================================================
STOP = {
    "the","and","for","with","that","this","from","you","your","are","was","were","will","have","has","had",
    "but","not","can","may","more","most","into","than","then","they","them","their","our","out","about",
    "also","over","under","between","within","near","where","when","what","why","how","who","which",
    "a","an","to","of","in","on","at","as","is","it","be","or","by","we","i","us"
}

def tokens(text: str) -> set[str]:
    ws = re.findall(r"[a-zA-Z]{3,}", (text or "").lower())
    return {w for w in ws if w not in STOP}

def first_two_sentences(text: str, max_len: int = 240) -> str:
    text = clean(text)
    if not text:
        return ""
    sents = re.split(r"(?<=[.!?])\s+", text)
    sents = [s.strip() for s in sents if len(s.strip()) > 35]
    base = " ".join(sents[:2]) if sents else text
    base = clean(base)
    if len(base) > max_len:
        base = base[:max_len].rstrip() + "..."
    return base

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

def update_mode_rows(bayut_nodes: list[dict], comp_nodes: list[dict], comp_url: str) -> list[dict]:
    """
    Rows:
    - Missing header rows for competitor H2/H3 not in Bayut
    - Content gap rows for headers present in both where competitor has meaningful extra coverage
    H4 are never separate rows; they are merged inside the parent row.
    """
    rows = []
    bayut_norm = collect_norm_set(bayut_nodes, keep_levels=(2,3))
    bayut_idx = index_by_norm(bayut_nodes, levels=(2,3))
    comp_idx = index_by_norm(comp_nodes, levels=(2,3))

    h4_map = group_h4_under_h3(comp_nodes)

    # 1) Missing headers (H2/H3)
    for x in flatten(comp_nodes):
        if x["level"] not in (2,3):
            continue
        nh = norm_header(x["header"])
        if not nh or nh in bayut_norm:
            continue

        brief = first_two_sentences(x.get("content",""), max_len=220)

        extra = ""
        if x["level"] == 3:
            h4s = h4_map.get(x["header"], [])
            if h4s:
                extra = " Includes: " + "; ".join(h4s[:10]) + "."
        else:
            h3s = [c["header"] for c in (x.get("children", []) or []) if c["level"] == 3 and not is_noise_header(c["header"])]
            if h3s:
                extra = " Covers: " + "; ".join(h3s[:10]) + ("…" if len(h3s) > 10 else "") + "."

        if not brief and not extra:
            brief = "Competitor includes this section — adding it would close a clear reader gap."

        text = clean(
            f"Competitor includes this section. {brief} "
            f"Bayut doesn’t cover it yet, so adding it would fill a clear gap.{extra}"
        )

        rows.append({
            "Header (Gap)": x["header"],
            "What to add": text,
            "Source": source_link(comp_url),
            "_key": f"missing::{nh}::{comp_url}"
        })

    # 2) Content gaps (present header, competitor has extra)
    content_gap_rows = []
    for nh, comp_item in comp_idx.items():
        if nh not in bayut_idx:
            continue

        b = bayut_idx[nh]
        comp_text = (comp_item.get("content","") or "")
        bayut_text = (b.get("content","") or "")

        if len(comp_text) < 180:
            continue

        comp_tok = tokens(comp_text)
        bay_tok = tokens(bayut_text)
        new_terms = [t for t in (comp_tok - bay_tok) if len(t) > 3]

        if (len(comp_text) > max(260, len(bayut_text) * 1.35)) and len(new_terms) >= 6:
            example = first_two_sentences(comp_text, max_len=240)
            msg = (
                f"Bayut covers this topic, but it misses some useful depth the competitor adds. "
                f"For example, the competitor explains: {example} "
                f"Add a short paragraph (or bullets) inside the existing section to cover the missing angles."
            )

            content_gap_rows.append({
                "Header (Gap)": f"{b['header']} (Content Gap)",
                "What to add": msg,
                "Source": source_link(comp_url),
                "_key": f"content::{nh}::{comp_url}"
            })

    # dedupe content gaps per (competitor, header)
    seen = set()
    for r in content_gap_rows:
        if r["_key"] in seen:
            continue
        seen.add(r["_key"])
        rows.append(r)

    # final: strip internal keys
    for r in rows:
        r.pop("_key", None)

    return rows

# =====================================================
# NEW POST MODE (competitor coverage) - SHORT OUTPUT
# =====================================================
def new_post_coverage_rows(comp_nodes: list[dict], comp_url: str) -> list[dict]:
    """
    SHORT + DIRECT:
    - one row H1
    - one row H2 (all H2s)
    - one row H3 (all H3s with H4 merged)
    """
    h1s = list_headers(comp_nodes, 1)
    h2s = list_headers(comp_nodes, 2)
    h3s = list_headers(comp_nodes, 3)
    h4_map = group_h4_under_h3(comp_nodes)

    h1 = h1s[0] if h1s else ""
    h1_brief = (
        "Competitors frame the article’s main promise in the H1. Use a clear, reader-friendly main angle."
        if h1 else
        "No clean H1 detected (site layout differs). Use a clear, reader-friendly main promise."
    )

    if h2s:
        h2_list = "; ".join(h2s[:12]) + ("…" if len(h2s) > 12 else "")
        h2_brief = f"Covers: {h2_list}."
    else:
        h2_brief = "No clear H2 sections detected."

    def h3_label(h3: str) -> str:
        h4s = h4_map.get(h3, [])
        if h4s:
            small = "; ".join(h4s[:5]) + ("…" if len(h4s) > 5 else "")
            return f"{h3} (includes: {small})"
        return h3

    if h3s:
        h3_labeled = [h3_label(x) for x in h3s[:14]]
        h3_list = "; ".join(h3_labeled) + ("…" if len(h3s) > 14 else "")
        h3_brief = f"Covers: {h3_list}."
    else:
        h3_brief = "No clear H3 subsections detected."

    return [
        {"Headers covered": "H1 (main angle)", "Content covered": (h1 or "—") + " — " + h1_brief, "Source": source_link(comp_url)},
        {"Headers covered": "H2 (sections covered)", "Content covered": h2_brief, "Source": source_link(comp_url)},
        {"Headers covered": "H3 (subsections covered)", "Content covered": h3_brief, "Source": source_link(comp_url)},
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
# MODE SELECTOR (CENTERED BUTTONS) + INTERNAL FETCH HIDDEN
# =====================================================
if "mode" not in st.session_state:
    st.session_state.mode = "update"  # "update" or "new"

# Centered mode buttons
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

# Optional INTERNAL ONLY (hidden by default)
show_internal_fetch = st.sidebar.checkbox("Admin: show internal fetch log", value=False)

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

        # INTERNAL ONLY: hidden from normal users
        if show_internal_fetch and internal_fetch:
            st.sidebar.markdown("### Internal fetch log")
            for u, s in internal_fetch:
                st.sidebar.write(u, "—", s)

        # If some competitors blocked, show a short user-friendly note (no URLs)
        blocked = sum(1 for _, s in internal_fetch if not str(s).startswith("ok"))
        if blocked:
            st.warning(f"{blocked} competitor URL(s) could not be fetched. Results are based on the pages that were accessible.")

        st.markdown("<div class='section-pill'>Content Gaps</div>", unsafe_allow_html=True)

        if not all_rows:
            st.info("No gaps found (or competitors blocked).")
        else:
            df = pd.DataFrame(all_rows)[["Header (Gap)", "What to add", "Source"]]
            render_table(df)

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

        # INTERNAL ONLY: hidden from normal users
        if show_internal_fetch and internal_fetch:
            st.sidebar.markdown("### Internal fetch log")
            for u, s in internal_fetch:
                st.sidebar.write(u, "—", s)

        blocked = sum(1 for _, s in internal_fetch if not str(s).startswith("ok"))
        if blocked:
            st.warning(f"{blocked} competitor URL(s) could not be fetched. Results are based on the pages that were accessible.")

        st.markdown("<div class='section-pill'>Competitor Coverage</div>", unsafe_allow_html=True)

        if not rows:
            st.info("No competitor coverage extracted (competitors blocked / headings not found).")
        else:
            df = pd.DataFrame(rows)[["Headers covered", "Content covered", "Source"]]
            render_table(df)
