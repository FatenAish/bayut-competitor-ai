# app.py (PART 1/3)
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
      .ai-summary {{
        background: white;
        border: 1px solid #E5E7EB;
        border-radius: 14px;
        padding: 14px 14px;
        margin: 6px 0 10px 0;
      }}
      .muted {{
        color:#6B7280;
        font-size: 13px;
      }}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    f"""
    <div class="hero">
      <h1><span class="bayut">Bayut</span> Competitor Gap Analysis</h1>
      <p>Identifies missing sections and incomplete coverage against competitor articles.</p>
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

        code, html = self._http_get(url)
        if code == 200 and html:
            text = self._extract_article_text_from_html(html)
            if self._validate_text(text, min_len=500):
                return FetchResult(True, "direct", code, html, text, None)

        ok, html2 = self._fetch_playwright_html(url)
        if ok and html2:
            text2 = self._extract_article_text_from_html(html2)
            if self._validate_text(text2, min_len=500):
                return FetchResult(True, "playwright", 200, html2, text2, None)

        jurl = self._jina_url(url)
        code3, txt3 = self._http_get(jurl)
        if code3 == 200 and txt3:
            text3 = self.clean(txt3)
            if self._validate_text(text3, min_len=500):
                return FetchResult(True, "jina", code3, "", text3, None)

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
                results[u] = FetchResult(True, "manual", 200, pasted.strip(), pasted.strip(), None)

    still_failed = [u for u in failed if not results[u].ok]
    if still_failed:
        st.stop()

    return results


# =====================================================
# HEADING TREE + FILTERS
# =====================================================
# (keep your existing heading extraction code exactly as you already have)
# ...
# (keep your HELPERS exactly as you already have)
# ...


# =====================================================
# CRASH-SAFETY HELPERS (NEW)
# =====================================================
def _fn_exists(name: str) -> bool:
    return name in globals() and callable(globals().get(name))

def safe_call_gap_engine(fn_name: str, *args, **kwargs) -> List[dict]:
    """
    Never crash the app on missing function.
    If the function is missing, return a single-row error that appears in the gaps table.
    """
    if not _fn_exists(fn_name):
        return [{
            "Headers": "System error: missing gap engine function",
            "Description": f"`{fn_name}` is not defined in this deployment. Paste PART 2/3 above the UI.",
            "Source": "System"
        }]
    try:
        return globals()[fn_name](*args, **kwargs) or []
    except Exception as e:
        return [{
            "Headers": "System error: gap engine failed",
            "Description": f"{fn_name} raised: {type(e).__name__}: {str(e)[:160]}",
            "Source": "System"
        }]
# app.py (PART 2/3)

def _header_similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, norm_header(a), norm_header(b)).ratio()

def _best_match_header(target: str, candidates: List[str], threshold: float = 0.82) -> Tuple[Optional[str], float]:
    best = None
    best_score = 0.0
    for c in candidates:
        sc = _header_similarity(target, c)
        if sc > best_score:
            best_score = sc
            best = c
    if best_score >= threshold:
        return best, best_score
    return None, best_score

def _collect_headers(nodes: List[dict], min_level: int = 2) -> List[str]:
    flat = flatten(nodes)
    headers = []
    for x in flat:
        lvl = x.get("level", 9)
        h = strip_label(x.get("header", ""))
        if not h:
            continue
        if lvl >= min_level:  # h2/h3/h4
            headers.append(h)
    return headers

def _content_snippet_for_header(nodes: List[dict], header: str) -> str:
    """
    Returns a short, cleaned content snippet under the matched header.
    """
    flat = flatten(nodes)
    header_n = norm_header(header)
    for x in flat:
        h = strip_label(x.get("header", ""))
        if not h:
            continue
        if norm_header(h) == header_n:
            c = clean(x.get("content", ""))
            if not c:
                return ""
            return c[:260] + ("…" if len(c) > 260 else "")
    return ""

def _missing_parts_under_shared_header(
    bayut_nodes: List[dict],
    comp_nodes: List[dict],
    shared_header: str
) -> Optional[str]:
    """
    Lightweight 'incomplete coverage' detector:
    - compare content lengths under a shared header
    - if competitor is much richer, we recommend expanding Bayut section
    """
    b = _content_snippet_for_header(bayut_nodes, shared_header)
    c = _content_snippet_for_header(comp_nodes, shared_header)

    b_len = len(clean(b))
    c_len = len(clean(c))

    if c_len >= max(220, b_len * 2.2):
        # competitor clearly has more detail
        return f"Competitor covers this section in more depth. Consider expanding '{shared_header}' with additional details/topics."
    return None

def update_mode_rows_header_first(
    bayut_nodes: List[dict],
    bayut_fr: FetchResult,
    comp_nodes: List[dict],
    comp_fr: FetchResult,
    comp_url: str,
    max_missing_headers: int = 7,
    max_missing_parts: int = 5,
) -> List[dict]:
    """
    Header-first gaps:
    1) Missing headers on Bayut vs competitor
    2) Then missing depth under shared headers (incomplete coverage)
    FAQs are NOT auto-added here (you can keep your strict FAQ logic elsewhere).
    """
    rows: List[dict] = []

    bayut_headers = _collect_headers(bayut_nodes, min_level=2)
    comp_headers = _collect_headers(comp_nodes, min_level=2)

    bayut_norm = {norm_header(h): h for h in bayut_headers}
    comp_norm = {norm_header(h): h for h in comp_headers}

    # 1) Missing headers first
    missing = []
    for c_h in comp_headers:
        c_n = norm_header(c_h)
        if c_n in bayut_norm:
            continue
        # allow fuzzy match (similar header names)
        match, sc = _best_match_header(c_h, bayut_headers, threshold=0.86)
        if match:
            continue
        missing.append(c_h)

    missing = missing[:max_missing_headers]
    for h in missing:
        snippet = _content_snippet_for_header(comp_nodes, h)
        desc = snippet if snippet else "Competitor includes this section. Add it to close the gap."
        rows.append({
            "Headers": h,
            "Description": desc,
            "Source": site_name(comp_url)
        })

    # 2) Missing parts (depth) under shared headers
    shared_candidates = []
    for c_h in comp_headers:
        c_n = norm_header(c_h)
        if c_n in bayut_norm:
            shared_candidates.append(comp_norm.get(c_n, c_h))
        else:
            match, sc = _best_match_header(c_h, bayut_headers, threshold=0.90)
            if match:
                shared_candidates.append(match)

    depth_rows = 0
    seen = set()
    for sh in shared_candidates:
        if depth_rows >= max_missing_parts:
            break
        key = norm_header(sh)
        if key in seen:
            continue
        seen.add(key)

        msg = _missing_parts_under_shared_header(bayut_nodes, comp_nodes, sh)
        if msg:
            rows.append({
                "Headers": f"{sh} (Missing parts)",
                "Description": msg,
                "Source": site_name(comp_url)
            })
            depth_rows += 1

    return rows
# =====================================================
# HTML TABLE RENDER (with hyperlinks)
# =====================================================
def render_table(df: pd.DataFrame, drop_internal_url: bool = True):
    if df is None or df.empty:
        st.info("No results to show.")
        return
    if drop_internal_url and "__url" in df.columns:
        df = df.drop(columns=["__url"])
    html = df.to_html(index=False, escape=False)
    st.markdown(html, unsafe_allow_html=True)

def section_header_with_ai_button(title: str, button_label: str, button_key: str) -> bool:
    c1, c2 = st.columns([4.2, 1.3])
    with c1:
        st.markdown(f"<div class='section-pill section-pill-tight'>{title}</div>", unsafe_allow_html=True)
    with c2:
        clicked = st.button(button_label, type="secondary", use_container_width=True, key=button_key)
    return clicked


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

show_internal_fetch = st.sidebar.checkbox("Admin: show internal fetch log", value=False)

# Keep last results visible
if "update_df" not in st.session_state:
    st.session_state.update_df = pd.DataFrame()
if "update_fetch" not in st.session_state:
    st.session_state.update_fetch = []
if "seo_update_df" not in st.session_state:
    st.session_state.seo_update_df = pd.DataFrame()
if "ai_update_df" not in st.session_state:
    st.session_state.ai_update_df = pd.DataFrame()
if "cq_update_df" not in st.session_state:
    st.session_state.cq_update_df = pd.DataFrame()
if "gaps_update_summary_text" not in st.session_state:
    st.session_state.gaps_update_summary_text = ""
if "seo_update_summary_text" not in st.session_state:
    st.session_state.seo_update_summary_text = ""

if "new_df" not in st.session_state:
    st.session_state.new_df = pd.DataFrame()
if "new_fetch" not in st.session_state:
    st.session_state.new_fetch = []
if "seo_new_df" not in st.session_state:
    st.session_state.seo_new_df = pd.DataFrame()
if "ai_new_df" not in st.session_state:
    st.session_state.ai_new_df = pd.DataFrame()
if "cq_new_df" not in st.session_state:
    st.session_state.cq_new_df = pd.DataFrame()
if "cov_new_summary_text" not in st.session_state:
    st.session_state.cov_new_summary_text = ""
if "seo_new_summary_text" not in st.session_state:
    st.session_state.seo_new_summary_text = ""


# =====================================================
# UI - UPDATE MODE
# =====================================================
if st.session_state.mode == "update":
    st.markdown("<div class='section-pill section-pill-tight'>Update Mode</div>", unsafe_allow_html=True)

    bayut_url = st.text_input("Bayut article URL", placeholder="https://www.bayut.com/mybayut/...")
    competitors_text = st.text_area(
        "Competitor URLs (one per line)",
        height=120,
        placeholder="https://example.com/article\nhttps://example.com/another"
    )
    competitors = [c.strip() for c in competitors_text.splitlines() if c.strip()]

    manual_fkw_update = st.text_input(
        "Optional: Focus Keyword (FKW) for analysis + UAE ranking",
        placeholder="e.g., pros and cons business bay"
    )

    run = st.button("Run analysis", type="primary")

    if run:
        if not bayut_url.strip():
            st.error("Bayut article URL is required.")
            st.stop()
        if not competitors:
            st.error("Add at least one competitor URL.")
            st.stop()

        with st.spinner("Fetching Bayut (no exceptions)…"):
            bayut_fr_map = resolve_all_or_require_manual(agent, [bayut_url.strip()], st_key_prefix="bayut")
            bayut_tree_map = ensure_headings_or_require_repaste([bayut_url.strip()], bayut_fr_map, st_key_prefix="bayut_tree")
        bayut_fr = bayut_fr_map[bayut_url.strip()]
        bayut_nodes = bayut_tree_map[bayut_url.strip()]["nodes"]

        with st.spinner("Fetching ALL competitors (no exceptions)…"):
            comp_fr_map = resolve_all_or_require_manual(agent, competitors, st_key_prefix="comp_update")
            comp_tree_map = ensure_headings_or_require_repaste(competitors, comp_fr_map, st_key_prefix="comp_update_tree")

        all_rows = []
        internal_fetch = []

        for comp_url in competitors:
            src = comp_fr_map[comp_url].source
            internal_fetch.append((comp_url, f"ok ({src})"))
            comp_nodes = comp_tree_map[comp_url]["nodes"]

            # SAFE: never crash on missing function or runtime exception
            all_rows.extend(
                safe_call_gap_engine(
                    "update_mode_rows_header_first",
                    bayut_nodes=bayut_nodes,
                    bayut_fr=bayut_fr,
                    comp_nodes=comp_nodes,
                    comp_fr=comp_fr_map[comp_url],
                    comp_url=comp_url,
                    max_missing_headers=7,
                    max_missing_parts=5,
                )
            )

        st.session_state.update_fetch = internal_fetch
        st.session_state.update_df = (
            pd.DataFrame(all_rows)[["Headers", "Description", "Source"]]
            if all_rows
            else pd.DataFrame(columns=["Headers", "Description", "Source"])
        )

        # These must exist in your codebase already
        st.session_state.seo_update_df = build_seo_analysis_update(
            bayut_url=bayut_url.strip(),
            bayut_fr=bayut_fr,
            bayut_nodes=bayut_nodes,
            competitors=competitors,
            comp_fr_map=comp_fr_map,
            comp_tree_map=comp_tree_map,
            manual_fkw=manual_fkw_update.strip()
        )

        with st.spinner("Fetching Google UAE ranking (desktop + mobile) + AI visibility…"):
            seo_enriched, ai_df = enrich_seo_df_with_rank_and_ai(
                st.session_state.seo_update_df,
                manual_query=manual_fkw_update.strip()
            )
            st.session_state.seo_update_df = seo_enriched
            st.session_state.ai_update_df = ai_df

        st.session_state.cq_update_df = build_content_quality_table_from_seo(
            seo_df=st.session_state.seo_update_df,
            fr_map_by_url={bayut_url.strip(): bayut_fr, **comp_fr_map},
            tree_map_by_url={bayut_url.strip(): {"nodes": bayut_nodes}, **{u: comp_tree_map[u] for u in competitors}},
            manual_query=manual_fkw_update.strip()
        )

    # Optional sidebar logs
    if show_internal_fetch and st.session_state.update_fetch:
        st.sidebar.markdown("### Internal fetch log (Update Mode)")
        st.sidebar.write(f"Playwright enabled: {PLAYWRIGHT_OK}")
        for u, s in st.session_state.update_fetch:
            st.sidebar.write(u, "—", s)

    # =========================
    # 1) Content Gaps Table
    # =========================
    gaps_clicked = section_header_with_ai_button("Content Gaps Table", "Summarize by AI", "btn_gaps_summary_update")
    if gaps_clicked:
        st.session_state.gaps_update_summary_text = ai_summary_from_df("gaps", st.session_state.update_df)

    if st.session_state.get("gaps_update_summary_text"):
        st.markdown(
            f"<div class='ai-summary'><b>AI Summary</b>"
            f"<div class='muted'>6–8 bullets only (real summary).</div>"
            f"<pre style='white-space:pre-wrap;margin:8px 0 0 0;'>{st.session_state.gaps_update_summary_text}</pre>"
            f"</div>",
            unsafe_allow_html=True
        )

    if st.session_state.update_df is None or st.session_state.update_df.empty:
        st.info("Run analysis to see results.")
    else:
        render_table(st.session_state.update_df)

    # =========================
    # 2) Content Quality (2nd table)
    # =========================
    st.markdown("<div class='section-pill section-pill-tight'>Content Quality</div>", unsafe_allow_html=True)
    if st.session_state.cq_update_df is None or st.session_state.cq_update_df.empty:
        st.info("Run analysis to see Content Quality signals.")
    else:
        render_table(st.session_state.cq_update_df, drop_internal_url=True)

    # =========================
    # 3) SEO Analysis
    # =========================
    seo_clicked = section_header_with_ai_button("SEO Analysis", "Summarize by AI", "btn_seo_summary_update")
    if seo_clicked:
        st.session_state.seo_update_summary_text = ai_summary_from_df("seo", st.session_state.seo_update_df)

    if st.session_state.get("seo_update_summary_text"):
        st.markdown(
            f"<div class='ai-summary'><b>AI Summary</b><div class='muted'>6–8 bullets only.</div>"
            f"<pre style='white-space:pre-wrap;margin:8px 0 0 0;'>{st.session_state.seo_update_summary_text}</pre></div>",
            unsafe_allow_html=True
        )

    if st.session_state.seo_update_df is None or st.session_state.seo_update_df.empty:
        st.info("Run analysis to see SEO comparison.")
    else:
        render_table(st.session_state.seo_update_df, drop_internal_url=True)

    # =========================
    # 4) AI Visibility
    # =========================
    st.markdown("<div class='section-pill section-pill-tight'>AI Visibility (Google AI Overview)</div>", unsafe_allow_html=True)
    if st.session_state.ai_update_df is None or st.session_state.ai_update_df.empty:
        st.info("Run analysis to see AI visibility signals.")
    else:
        render_table(st.session_state.ai_update_df, drop_internal_url=True)


# =====================================================
# UI - NEW POST MODE
# =====================================================
else:
    st.markdown("<div class='section-pill section-pill-tight'>New Post Mode</div>", unsafe_allow_html=True)

    new_title = st.text_input("New post title", placeholder="Pros & Cons of Living in Business Bay (2026)")
    competitors_text = st.text_area(
        "Competitor URLs (one per line)",
        height=120,
        placeholder="https://example.com/article\nhttps://example.com/another"
    )
    competitors = [c.strip() for c in competitors_text.splitlines() if c.strip()]

    manual_fkw_new = st.text_input("Optional: Focus Keyword (FKW) for SEO + UAE ranking", placeholder="e.g., pros and cons business bay")

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

        st.session_state.seo_new_df = build_seo_analysis_newpost(
            new_title=new_title.strip(),
            competitors=competitors,
            comp_fr_map=comp_fr_map,
            comp_tree_map=comp_tree_map,
            manual_fkw=manual_fkw_new.strip()
        )

        with st.spinner("Fetching Google UAE ranking (desktop + mobile) + AI visibility…"):
            seo_enriched, ai_df = enrich_seo_df_with_rank_and_ai(
                st.session_state.seo_new_df,
                manual_query=manual_fkw_new.strip()
            )
            st.session_state.seo_new_df = seo_enriched
            st.session_state.ai_new_df = ai_df

        st.session_state.cq_new_df = build_content_quality_table_from_seo(
            seo_df=st.session_state.seo_new_df,
            fr_map_by_url={**comp_fr_map},
            tree_map_by_url={u: comp_tree_map[u] for u in competitors},
            manual_query=manual_fkw_new.strip()
        )

    if show_internal_fetch and st.session_state.new_fetch:
        st.sidebar.markdown("### Internal fetch log (New Post Mode)")
        st.sidebar.write(f"Playwright enabled: {PLAYWRIGHT_OK}")
        for u, s in st.session_state.new_fetch:
            st.sidebar.write(u, "—", s)

    cov_clicked = section_header_with_ai_button("Competitor Coverage", "Summarize by AI", "btn_cov_summary_new")
    if cov_clicked:
        if st.session_state.new_df is None or st.session_state.new_df.empty:
            st.session_state.cov_new_summary_text = "No data."
        else:
            bullets = []
            for _, r in st.session_state.new_df.head(6).iterrows():
                bullets.append(f"• {r.get('Source','')}: {r.get('Headers covered','')}")
            st.session_state.cov_new_summary_text = openai_summarize_block(
                "Summarize competitor coverage (6–8 bullets).",
                "\n".join(bullets) if bullets else "No data."
            )

    if st.session_state.get("cov_new_summary_text"):
        st.markdown(
            f"<div class='ai-summary'><b>AI Summary</b><div class='muted'>6–8 bullets only.</div>"
            f"<pre style='white-space:pre-wrap;margin:8px 0 0 0;'>{st.session_state.cov_new_summary_text}</pre></div>",
            unsafe_allow_html=True
        )

    if st.session_state.new_df is None or st.session_state.new_df.empty:
        st.info("Generate competitor coverage to see results.")
    else:
        render_table(st.session_state.new_df)

    seo_clicked = section_header_with_ai_button("SEO Analysis", "Summarize by AI", "btn_seo_summary_new")
    if seo_clicked:
        st.session_state.seo_new_summary_text = ai_summary_from_df("seo", st.session_state.seo_new_df)

    if st.session_state.get("seo_new_summary_text"):
        st.markdown(
            f"<div class='ai-summary'><b>AI Summary</b><div class='muted'>6–8 bullets only.</div>"
            f"<pre style='white-space:pre-wrap;margin:8px 0 0 0;'>{st.session_state.seo_new_summary_text}</pre></div>",
            unsafe_allow_html=True
        )

    if st.session_state.seo_new_df is None or st.session_state.seo_new_df.empty:
        st.info("Generate competitor coverage to see SEO comparison.")
    else:
        render_table(st.session_state.seo_new_df, drop_internal_url=True)

    st.markdown("<div class='section-pill section-pill-tight'>AI Visibility (Google AI Overview)</div>", unsafe_allow_html=True)
    if st.session_state.ai_new_df is None or st.session_state.ai_new_df.empty:
        st.info("Generate competitor coverage to see AI visibility signals.")
    else:
        render_table(st.session_state.ai_new_df, drop_internal_url=True)

    st.markdown("<div class='section-pill section-pill-tight'>Content Quality</div>", unsafe_allow_html=True)
    if st.session_state.cq_new_df is None or st.session_state.cq_new_df.empty:
        st.info("Generate competitor coverage to see Content Quality signals.")
    else:
        render_table(st.session_state.cq_new_df, drop_internal_url=True)


# Helpful note (only if missing DataForSEO creds)
if (st.session_state.seo_update_df is not None and not st.session_state.seo_update_df.empty) or \
   (st.session_state.seo_new_df is not None and not st.session_state.seo_new_df.empty):
    if "_dataforseo_ready" in globals() and callable(globals()["_dataforseo_ready"]):
        if not _dataforseo_ready():
            st.warning("Google UAE ranking + AI visibility requires DataForSEO credentials in Streamlit secrets (DATAFORSEO login/password).")
