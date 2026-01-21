import base64
import html as html_lib
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
BAYUT_GREEN = "#0E8A6D"
BAYUT_GREEN_2 = "#26B89F"
INK = "#111827"
MUTED = "#6B7280"
PAGE_BG = "#F3FBF7"

# =====================================================
# NEW DESIGN (Tailwind/shadcn-inspired theme from your React CSS)
# =====================================================
st.markdown(
    """
<style>
/* ------------------------------
   Theme tokens (same spirit as src/index.css)
--------------------------------*/
:root{
  --bayut-primary: 163 82% 30%;
  --bayut-primary-light: 163 72% 40%;
  --bayut-primary-dark: 163 90% 22%;
  --bayut-glow: 163 60% 50%;

  --background: 150 55% 96%;
  --foreground: 220 25% 12%;

  --card: 0 0% 100%;
  --border: 150 20% 88%;
  --muted: 150 15% 92%;
  --muted-foreground: 220 10% 45%;

  --secondary: 150 30% 94%;
  --accent: 163 60% 92%;

  --primary: 163 82% 30%;
  --primary-foreground: 0 0% 100%;

  --radius: 0.75rem;

  --gradient-hero: linear-gradient(135deg, hsl(163 82% 30%) 0%, hsl(163 60% 45%) 50%, hsl(175 70% 40%) 100%);
  --gradient-card: linear-gradient(180deg, hsl(0 0% 100%) 0%, hsl(150 45% 96%) 100%);
  --gradient-surface: linear-gradient(135deg, hsl(150 55% 96%) 0%, hsl(163 40% 95%) 100%);

  --shadow-sm: 0 1px 2px 0 rgb(0 0 0 / 0.03);
  --shadow-md: 0 4px 6px -1px rgb(0 0 0 / 0.05), 0 2px 4px -2px rgb(0 0 0 / 0.03);
  --shadow-lg: 0 10px 15px -3px rgb(0 0 0 / 0.06), 0 4px 6px -4px rgb(0 0 0 / 0.04);
  --shadow-xl: 0 20px 25px -5px rgb(0 0 0 / 0.07), 0 8px 10px -6px rgb(0 0 0 / 0.04);
  --shadow-glow: 0 0 30px -5px hsl(163 82% 30% / 0.25);
}

/* ------------------------------
   App background + layout
--------------------------------*/
html, body { background: hsl(var(--background)) !important; }
[data-testid="stAppViewContainer"]{
  background: var(--gradient-surface) !important;
}
[data-testid="stHeader"] { background: transparent !important; }
section.main > div.block-container{
  max-width: 1040px !important;
  padding-top: 1.6rem !important;
  padding-bottom: 2.8rem !important;
}

/* Decorative background “blobs” like Index.tsx */
.bg-decor {
  position: fixed;
  inset: 0;
  z-index: -10;
  overflow: hidden;
}
.bg-decor .blob{
  position: absolute;
  border-radius: 9999px;
  filter: blur(60px);
  opacity: 0.42;
  animation: float 6s ease-in-out infinite;
}
.bg-decor .b1{ top:-80px; left:18%; width:520px; height:520px; background: hsl(var(--primary) / 0.3); }
.bg-decor .b2{ bottom:-120px; right:18%; width:460px; height:460px; background: hsl(var(--primary) / 0.22); animation-delay:-3s; }
.bg-decor .b3{ top:40%; left:-120px; width:420px; height:420px; background: hsl(163 60% 92% / 0.55); animation-delay:-1.5s; }
@keyframes float { 0%,100%{ transform: translateY(0);} 50%{ transform: translateY(-10px);} }

/* subtle grid overlay */
.grid-overlay{
  position:absolute; inset:0; opacity:0.035;
  background-image:
    linear-gradient(hsl(var(--foreground)) 1px, transparent 1px),
    linear-gradient(90deg, hsl(var(--foreground)) 1px, transparent 1px);
  background-size: 64px 64px;
}

/* ------------------------------
   Reusable classes (from your Tailwind components)
--------------------------------*/
.gradient-text{
  background: var(--gradient-hero);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
}
.card-elevated{
  background: var(--gradient-card);
  box-shadow: var(--shadow-lg);
  border: 1px solid hsl(var(--border));
  border-radius: 1.25rem;
}
.glow-hover{ transition: box-shadow 0.25s ease; }
.glow-hover:hover{ box-shadow: var(--shadow-glow); }
.pill{
  display:inline-flex; align-items:center; gap:.5rem;
  padding:.45rem .9rem;
  border-radius: 9999px;
  font-weight: 800;
  font-size: 13px;
  background: hsl(163 45% 95%);
  color: hsl(var(--primary));
  border: 1px solid hsl(var(--border));
}
.section-title{
  font-weight: 800;
  font-size: 18px;
  position: relative;
  display:inline-block;
  margin: 20px 0 10px 0;
}
.section-title:after{
  content:"";
  position:absolute;
  left:0; bottom:-6px;
  width:32px; height:3px;
  border-radius:9999px;
  background: hsl(var(--primary));
}

.footer-note{
  text-align: center;
  color: hsl(var(--muted-foreground));
  font-size: 13px;
  margin: 26px 0 8px 0;
}

/* ------------------------------
   Hero styling
--------------------------------*/
.hero-wrap{
  padding: 1.4rem 0 0.6rem 0;
  text-align:center;
  display:flex;
  flex-direction:column;
  align-items:center;
  position: relative;
}
.hero-wrap:before{
  content:"";
  position:absolute;
  top:-180px;
  left:50%;
  transform: translateX(-50%);
  width: 1000px;
  height: 520px;
  background: radial-gradient(circle at center, hsl(163 70% 82% / 0.65), transparent 62%);
  z-index: -1;
}
.hero-icon{
  width:76px; height:76px; margin:0 auto 18px auto;
  border-radius: 22px;
  background: linear-gradient(135deg, hsl(var(--primary)) 0%, hsl(var(--primary) / 0.75) 100%);
  box-shadow: var(--shadow-md), var(--shadow-glow);
  border: 1px solid hsl(var(--primary) / 0.15);
  display:flex; align-items:center; justify-content:center;
}
.hero-icon svg{
  width: 34px;
  height: 34px;
  color: white;
}
.hero-h1{
  font-size: 62px;
  line-height: 1.06;
  margin: 0;
  color: hsl(var(--foreground));
  font-weight: 900;
  letter-spacing: -0.02em;
}
.hero-sub{
  margin: 10px auto 0 auto;
  max-width: 560px;
  width: 100%;
  color: hsl(var(--muted-foreground));
  font-size: 16px;
  line-height: 1.6;
  text-align: center !important;
  margin-left: auto;
  margin-right: auto;
}
.hero-badges{ margin-top: 14px; display:flex; justify-content:center; gap:10px; flex-wrap:wrap; }
.hero-badge{
  display:inline-flex; align-items:center; gap:8px;
  padding: 8px 12px;
  border-radius: 9999px;
  border: 1px solid hsl(var(--border));
  background: hsl(var(--card));
  box-shadow: var(--shadow-sm);
  font-weight: 800;
  font-size: 13px;
  color: hsl(var(--foreground));
}
.hero-badge svg{
  width: 16px;
  height: 16px;
  color: hsl(var(--primary));
}

/* ------------------------------
   Streamlit widgets styling
--------------------------------*/
[data-testid="stTextInput"] input,
[data-testid="stTextArea"] textarea{
  background: hsl(var(--secondary) / 0.6) !important;
  border: 1px solid hsl(var(--border)) !important;
  border-radius: 16px !important;
  padding: 12px 14px !important;
  font-size: 15px !important;
}
[data-testid="stTextInput"] input:focus,
[data-testid="stTextArea"] textarea:focus{
  box-shadow: 0 0 0 4px hsl(var(--primary) / 0.18) !important;
  border-color: hsl(var(--primary)) !important;
}

/* Buttons (best-effort across Streamlit versions) */
div[data-testid="stButton"] > button,
div[data-testid="stFormSubmitButton"] > button{
  border-radius: 16px !important;
  padding: 0.85rem 1rem !important;
  font-weight: 900 !important;
  border: 1px solid hsl(var(--border)) !important;
  background: hsl(var(--secondary)) !important;
  color: hsl(var(--foreground)) !important;
  box-shadow: var(--shadow-sm) !important;
  transition: transform .1s ease, box-shadow .25s ease, background .2s ease !important;
}
div[data-testid="stButton"] > button:hover,
div[data-testid="stFormSubmitButton"] > button:hover{
  transform: translateY(-1px);
  box-shadow: var(--shadow-md), var(--shadow-glow) !important;
  background: hsl(var(--accent)) !important;
}
/* Primary */
div[data-testid="stButton"] > button[kind="primary"],
div[data-testid="stFormSubmitButton"] > button[kind="primary"]{
  background: hsl(var(--primary)) !important;
  color: hsl(var(--primary-foreground)) !important;
  border-color: hsl(var(--primary)) !important;
}
div[data-testid="stButton"] > button[kind="primary"]:hover,
div[data-testid="stFormSubmitButton"] > button[kind="primary"]:hover{
  background: hsl(var(--primary-dark)) !important;
}
/* Ensure form submit stays primary */
div[data-testid="stFormSubmitButton"] > button{
  background: hsl(var(--primary)) !important;
  color: hsl(var(--primary-foreground)) !important;
  border-color: hsl(var(--primary)) !important;
}
div[data-testid="stFormSubmitButton"] > button:hover{
  background: hsl(var(--primary-dark)) !important;
}

/* Mode toggle */
.mode-toggle{
  display:flex;
  justify-content:center;
  margin: 8px 0 2px 0;
}
.mode-hint{
  text-align:center;
  color: hsl(var(--muted-foreground));
  font-size: 14px;
  margin: 6px 0 24px 0;
}

/* Form card */
div[data-testid="stForm"]{
  background: hsl(var(--card)) !important;
  border: 1px solid hsl(var(--border)) !important;
  border-radius: 20px !important;
  padding: 22px 22px 6px 22px !important;
  box-shadow: var(--shadow-lg) !important;
  margin: 0 auto 24px auto !important;
  max-width: 860px !important;
  width: 100% !important;
}
div[data-testid="stForm"] form{
  padding: 0 !important;
}

/* Field labels */
.field-label{
  display:flex; align-items:center; gap:8px;
  font-size: 14px; font-weight: 800;
  color: hsl(var(--foreground));
  margin: 2px 0 6px 0;
}
.field-label .label-meta{
  font-size: 12px;
  font-weight: 600;
  color: hsl(var(--muted-foreground));
  margin-left: 6px;
}
.field-label .label-icon{
  display:inline-flex; align-items:center; justify-content:center;
  width: 20px; height: 20px;
  color: hsl(var(--primary));
}
.field-label .label-icon svg{
  width: 18px; height: 18px;
}

/* Empty state card */
.empty-card{
  background: hsl(var(--card));
  border: 1px solid hsl(var(--border));
  border-radius: 20px;
  box-shadow: var(--shadow-lg);
  padding: 34px 20px;
  text-align: center;
  margin: 24px auto 0 auto;
  max-width: 860px;
}
.empty-icon{
  width: 58px;
  height: 58px;
  margin: 0 auto 14px auto;
  border-radius: 18px;
  background: hsl(var(--secondary));
  display:flex;
  align-items:center;
  justify-content:center;
  color: hsl(var(--muted-foreground));
}
.empty-icon svg{
  width: 26px;
  height: 26px;
}
.empty-title{
  font-weight: 900;
  font-size: 20px;
  color: hsl(var(--foreground));
}
.empty-body{
  margin-top: 6px;
  color: hsl(var(--muted-foreground));
  font-size: 14px;
}


/* ------------------------------
   Tables (we’ll render with df.to_html(classes="data-table"))
--------------------------------*/
table.data-table{
  width:100% !important;
  border-collapse: separate !important;
  border-spacing: 0 !important;
  overflow: hidden !important;
  border-radius: 16px !important;
  border: 1px solid hsl(var(--border)) !important;
  background: hsl(var(--card)) !important;
}
table.data-table thead th{
  background: hsl(163 45% 95%) !important;
  color: hsl(var(--foreground)) !important;
  font-weight: 900 !important;
  padding: 10px 12px !important;
  text-align: left !important;
  border-bottom: 1px solid hsl(var(--border)) !important;
  white-space: nowrap;
}
table.data-table tbody td{
  padding: 10px 12px !important;
  border-top: 1px solid hsl(var(--border)) !important;
  color: hsl(var(--foreground)) !important;
  font-size: 13px !important;
  vertical-align: top !important;
}
table.data-table tbody tr:hover{
  background: hsl(var(--muted) / 0.5) !important;
}
a{
  color: hsl(var(--primary)) !important;
  font-weight: 900 !important;
  text-decoration: underline !important;
}
</style>
""",
    unsafe_allow_html=True,
)

# Background decor layer (like your React Index.tsx)
st.markdown(
    """
<div class="bg-decor">
  <div class="blob b1"></div>
  <div class="blob b2"></div>
  <div class="blob b3"></div>
  <div class="grid-overlay"></div>
</div>
""",
    unsafe_allow_html=True,
)

# Hero (like HeroSection.tsx)
st.markdown(
    """
<div class="hero-wrap">
  <div class="hero-icon">
    <svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
      <path d="M4 20V10M10 20V4M16 20V14M3 20H21" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
    </svg>
  </div>
  <h1 class="hero-h1">
    <span class="gradient-text">Bayut</span> <span>Competitor Gap Analysis</span>
  </h1>
  <p class="hero-sub">
    Identifies missing sections and incomplete coverage against competitor articles.
  </p>
  <div class="hero-badges">
    <span class="hero-badge">
      <svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
        <path d="M12 3l1.8 4.7L18 9l-4.2 1.3L12 15l-1.8-4.7L6 9l4.2-1.3L12 3z" stroke="currentColor" stroke-width="1.6" stroke-linejoin="round"/>
      </svg>
      SEO Analysis
    </span>
    <span class="hero-badge">
      <svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
        <path d="M12 3l1.8 4.7L18 9l-4.2 1.3L12 15l-1.8-4.7L6 9l4.2-1.3L12 3z" stroke="currentColor" stroke-width="1.6" stroke-linejoin="round"/>
      </svg>
      Content Quality
    </span>
    <span class="hero-badge">
      <svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
        <path d="M12 3l1.8 4.7L18 9l-4.2 1.3L12 15l-1.8-4.7L6 9l4.2-1.3L12 3z" stroke="currentColor" stroke-width="1.6" stroke-linejoin="round"/>
      </svg>
      Gap Detection
    </span>
  </div>
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

STOP = {
    "the","and","for","with","that","this","from","you","your","are","was","were","will","have","has","had",
    "but","not","can","may","more","most","into","than","then","they","them","their","our","out","about",
    "also","over","under","between","within","near","where","when","what","why","how","who","which",
    "a","an","to","of","in","on","at","as","is","it","be","or","by","we","i","us"
}

GENERIC_STOP = {
    "dubai","uae","business","bay","community","area","living","pros","cons",
    "property","properties","rent","sale","apartments","villas","guide"
}

def norm_header(h: str) -> str:
    h = clean(h).lower()
    h = re.sub(r"[^a-z0-9\s]", "", h)
    h = re.sub(r"\s+", " ", h).strip()
    return h

def header_is_faq(header: str) -> bool:
    nh = norm_header(header)
    if not nh:
        return False
    if nh in {"faq","faqs","frequently asked questions","frequently asked question"}:
        return True
    if "faq" in nh:
        return True
    if "frequently asked" in nh:
        return True
    return False

def is_noise_header(h: str) -> bool:
    s = clean(h)
    if not s:
        return True
    if header_is_faq(s):
        return False
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

def strip_label(h: str) -> str:
    return clean(re.sub(r"\s*:\s*$", "", (h or "").strip()))

def format_gap_list(items: List[str], limit: int = 6) -> str:
    cleaned = []
    seen = set()
    skip = {"other","other topics","other faq topics","faq topics","other faq topic","other faq","general","misc","miscellaneous"}
    for item in items or []:
        it = clean(item)
        if not it:
            continue
        k = norm_header(it)
        if k in skip:
            continue
        if k in seen:
            continue
        seen.add(k)
        cleaned.append(it)
    if not cleaned:
        return ""
    if limit <= 0 or len(cleaned) <= limit:
        return ", ".join(cleaned)
    return ", ".join(cleaned[:limit]) + f", and {len(cleaned) - limit} more"

def headings_blob(nodes: List[dict]) -> str:
    hs = []
    for x in flatten(nodes):
        h = clean(x.get("header", ""))
        if h and not is_noise_header(h):
            hs.append(h)
    return clean(" | ".join(hs))

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
    return "Not available"
# =====================================================
# STRICT FAQ DETECTION (REAL FAQ ONLY)
# =====================================================
FAQ_TITLES = {"faq","faqs","frequently asked questions","frequently asked question"}

def _looks_like_question(s: str) -> bool:
    s = clean(s)
    if not s or len(s) < 6:
        return False
    s_low = s.lower()
    if "?" in s:
        return True
    if re.match(r"^(what|where|when|why|how|who|which|can|is|are|do|does|did|should)\b", s_low):
        return True
    if any(p in s_low for p in ["what is", "how to", "is it", "are there", "can i", "should i"]):
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

def _faq_questions_from_schema(html: str) -> List[str]:
    if not html:
        return []
    qs: List[str] = []
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

            def add_q(q: str):
                qn = normalize_question(q)
                if not qn or len(qn) < 6 or len(qn) > 180:
                    return
                qs.append(qn)

            def walk(x):
                if isinstance(x, dict):
                    t = x.get("@type") or x.get("type")
                    t_list = []
                    if isinstance(t, list):
                        t_list = [str(z).lower() for z in t]
                    elif isinstance(t, str):
                        t_list = [t.lower()]

                    if any("question" == z or z.endswith("question") for z in t_list):
                        name = x.get("name") or x.get("text") or ""
                        if name:
                            add_q(name)

                    for v in x.values():
                        walk(v)
                elif isinstance(x, list):
                    for v in x:
                        walk(v)

            walk(j)
    except Exception:
        return []

    seen = set()
    out = []
    for q in qs:
        k = norm_header(q)
        if not k or k in seen:
            continue
        seen.add(k)
        out.append(q)
    return out

def _faq_questions_from_html(html: str) -> List[str]:
    if not html:
        return []
    soup = BeautifulSoup(html, "html.parser")
    for t in soup.find_all(list(IGNORE_TAGS)):
        t.decompose()

    qs: List[str] = []
    qs.extend(_faq_questions_from_schema(html))

    candidates = []
    for tag in soup.find_all(True):
        id_attr = (tag.get("id") or "").lower()
        cls_attr = " ".join(tag.get("class", []) or []).lower()
        if re.search(r"\bfaq\b|\bfaqs\b|\baccordion\b|\bquestions\b", id_attr + " " + cls_attr):
            candidates.append(tag)

    for h in soup.find_all(["h1", "h2", "h3", "h4"]):
        if header_is_faq(h.get_text(" ")):
            candidates.append(h.parent or h)

    for c in candidates[:10]:
        for el in c.find_all(["summary", "button", "h3", "h4", "h5", "strong", "p", "li", "dt"]):
            txt = clean(el.get_text(" "))
            if not txt or len(txt) < 6 or len(txt) > 180:
                continue
            if _looks_like_question(txt):
                qs.append(normalize_question(txt))

    seen = set()
    out = []
    for q in qs:
        k = norm_header(q)
        if not k or k in seen:
            continue
        seen.add(k)
        out.append(q)
    return out

def _faq_heading_nodes(nodes: List[dict]) -> List[dict]:
    out = []
    for x in flatten(nodes):
        if x.get("level") in (2, 3, 4) and header_is_faq(x.get("header", "")):
            out.append(x)
    return out

def _question_heading_children(node: dict) -> List[str]:
    qs = []
    for c in node.get("children", []) or []:
        hdr = clean(c.get("header", ""))
        if hdr and _looks_like_question(hdr):
            qs.append(normalize_question(hdr))
    return qs

def extract_questions_from_node(node: dict) -> List[str]:
    qs: List[str] = []
    qs.extend(_question_heading_children(node))

    def add_from_text_block(txt: str):
        txt = clean(txt or "")
        if not txt:
            return
        chunks = re.split(r"[\n\r]+|(?<=[\.\?\!])\s+", txt)
        for ch in chunks[:80]:
            ch = clean(ch)
            if not ch or len(ch) > 160:
                continue
            if _looks_like_question(ch):
                qs.append(normalize_question(ch))

    if len(qs) < 3:
        add_from_text_block(node.get("content", ""))

    seen = set()
    out = []
    for q in qs:
        k = norm_header(q)
        if not k or k in seen:
            continue
        seen.add(k)
        out.append(q)
    return out[:25]

def page_has_real_faq(fr: FetchResult, nodes: List[dict]) -> bool:
    if fr and fr.html:
        if _has_faq_schema(fr.html):
            return True
        if len(_faq_questions_from_html(fr.html)) >= 2:
            return True

    faq_nodes = _faq_heading_nodes(nodes)
    if not faq_nodes:
        return False

    for fn in faq_nodes:
        if len(extract_questions_from_node(fn)) >= 2:
            return True
        txt = clean(fn.get("content", ""))
        if txt and txt.count("?") >= 2:
            return True

    return True

def extract_faq_questions(fr: FetchResult, nodes: List[dict]) -> List[str]:
    qs: List[str] = []
    if fr and fr.html:
        qs.extend(_faq_questions_from_html(fr.html))
    for fn in _faq_heading_nodes(nodes):
        qs.extend(extract_questions_from_node(fn))

    seen = set()
    out = []
    for q in qs:
        k = norm_header(q)
        if not k or k in seen:
            continue
        seen.add(k)
        out.append(q)
    return out

def faq_topic_from_question(q: str) -> str:
    raw = normalize_question(q)
    if not raw:
        return ""
    topic = re.sub(
        r"^(what|where|when|why|how|who|which|can|is|are|do|does|did|should|could|would|will)\b",
        "",
        raw,
        flags=re.I,
    )
    topic = re.sub(
        r"^(is|are|do|does|did|can|should|could|would|will|has|have|had|there|it|this|that)\b",
        "",
        topic,
        flags=re.I,
    )
    topic = re.sub(r"^\s*(the|a|an)\b", "", topic, flags=re.I).strip()
    topic = re.sub(r"\?$", "", topic).strip()
    if len(topic) < 4:
        topic = raw.strip("?").strip()
    if len(topic) > 140:
        topic = topic[:140].rstrip()
    if not topic:
        return ""
    return topic[:1].upper() + topic[1:]

def faq_topics_from_questions(questions: List[str], limit: int = 10) -> List[str]:
    out: List[str] = []
    seen = set()
    for q in questions:
        subj = faq_topic_from_question(q)
        if not subj:
            continue
        k = norm_header(subj)
        if k in seen:
            continue
        seen.add(k)
        out.append(subj)
        if len(out) >= limit:
            break
    return out

def missing_faqs_row(
    bayut_nodes: List[dict],
    bayut_fr: FetchResult,
    comp_nodes: List[dict],
    comp_fr: FetchResult,
    comp_url: str
) -> Optional[dict]:
    if not page_has_real_faq(comp_fr, comp_nodes):
        return None

    comp_qs = extract_faq_questions(comp_fr, comp_nodes)
    comp_qs = [q for q in comp_qs if q and len(q) > 5]
    if not comp_qs:
        return None

    bayut_has = page_has_real_faq(bayut_fr, bayut_nodes)
    bayut_qs = []
    if bayut_has:
        bayut_qs = extract_faq_questions(bayut_fr, bayut_nodes)
    bayut_qs = [q for q in bayut_qs if q and len(q) > 5]

    def q_key(q: str) -> str:
        q2 = normalize_question(q)
        q2 = re.sub(r"[^a-z0-9\s]", "", q2.lower())
        q2 = re.sub(r"\s+", " ", q2).strip()
        return q2

    bayut_set = {q_key(q) for q in bayut_qs if q}

    missing_qs = [q for q in comp_qs if q_key(q) not in bayut_set]
    if not missing_qs:
        return None

    def as_question_list(items: List[str]) -> str:
        lis = "".join(f"<li>{html_lib.escape(clean(i))}</li>" for i in items if clean(i))
        return f"<ul>{lis}</ul>" if lis else ""

    return {
        "Headers": "FAQs",
        "Description": as_question_list(missing_qs),
        "Source": source_link(comp_url),
    }


# =====================================================
# SECTION EXTRACTION (HEADER-FIRST COMPARISON)
# =====================================================
def section_nodes(nodes: List[dict], levels=(2,3)) -> List[dict]:
    secs = []
    current_h2 = None
    for x in flatten(nodes):
        lvl = x["level"]
        h = strip_label(x.get("header",""))
        if not h or is_noise_header(h) or header_is_faq(h):
            continue
        if lvl == 2:
            current_h2 = h
        if lvl in levels:
            c = clean(x.get("content",""))
            secs.append({"level": lvl, "header": h, "content": c, "parent_h2": current_h2})

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

def find_best_bayut_match(comp_header: str, bayut_sections: List[dict], min_score: float = 0.73) -> Optional[dict]:
    best = None
    best_score = 0.0
    for b in bayut_sections:
        sc = header_similarity(comp_header, b["header"])
        if sc > best_score:
            best_score = sc
            best = b
    if best and best_score >= min_score:
        return {"bayut_section": best, "score": best_score}
    return None

def dedupe_rows(rows: List[dict]) -> List[dict]:
    out = []
    seen = set()
    for r in rows:
        hk = norm_header(r.get("Headers", ""))
        sk = norm_header(re.sub(r"<[^>]+>", "", r.get("Source", "")))
        k = hk + "||" + sk
        if k in seen:
            continue
        seen.add(k)
        out.append(r)
    return out


# =====================================================
# CONTENT GAP THEMES (NEUTRAL DESCRIPTION)
# =====================================================
def theme_flags(text: str) -> set:
    t = (text or "").lower()
    flags = set()

    def has_any(words: List[str]) -> bool:
        return any(w in t for w in words)

    if has_any(["metro", "public transport", "commute", "connectivity", "access", "highway", "roads", "bus", "train"]):
        flags.add("transport")
    if has_any(["parking", "traffic", "congestion", "rush hour", "gridlock"]):
        flags.add("traffic_parking")
    if has_any(["cost", "price", "pricing", "expensive", "afford", "budget", "rent", "fees", "charges"]):
        flags.add("cost")
    if has_any(["restaurants", "cafes", "nightlife", "vibe", "atmosphere", "social", "entertainment"]):
        flags.add("lifestyle")
    if has_any(["schools", "nursery", "kids", "family", "clinic", "hospital", "supermarket", "groceries", "pharmacy"]):
        flags.add("daily_life")
    if has_any(["safe", "safety", "security", "crime"]):
        flags.add("safety")
    if has_any(["pros", "cons", "advantages", "disadvantages", "weigh", "consider", "should you", "worth it"]):
        flags.add("decision_frame")
    if has_any(["compare", "comparison", "vs ", "versus", "alternative", "similar to"]):
        flags.add("comparison")

    return flags

def summarize_missing_section_action(header: str, subheaders: Optional[List[str]], comp_content: str) -> str:
    themes = list(theme_flags(comp_content))
    human_map = {
        "transport": "commute & connectivity",
        "traffic_parking": "traffic/parking realities",
        "cost": "cost considerations",
        "lifestyle": "lifestyle & vibe",
        "daily_life": "day-to-day convenience",
        "safety": "safety angle",
        "decision_frame": "decision framing",
        "comparison": "comparison context",
    }
    picks = [human_map.get(x, x) for x in themes]
    parts = []
    if subheaders:
        sub_list = format_gap_list(subheaders, limit=6)
        if sub_list:
            parts.append(f"Missing subtopics: {sub_list}.")
    if picks:
        theme_list = format_gap_list(picks, limit=4)
        if theme_list:
            parts.append(f"Missing coverage on: {theme_list}.")
    if not parts:
        parts.append("Missing this section.")
    return " ".join(parts)

def summarize_content_gap_action(header: str, comp_content: str, bayut_content: str) -> str:
    comp_flags = theme_flags(comp_content)
    bayut_flags = theme_flags(bayut_content)
    missing = list(comp_flags - bayut_flags)

    human_map = {
        "transport": "commute & connectivity",
        "traffic_parking": "traffic/parking realities",
        "cost": "cost considerations",
        "lifestyle": "lifestyle & vibe",
        "daily_life": "day-to-day convenience",
        "safety": "safety angle",
        "decision_frame": "decision framing",
        "comparison": "comparison context",
    }
    missing_human = [human_map.get(x, x) for x in missing]
    missing_list = format_gap_list(missing_human, limit=4)
    if missing_list:
        return "Missing depth on: " + missing_list + "."
    return "Missing depth and practical specifics in this section."


# =====================================================
# UPDATE MODE ENGINE (Headers | Description | Source)
# =====================================================
def update_mode_rows_header_first(
    bayut_nodes: List[dict],
    bayut_fr: FetchResult,
    comp_nodes: List[dict],
    comp_fr: FetchResult,
    comp_url: str,
    max_missing_headers: Optional[int] = None
) -> List[dict]:
    rows_map: Dict[str, dict] = {}
    source = source_link(comp_url)

    def add_row(header: str, parts: List[str]):
        if not header or not parts:
            return
        key = norm_header(header) + "||" + norm_header(re.sub(r"<[^>]+>", "", source))
        if key not in rows_map:
            rows_map[key] = {"Headers": header, "DescriptionParts": [], "Source": source}
        for p in parts:
            p = clean(p)
            if not p:
                continue
            if not p.endswith("."):
                p = p + "."
            if p not in rows_map[key]["DescriptionParts"]:
                rows_map[key]["DescriptionParts"].append(p)

    def children_map(h3_list: List[dict]) -> Dict[str, List[dict]]:
        cmap: Dict[str, List[dict]] = {}
        for h3 in h3_list:
            parent = h3.get("parent_h2") or ""
            pk = norm_header(parent)
            if not pk:
                continue
            cmap.setdefault(pk, []).append(h3)
        return cmap

    def child_headers(cmap: Dict[str, List[dict]], parent_header: str) -> List[str]:
        pk = norm_header(parent_header)
        return [c.get("header", "") for c in cmap.get(pk, [])]

    def combined_h2_content(h2_header: str, h2_list: List[dict], cmap: Dict[str, List[dict]]) -> str:
        pk = norm_header(h2_header)
        h2_content = ""
        for h2 in h2_list:
            if norm_header(h2.get("header", "")) == pk:
                h2_content = h2.get("content", "")
                break
        child_content = " ".join(c.get("content", "") for c in cmap.get(pk, []))
        return clean(" ".join([h2_content, child_content]))

    def missing_children(comp_children: List[str], bayut_children: List[str]) -> List[str]:
        missing = []
        for ch in comp_children:
            if not any(header_similarity(ch, bh) >= 0.73 for bh in bayut_children):
                missing.append(ch)
        return missing

    def depth_gap_summary(comp_text: str, bayut_text: str) -> str:
        c_txt = clean(comp_text or "")
        b_txt = clean(bayut_text or "")
        if len(c_txt) < 140:
            return ""
        if len(c_txt) < (1.30 * max(len(b_txt), 1)):
            return ""
        comp_flags = theme_flags(c_txt)
        bayut_flags = theme_flags(b_txt)
        if len(comp_flags - bayut_flags) < 1 and len(c_txt) < 650:
            return ""
        return summarize_content_gap_action("", c_txt, b_txt)

    bayut_secs = section_nodes(bayut_nodes, levels=(2, 3))
    comp_secs = section_nodes(comp_nodes, levels=(2, 3))

    bayut_h2 = [s for s in bayut_secs if s["level"] == 2]
    bayut_h3 = [s for s in bayut_secs if s["level"] == 3]
    comp_h2 = [s for s in comp_secs if s["level"] == 2]
    comp_h3 = [s for s in comp_secs if s["level"] == 3]

    bayut_children_map = children_map(bayut_h3)
    comp_children_map = children_map(comp_h3)

    for cs in comp_h2:
        comp_header = cs.get("header", "")
        comp_children = child_headers(comp_children_map, comp_header)
        comp_text = combined_h2_content(comp_header, comp_h2, comp_children_map) or cs.get("content", "")

        m = find_best_bayut_match(comp_header, bayut_h2, min_score=0.73)
        if not m:
            desc = summarize_missing_section_action(comp_header, comp_children, comp_text)
            add_row(comp_header, [desc])
            continue

        bayut_header = m["bayut_section"]["header"]
        bayut_children = child_headers(bayut_children_map, bayut_header)
        missing_sub = missing_children(comp_children, bayut_children)

        parts = []
        if missing_sub:
            sub_list = format_gap_list(missing_sub, limit=6)
            if sub_list:
                parts.append(f"Missing subtopics: {sub_list}.")

        bayut_text = combined_h2_content(bayut_header, bayut_h2, bayut_children_map)
        depth_note = depth_gap_summary(comp_text, bayut_text)
        if depth_note:
            parts.append(depth_note)

        if parts:
            add_row(comp_header, parts)

    comp_h2_norms = {norm_header(h.get("header", "")) for h in comp_h2}
    for cs in comp_h3:
        parent = cs.get("parent_h2") or ""
        if parent and norm_header(parent) in comp_h2_norms:
            continue
        m = find_best_bayut_match(cs["header"], bayut_h3 + bayut_h2, min_score=0.73)
        if m:
            continue
        desc = summarize_missing_section_action(cs["header"], None, cs.get("content", ""))
        add_row(cs["header"], [desc])

    rows = []
    for r in rows_map.values():
        desc = " ".join(r.get("DescriptionParts", [])).strip()
        rows.append({"Headers": r.get("Headers", ""), "Description": desc, "Source": r.get("Source", "")})

    if max_missing_headers and len(rows) > max_missing_headers:
        rows = rows[:max_missing_headers]

    faq_row = missing_faqs_row(bayut_nodes, bayut_fr, comp_nodes, comp_fr, comp_url)
    if faq_row:
        rows.append(faq_row)

    return dedupe_rows(rows)


# =====================================================
# SEO ANALYSIS (YOUR SAME LOGIC)
# =====================================================
def _secrets_get(key: str, default=None):
    try:
        if hasattr(st, "secrets") and key in st.secrets:
            return st.secrets[key]
    except Exception:
        pass
    return default

SERPAPI_API_KEY = _secrets_get("SERPAPI_API_KEY", None)

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

def extract_media_used(html: str) -> str:
    if not html:
        return "Not available"
    soup = BeautifulSoup(html, "html.parser")
    for t in soup.find_all(list(IGNORE_TAGS)):
        t.decompose()
    root = soup.find("article") or soup

    imgs = len(root.find_all("img"))
    videos = len(root.find_all("video"))
    tables = len(root.find_all("table"))

    for x in root.find_all("iframe"):
        src = (x.get("src") or "").lower()
        if any(k in src for k in ["youtube", "youtu.be", "vimeo", "dailymotion"]):
            videos += 1

    types = []
    if imgs:
        types.append("Images")
    if videos:
        types.append("Video")
    if tables:
        types.append("Tables")
    return ", ".join(types) if types else "None detected"

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
            if all(w in STOP or w in GENERIC_STOP for w in chunk):
                continue
            phrase = " ".join(chunk)
            if len(phrase) < 8:
                continue
            freq[phrase] = freq.get(phrase, 0) + 1
    return freq

def pick_fkw_only(seo_title: str, h1: str, headings_blob_text: str, body_text: str, manual_fkw: str = "") -> str:
    manual_fkw = clean(manual_fkw)
    if manual_fkw:
        return manual_fkw.lower()

    base = " ".join([seo_title or "", h1 or "", headings_blob_text or "", body_text or ""])
    freq = phrase_candidates(base, n_min=2, n_max=4)
    if not freq:
        return "Not available"

    title_low = (seo_title or "").lower()
    h1_low = (h1 or "").lower()

    scored = []
    for ph, c in freq.items():
        boost = 1.0
        if ph in title_low:
            boost += 0.9
        if ph in h1_low:
            boost += 0.6
        scored.append((c * boost, ph))
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[0][1] if scored else "Not available"

def word_count_from_text(text: str) -> int:
    t = clean(text or "")
    if not t:
        return 0
    return len(re.findall(r"\b\w+\b", t))

def compute_kw_repetition(text: str, phrase: str) -> str:
    if not text or not phrase or phrase == "Not available":
        return "Not available"
    t = " " + re.sub(r"\s+", " ", (text or "").lower()) + " "
    p = " " + re.sub(r"\s+", " ", (phrase or "").lower()) + " "
    return str(t.count(p))

def kw_usage_summary(seo_title: str, h1: str, headings_blob_text: str, body_text: str, fkw: str) -> str:
    fkw = clean(fkw or "").lower()
    if not fkw or fkw == "not available":
        return "Not available"

    text = clean(body_text or "")
    wc = word_count_from_text(text)

    rep = compute_kw_repetition(text, fkw)
    try:
        rep_i = int(rep)
    except Exception:
        rep_i = None

    per_1k = "Not available"
    if wc and rep_i is not None:
        per_1k = f"{(rep_i / max(wc,1))*1000:.1f}/1k"

    title_hit = "Yes" if fkw in (seo_title or "").lower() else "No"
    h1_hit = "Yes" if fkw in (h1 or "").lower() else "No"
    headings_hit = "Yes" if fkw in (headings_blob_text or "").lower() else "No"
    intro_raw = (body_text or "")[:1200].lower()
    intro_hit = "Yes" if fkw in intro_raw else "No"

    return f"Repeats:{rep} | {per_1k} | Title:{title_hit} H1:{h1_hit} Headings:{headings_hit} Intro:{intro_hit}"

def domain_of(url: str) -> str:
    try:
        host = urlparse(url).netloc.lower().replace("www.", "")
        return host.split(":")[0]
    except Exception:
        return ""

def _extract_canonical_and_robots(html: str) -> Tuple[str, str]:
    if not html:
        return ("Not available", "Not available")

    soup = BeautifulSoup(html, "html.parser")

    canonical = ""
    can = soup.find("link", attrs={"rel": "canonical"})
    if can and can.get("href"):
        canonical = clean(can.get("href"))

    robots = ""
    mr = soup.find("meta", attrs={"name": re.compile("^robots$", re.I)})
    if mr and mr.get("content"):
        robots = clean(mr.get("content"))

    return (canonical or "Not available", robots or "Not available")

def _count_headers(html: str) -> str:
    if not html:
        return "H1:0 / H2:0 / H3:0 / Total:0"
    soup = BeautifulSoup(html, "html.parser")
    for t in soup.find_all(list(IGNORE_TAGS)):
        t.decompose()
    h1 = len(soup.find_all("h1"))
    h2 = len(soup.find_all("h2"))
    h3 = len(soup.find_all("h3"))
    total = h1 + h2 + h3
    return f"H1:{h1} / H2:{h2} / H3:{h3} / Total:{total}"

from urllib.parse import urljoin

def _count_internal_outbound_links(html: str, page_url: str) -> Tuple[int, int]:
    if not html:
        return (0, 0)

    soup = BeautifulSoup(html, "html.parser")

    for t in soup.find_all(["nav","footer","header","aside","script","style","noscript","form"]):
        t.decompose()

    root = soup.find("article") or soup.find("main") or soup
    for bad in root.find_all(["nav","footer","header","aside"]):
        bad.decompose()

    body_blocks = root.find_all(["p","li","td","th","blockquote","figcaption"])

    internal = 0
    outbound = 0

    base_dom = domain_of(page_url)
    base_root = ".".join(base_dom.split(".")[-2:]) if base_dom else ""

    for blk in body_blocks:
        for a in blk.find_all("a", href=True):
            href = (a.get("href") or "").strip()
            if not href:
                continue
            hlow = href.lower()
            if hlow.startswith("#") or hlow.startswith("mailto:") or hlow.startswith("tel:") or hlow.startswith("javascript:"):
                continue
            full = urljoin(page_url, href)
            try:
                p = urlparse(full)
            except Exception:
                continue
            dom = (p.netloc or "").lower().replace("www.", "")
            if not dom:
                internal += 1
                continue
            if base_root and dom.endswith(base_root):
                internal += 1
            elif dom == base_dom:
                internal += 1
            else:
                outbound += 1

    return (internal, outbound)

def _schema_present(html: str) -> str:
    if not html:
        return "None detected"
    soup = BeautifulSoup(html, "html.parser")
    scripts = soup.find_all("script", attrs={"type": re.compile(r"ld\+json", re.I)})
    types = set()
    for s in scripts:
        raw = (s.string or "").strip()
        if not raw:
            continue
        try:
            j = json.loads(raw)
        except Exception:
            continue

        def walk(x):
            if isinstance(x, dict):
                t = x.get("@type") or x.get("type")
                if t:
                    if isinstance(t, list):
                        for z in t:
                            types.add(str(z))
                    else:
                        types.add(str(t))
                for v in x.values():
                    walk(v)
            elif isinstance(x, list):
                for v in x:
                    walk(v)
        walk(j)
    return ", ".join(sorted(types)) if types else "None detected"

def seo_row_for_page_extended(label: str, url: str, fr: FetchResult, nodes: List[dict], manual_fkw: str = "") -> dict:
    seo_title, meta_desc = extract_head_seo(fr.html or "")
    slug = url_slug(url) if url and url != "Not applicable" else "Not applicable"
    h_blob = headings_blob(nodes)
    h_counts = _count_headers(fr.html or fr.text or "")
    fkw = pick_fkw_only(seo_title, get_first_h1(nodes), h_blob, fr.text or "", manual_fkw=manual_fkw)
    kw_usage = kw_usage_summary(seo_title, get_first_h1(nodes), h_blob, fr.text or "", fkw)
    _, outbound_links_count = _count_internal_outbound_links(fr.html or "", url or "")
    media = extract_media_used(fr.html or "")
    schema = _schema_present(fr.html or "")
    _, robots = _extract_canonical_and_robots(fr.html or "")

    return {
        "Page": label,
        "UAE Rank (Mobile)": "Not available",
        "SEO Title": seo_title,
        "Meta Description": meta_desc,
        "URL Slug": slug,
        "Headers (H1/H2/H3/Total)": h_counts,
        "FKW Usage": kw_usage,
        "Robots Meta (index/follow)": robots,
        "Outbound Links Count": str(outbound_links_count),
        "Media (Images/Video/Tables)": media,
        "Schema Present": schema,
        "__fkw": fkw,
        "__url": url,
    }

def build_seo_analysis_update(
    bayut_url: str,
    bayut_fr: FetchResult,
    bayut_nodes: List[dict],
    competitors: List[str],
    comp_fr_map: Dict[str, FetchResult],
    comp_tree_map: Dict[str, dict],
    manual_fkw: str = ""
) -> pd.DataFrame:
    rows = []
    rows.append(seo_row_for_page_extended("Bayut", bayut_url, bayut_fr, bayut_nodes, manual_fkw=manual_fkw))
    for cu in competitors:
        fr = comp_fr_map.get(cu)
        nodes = (comp_tree_map.get(cu) or {}).get("nodes", [])
        rows.append(seo_row_for_page_extended(site_name(cu), cu, fr, nodes, manual_fkw=manual_fkw))
    df = pd.DataFrame(rows)
    cols = [
        "Page","UAE Rank (Mobile)","SEO Title","Meta Description","URL Slug",
        "Headers (H1/H2/H3/Total)","FKW Usage","Robots Meta (index/follow)",
        "Outbound Links Count","Media (Images/Video/Tables)",
        "Schema Present","__fkw","__url"
    ]
    for c in cols:
        if c not in df.columns:
            df[c] = ""
    return df[cols]

def build_seo_analysis_newpost(
    new_title: str,
    competitors: List[str],
    comp_fr_map: Dict[str, FetchResult],
    comp_tree_map: Dict[str, dict],
    manual_fkw: str = ""
) -> pd.DataFrame:
    rows = []
    fake_fr = FetchResult(True, "manual", 200, "", new_title, None)
    rows.append(seo_row_for_page_extended("Target page (new)", "Not applicable", fake_fr, [], manual_fkw=manual_fkw))
    for cu in competitors:
        fr = comp_fr_map.get(cu)
        nodes = (comp_tree_map.get(cu) or {}).get("nodes", [])
        rows.append(seo_row_for_page_extended(site_name(cu), cu, fr, nodes, manual_fkw=manual_fkw))
    df = pd.DataFrame(rows)
    cols = [
        "Page","UAE Rank (Mobile)","SEO Title","Meta Description","URL Slug",
        "Headers (H1/H2/H3/Total)","FKW Usage","Robots Meta (index/follow)",
        "Outbound Links Count","Media (Images/Video/Tables)",
        "Schema Present","__fkw","__url"
    ]
    for c in cols:
        if c not in df.columns:
            df[c] = ""
    return df[cols]

def enrich_seo_df_with_rank_and_ai(seo_df: pd.DataFrame, manual_query: str = "") -> Tuple[pd.DataFrame, pd.DataFrame]:
    ai_df = pd.DataFrame(columns=["Note"])
    return seo_df, ai_df


# =====================================================
# AI VISIBILITY (AIO) TABLE
# =====================================================
AI_OVERVIEW_KEYS = {"ai_overview", "ai_overviews", "ai overview", "ai overviews"}
URL_RE = re.compile(r"https?://[^\s\"'>\)]+")

def _find_ai_overview_block(data):
    if isinstance(data, dict):
        for k, v in data.items():
            if any(key in k.lower() for key in AI_OVERVIEW_KEYS):
                return v
        for v in data.values():
            found = _find_ai_overview_block(v)
            if found is not None:
                return found
    elif isinstance(data, list):
        for v in data:
            found = _find_ai_overview_block(v)
            if found is not None:
                return found
    return None

def _collect_urls(obj) -> List[str]:
    urls: List[str] = []
    if isinstance(obj, str):
        urls.extend(URL_RE.findall(obj))
    elif isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, str) and k.lower() in {"link", "source", "url"}:
                urls.extend(URL_RE.findall(v) or [v])
            else:
                urls.extend(_collect_urls(v))
    elif isinstance(obj, list):
        for v in obj:
            urls.extend(_collect_urls(v))
    return urls

def _serp_features_present(data: dict) -> List[str]:
    if not isinstance(data, dict):
        return []
    features = []
    feature_map = {
        "ai_overview": "AI Overview",
        "answer_box": "Answer box",
        "featured_snippet": "Featured snippet",
        "knowledge_graph": "Knowledge panel",
        "related_questions": "People also ask",
        "local_results": "Local pack",
        "top_stories": "Top stories",
        "news_results": "News results",
        "images_results": "Image pack",
        "image_results": "Image pack",
        "video_results": "Video results",
        "shopping_results": "Shopping results",
        "recipes_results": "Recipes",
    }
    for k, label in feature_map.items():
        if data.get(k):
            features.append(label)
    if _find_ai_overview_block(data) is not None and "AI Overview" not in features:
        features.append("AI Overview")
    return features

@st.cache_data(show_spinner=False, ttl=1800)
def serpapi_serp_cached(query: str, device: str) -> dict:
    if not SERPAPI_API_KEY:
        return {"_error": "missing_serpapi_key"}
    params = {
        "engine": "google",
        "q": query,
        "google_domain": "google.ae",
        "gl": "ae",
        "hl": "en",
        "api_key": SERPAPI_API_KEY,
        "num": 20,
        "device": device,
    }
    try:
        r = requests.get("https://serpapi.com/search.json", params=params, timeout=35)
        if r.status_code != 200:
            return {"_error": f"serpapi_http_{r.status_code}", "_text": r.text[:400]}
        return r.json()
    except Exception as e:
        return {"_error": str(e)}

def build_ai_visibility_table(query: str, target_url: str, competitors: List[str], device: str = "mobile") -> pd.DataFrame:
    cols = ["Target URL Cited in AIO","Cited Domains","# AIO Citations","Top Competitor Domains","SERP Features Present"]
    if not query or not SERPAPI_API_KEY:
        return pd.DataFrame([{c: "Not available" for c in cols}], columns=cols)

    data = serpapi_serp_cached(query, device=device)
    if not data or (isinstance(data, dict) and data.get("_error")):
        return pd.DataFrame([{c: "Not available" for c in cols}], columns=cols)

    ai_block = _find_ai_overview_block(data)
    cited_urls = _collect_urls(ai_block) if ai_block is not None else []
    cited_urls = list(dict.fromkeys([u for u in cited_urls if u.startswith("http")]))
    cited_domains = []
    for u in cited_urls:
        d = domain_of(u)
        if d and d not in cited_domains:
            cited_domains.append(d)

    target_dom = domain_of(target_url) if target_url and target_url != "Not applicable" else ""
    if ai_block is None:
        target_cited = "Not available"
        cited_domains_txt = "Not available"
        cited_count = "Not available"
    else:
        if not target_dom:
            target_cited = "Not applicable"
        else:
            target_cited = "Yes" if target_dom in cited_domains else "No"
        cited_domains_txt = format_gap_list(cited_domains, limit=6) if cited_domains else "None detected"
        cited_count = str(len(cited_urls))

    top_comp_domains = []
    organic = data.get("organic_results") or []
    for it in organic:
        link = it.get("link") or ""
        dom = domain_of(link)
        if not dom:
            continue
        if target_dom and dom == target_dom:
            continue
        if dom not in top_comp_domains:
            top_comp_domains.append(dom)
        if len(top_comp_domains) >= 6:
            break

    serp_features = _serp_features_present(data)
    serp_features_txt = format_gap_list(serp_features, limit=6) if serp_features else "None detected"

    row = {
        "Target URL Cited in AIO": target_cited,
        "Cited Domains": cited_domains_txt or "Not available",
        "# AIO Citations": cited_count,
        "Top Competitor Domains": format_gap_list(top_comp_domains, limit=6) if top_comp_domains else "Not available",
        "SERP Features Present": serp_features_txt,
    }
    return pd.DataFrame([row], columns=cols)


# =====================================================
# CONTENT QUALITY (same as your code)
# =====================================================
@st.cache_data(show_spinner=False, ttl=86400)
def _head_last_modified(url: str) -> str:
    try:
        r = requests.head(url, headers=DEFAULT_HEADERS, allow_redirects=True, timeout=18)
        return r.headers.get("Last-Modified", "") or ""
    except Exception:
        return ""

def _extract_last_modified_from_html(html: str) -> str:
    if not html:
        return ""
    soup = BeautifulSoup(html, "html.parser")

    meta_candidates = [
        ("meta", {"property": "article:modified_time"}, "content"),
        ("meta", {"property": "article:published_time"}, "content"),
        ("meta", {"name": "lastmod"}, "content"),
        ("meta", {"name": "last-modified"}, "content"),
        ("meta", {"name": "date"}, "content"),
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
        v = clean(tm.get("datetime"))
        if v:
            return v

    return ""

def get_last_modified(url: str, html: str) -> str:
    v = _extract_last_modified_from_html(html or "")
    if v:
        return v
    h = _head_last_modified(url)
    return h if h else "Not available"

def _kw_stuffing_label(word_count: int, repeats: int) -> str:
    if word_count <= 0:
        return "Not available"
    per_1k = (repeats / max(word_count, 1)) * 1000.0
    if per_1k >= 18:
        return f"High ({repeats} repeats, {per_1k:.1f}/1k words)"
    if per_1k >= 10:
        return f"Moderate ({repeats} repeats, {per_1k:.1f}/1k words)"
    return f"Low ({repeats} repeats, {per_1k:.1f}/1k words)"

def _latest_year_mentioned(text: str) -> int:
    years = re.findall(r"\b(19\d{2}|20\d{2})\b", (text or ""))
    ys = []
    for y in years:
        try:
            ys.append(int(y))
        except Exception:
            pass
    return max(ys) if ys else 0

def _has_brief_summary(nodes: List[dict], text: str) -> str:
    blob = (headings_blob(nodes) or "").lower()
    t = (text or "").lower()
    cues = ["tl;dr", "tldr", "key takeaways", "in summary", "summary", "quick summary", "at a glance"]
    if any(c in blob for c in cues) or any(c in t[:1400] for c in cues):
        return "Yes"
    return "No"

def _count_source_links(html: str) -> int:
    if not html:
        return 0
    soup = BeautifulSoup(html, "html.parser")
    links = soup.find_all("a", href=True)
    cnt = 0
    for a in links:
        href = a.get("href") or ""
        if href.startswith("http"):
            cnt += 1
    return cnt

CREDIBLE_KEYWORDS = ["gov", "edu", "who.int", "un.org", "worldbank", "statista", "imf", "oecd", "bbc", "nytimes", "guardian", "reuters", "wsj", "ft"]

def _credible_sources_count(html: str, page_url: str) -> int:
    if not html:
        return 0
    soup = BeautifulSoup(html, "html.parser")
    links = soup.find_all("a", href=True)
    seen = set()
    base_dom = domain_of(page_url)
    for a in links:
        href = a.get("href") or ""
        if not href.startswith("http"):
            continue
        dom = urlparse(href).netloc.lower().replace("www.", "")
        if dom == base_dom:
            continue
        if dom in seen:
            continue
        if any(k in dom for k in CREDIBLE_KEYWORDS) or dom.endswith(".gov") or dom.endswith(".edu") or dom.endswith(".org"):
            seen.add(dom)
    return len(seen)

def _references_section_present(nodes: List[dict], html: str) -> str:
    blob = headings_blob(nodes).lower()
    if any(k in blob for k in ["references", "sources", "further reading", "bibliography"]):
        return "Yes"
    if html:
        soup = BeautifulSoup(html, "html.parser")
        footers = soup.find_all(["footer", "section"])
        for s in footers[-3:]:
            txt = (s.get_text(" ") or "").lower()
            if "references" in txt or "sources" in txt:
                return "Yes"
    return "No"

def _data_points_count(text: str) -> int:
    if not text:
        return 0
    matches = re.findall(r"\b\d{1,3}(?:[,\d]{0,})(?:\.\d+)?\b|\b\d+%|\b\d+\.\d+%", text)
    return len(matches)

def _data_backed_claims_count(text: str) -> int:
    if not text:
        return 0
    patterns = [r"according to", r"data from", r"study", r"survey", r"research", r"reported that", r"found that", r"statistics"]
    cnt = 0
    for p in patterns:
        cnt += len(re.findall(p, text, flags=re.I))
    return cnt

STRONG_WORDS_RE = r"\b(best|worst|always|never|guarantee|guaranteed|unbeatable|the most|the best|huge|massive)\b"

def _unsupported_strong_claims_count(text: str) -> int:
    if not text:
        return 0
    sentences = re.split(r"(?<=[.!?])\s+", text)
    cnt = 0
    for s in sentences:
        if re.search(STRONG_WORDS_RE, s, flags=re.I):
            if not re.search(r"\d", s):
                cnt += 1
    return cnt

def _latest_information_label(last_modified: str, text: str) -> str:
    lm = (last_modified or "").lower()
    y = _latest_year_mentioned(text or "")
    if ("2026" in lm) or ("2025" in lm) or y >= 2025:
        return "Likely up-to-date"
    if y >= 2024:
        return "Somewhat recent"
    return "Unclear/Older"

def _split_sentences(text: str) -> List[str]:
    if not text:
        return []
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [clean(p) for p in parts if clean(p)]

def _extract_years(s: str) -> List[int]:
    years = []
    for y in re.findall(r"\b(19\d{2}|20\d{2})\b", s or ""):
        try:
            years.append(int(y))
        except Exception:
            continue
    return years

def _outdated_snippets(text: str, max_year: int = 2023, limit: int = 6) -> List[str]:
    if not text:
        return []
    out = []
    seen = set()
    for s in _split_sentences(text):
        yrs = _extract_years(s)
        if not yrs:
            continue
        if any(y <= max_year for y in yrs):
            key = norm_header(s)
            if key and key not in seen:
                seen.add(key)
                out.append(s)
        if len(out) >= limit:
            break
    return out

def _strong_claim_snippets(text: str, limit: int = 6) -> List[str]:
    if not text:
        return []
    out = []
    seen = set()
    for s in _split_sentences(text):
        if re.search(STRONG_WORDS_RE, s, flags=re.I) and not re.search(r"\d", s):
            key = norm_header(s)
            if key and key not in seen:
                seen.add(key)
                out.append(s)
        if len(out) >= limit:
            break
    return out

def _outdated_misleading_cell(last_modified: str, text: str) -> str:
    lm = clean(last_modified or "")
    lm_years = _extract_years(lm)
    outdated_items = _outdated_snippets(text, max_year=2023, limit=6)
    if lm_years and max(lm_years) <= 2023:
        outdated_items.insert(0, f"Last modified date: {lm}")

    wrong_items = _strong_claim_snippets(text, limit=6)

    if not outdated_items and not wrong_items:
        return "No obvious issues"

    if outdated_items and wrong_items:
        label = "Outdated + Wrong info"
    elif outdated_items:
        label = "Outdated info"
    else:
        label = "Wrong info"

    def as_list(items: List[str]) -> str:
        lis = "".join(f"<li>{html_lib.escape(i)}</li>" for i in items)
        return f"<ul>{lis}</ul>" if lis else ""

    details = []
    if outdated_items:
        details.append("<div><strong>Outdated signals</strong>" + as_list(outdated_items) + "</div>")
    if wrong_items:
        details.append("<div><strong>Potentially wrong or unsupported claims</strong>" + as_list(wrong_items) + "</div>")

    detail_html = "".join(details)
    return (
        "<details class='details-link'>"
        f"<summary><span class='link-like'>{html_lib.escape(label)}</span></summary>"
        f"<div class='details-box'>{detail_html}</div>"
        "</details>"
    )

def normalize_url_for_match(u: str) -> str:
    try:
        p = urlparse(u)
        host = p.netloc.lower().replace("www.", "")
        path = (p.path or "").rstrip("/")
        return host + path
    except Exception:
        return (u or "").strip().lower().replace("www.", "").rstrip("/")

def _topic_cannibalization_label(query: str, page_url: str) -> str:
    if not SERPAPI_API_KEY:
        return "Not available (no SERPAPI_API_KEY)"
    dom = domain_of(page_url)
    if not dom or not query or query == "Not available":
        return "Not available"
    site_q = f"site:{dom} {query}"
    data = serpapi_serp_cached(site_q, device="desktop")
    if not data or data.get("_error"):
        return f"Not available ({data.get('_error')})" if isinstance(data, dict) else "Not available"

    organic = data.get("organic_results") or []
    target = normalize_url_for_match(page_url)
    others = []
    for it in organic:
        link = it.get("link") or ""
        if not link:
            continue
        nm = normalize_url_for_match(link)
        if dom in nm and nm != target:
            others.append(link)

    cnt = len(set(others))
    if cnt >= 3:
        return f"High risk (≈{cnt} other pages on same domain)"
    if cnt >= 1:
        return f"Medium risk (≈{cnt} other page(s) on same domain)"
    return "Low risk"

def build_content_quality_table_from_seo(
    seo_df: pd.DataFrame,
    fr_map_by_url: Dict[str, FetchResult],
    tree_map_by_url: Dict[str, dict],
    manual_query: str = ""
) -> pd.DataFrame:
    if seo_df is None or seo_df.empty:
        return pd.DataFrame()

    cols = [
        "Page","Word Count","Last Updated / Modified","Topic Cannibalization","Keyword Stuffing",
        "Brief Summary","FAQs","References Section",
        "Source Links Count","Data-Backed Claims","Latest Information Score",
        "Outdated / Misleading Info","Styling / Layout",
    ]

    rows = []
    for _, r in seo_df.iterrows():
        page = str(r.get("Page", "")).strip()
        page_url = str(r.get("__url", "")).strip()

        if not page_url or page_url == "Not applicable":
            rows.append({c: "Not applicable" for c in cols})
            rows[-1]["Page"] = page
            continue

        fr = fr_map_by_url.get(page_url)
        tr = tree_map_by_url.get(page_url) or {}
        nodes = tr.get("nodes", []) if isinstance(tr, dict) else []

        html = (fr.html if fr else "") or ""
        text = (fr.text if fr else "") or ""

        wc = word_count_from_text(text)
        lm = get_last_modified(page_url, html)

        fkw = clean(manual_query) if clean(manual_query) else str(r.get("__fkw", ""))
        rep_s = compute_kw_repetition(text, fkw) if fkw and fkw != "Not available" else "0"
        try:
            rep_i = int(rep_s)
        except Exception:
            rep_i = 0

        topic_cann = _topic_cannibalization_label(fkw, page_url) if fkw else "Not available"
        kw_stuff = _kw_stuffing_label(wc, rep_i)

        brief = _has_brief_summary(nodes, text)
        faqs = "Yes" if (fr and page_has_real_faq(fr, nodes)) else "No"
        refs = _references_section_present(nodes, html)
        src_links = _count_source_links(html)
        data_backed = _data_backed_claims_count(text)
        latest_score = _latest_information_label(lm, text)
        outdated = _outdated_misleading_cell(lm, text)
        styling = "OK"

        rows.append({
            "Page": page,
            "Word Count": str(wc) if wc else "Not available",
            "Last Updated / Modified": lm,
            "Topic Cannibalization": topic_cann,
            "Keyword Stuffing": kw_stuff,
            "Brief Summary": brief,
            "FAQs": faqs,
            "References Section": refs,
            "Source Links Count": str(src_links),
            "Data-Backed Claims": str(data_backed),
            "Latest Information Score": latest_score,
            "Outdated / Misleading Info": outdated,
            "Styling / Layout": styling,
        })

    return pd.DataFrame(rows, columns=cols)


# =====================================================
# NEW POST MODE helpers
# =====================================================
def list_headers(nodes: List[dict], level: int) -> List[str]:
    return [x["header"] for x in flatten(nodes) if x["level"] == level and not is_noise_header(x["header"])]

def detect_main_angle(comp_nodes: List[dict]) -> str:
    h2s = [norm_header(h) for h in list_headers(comp_nodes, 2)]
    blob = " ".join(h2s)
    if ("pros" in blob and "cons" in blob) or ("advantages" in blob and "disadvantages" in blob):
        return "pros-and-cons decision guide"
    if "payment plan" in blob:
        return "buyer decision / payment-plan-led guide"
    if "amenities" in blob and "location" in blob:
        return "community overview for buyers"
    return "decision-led overview"

def new_post_coverage_rows(comp_nodes: List[dict], comp_url: str) -> List[dict]:
    h1s = list_headers(comp_nodes, 1)
    h1_title = strip_label(h1s[0]) if h1s else ""
    angle = detect_main_angle(comp_nodes)
    h1_text = f"{h1_title} — The competitor frames the page as a {angle}." if h1_title else f"The competitor frames the page as a {angle}."

    h2s = [strip_label(h) for h in list_headers(comp_nodes, 2)]
    h2_main = [h for h in h2s if h and not header_is_faq(h)]
    h2_main = h2_main[:6]
    h2_text = "Major sections include: " + " → ".join(h2_main) + "." if h2_main else "Major sections introduce the topic, break down key points, and end with wrap-up context."

    h3s = [strip_label(h) for h in list_headers(comp_nodes, 3)]
    themes, seen = [], set()
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
    h3_text = "Subsections break sections into practical themes such as: " + ", ".join(themes) + "." if themes else "Subsections add practical depth inside each major section."

    return [
        {"Headers covered": "H1 (main angle)", "Content covered": h1_text, "Source": site_name(comp_url)},
        {"Headers covered": "H2 (sections covered)", "Content covered": h2_text, "Source": site_name(comp_url)},
        {"Headers covered": "H3 (subsections covered)", "Content covered": h3_text, "Source": site_name(comp_url)},
    ]


# =====================================================
# HTML TABLE RENDER (UPDATED: use data-table class)
# =====================================================
def render_table(df: pd.DataFrame, drop_internal_url: bool = True):
    if df is None or df.empty:
        st.info("No results to show.")
        return
    if drop_internal_url:
        drop_cols = [c for c in df.columns if c.startswith("__")]
        if drop_cols:
            df = df.drop(columns=drop_cols)
    html = df.to_html(index=False, escape=False, classes="data-table")
    st.markdown(html, unsafe_allow_html=True)

ICON_LINK = """
<svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
  <path d="M10 13a5 5 0 0 1 0-7l1.6-1.6a5 5 0 0 1 7.1 7.1L17 12" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
  <path d="M14 11a5 5 0 0 1 0 7l-1.6 1.6a5 5 0 0 1-7.1-7.1L7 12" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
</svg>
"""

ICON_LIST = """
<svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
  <path d="M8 6h12M8 12h12M8 18h12" stroke="currentColor" stroke-width="2" stroke-linecap="round"/>
  <circle cx="4" cy="6" r="1.5" fill="currentColor"/>
  <circle cx="4" cy="12" r="1.5" fill="currentColor"/>
  <circle cx="4" cy="18" r="1.5" fill="currentColor"/>
</svg>
"""

ICON_SEARCH = """
<svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
  <circle cx="11" cy="11" r="6" stroke="currentColor" stroke-width="2"/>
  <path d="M20 20l-3.6-3.6" stroke="currentColor" stroke-width="2" stroke-linecap="round"/>
</svg>
"""

ICON_DOC = """
<svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
  <path d="M14 2H7a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V8z" stroke="currentColor" stroke-width="2" stroke-linejoin="round"/>
  <path d="M14 2v6h6" stroke="currentColor" stroke-width="2" stroke-linejoin="round"/>
  <path d="M9 12h6M9 16h6" stroke="currentColor" stroke-width="2" stroke-linecap="round"/>
</svg>
"""

def render_field_label(title: str, meta: str = "", icon_svg: str = ""):
    meta_html = f"<span class='label-meta'>{html_lib.escape(meta)}</span>" if meta else ""
    icon_html = f"<span class='label-icon'>{icon_svg}</span>" if icon_svg else ""
    st.markdown(
        f"<div class='field-label'>{icon_html}<span>{html_lib.escape(title)}</span>{meta_html}</div>",
        unsafe_allow_html=True,
    )

def render_empty_state(title: str, body: str):
    st.markdown(
        f"""
<div class="empty-card">
  <div class="empty-icon">{ICON_SEARCH}</div>
  <div class="empty-title">{html_lib.escape(title)}</div>
  <div class="empty-body">{html_lib.escape(body)}</div>
</div>
""",
        unsafe_allow_html=True,
    )

def section_header_pill(title: str):
    st.markdown(f"<div class='section-title'>{html_lib.escape(title)}</div>", unsafe_allow_html=True)


# =====================================================
# MODE SELECTOR (CENTERED BUTTONS)
# =====================================================
if "mode" not in st.session_state:
    st.session_state.mode = "update"  # "update" or "new"

st.markdown("<div class='mode-toggle'>", unsafe_allow_html=True)
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

mode_hint = (
    "Compare an existing Bayut article against competitors to find gaps."
    if st.session_state.mode == "update"
    else "Use competitor coverage to shape a new Bayut post outline."
)
st.markdown(f"<div class='mode-hint'>{html_lib.escape(mode_hint)}</div>", unsafe_allow_html=True)

show_internal_fetch = st.sidebar.checkbox("Admin: show internal fetch log", value=False)

# session state
for k, default in [
    ("update_df", pd.DataFrame()),
    ("update_fetch", []),
    ("seo_update_df", pd.DataFrame()),
    ("ai_update_df", pd.DataFrame()),
    ("cq_update_df", pd.DataFrame()),
    ("ai_vis_update_df", pd.DataFrame()),
    ("new_df", pd.DataFrame()),
    ("new_fetch", []),
    ("seo_new_df", pd.DataFrame()),
    ("ai_new_df", pd.DataFrame()),
    ("cq_new_df", pd.DataFrame()),
    ("ai_vis_new_df", pd.DataFrame()),
]:
    if k not in st.session_state:
        st.session_state[k] = default


# =====================================================
# UI - UPDATE MODE
# =====================================================
if st.session_state.mode == "update":
    with st.form("update_form"):
        render_field_label("Bayut Article URL", icon_svg=ICON_LINK)
        bayut_url = st.text_input(
            "Bayut Article URL",
            placeholder="https://www.bayut.com/mybayut/...",
            label_visibility="collapsed",
        )
        render_field_label("Competitor URLs", meta="(one per line)", icon_svg=ICON_LIST)
        competitors_text = st.text_area(
            "Competitor URLs",
            height=130,
            placeholder="https://example.com/article\nhttps://example.com/another",
            label_visibility="collapsed",
        )
        render_field_label("Focus Keyword", meta="(optional)", icon_svg=ICON_SEARCH)
        manual_fkw_update = st.text_input(
            "Focus Keyword",
            placeholder="e.g., pros and cons business bay",
            label_visibility="collapsed",
        )
        run = st.form_submit_button("Run Analysis", type="primary", use_container_width=True)

    competitors = [c.strip() for c in competitors_text.splitlines() if c.strip()]

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

            all_rows.extend(update_mode_rows_header_first(
                bayut_nodes=bayut_nodes,
                bayut_fr=bayut_fr,
                comp_nodes=comp_nodes,
                comp_fr=comp_fr_map[comp_url],
                comp_url=comp_url,
            ))

        st.session_state.update_fetch = internal_fetch
        st.session_state.update_df = (
            pd.DataFrame(all_rows)[["Headers", "Description", "Source"]]
            if all_rows
            else pd.DataFrame(columns=["Headers", "Description", "Source"])
        )

        st.session_state.seo_update_df = build_seo_analysis_update(
            bayut_url=bayut_url.strip(),
            bayut_fr=bayut_fr,
            bayut_nodes=bayut_nodes,
            competitors=competitors,
            comp_fr_map=comp_fr_map,
            comp_tree_map=comp_tree_map,
            manual_fkw=manual_fkw_update.strip()
        )

        st.session_state.seo_update_df, st.session_state.ai_update_df = enrich_seo_df_with_rank_and_ai(
            st.session_state.seo_update_df,
            manual_query=manual_fkw_update.strip()
        )

        st.session_state.cq_update_df = build_content_quality_table_from_seo(
            seo_df=st.session_state.seo_update_df,
            fr_map_by_url={bayut_url.strip(): bayut_fr, **comp_fr_map},
            tree_map_by_url={bayut_url.strip(): {"nodes": bayut_nodes}, **{u: comp_tree_map[u] for u in competitors}},
            manual_query=manual_fkw_update.strip()
        )

        query_for_ai = manual_fkw_update.strip() or get_first_h1(bayut_nodes)
        st.session_state.ai_vis_update_df = build_ai_visibility_table(
            query=query_for_ai,
            target_url=bayut_url.strip(),
            competitors=competitors,
            device="mobile",
        )

    if show_internal_fetch and st.session_state.update_fetch:
        st.sidebar.markdown("### Internal fetch log (Update Mode)")
        st.sidebar.write(f"Playwright enabled: {PLAYWRIGHT_OK}")
        for u, s in st.session_state.update_fetch:
            st.sidebar.write(u, "—", s)

    has_update_results = any(
        df is not None and not df.empty
        for df in [
            st.session_state.update_df,
            st.session_state.cq_update_df,
            st.session_state.seo_update_df,
            st.session_state.ai_vis_update_df,
        ]
    )

    if not has_update_results:
        render_empty_state(
            "No Analysis Yet",
            "Enter your URLs above and run the analysis to see competitor gap insights, SEO comparison, and content quality signals.",
        )
    else:
        section_header_pill("Gaps Table")
        if st.session_state.update_df is None or st.session_state.update_df.empty:
            st.info("No gaps detected yet. Add more competitors or refine URLs.")
        else:
            render_table(st.session_state.update_df)

        section_header_pill("Content Quality")
        if st.session_state.cq_update_df is None or st.session_state.cq_update_df.empty:
            st.info("Run analysis to see Content Quality signals.")
        else:
            render_table(st.session_state.cq_update_df, drop_internal_url=True)

        section_header_pill("SEO Analysis")
        if st.session_state.seo_update_df is None or st.session_state.seo_update_df.empty:
            st.info("Run analysis to see SEO comparison.")
        else:
            render_table(st.session_state.seo_update_df, drop_internal_url=True)

        section_header_pill("AI Visibility")
        if st.session_state.ai_vis_update_df is None or st.session_state.ai_vis_update_df.empty:
            st.info("Run analysis to see AI visibility signals.")
        else:
            render_table(st.session_state.ai_vis_update_df, drop_internal_url=True)


# =====================================================
# UI - NEW POST MODE
# =====================================================
else:
    with st.form("new_post_form"):
        render_field_label("New Post Title", icon_svg=ICON_DOC)
        new_title = st.text_input(
            "New Post Title",
            placeholder="Pros & Cons of Living in Business Bay (2026)",
            label_visibility="collapsed",
        )
        render_field_label("Competitor URLs", meta="(one per line)", icon_svg=ICON_LIST)
        competitors_text = st.text_area(
            "Competitor URLs",
            height=130,
            placeholder="https://example.com/article\nhttps://example.com/another",
            label_visibility="collapsed",
        )
        render_field_label("Focus Keyword", meta="(optional)", icon_svg=ICON_SEARCH)
        manual_fkw_new = st.text_input(
            "Focus Keyword",
            placeholder="e.g., pros and cons business bay",
            label_visibility="collapsed",
        )
        run = st.form_submit_button("Generate Coverage", type="primary", use_container_width=True)

    competitors = [c.strip() for c in competitors_text.splitlines() if c.strip()]

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

        st.session_state.seo_new_df, st.session_state.ai_new_df = enrich_seo_df_with_rank_and_ai(
            st.session_state.seo_new_df,
            manual_query=manual_fkw_new.strip()
        )

        st.session_state.cq_new_df = build_content_quality_table_from_seo(
            seo_df=st.session_state.seo_new_df,
            fr_map_by_url={u: comp_fr_map[u] for u in competitors},
            tree_map_by_url={u: comp_tree_map[u] for u in competitors},
            manual_query=manual_fkw_new.strip()
        )

        query_for_ai = manual_fkw_new.strip() or new_title.strip()
        st.session_state.ai_vis_new_df = build_ai_visibility_table(
            query=query_for_ai,
            target_url="Not applicable",
            competitors=competitors,
            device="mobile",
        )

    if show_internal_fetch and st.session_state.new_fetch:
        st.sidebar.markdown("### Internal fetch log (New Post Mode)")
        st.sidebar.write(f"Playwright enabled: {PLAYWRIGHT_OK}")
        for u, s in st.session_state.new_fetch:
            st.sidebar.write(u, "—", s)

    has_new_results = any(
        df is not None and not df.empty
        for df in [
            st.session_state.new_df,
            st.session_state.cq_new_df,
            st.session_state.seo_new_df,
            st.session_state.ai_vis_new_df,
        ]
    )

    if not has_new_results:
        render_empty_state(
            "No Analysis Yet",
            "Enter your competitor URLs above and generate coverage to see insights.",
        )
    else:
        section_header_pill("Competitor Coverage")
        if st.session_state.new_df is None or st.session_state.new_df.empty:
            st.info("Generate competitor coverage to see results.")
        else:
            render_table(st.session_state.new_df)

        section_header_pill("Content Quality")
        if st.session_state.cq_new_df is None or st.session_state.cq_new_df.empty:
            st.info("Generate competitor coverage to see Content Quality signals.")
        else:
            render_table(st.session_state.cq_new_df, drop_internal_url=True)

        section_header_pill("SEO Analysis")
        if st.session_state.seo_new_df is None or st.session_state.seo_new_df.empty:
            st.info("Generate competitor coverage to see SEO comparison.")
        else:
            render_table(st.session_state.seo_new_df, drop_internal_url=True)

        section_header_pill("AI Visibility")
        if st.session_state.ai_vis_new_df is None or st.session_state.ai_vis_new_df.empty:
            st.info("Generate competitor coverage to see AI visibility signals.")
        else:
            render_table(st.session_state.ai_vis_new_df, drop_internal_url=True)

if (st.session_state.seo_update_df is not None and not st.session_state.seo_update_df.empty) or \
   (st.session_state.seo_new_df is not None and not st.session_state.seo_new_df.empty):
    if not SERPAPI_API_KEY:
        st.warning("Note: SERPAPI_API_KEY is optional now (used for Topic Cannibalization and AI Visibility).")

st.markdown(
    "<div class='footer-note'>Bayut Competitor Gap Analysis Tool - Built for content optimization</div>",
    unsafe_allow_html=True,
)
