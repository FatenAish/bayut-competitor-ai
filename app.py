import streamlit as st
import requests
import re
from bs4 import BeautifulSoup
from urllib.parse import quote_plus

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="Bayut Header Gap (Per Competitor)", layout="wide")

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
    # r.jina.ai/http(s)://...
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

def fetch_html_best_effort(url: str) -> dict:
    """
    Returns dict:
      { ok, source, html, text }
    html is only present when we really have HTML.
    text is always present when ok=True (could be reader-text).
    """
    url = (url or "").strip()
    if not url:
        return {"ok": False, "source": None, "html": "", "text": ""}

    # 1) direct
    try:
        code, html = fetch_direct(url)
        if code == 200 and html:
            # keep html + plain text
            soup = BeautifulSoup(html, "lxml")
            for t in soup.find_all(list(IGNORE)):
                t.decompose()
            article = soup.find("article") or soup
            text = clean(article.get_text(" "))
            if len(text) > 400 and not looks_blocked(text):
                return {"ok": True, "source": "direct", "html": html, "text": text}
    except Exception:
        pass

    # 2) jina reader (text-like, sometimes markdown)
    try:
        code, txt = fetch_jina(url)
        if code == 200 and txt:
            text = clean(txt)
            if len(text) > 400 and not looks_blocked(text):
                return {"ok": True, "source": "jina", "html": "", "text": text}
    except Exception:
        pass

    # 3) textise (encoded)
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
# HEADERS EXTRACTION
# =========================
def norm_header(h: str) -> str:
    h = clean(h).lower()
    h = re.sub(r"[^a-z0-9\s]", "", h)
    h = re.sub(r"\s+", " ", h).strip()
    return h

def extract_sections_from_html(html: str) -> list[dict]:
    soup = BeautifulSoup(html, "lxml")
    for t in soup.find_all(list(IGNORE)):
        t.decompose()

    root = soup.find("article") or soup
    headings = root.find_all(["h2", "h3", "h4"])
    if not headings:
        headings = root.find_all(["h1", "h2", "h3", "h4"])

    sections = []
    for h in headings:
        header = clean(h.get_text(" "))
        if not header or len(header) < 3:
            continue

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

        content = clean(" ".join(content_parts))
        if len(content) < 80:
            continue

        sections.append({"header": header, "content": content})

    return sections

def extract_sections_from_reader_text(text: str) -> list[dict]:
    """
    Works for Jina markdown-like output:
    ## Heading
    content...
    """
    lines = [l.rstrip() for l in (text or "").splitlines()]
    sections = []
    current_h = None
    buff = []

    def flush():
        nonlocal current_h, buff
        if current_h and buff:
            content = clean(" ".join(buff))
            if len(content) >= 80:
                sections.append({"header": current_h, "content": content})
        current_h = None
        buff = []

    for line in lines:
        s = line.strip()
        if not s:
            continue

        m = re.match(r"^(#{1,4})\s+(.*)$", s)
        if m:
            flush()
            current_h = clean(m.group(2))
            continue

        if current_h:
            buff.append(s)

    flush()
    return sections

def brief(content: str) -> str:
    # 1–2 sentences only
    content = clean(content)
    parts = re.split(r"(?<=[.!?])\s+", content)
    parts = [p.strip() for p in parts if len(p.strip()) > 30]
    out = " ".join(parts[:2])
    if len(out) > 220:
        out = out[:220].rstrip() + "..."
    return out

def get_sections(url: str) -> dict:
    """
    Returns:
      { ok, source, sections }
    """
    fetched = fetch_html_best_effort(url)
    if not fetched["ok"]:
        return {"ok": False, "source": None, "sections": []}

    if fetched["html"]:
        secs = extract_sections_from_html(fetched["html"])
        return {"ok": True, "source": fetched["source"], "sections": secs}

    # reader text
    secs = extract_sections_from_reader_text(fetched["text"])
    return {"ok": True, "source": fetched["source"], "sections": secs}

# =========================
# UI
# =========================
st.title("Bayut Header Gap Analysis")
st.caption("Per competitor: missing headers in Bayut + brief content + source")

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
        bayut_data = get_sections(bayut_url.strip())
    if not bayut_data["ok"] or not bayut_data["sections"]:
        st.error("Could not extract headings from Bayut page (blocked or no headings found).")
        st.stop()

    bayut_headers_norm = set(norm_header(s["header"]) for s in bayut_data["sections"] if norm_header(s["header"]))

    rows = []
    fetch_report = []

    for comp_url in competitors:
        with st.spinner(f"Fetching competitor: {comp_url}"):
            comp_data = get_sections(comp_url)

        if not comp_data["ok"] or not comp_data["sections"]:
            fetch_report.append((comp_url, "blocked/no headings"))
            continue

        fetch_report.append((comp_url, f"ok ({comp_data['source']})"))

        # ✅ RUN EACH COMPETITOR ALONE:
        # Just add headers that this competitor has and Bayut doesn't.
        for sec in comp_data["sections"]:
            h = sec["header"]
            hn = norm_header(h)
            if not hn:
                continue
            if hn not in bayut_headers_norm:
                rows.append({
                    "Missing Header": h,
                    "What it contains (brief)": brief(sec["content"]),
                    "Source": comp_url
                })

    # show fetch status (not design change; just useful)
    st.subheader("Fetch status")
    for u, status in fetch_report:
        if status.startswith("ok"):
            st.success(f"✔ {u} — {status}")
        else:
            st.warning(f"⚠ {u} — {status}")

    st.subheader("Missing headers in Bayut (per competitor)")
    if not rows:
        st.info("No missing headers found (or competitors blocked / headings not extractable).")
    else:
        st.dataframe(rows, use_container_width=True)
