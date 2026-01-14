import streamlit as st
import requests
import re
from bs4 import BeautifulSoup
from urllib.parse import quote_plus
from collections import defaultdict, Counter

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="Bayut Header Gap Analyzer", layout="wide")

# =========================
# FETCH / EXTRACT (ROBUST)
# =========================
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

IGNORE = {"nav", "footer", "header", "aside", "script", "style", "noscript"}

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

def fetch_url_raw(url: str, timeout: int = 20) -> tuple[int, str]:
    r = requests.get(url, headers=DEFAULT_HEADERS, timeout=timeout, allow_redirects=True)
    return r.status_code, r.text

def fetch_jina(url: str, timeout: int = 20) -> tuple[int, str]:
    # https://r.jina.ai/http(s)://...
    if url.startswith("https://"):
        jina_url = "https://r.jina.ai/https://" + url[len("https://"):]
    elif url.startswith("http://"):
        jina_url = "https://r.jina.ai/http://" + url[len("http://"):]
    else:
        jina_url = "https://r.jina.ai/https://" + url
    r = requests.get(jina_url, headers=DEFAULT_HEADERS, timeout=timeout, allow_redirects=True)
    return r.status_code, r.text

def extract_article_text_from_html(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    for t in soup.find_all(list(IGNORE)):
        t.decompose()
    article = soup.find("article")
    text = article.get_text(" ") if article else soup.get_text(" ")
    return clean(text)

def html_headings_and_sections(html: str) -> list[dict]:
    """
    Return list of {header, content}
    Uses h2/h3/h4 primarily; falls back to h1 if needed.
    """
    soup = BeautifulSoup(html, "lxml")
    for t in soup.find_all(list(IGNORE)):
        t.decompose()

    # prefer inside <article> if present
    root = soup.find("article") or soup

    # get headings in order
    headings = root.find_all(["h2", "h3", "h4"])
    if not headings:
        headings = root.find_all(["h1", "h2", "h3", "h4"])

    sections = []
    for i, h in enumerate(headings):
        header = clean(h.get_text(" "))
        if not header or len(header) < 3:
            continue

        # collect content until next heading
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
        # ignore tiny sections
        if len(content) < 80:
            continue

        sections.append({"header": header, "content": content})

    return sections

def markdown_headings_and_sections(md_text: str) -> list[dict]:
    """
    Jina often returns markdown-like text with headings:
    #, ##, ### etc.
    """
    lines = [l.rstrip() for l in (md_text or "").splitlines()]
    sections = []
    current_h = None
    buff = []

    def flush():
        nonlocal current_h, buff, sections
        if current_h and buff:
            content = clean(" ".join(buff))
            if len(content) >= 80:
                sections.append({"header": current_h, "content": content})
        current_h = None
        buff = []

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue

        # markdown heading
        m = re.match(r"^(#{1,4})\s+(.*)$", stripped)
        if m:
            flush()
            current_h = clean(m.group(2))
            continue

        # otherwise content
        if current_h:
            buff.append(stripped)

    flush()
    return sections

def heuristic_headings_from_text(text: str) -> list[dict]:
    """
    Fallback when only plain text exists.
    Detect 'heading-like' lines.
    """
    raw_lines = [clean(l) for l in (text or "").split("\n")]
    lines = [l for l in raw_lines if l]

    def looks_like_heading(line: str) -> bool:
        if len(line) < 6 or len(line) > 80:
            return False
        if line.endswith("."):
            return False
        # too many words = likely paragraph
        if len(line.split()) > 12:
            return False
        # must contain letters
        if not re.search(r"[A-Za-z]", line):
            return False
        return True

    # Build sections by heading-like lines
    sections = []
    current_h = None
    buff = []

    def flush():
        nonlocal current_h, buff, sections
        if current_h and buff:
            content = clean(" ".join(buff))
            if len(content) >= 80:
                sections.append({"header": current_h, "content": content})
        current_h = None
        buff = []

    for line in lines:
        if looks_like_heading(line):
            flush()
            current_h = line
        else:
            if current_h:
                buff.append(line)

    flush()
    return sections

def fetch_page_structured(url: str) -> dict:
    """
    Returns:
      { url, ok, source, text, sections:[{header,content}] }
    """
    url = (url or "").strip()
    if not url:
        return {"url": url, "ok": False, "source": None, "error": "Empty URL", "text": "", "sections": []}

    # 1) direct html
    try:
        code, html = fetch_url_raw(url)
        if code == 200 and html:
            text = extract_article_text_from_html(html)
            sections = html_headings_and_sections(html)
            # if headings missing, still ok
            if len(text) > 500:
                # if no sections found, build heuristics from text
                if not sections:
                    sections = heuristic_headings_from_text(text)
                return {"url": url, "ok": True, "source": "direct", "error": None, "text": text, "sections": sections}
    except Exception:
        pass

    # 2) jina reader
    try:
        code, md = fetch_jina(url)
        if code == 200 and md:
            text = clean(md)
            sections = markdown_headings_and_sections(md)
            if not sections:
                sections = heuristic_headings_from_text(text)
            if len(text) > 500:
                return {"url": url, "ok": True, "source": "jina", "error": None, "text": text, "sections": sections}
    except Exception:
        pass

    # 3) textise (encoded)
    try:
        encoded = quote_plus(url)
        t_url = f"https://textise.org/showtext.aspx?strURL={encoded}"
        code, html = fetch_url_raw(t_url)
        if code == 200 and html:
            text = clean(BeautifulSoup(html, "lxml").get_text(" "))
            sections = heuristic_headings_from_text(text)
            if len(text) > 400:
                return {"url": url, "ok": True, "source": "textise", "error": None, "text": text, "sections": sections}
    except Exception:
        pass

    return {"url": url, "ok": False, "source": None, "error": "Blocked / unreachable", "text": "", "sections": []}

# =========================
# SECTION SUMMARIES + GAP
# =========================
def brief_section_summary(content: str, max_sentences: int = 2) -> str:
    # take first 1–2 meaningful sentences
    sents = re.split(r"(?<=[.!?])\s+", clean(content))
    sents = [s.strip() for s in sents if len(s.strip()) > 40]
    return " ".join(sents[:max_sentences])[:220] + ("..." if len(" ".join(sents[:max_sentences])) > 220 else "")

def key_phrases(content: str, top_n: int = 8) -> list[str]:
    # lightweight keyword extraction
    words = re.findall(r"[a-zA-Z]{3,}", content.lower())
    words = [w for w in words if w not in STOP]
    c = Counter(words)
    return [w for w, _ in c.most_common(top_n)]

def find_matching_section(sections: list[dict], target_header_norm: str) -> dict | None:
    # exact or fuzzy-ish match
    for s in sections:
        if norm_header(s["header"]) == target_header_norm:
            return s
    # fuzzy: header contains most terms
    tgt_terms = set(target_header_norm.split())
    best = None
    best_score = 0
    for s in sections:
        hn = norm_header(s["header"])
        terms = set(hn.split())
        if not terms:
            continue
        score = len(tgt_terms & terms) / max(len(tgt_terms), 1)
        if score > best_score:
            best = s
            best_score = score
    return best if best_score >= 0.6 else None

def compute_section_gap(bayut_section: dict, comp_section: dict) -> str:
    """
    Return short missing content bullets (as a string) if competitor has extra signals.
    """
    bayut_text = (bayut_section.get("content") or "").lower()
    comp_text = (comp_section.get("content") or "")
    comp_keys = key_phrases(comp_text, top_n=10)

    missing = [k for k in comp_keys if k not in bayut_text]
    missing = missing[:5]

    if not missing:
        return ""

    # Make it editor-friendly
    return "Missing points: " + ", ".join(missing)

# =========================
# UI
# =========================
st.title("Bayut Header Content Gap Analysis")
st.caption("Compare competitor headers → detect missing headers + section-level content gaps (organized, not noisy).")

bayut_url = st.text_input("Bayut article URL")

competitors_text = st.text_area(
    "Competitor URLs (one per line, up to 5)",
    placeholder="https://example.com/article\nhttps://example.com/another"
)
competitors = [c.strip() for c in competitors_text.splitlines() if c.strip()][:5]

# Output controls (to avoid too much)
st.markdown("### Output controls")
min_competitors_for_header = st.slider("Show headers that appear in at least N competitors", 1, 5, 2)
max_rows = st.slider("Maximum rows to show", 5, 30, 12)

if st.button("Run header gap analysis"):
    if not bayut_url.strip():
        st.error("Bayut URL is required.")
        st.stop()
    if not competitors:
        st.error("At least one competitor is required.")
        st.stop()

    with st.spinner("Fetching pages (auto fallbacks)..."):
        bayut = fetch_page_structured(bayut_url.strip())
        if not bayut["ok"]:
            st.error(f"Bayut fetch failed: {bayut.get('error')}")
            st.stop()

        comp_results = [fetch_page_structured(u) for u in competitors]

    st.subheader("Fetch status")
    for c in comp_results:
        if c["ok"]:
            st.success(f"✔ {c['url']} — source: {c['source']} — sections found: {len(c['sections'])}")
        else:
            st.warning(f"⚠ {c['url']} — {c.get('error')}")

    valid_comps = [c for c in comp_results if c["ok"] and c["sections"]]
    if not valid_comps:
        st.error("No competitor sections could be extracted (blocked or no headings detected).")
        st.stop()

    # =========================
    # BUILD MARKET HEADERS
    # =========================
    header_to_comp_sections = defaultdict(list)  # norm_header -> list of (comp_url, header, content)
    header_counts = Counter()

    for comp in valid_comps:
        seen_norms = set()
        for sec in comp["sections"]:
            hn = norm_header(sec["header"])
            if not hn:
                continue
            # avoid counting same header multiple times per competitor
            if hn in seen_norms:
                continue
            seen_norms.add(hn)
            header_counts[hn] += 1
            header_to_comp_sections[hn].append((comp["url"], sec["header"], sec["content"]))

    # Filter headers by frequency across competitors
    market_headers = [h for h, cnt in header_counts.items() if cnt >= min_competitors_for_header]
    # sort by frequency desc
    market_headers.sort(key=lambda h: header_counts[h], reverse=True)

    # =========================
    # COMPARE AGAINST BAYUT
    # =========================
    rows = []
    for hn in market_headers:
        # Build competitor brief summary from first example section
        comp_examples = header_to_comp_sections[hn]
        comp_url, comp_header, comp_content = comp_examples[0]
        comp_brief = brief_section_summary(comp_content)

        # Match in Bayut
        bayut_match = find_matching_section(bayut["sections"], hn)

        if bayut_match is None:
            # Missing header entirely
            rows.append({
                "Header": comp_header,
                "What this header contains (competitors)": comp_brief,
                "Gap": "Missing Header",
                "What Bayut should add (brief)": "Add this header + write this section (same intent, structured).",
                "Evidence": f"{header_counts[hn]}/{len(valid_comps)} competitors have it"
            })
        else:
            # Header exists; check content gap
            gap_text = compute_section_gap(bayut_match, {"header": comp_header, "content": comp_content})
            if gap_text:
                rows.append({
                    "Header": f"{bayut_match['header']} (Content Gap)",
                    "What this header contains (competitors)": comp_brief,
                    "Gap": "Content Gap",
                    "What Bayut should add (brief)": gap_text,
                    "Evidence": f"{header_counts[hn]}/{len(valid_comps)} competitors have it"
                })

        if len(rows) >= max_rows:
            break

    # =========================
    # OUTPUT
    # =========================
    st.subheader("Header-based content gaps (organized)")
    if not rows:
        st.success("No header-level gaps found under current filters.")
        st.info("Try lowering 'Show headers that appear in at least N competitors' to 1.")
    else:
        st.dataframe(rows, use_container_width=True)

    # Optional debug: show extracted headers
    with st.expander("Debug: extracted headers (Bayut + competitors)"):
        st.markdown("**Bayut headers:**")
        st.write([s["header"] for s in bayut["sections"]][:80])

        st.markdown("**Competitor headers:**")
        for comp in valid_comps:
            st.write(comp["url"])
            st.write([s["header"] for s in comp["sections"]][:60])
