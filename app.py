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
# GROUP SUBPOINTS
# =========================
def is_subpoint_heading(h: str) -> bool:
    s = clean(h)
    if not s:
        return False
    if is_noise_header(s):
        return False

    if s.endswith(":"):
        return True

    words = s.split()
    if 2 <= len(words) <= 6:
        cap_ratio = sum(1 for w in words if w[:1].isupper()) / len(words)
        if cap_ratio >= 0.6:
            return True

    return False


def collect_headers_norm(nodes: list[dict], keep_levels=(2, 3)) -> set:
    out = set()

    def walk(n: dict):
        if n["level"] in keep_levels:
            hn = norm_header(n["header"])
            if hn:
                out.add(hn)
        for c in n["children"]:
            walk(c)

    for n in nodes:
        walk(n)

    return out

# =========================
# DISCUSSION-STYLE BRIEF
# =========================
STOP = {
    "the","and","for","with","that","this","from","you","your","are","was","were","will","have","has","had",
    "but","not","can","may","more","most","into","than","then","they","them","their","our","out","about",
    "also","over","under","between","within","near","where","when","what","why","how","who","which",
    "a","an","to","of","in","on","at","as","is","it","be","or","by","we","i","us",
    "business","bay","dubai"
}

def extract_key_phrases(text: str, top_n: int = 6) -> list[str]:
    words = re.findall(r"[a-zA-Z]{3,}", (text or "").lower())
    words = [w for w in words if w not in STOP]
    freq = {}
    for w in words:
        freq[w] = freq.get(w, 0) + 1
    ranked = sorted(freq.items(), key=lambda x: (x[1], len(x[0])), reverse=True)
    return [w for w, _ in ranked[:top_n]]

def brief_text(content: str) -> str:
    content = clean(content)
    if not content:
        return ""
    sents = re.split(r"(?<=[.!?])\s+", content)
    sents = [s.strip() for s in sents if len(s.strip()) > 40]
    base = sents[0] if sents else (content[:180] + ("..." if len(content) > 180 else ""))
    keys = extract_key_phrases(content, top_n=6)
    if keys:
        brief = f"Competitor discusses {', '.join(keys)}. {base}"
    else:
        brief = f"Competitor discusses this section. {base}"
    brief = clean(brief)
    if len(brief) > 320:
        brief = brief[:320].rstrip() + "..."
    return brief

# =========================
# FAQ QUESTION GAP
# =========================
FAQ_HEAD_RE = re.compile(r"\b(faq|faqs|frequently asked questions)\b", re.I)

def is_question(s: str) -> bool:
    s = clean(s)
    if not s:
        return False
    if s.endswith("?"):
        return True
    # common question starters
    return bool(re.match(r"^(what|why|how|where|when|who|which|is|are|can|do|does|did|should|will)\b", s.lower()))

def normalize_q(s: str) -> str:
    s = clean(s).lower()
    s = re.sub(r"\s+", " ", s)
    s = s.strip(" .!?")
    return s

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

    # child headings often contain questions
    def walk(n: dict):
        h = clean(n.get("header", ""))
        if h and is_question(h):
            qs.append(h)
        # also: sometimes content contains questions in a list-like manner
        content = n.get("content", "")
        if content:
            for part in re.split(r"[•\n]", content):
                p = clean(part)
                if is_question(p) and 6 <= len(p) <= 140:
                    qs.append(p)
        for c in n.get("children", []):
            walk(c)

    walk(node)

    # dedupe preserving order
    seen = set()
    out = []
    for q in qs:
        nq = normalize_q(q)
        if nq and nq not in seen:
            seen.add(nq)
            out.append(q)
    return out

# =========================
# MAIN OUTPUT
# =========================
def rows_missing_headers(bayut_nodes, comp_nodes, comp_url):
    bayut_norm = collect_headers_norm(bayut_nodes, keep_levels=(2, 3))
    rows = []

    # FAQ special: build question sets once
    bayut_faq = find_faq_node(bayut_nodes)
    comp_faq = find_faq_node(comp_nodes)
    bayut_qs = extract_faq_questions(bayut_faq) if bayut_faq else []
    comp_qs = extract_faq_questions(comp_faq) if comp_faq else []

    bayut_q_norm = {normalize_q(q) for q in bayut_qs}
    comp_q_norm = [(q, normalize_q(q)) for q in comp_qs]

    missing_faq_qs = [q for q, nq in comp_q_norm if nq and nq not in bayut_q_norm]

    def walk(node: dict, parent: dict | None):
        lvl = node["level"]
        header = node.get("header", "")

        # FAQ section row: show missing questions instead of generic brief
        if FAQ_HEAD_RE.search(header):
            # only add row if competitor has FAQ and Bayut is missing some questions or missing section entirely
            if missing_faq_qs or not bayut_faq:
                if missing_faq_qs:
                    brief = "Missing questions in Bayut: " + "; ".join(missing_faq_qs[:10])
                    if len(missing_faq_qs) > 10:
                        brief += f" (+{len(missing_faq_qs) - 10} more)"
                else:
                    brief = "Competitor includes an FAQ section; Bayut should add FAQs relevant to this topic."
                rows.append({
                    "Missing Header": "FAQs",
                    "What it contains (brief)": brief,
                    "Source": comp_url
                })
            return  # don't further list FAQ children as headers

        # Normal rows for H2/H3, skip subpoints
        if lvl in (2, 3):
            hn = norm_header(header)
            if hn and hn not in bayut_norm:
                if parent and is_subpoint_heading(header):
                    pass
                else:
                    base = brief_text(node.get("content", ""))

                    subs = []
                    for ch in node.get("children", []):
                        if is_subpoint_heading(ch.get("header", "")):
                            subs.append(clean(ch["header"]))

                    extra = ""
                    if subs:
                        extra = " Subpoints: " + "; ".join(subs[:12])

                    brief_full = (base + (" " if base and extra else "") + extra).strip()
                    if not brief_full:
                        brief_full = extra.strip() or "Section present in competitor."

                    rows.append({
                        "Missing Header": header,
                        "What it contains (brief)": brief_full[:320] + ("..." if len(brief_full) > 320 else ""),
                        "Source": comp_url
                    })

        for ch in node.get("children", []):
            walk(ch, node)

    for n in comp_nodes:
        walk(n, None)

    # If competitor has FAQ but our walk didn't add it (rare), add it here
    if comp_faq and not any(r["Missing Header"] == "FAQs" and r["Source"] == comp_url for r in rows):
        if missing_faq_qs or not bayut_faq:
            if missing_faq_qs:
                brief = "Missing questions in Bayut: " + "; ".join(missing_faq_qs[:10])
                if len(missing_faq_qs) > 10:
                    brief += f" (+{len(missing_faq_qs) - 10} more)"
            else:
                brief = "Competitor includes an FAQ section; Bayut should add FAQs relevant to this topic."
            rows.append({
                "Missing Header": "FAQs",
                "What it contains (brief)": brief,
                "Source": comp_url
            })

    return rows

# =========================
# UI
# =========================
st.title("Bayut Header Gap Analysis")
st.caption("Per competitor: Missing editorial headers in Bayut • filters out CTA/share/widgets • groups subpoints under parent • FAQs show missing questions.")

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

    rows = []
    fetch_report = []

    for comp_url in competitors:
        with st.spinner(f"Fetching competitor: {comp_url}"):
            comp_data = get_tree(comp_url)

        if not comp_data["ok"] or not comp_data["nodes"]:
            fetch_report.append((comp_url, "blocked/no headings"))
            continue

        fetch_report.append((comp_url, f"ok ({comp_data['source']})"))
        rows.extend(rows_missing_headers(
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

    st.subheader("Missing headers in Bayut (per competitor)")
    if not rows:
        st.info("No missing editorial headers found (or competitors blocked / headings not extractable).")
    else:
        st.dataframe(rows, use_container_width=True)
