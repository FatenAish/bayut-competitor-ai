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
# HEADING TREE
# =========================
def norm_header(h: str) -> str:
    h = clean(h).lower()
    h = re.sub(r"[^a-z0-9\s]", "", h)
    h = re.sub(r"\s+", " ", h).strip()
    return h

def brief_text(content: str) -> str:
    content = clean(content)
    parts = re.split(r"(?<=[.!?])\s+", content)
    parts = [p.strip() for p in parts if len(p.strip()) > 30]
    out = " ".join(parts[:2])
    if len(out) > 220:
        out = out[:220].rstrip() + "..."
    return out

def level_of(tag_name: str) -> int:
    try:
        return int(tag_name[1])
    except:
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

    def pop_to_level(lvl):
        while stack and stack[-1]["level"] >= lvl:
            stack.pop()

    def add_node(node):
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
        return len(m.group(1)), clean(m.group(2))

    def pop_to_level(lvl):
        while stack and stack[-1]["level"] >= lvl:
            stack.pop()

    def add_node(node):
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
            pop_to_level(lvl)
            node = {"level": lvl, "header": header, "content": "", "children": []}
            add_node(node)
            current = node
        else:
            if current:
                current["content"] += " " + s

    # cleanup
    def walk(n):
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
# FORCE GROUPING LOGIC
# =========================
def is_subpoint_heading(h: str) -> bool:
    """
    Treat as "should be grouped under parent" even if it's H3.
    Examples: 'Dubai Marina:' 'Downtown Dubai:' 'JLT (Jumeirah Lakes Towers):'
    """
    s = clean(h)
    if not s:
        return False

    # ends with colon
    if s.endswith(":"):
        return True

    # very short place-like headings
    # (2-6 words, mostly capitalized words)
    words = s.split()
    if 2 <= len(words) <= 6:
        cap_ratio = sum(1 for w in words if w[:1].isupper()) / len(words)
        if cap_ratio >= 0.6:
            return True

    return False

def collect_headers_norm(nodes: list[dict], keep_levels=(2, 3)) -> set:
    out = set()
    def walk(n):
        if n["level"] in keep_levels:
            hn = norm_header(n["header"])
            if hn:
                out.add(hn)
        for c in n["children"]:
            walk(c)
    for n in nodes:
        walk(n)
    return out

def gather_child_titles(node: dict) -> list[str]:
    titles = []
    def walk(n):
        for ch in n.get("children", []):
            t = clean(ch.get("header", ""))
            if t:
                titles.append(t)
            walk(ch)
    walk(node)
    # dedupe
    seen = set()
    out = []
    for t in titles:
        nt = norm_header(t)
        if nt and nt not in seen:
            seen.add(nt)
            out.append(t)
    return out

def rows_missing_headers(bayut_nodes, comp_nodes, comp_url):
    bayut_norm = collect_headers_norm(bayut_nodes, keep_levels=(2, 3))
    rows = []

    def walk(node: dict, parent: dict | None):
        lvl = node["level"]

        # Only output rows for H2/H3, BUT:
        # - If it's subpoint heading, NEVER output it (merge into parent)
        if lvl in (2, 3):
            hn = norm_header(node["header"])
            if hn and hn not in bayut_norm:
                # if this node is a subpoint, skip row; parent will include it
                if parent and is_subpoint_heading(node["header"]):
                    pass
                else:
                    base = brief_text(node.get("content", ""))

                    # Include subpoints under this header (any child headings that are subpoints)
                    subs = []
                    for ch in node.get("children", []):
                        if is_subpoint_heading(ch.get("header", "")):
                            subs.append(clean(ch["header"]))
                    # Also include deeper subpoints
                    if not subs:
                        # gather any child titles if there are no "subpoint" marked children
                        pass

                    extra = ""
                    if subs:
                        extra = " Subpoints: " + "; ".join(subs[:12])

                    brief_full = (base + (" " if base and extra else "") + extra).strip()
                    if not brief_full:
                        # fallback: show child titles if no content
                        child_titles = gather_child_titles(node)
                        if child_titles:
                            brief_full = "Includes: " + "; ".join(child_titles[:12])
                        else:
                            brief_full = "Section present in competitor."

                    rows.append({
                        "Missing Header": node["header"],
                        "What it contains (brief)": brief_full[:320] + ("..." if len(brief_full) > 320 else ""),
                        "Source": comp_url
                    })

        # traverse children, but pass current as parent
        for ch in node.get("children", []):
            walk(ch, node)

    for n in comp_nodes:
        walk(n, None)

    return rows

# =========================
# UI
# =========================
st.title("Bayut Header Gap Analysis")
st.caption("Per competitor: Missing headers in Bayut • subpoint headings get grouped under the parent section (not separate rows).")

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
        st.info("No missing headers found (or competitors blocked / headings not extractable).")
    else:
        st.dataframe(rows, use_container_width=True)
