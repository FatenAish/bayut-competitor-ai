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
    """
    Returns: { ok, source, html, text }
    - html is provided only when we have real HTML.
    - text provided for reader-style fallbacks.
    """
    url = (url or "").strip()
    if not url:
        return {"ok": False, "source": None, "html": "", "text": ""}

    # 1) direct HTML
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

    # 2) jina reader
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
# HIERARCHY-AWARE HEADINGS
# =========================
def norm_header(h: str) -> str:
    h = clean(h).lower()
    h = re.sub(r"[^a-z0-9\s]", "", h)
    h = re.sub(r"\s+", " ", h).strip()
    return h

def is_faq_header(h: str) -> bool:
    hn = norm_header(h)
    return "faq" in hn or "frequently asked" in hn

def brief_text(content: str) -> str:
    # 1–2 sentences max
    content = clean(content)
    parts = re.split(r"(?<=[.!?])\s+", content)
    parts = [p.strip() for p in parts if len(p.strip()) > 30]
    out = " ".join(parts[:2])
    if len(out) > 220:
        out = out[:220].rstrip() + "..."
    return out

def build_heading_tree_from_html(html: str) -> list[dict]:
    """
    Build hierarchical nodes from H2/H3/H4.
    Output: list of nodes:
      { level, header, content, children:[...] }
    """
    soup = BeautifulSoup(html, "lxml")
    for t in soup.find_all(list(IGNORE)):
        t.decompose()

    root = soup.find("article") or soup
    headings = root.find_all(["h2", "h3", "h4"])
    if not headings:
        headings = root.find_all(["h1", "h2", "h3", "h4"])

    def level_of(tag_name: str) -> int:
        # map h1->1, h2->2, h3->3, h4->4
        try:
            return int(tag_name[1])
        except:
            return 9

    nodes = []
    stack = []  # stack of nodes

    def push_node(node):
        nonlocal nodes, stack
        # attach to parent if exists
        if stack:
            stack[-1]["children"].append(node)
        else:
            nodes.append(node)
        stack.append(node)

    def pop_to_level(lvl):
        nonlocal stack
        while stack and stack[-1]["level"] >= lvl:
            stack.pop()

    # Walk headings, gather content until next heading
    for h in headings:
        header = clean(h.get_text(" "))
        if not header or len(header) < 3:
            continue
        lvl = level_of(h.name)

        pop_to_level(lvl)

        node = {"level": lvl, "header": header, "content": "", "children": []}
        push_node(node)

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

        node["content"] = clean(" ".join(content_parts))

    # drop tiny content-only nodes (keep header + children though)
    def prune(n: dict) -> dict | None:
        # prune children first
        kept_children = []
        for ch in n["children"]:
            pr = prune(ch)
            if pr:
                kept_children.append(pr)
        n["children"] = kept_children

        # keep if it has children (even if short content)
        if n["children"]:
            return n

        # keep if content is meaningful
        if len(n["content"]) >= 80:
            return n

        return None

    pruned = []
    for n in nodes:
        p = prune(n)
        if p:
            pruned.append(p)

    return pruned

def build_heading_tree_from_reader_text(text: str) -> list[dict]:
    """
    For Jina markdown-like text.
    Headings look like:
      ## Header
      ### Subheader
      #### Question
    """
    lines = [l.rstrip() for l in (text or "").splitlines()]

    def level_from_md(line: str) -> int | None:
        m = re.match(r"^(#{1,4})\s+(.*)$", line.strip())
        if not m:
            return None
        return len(m.group(1))

    nodes = []
    stack = []

    def push_node(node):
        nonlocal nodes, stack
        if stack:
            stack[-1]["children"].append(node)
        else:
            nodes.append(node)
        stack.append(node)

    def pop_to_level(lvl):
        nonlocal stack
        while stack and stack[-1]["level"] >= lvl:
            stack.pop()

    current = None
    for line in lines:
        s = line.strip()
        if not s:
            continue

        lvl = level_from_md(s)
        if lvl is not None:
            header = clean(re.sub(r"^#{1,4}\s+", "", s))
            if not header:
                continue
            pop_to_level(lvl)
            node = {"level": lvl, "header": header, "content": "", "children": []}
            push_node(node)
            current = node
        else:
            if current:
                current["content"] += " " + s

    # clean contents
    def post(n):
        n["content"] = clean(n["content"])
        n["children"] = [post(c) for c in n["children"]]
        return n
    nodes = [post(n) for n in nodes]

    # prune like HTML version
    def prune(n: dict) -> dict | None:
        kept_children = []
        for ch in n["children"]:
            pr = prune(ch)
            if pr:
                kept_children.append(pr)
        n["children"] = kept_children

        if n["children"]:
            return n
        if len(n["content"]) >= 80:
            return n
        return None

    pruned = []
    for n in nodes:
        p = prune(n)
        if p:
            pruned.append(p)

    return pruned

def get_tree(url: str) -> dict:
    """
    Returns:
      { ok, source, nodes:[...] }
    """
    fetched = fetch_best_effort(url)
    if not fetched["ok"]:
        return {"ok": False, "source": None, "nodes": []}

    if fetched["html"]:
        nodes = build_heading_tree_from_html(fetched["html"])
        return {"ok": True, "source": fetched["source"], "nodes": nodes}

    nodes = build_heading_tree_from_reader_text(fetched["text"])
    return {"ok": True, "source": fetched["source"], "nodes": nodes}

# =========================
# FLATTEN FOR OUTPUT
# =========================
def collect_headers_norm(nodes: list[dict], keep_levels=(2,3)) -> set:
    s = set()
    def walk(n):
        if n["level"] in keep_levels:
            hn = norm_header(n["header"])
            if hn:
                s.add(hn)
        for c in n["children"]:
            walk(c)
    for n in nodes:
        walk(n)
    return s

def output_rows_missing_headers_per_comp(bayut_nodes, comp_nodes, comp_url):
    """
    Output rows for missing H2/H3 only.
    If missing header is H3, include H4 titles inside the brief (not separate rows).
    FAQs: show one row with the list of questions (H4) in the brief.
    """
    bayut_norm = collect_headers_norm(bayut_nodes, keep_levels=(2,3))

    rows = []

    def h4_titles(node: dict) -> list[str]:
        titles = []
        for ch in node.get("children", []):
            if ch["level"] == 4:
                t = clean(ch["header"])
                if t:
                    titles.append(t)
            # sometimes H4 might be nested differently; collect any level-4 in subtree
            titles.extend(h4_titles(ch))
        # dedupe preserve order
        out = []
        seen = set()
        for t in titles:
            nt = norm_header(t)
            if nt and nt not in seen:
                seen.add(nt)
                out.append(t)
        return out

    def walk(node: dict):
        lvl = node["level"]

        # only consider H2/H3 as "headers" rows
        if lvl in (2,3):
            hn = norm_header(node["header"])
            if hn and hn not in bayut_norm:
                base = brief_text(node.get("content", ""))

                # attach H4 questions/subheaders inside the brief for H3
                extra = ""
                if lvl == 3:
                    subs = h4_titles(node)
                    if subs:
                        if is_faq_header(node["header"]):
                            extra = " FAQs: " + "; ".join(subs[:12])
                        else:
                            extra = " Subpoints: " + "; ".join(subs[:10])

                brief_full = (base + (" " if base and extra else "") + extra).strip()
                if not brief_full:
                    # if no content, at least show subpoints
                    brief_full = extra.strip() or "Section present in competitor."

                rows.append({
                    "Missing Header": node["header"],
                    "What it contains (brief)": brief_full[:300] + ("..." if len(brief_full) > 300 else ""),
                    "Source": comp_url
                })

        # continue traversal
        for ch in node.get("children", []):
            walk(ch)

    for n in comp_nodes:
        walk(n)

    return rows

# =========================
# UI
# =========================
st.title("Bayut Header Gap Analysis")
st.caption("Per competitor: Missing H2/H3 headers in Bayut • H4 gets grouped under its parent H3 (e.g., FAQs questions).")

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

        # ✅ run competitor alone: missing headers only
        rows.extend(output_rows_missing_headers_per_comp(
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
        st.info("No missing H2/H3 headers found (or competitors blocked / headings not extractable).")
    else:
        st.dataframe(rows, use_container_width=True)
