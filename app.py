# =========================
# FULL APP.PY UPDATE — HALF 1/2
# Paste this HALF to REPLACE your current "CONTENT GAP ENGINE (ONE BLOCK)" section
# AND also paste the 2 compute functions below it.
# (Keep the rest of your file as-is.)
# =========================

# ------------------------------------------------------------
# CONTENT GAP ENGINE (OLD LOGIC — HEADER FIRST + (MISSING PARTS) + STRICT FAQ)
# ------------------------------------------------------------

def _subheaders_for_comp_parent_h2(comp_secs: List[dict], parent_h2: str, limit: int = 5) -> List[str]:
    if not parent_h2:
        return []
    out = []
    for s in comp_secs:
        if s.get("level") == 3 and clean(s.get("parent_h2","")) == parent_h2:
            h = clean(s.get("header",""))
            if h and h not in out:
                out.append(h)
        if len(out) >= limit:
            break
    return out

def _content_diff_missing_parts(comp_text: str, bayut_text: str, max_parts: int = 3) -> List[str]:
    comp_text = clean(comp_text or "")
    bayut_text = clean(bayut_text or "")
    if not comp_text:
        return []

    comp_flags = theme_flags(comp_text)
    bayut_flags = theme_flags(bayut_text)
    missing = list(comp_flags - bayut_flags)

    human_map = {
        "transport":"commute & connectivity",
        "traffic_parking":"traffic/parking realities",
        "cost":"cost considerations",
        "lifestyle":"lifestyle & vibe",
        "daily_life":"day-to-day convenience",
        "safety":"safety angle",
        "decision_frame":"decision framing",
        "comparison":"comparison context",
    }
    return [human_map.get(x, x) for x in missing][:max_parts]

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
    OLD logic:
    - Missing headers first (competitor header not in Bayut)
    - Then (missing parts) rows when headers match but competitor goes deeper
    - FAQs row only if competitor has REAL FAQ
    """
    rows: List[dict] = []

    bayut_secs = section_nodes(bayut_nodes, levels=(2,3))
    comp_secs  = section_nodes(comp_nodes,  levels=(2,3))

    bayut_h2 = [s for s in bayut_secs if s.get("level") == 2]
    comp_h2  = [s for s in comp_secs  if s.get("level") == 2]

    # 1) Missing headers (competitor H2 not found in Bayut)
    missing_headers = []
    for c in comp_h2:
        hit = find_best_bayut_match(c["header"], bayut_h2, min_score=0.73)
        if not hit:
            missing_headers.append(c)

    for c in missing_headers[:max_missing_headers]:
        subhs = _subheaders_for_comp_parent_h2(comp_secs, c["header"], limit=5)
        desc = summarize_missing_section_action(c["header"], subhs, c.get("content",""))
        rows.append({
            "Headers": c["header"],
            "Description": desc,
            "Source": source_link(comp_url),
        })

    # 2) Missing parts under matching headers (competitor deeper)
    # Only compare H2 level (same as old behavior)
    parts_rows = 0
    for c in comp_h2:
        hit = find_best_bayut_match(c["header"], bayut_h2, min_score=0.73)
        if not hit:
            continue

        b = hit["bayut_section"]
        comp_text = c.get("content","")
        bayut_text = b.get("content","")

        # If competitor has useful themes not present in Bayut, add "(missing parts)"
        parts = _content_diff_missing_parts(comp_text, bayut_text, max_parts=3)
        if not parts:
            continue

        desc = summarize_content_gap_action(c["header"], comp_text, bayut_text)
        # Strengthen description with explicit missing themes (old-style)
        desc = desc if desc else ("Competitor goes deeper on: " + ", ".join(parts) + ".")

        rows.append({
            "Headers": f"{c['header']} (missing parts)",
            "Description": desc,
            "Source": source_link(comp_url),
        })
        parts_rows += 1
        if parts_rows >= max_missing_parts:
            break

    # 3) FAQs — ONE ROW only if competitor has REAL FAQ
    faq_row = missing_faqs_row(bayut_nodes, bayut_fr, comp_nodes, comp_fr, comp_url)
    if faq_row:
        rows.append(faq_row)

    return rows

def compute_gaps_header_first(
    bayut_url: str,
    bayut_fr: FetchResult,
    bayut_nodes: List[dict],
    competitors: List[str],
    comp_fr_map: Dict[str, FetchResult],
    comp_tree_map: Dict[str, dict],
) -> pd.DataFrame:
    """
    Returns ONE table with columns:
    Headers | Description | Source
    Deduped by header+source (same as old).
    """
    all_rows: List[dict] = []

    for cu in competitors:
        comp_nodes = (comp_tree_map.get(cu) or {}).get("nodes", []) or []
        comp_fr = comp_fr_map.get(cu)

        rows = update_mode_rows_header_first(
            bayut_nodes=bayut_nodes,
            bayut_fr=bayut_fr,
            comp_nodes=comp_nodes,
            comp_fr=comp_fr,
            comp_url=cu,
            max_missing_headers=7,
            max_missing_parts=5,
        )
        all_rows.extend(rows)

    all_rows = dedupe_rows(all_rows)
    df = pd.DataFrame(all_rows, columns=["Headers", "Description", "Source"])
    return df
# =========================
# FULL APP.PY UPDATE — HALF 2/2
# Paste this HALF to FIX New Post Mode to use the SAME OLD LOGIC as Update Mode.
# Replace your current `else:` block (New Post Mode UI) بالكامل بهذا.
# =========================

else:
    st.markdown("<div class='section-pill section-pill-tight'>New Post Mode (Header coverage)</div>", unsafe_allow_html=True)

    topic_title = st.text_input("Topic / Post title", placeholder="e.g., Pros & Cons of Living in Business Bay")
    competitors_text = st.text_area("Competitor URLs (one per line, max 5)", height=120)
    competitors = [c.strip() for c in competitors_text.splitlines() if c.strip()][:5]
    manual_fkw_new = st.text_input("Optional: Focus Keyword (FKW) for analysis + UAE ranking", placeholder="e.g., living in Business Bay")

    current_sig = signature("new", topic_title, competitors_text, manual_fkw_new)
    if st.session_state.last_sig_new and st.session_state.last_sig_new != current_sig:
        clear_new_results()
        st.session_state.new_fetch = []

    run = st.button("Run analysis", type="primary")

    if run:
        if not competitors:
            st.error("Add at least one competitor URL.")
            st.stop()

        clear_new_results()
        st.session_state.new_fetch = []
        st.session_state.last_sig_new = current_sig

        # Fetch all competitors (force HTML for quality tables/FAQ)
        with st.spinner("Fetching ALL competitors (no exceptions)…"):
            comp_fr_map = resolve_all_or_require_manual(agent, competitors, st_key_prefix="comp_new")
            comp_fr_map = ensure_html_for_quality(competitors, comp_fr_map, st_key_prefix="comp_new_html")
            comp_tree_map = ensure_headings_or_require_repaste(competitors, comp_fr_map, st_key_prefix="comp_new_tree")

        st.session_state.new_fetch = [(u, f"ok ({comp_fr_map[u].source})") for u in competitors]

        # Build a "coverage" table using the SAME engine:
        # We simulate Bayut as "empty" by using the FIRST competitor as baseline bayut_nodes with blank content
        # BUT to keep the OLD behavior (missing headers first), we instead:
        # - pick the competitor with MOST headings as "master outline"
        # - compare all other competitors against it and list missing headers/parts + FAQ topics.
        # This gives a clean "what to include" outline for the new post.

        # Pick master competitor = most H2 sections
        best_u = None
        best_count = -1
        for u in competitors:
            nodes = (comp_tree_map.get(u) or {}).get("nodes", []) or []
            cnt = len([s for s in section_nodes(nodes, levels=(2,3)) if s.get("level") == 2])
            if cnt > best_count:
                best_count = cnt
                best_u = u

        master_url = best_u or competitors[0]
        master_fr = comp_fr_map[master_url]
        master_nodes = comp_tree_map[master_url]["nodes"]

        # Compare other competitors vs master using SAME OLD logic
        other_urls = [u for u in competitors if u != master_url]

        st.session_state.new_df = compute_gaps_header_first(
            bayut_url=master_url,
            bayut_fr=master_fr,
            bayut_nodes=master_nodes,
            competitors=other_urls,
            comp_fr_map=comp_fr_map,
            comp_tree_map=comp_tree_map,
        )

        # SEO table for New Mode (master + others)
        rows = []
        rm = seo_row_for_page(site_name(master_url), master_url, master_fr, master_nodes, manual_fkw=manual_fkw_new.strip())
        rm["__url"] = master_url
        rows.append(rm)
        for u in other_urls:
            rr = seo_row_for_page(site_name(u), u, comp_fr_map[u], comp_tree_map[u]["nodes"], manual_fkw=manual_fkw_new.strip())
            rr["__url"] = u
            rows.append(rr)

        st.session_state.seo_new_df = pd.DataFrame(rows)

        with st.spinner("Fetching Google UAE ranking (desktop + mobile) + AI visibility via DataForSEO…"):
            seo_enriched, ai_df = enrich_seo_df_with_rank_and_ai(
                st.session_state.seo_new_df,
                manual_query=manual_fkw_new.strip(),
                location_name=DEFAULT_LOCATION_NAME
            )
            st.session_state.seo_new_df = seo_enriched
            st.session_state.ai_new_df = ai_df

        st.session_state.cq_new_df = build_content_quality_table_from_seo(
            seo_df=st.session_state.seo_new_df,
            fr_map_by_url={**comp_fr_map},
            tree_map_by_url={u: comp_tree_map[u] for u in competitors},
            manual_query=manual_fkw_new.strip(),
            location_name=DEFAULT_LOCATION_NAME
        )

    if show_internal_fetch and st.session_state.new_fetch:
        st.sidebar.markdown("### Internal fetch log (New Post Mode)")
        st.sidebar.write(f"Playwright enabled: {PLAYWRIGHT_OK}")
        for u, s in st.session_state.new_fetch:
            st.sidebar.write(u, "—", s)

    # Render results
    if st.session_state.new_df is None or st.session_state.new_df.empty:
        st.info("Run analysis to see results.")
    else:
        st.markdown("<div class='section-pill section-pill-tight'>Content Outline Gaps (vs master competitor)</div>", unsafe_allow_html=True)
        render_table(st.session_state.new_df)

    st.markdown("<div class='section-pill section-pill-tight'>SEO Analysis</div>", unsafe_allow_html=True)
    render_table(st.session_state.seo_new_df, drop_internal_url=True)

    st.markdown("<div class='section-pill section-pill-tight'>AI Visibility (Google AI Overview)</div>", unsafe_allow_html=True)
    render_table(st.session_state.ai_new_df, drop_internal_url=True)

    st.markdown("<div class='section-pill section-pill-tight'>Content Quality</div>", unsafe_allow_html=True)
    render_table(st.session_state.cq_new_df, drop_internal_url=True)
