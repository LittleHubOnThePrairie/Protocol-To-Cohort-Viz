"""Streamlit comparison viewer: original vs extracted side-by-side (PTCV-148).

Reads from the analysis SQLite data store (PTCV-147) to display
original protocol text alongside query pipeline extracted content.
Supports cross-protocol corpus statistics, filtering, and
before/after run comparison.

Standalone usage::

    streamlit run src/ptcv/ui/pages/comparison_viewer.py

Pure-Python helpers are separated from Streamlit rendering for
testability — only the ``render_*`` and ``main`` functions import
Streamlit.
"""

from __future__ import annotations

import html as html_mod
import os
from pathlib import Path
from typing import Any

# ====================================================================
# Pure-Python data helpers (no Streamlit imports — fully testable)
# ====================================================================

# Confidence thresholds per PTCV-148 requirements.
# Different from confidence_badge.py (0.85/0.70) which aligns with
# SectionMatcher tiers; these are for the comparison viewer display.
CONF_HIGH = 0.80
CONF_MODERATE = 0.60


def viewer_color(confidence: float | None) -> str:
    """Return CSS color name for a confidence score.

    Args:
        confidence: Value 0.0-1.0, or ``None`` for missing data.

    Returns:
        ``'green'``, ``'goldenrod'``, ``'red'``, or ``'grey'``.
    """
    if confidence is None:
        return "grey"
    if confidence >= CONF_HIGH:
        return "green"
    if confidence >= CONF_MODERATE:
        return "goldenrod"
    return "red"


def viewer_label(confidence: float | None) -> str:
    """Return tier label for a confidence score.

    Args:
        confidence: Value 0.0-1.0, or ``None`` for missing data.

    Returns:
        ``'HIGH'``, ``'MODERATE'``, ``'LOW'``, or ``'MISSING'``.
    """
    if confidence is None:
        return "MISSING"
    if confidence >= CONF_HIGH:
        return "HIGH"
    if confidence >= CONF_MODERATE:
        return "MODERATE"
    return "LOW"


# ------------------------------------------------------------------
# Corpus summary
# ------------------------------------------------------------------

def compute_corpus_summary(
    section_stats: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Format ``get_section_stats()`` output for display.

    Adds color coding, hit-rate percentage, and score variance.

    Args:
        section_stats: Output from ``AnalysisStore.get_section_stats()``.

    Returns:
        List of dicts with keys: ``ich_section_code``,
        ``ich_section_name``, ``avg_confidence``, ``color``,
        ``label``, ``hit_rate_pct``, ``variance``,
        ``protocol_count``, ``total_protocols``.
    """
    result: list[dict[str, Any]] = []
    for s in section_stats:
        avg = s.get("avg_boosted", 0.0)
        mn = s.get("min_boosted", 0.0)
        mx = s.get("max_boosted", 0.0)
        hit_rate = s.get("hit_rate", 0.0)
        result.append({
            "ich_section_code": s.get("ich_section_code", ""),
            "ich_section_name": s.get("ich_section_name", ""),
            "avg_confidence": round(avg, 3),
            "color": viewer_color(avg),
            "label": viewer_label(avg),
            "hit_rate_pct": round(hit_rate * 100, 1),
            "variance": round(mx - mn, 3),
            "protocol_count": s.get("protocol_count", 0),
            "total_protocols": s.get("total_protocols", 0),
        })
    return result


# ------------------------------------------------------------------
# Comparison pair enrichment
# ------------------------------------------------------------------

def enrich_comparison_pairs(
    pairs: list[dict[str, Any]],
    section_matches: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Join comparison pairs with section match metadata.

    Adds ``similarity_score``, ``boosted_score``, ``match_method``,
    ``auto_mapped``, ``color``, and ``label`` to each pair.

    Args:
        pairs: From ``AnalysisStore.get_comparison_pairs()``.
        section_matches: From protocol summary ``section_matches``.

    Returns:
        New list with enriched dicts (originals unmodified).
    """
    # Build lookup: ich_section_code → best section_match
    match_lookup: dict[str, dict[str, Any]] = {}
    for sm in section_matches:
        code = sm.get("ich_section_code", "")
        existing = match_lookup.get(code)
        if existing is None or sm.get("boosted_score", 0.0) > existing.get(
            "boosted_score", 0.0
        ):
            match_lookup[code] = sm

    enriched: list[dict[str, Any]] = []
    for pair in pairs:
        code = pair.get("ich_section_code", "")
        sm = match_lookup.get(code, {})
        conf = pair.get("confidence")
        enriched.append({
            **pair,
            "similarity_score": sm.get("similarity_score", 0.0),
            "boosted_score": sm.get("boosted_score", 0.0),
            "match_method": sm.get("match_method", ""),
            "auto_mapped": bool(sm.get("auto_mapped", False)),
            "color": viewer_color(conf),
            "label": viewer_label(conf),
        })
    return enriched


# ------------------------------------------------------------------
# Filtering
# ------------------------------------------------------------------

def filter_pairs(
    pairs: list[dict[str, Any]],
    *,
    confidence_min: float = 0.0,
    confidence_max: float = 1.0,
    match_quality: str | None = None,
    ich_section: str | None = None,
    populated_only: bool = False,
    unpopulated_only: bool = False,
) -> list[dict[str, Any]]:
    """Filter enriched comparison pairs by criteria.

    Args:
        pairs: Enriched comparison pairs.
        confidence_min: Minimum confidence (inclusive).
        confidence_max: Maximum confidence (inclusive).
        match_quality: If set, keep only this match_quality value.
        ich_section: If set, keep only this ICH section code.
        populated_only: Keep only pairs with extracted_text.
        unpopulated_only: Keep only pairs without extracted_text.

    Returns:
        Filtered list.
    """
    result: list[dict[str, Any]] = []
    for p in pairs:
        conf = p.get("confidence")
        if conf is not None:
            if conf < confidence_min or conf > confidence_max:
                continue

        if match_quality and p.get("match_quality") != match_quality:
            continue

        if ich_section and p.get("ich_section_code") != ich_section:
            continue

        has_text = bool(p.get("extracted_text", "").strip())
        if populated_only and not has_text:
            continue
        if unpopulated_only and has_text:
            continue

        result.append(p)
    return result


# ------------------------------------------------------------------
# Run comparison formatting
# ------------------------------------------------------------------

def format_run_comparison(
    comparison: dict[str, Any],
) -> list[dict[str, Any]]:
    """Format ``compare_runs()`` output for tabular display.

    Adds color coding based on improvement/regression status.

    Args:
        comparison: Output from ``AnalysisStore.compare_runs()``.

    Returns:
        List of dicts with keys: ``ich_section``, ``run_a_score``,
        ``run_b_score``, ``delta``, ``status``, ``color``.
    """
    rows: list[dict[str, Any]] = []
    for s in comparison.get("sections", []):
        status = s.get("status", "unchanged")
        if status == "improved":
            color = "green"
        elif status == "regressed":
            color = "red"
        else:
            color = "grey"
        rows.append({
            "ich_section": s.get("ich_section_code", ""),
            "run_a_score": round(s.get("run_a_avg_score", 0.0), 3),
            "run_b_score": round(s.get("run_b_avg_score", 0.0), 3),
            "delta": round(s.get("delta", 0.0), 3),
            "status": status,
            "color": color,
        })
    return rows


# ------------------------------------------------------------------
# Navigation helpers
# ------------------------------------------------------------------

def get_protocol_index(
    protocols: list[dict[str, Any]],
    current_nct_id: str,
) -> int:
    """Find the index of a protocol in the list.

    Returns:
        0-based index, or 0 if not found.
    """
    for i, p in enumerate(protocols):
        if p.get("nct_id") == current_nct_id:
            return i
    return 0


def get_next_protocol(
    protocols: list[dict[str, Any]],
    current_nct_id: str,
) -> str | None:
    """Get the NCT ID of the next protocol.

    Returns:
        Next NCT ID, or ``None`` if at end of list.
    """
    idx = get_protocol_index(protocols, current_nct_id)
    if idx + 1 < len(protocols):
        return protocols[idx + 1].get("nct_id")
    return None


def get_prev_protocol(
    protocols: list[dict[str, Any]],
    current_nct_id: str,
) -> str | None:
    """Get the NCT ID of the previous protocol.

    Returns:
        Previous NCT ID, or ``None`` if at start of list.
    """
    idx = get_protocol_index(protocols, current_nct_id)
    if idx > 0:
        return protocols[idx - 1].get("nct_id")
    return None


# ------------------------------------------------------------------
# HTML comparison builder
# ------------------------------------------------------------------

def build_comparison_html(
    pairs: list[dict[str, Any]],
    height: int = 600,
) -> str:
    """Build side-by-side HTML with synced scrolling and color coding.

    Each section shows a confidence badge, match metadata overlay,
    and original vs extracted text. Missing sections get grey
    backgrounds.

    Uses the CSS/JS sync-scroll pattern from ``section_align.py``.

    Args:
        pairs: Enriched comparison pairs from ``enrich_comparison_pairs()``.
        height: Panel height in pixels.

    Returns:
        Complete HTML string with embedded CSS and JavaScript.
    """
    if not pairs:
        return "<p>No comparison pairs available.</p>"

    nav_items: list[str] = []
    left_items: list[str] = []
    right_items: list[str] = []

    for p in pairs:
        code = html_mod.escape(p.get("ich_section_code", "?"))
        conf = p.get("confidence")
        color = p.get("color", viewer_color(conf))
        label = p.get("label", viewer_label(conf))
        method = html_mod.escape(p.get("match_method", ""))
        sim = p.get("similarity_score", 0.0)
        boosted = p.get("boosted_score", 0.0)
        quality = html_mod.escape(p.get("match_quality", ""))

        orig = p.get("original_text", "") or ""
        extr = p.get("extracted_text", "") or ""
        esc_orig = html_mod.escape(orig) if orig.strip() else "(no original content)"
        esc_extr = html_mod.escape(extr) if extr.strip() else "(no extracted content)"

        missing_bg = "background:#f5f5f5;" if not extr.strip() else ""

        badge = (
            f'<span style="color:{color};font-weight:600;">'
            f'{conf:.2f} ({label})</span>' if conf is not None
            else f'<span style="color:grey;">MISSING</span>'
        )

        meta_line = (
            f'<div class="meta">Method: {method} | '
            f'Sim: {sim:.2f} | Boost: {boosted:.2f} | '
            f'Quality: {quality}</div>'
        )

        # Nav item with color dot
        nav_items.append(
            f'<div class="nav-item" onclick="jumpTo(\'{code}\')">'
            f'<span style="color:{color};">&#9679;</span> '
            f'<b>{code}</b></div>'
        )

        # Left panel — original
        left_items.append(
            f'<div id="left-{code}" class="section">'
            f'<h4>{code} {badge}</h4>'
            f'{meta_line}'
            f'<pre>{esc_orig}</pre></div>'
        )

        # Right panel — extracted
        right_items.append(
            f'<div id="right-{code}" class="section" '
            f'style="{missing_bg}">'
            f'<h4>{code} {badge}</h4>'
            f'{meta_line}'
            f'<pre>{esc_extr}</pre></div>'
        )

    nav_html = "\n".join(nav_items)
    left_html = "\n".join(left_items)
    right_html = "\n".join(right_items)

    return _HTML_TEMPLATE.format(
        height=height,
        nav_html=nav_html,
        left_html=left_html,
        right_html=right_html,
    )


_HTML_TEMPLATE = """\
<style>
  .cv-container {{
    display: flex;
    gap: 6px;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI",
                 Roboto, sans-serif;
    font-size: 13px;
  }}
  .cv-nav {{
    width: 160px;
    min-width: 160px;
    overflow-y: auto;
    max-height: {height}px;
    border: 1px solid #ddd;
    border-radius: 4px;
    padding: 4px;
    background: #fafafa;
  }}
  .nav-item {{
    padding: 4px 6px;
    cursor: pointer;
    border-radius: 3px;
    margin-bottom: 2px;
    font-size: 12px;
  }}
  .nav-item:hover, .nav-item.active {{
    background: #e3f2fd;
  }}
  .cv-scroll {{
    flex: 1;
    overflow-y: auto;
    max-height: {height}px;
    border: 1px solid #ddd;
    border-radius: 4px;
    padding: 8px;
  }}
  .cv-scroll h4 {{
    position: sticky;
    top: 0;
    background: #fff;
    margin: 0 0 4px 0;
    padding: 4px 0;
    border-bottom: 1px solid #eee;
  }}
  .section {{
    margin-bottom: 16px;
  }}
  .section pre {{
    white-space: pre-wrap;
    word-wrap: break-word;
    font-family: inherit;
    margin: 4px 0;
    line-height: 1.5;
  }}
  .meta {{
    font-size: 11px;
    color: #666;
    margin-bottom: 4px;
  }}
  .cv-label {{
    font-weight: 600;
    text-align: center;
    padding: 4px;
    background: #f5f5f5;
    border-radius: 4px 4px 0 0;
    border: 1px solid #ddd;
    border-bottom: none;
    font-size: 12px;
  }}
</style>

<div class="cv-container">
  <div class="cv-nav" id="cv-nav">
    <div style="font-weight:600; padding:4px 6px; margin-bottom:4px;">
      Sections
    </div>
    {nav_html}
  </div>
  <div style="flex:1; display:flex; flex-direction:column;">
    <div class="cv-label">Original Protocol Text</div>
    <div class="cv-scroll" id="cv-left" onscroll="cvSync('cv-left')">
      {left_html}
    </div>
  </div>
  <div style="flex:1; display:flex; flex-direction:column;">
    <div class="cv-label">Extracted Content</div>
    <div class="cv-scroll" id="cv-right" onscroll="cvSync('cv-right')">
      {right_html}
    </div>
  </div>
</div>

<script>
  let cvSyncing = false;

  function cvSync(source) {{
    if (cvSyncing) return;
    cvSyncing = true;
    const src = document.getElementById(source);
    const tgt = document.getElementById(
      source === 'cv-left' ? 'cv-right' : 'cv-left'
    );
    const ratio = src.scrollTop / (src.scrollHeight - src.clientHeight || 1);
    tgt.scrollTop = ratio * (tgt.scrollHeight - tgt.clientHeight);
    cvHighlight(source);
    setTimeout(() => {{ cvSyncing = false; }}, 50);
  }}

  function jumpTo(code) {{
    const left = document.getElementById('left-' + code);
    const right = document.getElementById('right-' + code);
    if (left) left.scrollIntoView({{ behavior: 'smooth', block: 'start' }});
    if (right) right.scrollIntoView({{ behavior: 'smooth', block: 'start' }});
    document.querySelectorAll('.nav-item').forEach(
      el => el.classList.remove('active')
    );
    document.querySelectorAll('.nav-item').forEach(el => {{
      if (el.textContent.trim().includes(code)) el.classList.add('active');
    }});
  }}

  function cvHighlight(panelId) {{
    const panel = document.getElementById(panelId);
    const sections = panel.querySelectorAll('.section');
    let closest = null;
    let minDist = Infinity;
    sections.forEach(sec => {{
      const rect = sec.getBoundingClientRect();
      const panelRect = panel.getBoundingClientRect();
      const dist = Math.abs(rect.top - panelRect.top);
      if (dist < minDist) {{
        minDist = dist;
        closest = sec;
      }}
    }});
    if (closest) {{
      const code = closest.id.replace(panelId.replace('cv-', '') + '-', '');
      document.querySelectorAll('.nav-item').forEach(el => {{
        el.classList.remove('active');
        if (el.textContent.trim().includes(code)) el.classList.add('active');
      }});
    }}
  }}
</script>
"""


# ====================================================================
# Streamlit rendering layer
# ====================================================================

# Default database path, overridable via environment variable.
_DEFAULT_DB = Path(
    os.environ.get("PTCV_ANALYSIS_DB", "data/analysis/results.db")
)

# Session state key prefix to avoid collisions with main app.
_SK = "cv_"

_STATE_DEFAULTS: dict[str, Any] = {
    f"{_SK}run_id": None,
    f"{_SK}nct_id": None,
    f"{_SK}ich_section": None,
    f"{_SK}compare_run_id": None,
    f"{_SK}conf_min": 0.0,
    f"{_SK}conf_max": 1.0,
    f"{_SK}match_quality": None,
    f"{_SK}populated_filter": "all",
}


def _init_state() -> None:
    """Initialize session state keys with defaults if missing."""
    import streamlit as st  # noqa: local import

    for key, default in _STATE_DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = default


def render_sidebar(store: Any) -> None:
    """Render sidebar navigation: run, protocol, section, filters.

    Modifies ``st.session_state`` with selected values.
    """
    import streamlit as st

    st.sidebar.header("Comparison Viewer")

    # --- Batch run selector ---
    runs = store.list_runs()
    if not runs:
        st.sidebar.warning(
            "No batch runs found. Run the batch runner first:\n\n"
            "```\npython -m ptcv.analysis.batch_runner "
            "--protocol-dir data/protocols/clinicaltrials\n```"
        )
        return

    run_ids = [r["run_id"] for r in runs]
    run_labels = [
        f"{r['run_id']} ({r.get('timestamp', '')[:10]})"
        for r in runs
    ]
    current_run = st.session_state[f"{_SK}run_id"]
    run_index = (
        run_ids.index(current_run) if current_run in run_ids else 0
    )
    selected_run_label = st.sidebar.selectbox(
        "Batch Run",
        run_labels,
        index=run_index,
    )
    selected_run = run_ids[run_labels.index(selected_run_label)]
    st.session_state[f"{_SK}run_id"] = selected_run

    # --- Protocol selector ---
    protocols = store.list_protocols(selected_run, status="pass")
    if not protocols:
        st.sidebar.info("No passing protocols in this run.")
        return

    nct_ids = [p["nct_id"] for p in protocols]
    current_nct = st.session_state[f"{_SK}nct_id"]
    nct_index = (
        nct_ids.index(current_nct) if current_nct in nct_ids else 0
    )
    selected_nct = st.sidebar.selectbox(
        "Protocol (NCT ID)",
        nct_ids,
        index=nct_index,
    )
    st.session_state[f"{_SK}nct_id"] = selected_nct

    # --- ICH section filter ---
    sections = store.list_ich_sections(selected_run)
    section_options = ["All"] + [
        f"{s['ich_section_code']} - {s['ich_section_name']}"
        for s in sections
    ]
    current_sec = st.session_state[f"{_SK}ich_section"]
    sec_index = 0
    if current_sec:
        for i, opt in enumerate(section_options):
            if opt.startswith(current_sec):
                sec_index = i
                break
    selected_sec_label = st.sidebar.selectbox(
        "ICH Section",
        section_options,
        index=sec_index,
    )
    if selected_sec_label == "All":
        st.session_state[f"{_SK}ich_section"] = None
    else:
        st.session_state[f"{_SK}ich_section"] = (
            selected_sec_label.split(" - ")[0]
        )

    # --- Filters ---
    st.sidebar.markdown("---")
    st.sidebar.subheader("Filters")

    conf_range = st.sidebar.slider(
        "Confidence Range",
        0.0, 1.0,
        (
            st.session_state[f"{_SK}conf_min"],
            st.session_state[f"{_SK}conf_max"],
        ),
        0.05,
    )
    st.session_state[f"{_SK}conf_min"] = conf_range[0]
    st.session_state[f"{_SK}conf_max"] = conf_range[1]

    quality_options = ["All", "good", "partial", "poor", "missing"]
    quality_sel = st.sidebar.selectbox(
        "Match Quality", quality_options,
    )
    st.session_state[f"{_SK}match_quality"] = (
        None if quality_sel == "All" else quality_sel
    )

    pop_options = ["All", "Populated only", "Unpopulated only"]
    pop_sel = st.sidebar.radio("Content Filter", pop_options)
    st.session_state[f"{_SK}populated_filter"] = pop_sel.lower().split()[0]

    # --- Compare mode ---
    st.sidebar.markdown("---")
    st.sidebar.subheader("Run Comparison")
    compare_options = ["None"] + [
        rid for rid in run_ids if rid != selected_run
    ]
    compare_sel = st.sidebar.selectbox(
        "Compare with", compare_options,
    )
    st.session_state[f"{_SK}compare_run_id"] = (
        None if compare_sel == "None" else compare_sel
    )


def render_corpus_summary(store: Any) -> None:
    """Render cross-protocol per-section statistics table."""
    import streamlit as st

    run_id = st.session_state.get(f"{_SK}run_id")
    if not run_id:
        return

    stats = store.get_section_stats(run_id)
    if not stats:
        st.info("No section statistics available for this run.")
        return

    summary = compute_corpus_summary(stats)

    with st.expander("Corpus Summary (per ICH section)", expanded=False):
        header = (
            "| Section | Name | Avg Conf | Tier | "
            "Hit Rate | Variance | Protocols |\n"
            "|---|---|---|---|---|---|---|\n"
        )
        rows_md = []
        for s in summary:
            rows_md.append(
                f"| {s['ich_section_code']} "
                f"| {s['ich_section_name'][:30]} "
                f"| {s['avg_confidence']:.3f} "
                f"| {s['label']} "
                f"| {s['hit_rate_pct']:.1f}% "
                f"| {s['variance']:.3f} "
                f"| {s['protocol_count']}/{s['total_protocols']} |"
            )
        st.markdown(header + "\n".join(rows_md))

        # Coverage distribution
        dist = store.get_coverage_distribution(run_id)
        if dist.get("protocol_count", 0) > 0:
            st.markdown(
                f"**Corpus:** {dist['protocol_count']} protocols | "
                f"Avg coverage: {dist['avg_coverage_pct']:.1f}% | "
                f"Avg confidence: {dist['avg_confidence']:.3f}"
            )


def render_comparison_view(store: Any) -> None:
    """Render the main side-by-side comparison with navigation."""
    import streamlit as st

    run_id = st.session_state.get(f"{_SK}run_id")
    nct_id = st.session_state.get(f"{_SK}nct_id")
    if not run_id or not nct_id:
        st.info("Select a batch run and protocol from the sidebar.")
        return

    # Navigation buttons
    protocols = store.list_protocols(run_id, status="pass")
    col_prev, col_info, col_next = st.columns([1, 4, 1])

    with col_prev:
        prev_nct = get_prev_protocol(protocols, nct_id)
        if st.button(
            "Prev", disabled=prev_nct is None, use_container_width=True,
        ):
            st.session_state[f"{_SK}nct_id"] = prev_nct
            st.rerun()

    with col_info:
        idx = get_protocol_index(protocols, nct_id)
        st.markdown(
            f"**{nct_id}** &mdash; "
            f"Protocol {idx + 1} of {len(protocols)}"
        )

    with col_next:
        next_nct = get_next_protocol(protocols, nct_id)
        if st.button(
            "Next", disabled=next_nct is None, use_container_width=True,
        ):
            st.session_state[f"{_SK}nct_id"] = next_nct
            st.rerun()

    # Load comparison data
    ich_filter = st.session_state.get(f"{_SK}ich_section")
    pairs = store.get_comparison_pairs(run_id, nct_id, ich_filter)

    if not pairs:
        st.warning(
            f"No comparison pairs for {nct_id}"
            + (f" section {ich_filter}" if ich_filter else "")
            + "."
        )
        return

    # Enrich with section match metadata
    summary = store.get_protocol_summary(run_id, nct_id)
    section_matches = summary.get("section_matches", [])
    enriched = enrich_comparison_pairs(pairs, section_matches)

    # Apply filters
    pop_filter = st.session_state.get(f"{_SK}populated_filter", "all")
    filtered = filter_pairs(
        enriched,
        confidence_min=st.session_state.get(f"{_SK}conf_min", 0.0),
        confidence_max=st.session_state.get(f"{_SK}conf_max", 1.0),
        match_quality=st.session_state.get(f"{_SK}match_quality"),
        populated_only=(pop_filter == "populated"),
        unpopulated_only=(pop_filter == "unpopulated"),
    )

    if not filtered:
        st.info("No pairs match the current filters.")
        return

    # Metrics row
    conf_values = [
        p["confidence"] for p in filtered
        if p.get("confidence") is not None
    ]
    avg_conf = sum(conf_values) / len(conf_values) if conf_values else 0.0
    populated = sum(
        1 for p in filtered if p.get("extracted_text", "").strip()
    )

    m1, m2, m3 = st.columns(3)
    m1.metric("Sections", len(filtered))
    m2.metric("Populated", f"{populated}/{len(filtered)}")
    m3.metric("Avg Confidence", f"{avg_conf:.3f}")

    # Render HTML comparison
    comparison_html = build_comparison_html(filtered)
    st.components.v1.html(comparison_html, height=650, scrolling=False)


def render_run_diff(store: Any) -> None:
    """Render before/after comparison between two batch runs."""
    import streamlit as st

    run_a = st.session_state.get(f"{_SK}run_id")
    run_b = st.session_state.get(f"{_SK}compare_run_id")
    if not run_a or not run_b:
        return

    st.subheader(f"Run Comparison: {run_a} vs {run_b}")

    comparison = store.compare_runs(run_a, run_b)
    rows = format_run_comparison(comparison)

    if not rows:
        st.info("No overlapping sections between runs.")
        return

    # Aggregate metrics
    cov_delta = comparison.get("coverage_delta", 0.0)
    conf_delta = comparison.get("confidence_delta", 0.0)
    improved = sum(1 for r in rows if r["status"] == "improved")
    regressed = sum(1 for r in rows if r["status"] == "regressed")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Coverage Delta", f"{cov_delta:+.1f}%")
    c2.metric("Confidence Delta", f"{conf_delta:+.4f}")
    c3.metric("Improved", improved)
    c4.metric("Regressed", regressed)

    # Table
    header = (
        "| Section | Run A | Run B | Delta | Status |\n"
        "|---|---|---|---|---|\n"
    )
    table_rows = []
    for r in rows:
        delta_str = f"{r['delta']:+.3f}"
        table_rows.append(
            f"| {r['ich_section']} "
            f"| {r['run_a_score']:.3f} "
            f"| {r['run_b_score']:.3f} "
            f"| {delta_str} "
            f"| {r['status']} |"
        )
    st.markdown(header + "\n".join(table_rows))


def main() -> None:
    """Streamlit page entry point."""
    import streamlit as st

    st.set_page_config(
        page_title="PTCV Comparison Viewer",
        page_icon="🔍",
        layout="wide",
    )

    _init_state()

    # Open data store
    db_path = _DEFAULT_DB
    if not db_path.exists():
        st.error(
            f"Analysis database not found at `{db_path}`.\n\n"
            "Run the batch runner first:\n\n"
            "```\npython -m ptcv.analysis.batch_runner "
            "--protocol-dir data/protocols/clinicaltrials "
            "--output-db data/analysis/results.db\n```"
        )
        return

    from ptcv.analysis.data_store import AnalysisStore

    store = AnalysisStore(db_path)

    try:
        st.title("Protocol Comparison Viewer")
        render_sidebar(store)

        compare_run = st.session_state.get(f"{_SK}compare_run_id")
        if compare_run:
            render_run_diff(store)
        else:
            render_corpus_summary(store)
            render_comparison_view(store)
    finally:
        store.close()


if __name__ == "__main__":
    main()
