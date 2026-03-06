"""Streamlit UI panel for mock SDTM data display (PTCV-122).

Renders mock data generation results with per-domain DataFrame
previews, validation status badges, and CSV download.

Risk tier: LOW — UI component, no side effects.
"""

from __future__ import annotations

import io
import zipfile
from dataclasses import dataclass
from typing import Any

import pandas as pd
import streamlit as st


# ---------------------------------------------------------------------------
# Data models for panel input
# ---------------------------------------------------------------------------


@dataclass
class DomainValidationResult:
    """Validation result for a single domain.

    Attributes:
        domain_code: Two-letter SDTM domain code.
        passed: True if all expectations passed.
        total_expectations: Number of expectations evaluated.
        failed_expectations: List of failed expectation details.
    """

    domain_code: str
    passed: bool
    total_expectations: int
    failed_expectations: list[dict[str, Any]]


@dataclass
class MockPipelineResult:
    """Result container for the mock data panel.

    Attributes:
        domain_dataframes: Mapping of domain code to DataFrame.
        validation_results: Per-domain validation results.
        run_id: Generation run UUID.
        num_subjects: Number of subjects generated.
        synthesizer_type: Synthesizer backend used.
    """

    domain_dataframes: dict[str, pd.DataFrame]
    validation_results: dict[str, DomainValidationResult]
    run_id: str
    num_subjects: int
    synthesizer_type: str


# ---------------------------------------------------------------------------
# CSV ZIP export
# ---------------------------------------------------------------------------


def _build_csv_zip(
    domain_dfs: dict[str, pd.DataFrame],
) -> bytes:
    """Build a ZIP archive containing one CSV per domain.

    Args:
        domain_dfs: Mapping of domain code to DataFrame.

    Returns:
        ZIP file contents as bytes.
    """
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for domain_code, df in sorted(domain_dfs.items()):
            csv_bytes = df.to_csv(index=False).encode("utf-8")
            zf.writestr(f"{domain_code}.csv", csv_bytes)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Render functions
# ---------------------------------------------------------------------------


def render_mock_data_panel(
    pipeline_result: MockPipelineResult,
) -> None:
    """Render the mock data panel with domain previews and validation.

    Args:
        pipeline_result: Result from mock data generation pipeline.
    """
    domain_dfs = pipeline_result.domain_dataframes
    validation = pipeline_result.validation_results

    # --- Summary metrics ---
    total_rows = sum(len(df) for df in domain_dfs.values())
    n_domains = len(domain_dfs)

    if validation:
        n_passed = sum(
            1 for v in validation.values() if v.passed
        )
        n_failed = len(validation) - n_passed
        pass_rate = (
            n_passed / len(validation) * 100
            if validation
            else 0
        )
    else:
        n_passed = 0
        n_failed = 0
        pass_rate = 0.0

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Domains", n_domains)
    col2.metric("Total Rows", f"{total_rows:,}")
    col3.metric("Subjects", pipeline_result.num_subjects)

    if validation:
        col4.metric(
            "Validation",
            f"{pass_rate:.0f}% pass",
        )
    else:
        col4.metric("Validation", "N/A")

    st.caption(
        f"Run: {pipeline_result.run_id[:8]}... | "
        f"Synthesizer: {pipeline_result.synthesizer_type}"
    )

    # --- Download button ---
    zip_bytes = _build_csv_zip(domain_dfs)
    st.download_button(
        label="Download All Domains (CSV ZIP)",
        data=zip_bytes,
        file_name="mock_sdtm_data.zip",
        mime="application/zip",
        key="btn_download_mock_csv",
    )

    st.divider()

    # --- Per-domain expandable sections ---
    for domain_code in sorted(domain_dfs.keys()):
        df = domain_dfs[domain_code]
        val = validation.get(domain_code)

        # Build header with pass/fail badge.
        if val is not None:
            badge = "PASS" if val.passed else "FAIL"
            header = (
                f"{domain_code} — {len(df):,} rows — "
                f"{badge}"
            )
        else:
            header = f"{domain_code} — {len(df):,} rows"

        with st.expander(header, expanded=False):
            # DataFrame preview (first 20 rows).
            st.dataframe(
                df.head(20),
                use_container_width=True,
                hide_index=True,
            )

            if len(df) > 20:
                st.caption(
                    f"Showing 20 of {len(df):,} rows."
                )

            # Validation details.
            if val is not None:
                if val.passed:
                    st.success(
                        f"All {val.total_expectations} "
                        f"expectations passed."
                    )
                else:
                    st.error(
                        f"{len(val.failed_expectations)} of "
                        f"{val.total_expectations} "
                        f"expectations failed."
                    )

                    # Show failed expectations.
                    for fail in val.failed_expectations:
                        col_name = fail.get(
                            "column", "unknown"
                        )
                        expectation = fail.get(
                            "expectation", "unknown"
                        )
                        observed = fail.get(
                            "observed", "N/A"
                        )
                        st.markdown(
                            f"- **{col_name}**: "
                            f"{expectation} "
                            f"(observed: `{observed}`)"
                        )


# ---------------------------------------------------------------------------
# Sidebar configuration controls
# ---------------------------------------------------------------------------


def render_mock_data_sidebar() -> dict[str, Any]:
    """Render sidebar configuration controls for mock data generation.

    Returns:
        Dict with configuration values:
        ``num_subjects``, ``synthesizer_type``.
    """
    st.sidebar.subheader("Mock Data Config")

    num_subjects = st.sidebar.slider(
        "Number of subjects",
        min_value=5,
        max_value=100,
        value=10,
        step=5,
        key="slider_mock_subjects",
    )

    synthesizer_type = st.sidebar.radio(
        "Synthesizer",
        options=["gaussian_copula", "ctgan"],
        index=0,
        key="radio_mock_synth",
    )

    return {
        "num_subjects": num_subjects,
        "synthesizer_type": synthesizer_type,
    }
