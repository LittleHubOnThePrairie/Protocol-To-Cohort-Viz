"""Validation results and domain checklist panel (PTCV-272).

Renders required domain checklist with pass/fail indicators and
validation findings grouped by severity with remediation guidance.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from ptcv.sdtm.validation.required_domain_checker import DomainCheckResult
    from ptcv.sdtm.validation.models import ValidationResult


def render_domain_checklist(
    domain_check: Optional["DomainCheckResult"],
) -> None:
    """Render the required domain checklist in Streamlit.

    Shows always-required domains with green check / red X,
    and conditional domains with amber warning if missing.

    Args:
        domain_check: DomainCheckResult from ValidationResult.domain_check.
    """
    try:
        import pandas as pd
        import streamlit as st
    except ImportError:
        return

    if domain_check is None:
        return

    st.subheader("Required Domain Checklist")

    cols = st.columns(3)
    cols[0].metric("Errors", domain_check.error_count)
    cols[1].metric("Warnings", domain_check.warning_count)
    cols[2].metric(
        "Status",
        "Pass" if domain_check.passed else "Fail",
    )

    if not domain_check.findings:
        st.success("All required domains are present.")
        return

    rows = []
    for finding in domain_check.findings:
        if finding.severity == "Error":
            icon = "❌"
        else:
            icon = "⚠️"

        rows.append({
            "": icon,
            "Domain": f"{finding.domain_code} ({finding.domain_name})",
            "Severity": finding.severity,
            "Message": finding.message,
            "Trigger": finding.trigger,
        })

    st.dataframe(
        pd.DataFrame(rows),
        use_container_width=True,
        hide_index=True,
    )


def _attr(obj: Any, name: str, default: Any = None) -> Any:
    """Get attribute from object or dict (PTCV-289).

    The query pipeline caches validation results as plain dicts,
    but ValidationService.validate() returns a ValidationResult
    dataclass. This helper supports both access patterns.
    """
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def render_validation_results(
    validation_result: Optional["ValidationResult"],
) -> None:
    """Render validation findings grouped by severity.

    Shows P21 issues, TCG completeness, schedule validation,
    and Define-XML issues with remediation guidance.

    Accepts both ValidationResult dataclass and dict (from query
    pipeline cache) via _attr() accessor (PTCV-289).

    Args:
        validation_result: ValidationResult or dict from cache.
    """
    try:
        import pandas as pd
        import streamlit as st
    except ImportError:
        return

    if validation_result is None:
        return

    st.subheader("Validation Results")

    # Summary metrics
    cols = st.columns(4)
    cols[0].metric(
        "P21 Errors", _attr(validation_result, "p21_error_count", 0),
    )
    cols[1].metric(
        "P21 Warnings", _attr(validation_result, "p21_warning_count", 0),
    )
    cols[2].metric(
        "TCG",
        "Pass" if _attr(validation_result, "tcg_passed", True) else "Fail",
    )
    cols[3].metric(
        "Schedule",
        "Feasible"
        if _attr(validation_result, "schedule_feasible", True)
        else "Infeasible",
    )

    # P21 issues by severity
    p21_issues = _attr(validation_result, "p21_issues", [])
    if p21_issues:
        errors = [
            i for i in p21_issues
            if _attr(i, "severity") == "Error"
        ]
        warnings = [
            i for i in p21_issues
            if _attr(i, "severity") == "Warning"
        ]
        notices = [
            i for i in p21_issues
            if _attr(i, "severity") == "Notice"
        ]

        if errors:
            with st.expander(
                f"Errors ({len(errors)})", expanded=True,
            ):
                _render_issue_table(errors)

        if warnings:
            with st.expander(
                f"Warnings ({len(warnings)})", expanded=False,
            ):
                _render_issue_table(warnings)

        if notices:
            with st.expander(
                f"Notices ({len(notices)})", expanded=False,
            ):
                _render_issue_table(notices)
    else:
        st.success("No P21 validation issues found.")

    # TCG missing parameters
    tcg_missing = _attr(validation_result, "tcg_missing_params", [])
    if tcg_missing:
        with st.expander(
            f"Missing TS Parameters ({len(tcg_missing)})",
            expanded=False,
        ):
            st.markdown(
                "Required by FDA TCG v5.9 Appendix B:"
            )
            for param in tcg_missing:
                st.markdown(f"- `{param}`")

    # Schedule issues
    schedule_issues = _attr(validation_result, "schedule_issues", [])
    if schedule_issues:
        with st.expander(
            f"Schedule Issues ({len(schedule_issues)})",
            expanded=False,
        ):
            import pandas as pd
            sched_rows = [
                {
                    "Rule": _attr(i, "rule_id", ""),
                    "Severity": _attr(i, "severity", ""),
                    "Description": _attr(i, "description", ""),
                    "Remediation": _attr(i, "remediation_guidance", ""),
                }
                for i in schedule_issues
            ]
            st.dataframe(
                pd.DataFrame(sched_rows),
                use_container_width=True,
                hide_index=True,
            )


def _render_issue_table(issues: list) -> None:
    """Render a list of P21Issue (or dicts) as a dataframe."""
    import pandas as pd
    import streamlit as st

    rows = [
        {
            "Rule": _attr(i, "rule_id", ""),
            "Domain": _attr(i, "domain", ""),
            "Variable": _attr(i, "variable", ""),
            "Description": _attr(i, "description", ""),
            "Remediation": _attr(i, "remediation_guidance", ""),
        }
        for i in issues
    ]
    st.dataframe(
        pd.DataFrame(rows),
        use_container_width=True,
        hide_index=True,
    )


__all__ = [
    "render_domain_checklist",
    "render_validation_results",
]
