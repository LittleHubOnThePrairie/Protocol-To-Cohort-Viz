"""Observation domain specs and EX domain display panel (PTCV-271).

Renders observation domain specs (VS, LB, EG, PE) and EX domain
intervention details from SdtmGenerationResult in Streamlit.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from ptcv.sdtm.domain_spec_builder import DomainSpecResult
    from ptcv.sdtm.ex_domain_builder import ExDomainSpec


def render_observation_specs(
    domain_specs: Optional["DomainSpecResult"],
) -> None:
    """Render observation domain specs (VS, LB, EG, PE) in Streamlit.

    Args:
        domain_specs: DomainSpecResult from SdtmGenerationResult.domain_specs.
    """
    try:
        import pandas as pd
        import streamlit as st
    except ImportError:
        return

    if domain_specs is None:
        st.info(
            "No observation domain specs available. "
            "Run SDTM generation with an SoA table to produce them."
        )
        return

    st.subheader("Observation Domain Specs")
    st.caption(
        f"{len(domain_specs.specs)} domains mapped from "
        f"{domain_specs.total_assessments} SoA assessments "
        f"({domain_specs.mapped_count} mapped, "
        f"{len(domain_specs.unmapped)} unmapped)"
    )

    # Domain spec cards
    for spec in domain_specs.specs:
        with st.expander(
            f"{spec.domain_code} — {spec.domain_name} "
            f"({spec.test_count} test codes)",
            expanded=False,
        ):
            # Variables table
            var_data = [
                {
                    "Variable": v.name,
                    "Label": v.label,
                    "Type": v.type,
                    "Required": "Yes" if v.required else "No",
                }
                for v in spec.variables
            ]
            if var_data:
                st.markdown("**CDISC Variables**")
                st.dataframe(
                    pd.DataFrame(var_data),
                    use_container_width=True,
                    hide_index=True,
                )

            # Test codes table
            if spec.test_codes:
                tc_data = [
                    {
                        "TESTCD": tc.testcd,
                        "Test Name": tc.test,
                        "Source Assessment": tc.source_assessment,
                        "Visits": ", ".join(tc.visit_schedule) if tc.visit_schedule else "",
                    }
                    for tc in spec.test_codes
                ]
                st.markdown("**Mapped Test Codes**")
                st.dataframe(
                    pd.DataFrame(tc_data),
                    use_container_width=True,
                    hide_index=True,
                )

            # Specimen type (LB only)
            if spec.specimen_type:
                st.markdown(f"**Specimen Type:** {spec.specimen_type}")

            # Source assessments
            if spec.source_assessments:
                st.markdown(
                    f"**Source Assessments:** "
                    f"{', '.join(spec.source_assessments)}"
                )

    # Unmapped assessments
    if domain_specs.unmapped:
        st.divider()
        st.markdown("**Unmapped Assessments** (review required)")
        unmapped_data = [
            {
                "Assessment": u.assessment_name,
                "Suggested Domain": u.suggested_domain,
                "Reason": u.reason,
            }
            for u in domain_specs.unmapped
        ]
        st.dataframe(
            pd.DataFrame(unmapped_data),
            use_container_width=True,
            hide_index=True,
        )


def render_ex_domain_spec(
    ex_spec: Optional["ExDomainSpec"],
) -> None:
    """Render EX domain intervention details in Streamlit.

    Args:
        ex_spec: ExDomainSpec from SdtmGenerationResult.ex_domain_spec.
    """
    try:
        import pandas as pd
        import streamlit as st
    except ImportError:
        return

    if ex_spec is None:
        return

    st.subheader("EX Domain — Exposure")
    st.caption(
        f"{ex_spec.treatment_count} intervention(s) identified"
    )

    if ex_spec.interventions:
        intv_data = [
            {
                "Treatment": i.name,
                "Type": i.intervention_type,
                "Dose": f"{i.dose} {i.dose_unit}" if i.dose else "",
                "Route": i.route,
                "Frequency": i.frequency,
                "Arms": ", ".join(i.arm_labels) if i.arm_labels else "",
            }
            for i in ex_spec.interventions
        ]
        st.dataframe(
            pd.DataFrame(intv_data),
            use_container_width=True,
            hide_index=True,
        )

    # Arm-treatment mapping
    if ex_spec.arm_treatment_map:
        st.markdown("**Arm → Treatment Mapping**")
        arm_data = [
            {"Arm": arm, "Treatments": ", ".join(treatments)}
            for arm, treatments in ex_spec.arm_treatment_map.items()
        ]
        st.dataframe(
            pd.DataFrame(arm_data),
            use_container_width=True,
            hide_index=True,
        )


__all__ = [
    "render_ex_domain_spec",
    "render_observation_specs",
]
