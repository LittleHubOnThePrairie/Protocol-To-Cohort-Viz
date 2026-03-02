"""Define-XML v2.1 structural validator (PTCV-23).

Compares the variables declared in define.xml against the columns
actually present in each XPT DataFrame. Reports:
- Structural mismatches: variables in define.xml absent from the XPT.
- Codelist reference issues: CodelistRef OIDs in define.xml that point to
  unknown CDISC CT codelists (basic format check).

Risk tier: MEDIUM — regulatory submission structural conformance; no patient data.

Regulatory references:
- CDISC Define-XML v2.1 Specification
- SDTMIG v3.3 §7.4 — Define-XML must be consistent with submitted XPT
- FDA Study Data Technical Conformance Guide v5.0 §4.1.2
"""

from __future__ import annotations

import re

import pandas as pd

from .models import DefineXmlIssue

try:
    from lxml import etree as ET  # type: ignore[import-untyped]
    _LXML = True
except ImportError:
    import xml.etree.ElementTree as ET  # type: ignore[assignment]
    _LXML = False

_ODM_NS = "http://www.cdisc.org/ns/odm/v1.3"
_DEF_NS = "http://www.cdisc.org/ns/def/v2.1"

# CDISC CT codelist OID prefix pattern expected in Define-XML
# e.g. "CL.C66737.INTTYPE" or custom "CL.BLIND"
_CODELIST_OID_RE = re.compile(r"^CL\.")


class DefineXmlValidator:
    """Validates a Define-XML v2.1 document against actual XPT column sets.

    Args:
        define_xml_bytes: Raw bytes of the define.xml file.
        datasets: Mapping from domain name (e.g. "TS") to DataFrame
            representing the actual XPT content.
    """

    def __init__(
        self,
        define_xml_bytes: bytes,
        datasets: dict[str, pd.DataFrame],
    ) -> None:
        self._xml_bytes = define_xml_bytes
        self._datasets = {k.upper(): v for k, v in datasets.items()}

    def validate(self) -> list[DefineXmlIssue]:
        """Run structural and codelist reference validation.

        Returns:
            List of DefineXmlIssue records. Empty list means no issues.
        [PTCV-23 Scenario: Define-XML validated against actual dataset structure]
        """
        issues: list[DefineXmlIssue] = []
        try:
            root = ET.fromstring(self._xml_bytes)
        except Exception as exc:
            issues.append(
                DefineXmlIssue(
                    issue_type="parse_error",
                    domain="",
                    variable="",
                    description=f"Define-XML could not be parsed: {exc}",
                )
            )
            return issues

        issues.extend(self._check_structural_mismatches(root))
        issues.extend(self._check_codelist_refs(root))
        return issues

    # ------------------------------------------------------------------
    # Structural mismatch check
    # ------------------------------------------------------------------

    def _check_structural_mismatches(
        self, root: ET.Element
    ) -> list[DefineXmlIssue]:
        """Find variables declared in define.xml absent from the XPT."""
        issues: list[DefineXmlIssue] = []

        # Build map: domain → set of ItemDef OIDs declared
        # ItemGroupDef Name attribute is the domain; each ItemRef/@ItemOID
        # references an ItemDef; ItemDef/@Name is the variable name.

        # Parse ItemDefs: OID → Name
        item_oid_to_name: dict[str, str] = {}
        for item_def in root.iter(f"{{{_ODM_NS}}}ItemDef"):
            oid = item_def.get("OID", "")
            name = item_def.get("Name", "")
            if oid and name:
                item_oid_to_name[oid] = name

        # Iterate ItemGroupDefs
        for ig_def in root.iter(f"{{{_ODM_NS}}}ItemGroupDef"):
            domain = ig_def.get("Name", "").upper()
            if not domain:
                continue
            df = self._datasets.get(domain)
            xpt_cols: set[str] = set(df.columns) if df is not None else set()

            for item_ref in ig_def.iter(f"{{{_ODM_NS}}}ItemRef"):
                item_oid = item_ref.get("ItemOID", "")
                var_name = item_oid_to_name.get(item_oid, "")
                if not var_name:
                    continue

                if df is not None and var_name not in xpt_cols:
                    issues.append(
                        DefineXmlIssue(
                            issue_type="structural_mismatch",
                            domain=domain,
                            variable=var_name,
                            description=(
                                f"Variable {var_name} declared in define.xml "
                                f"for domain {domain} is absent from the XPT "
                                "dataset columns."
                            ),
                        )
                    )

        return issues

    # ------------------------------------------------------------------
    # Codelist reference check
    # ------------------------------------------------------------------

    def _check_codelist_refs(self, root: ET.Element) -> list[DefineXmlIssue]:
        """Find CodelistRef OIDs that do not follow CDISC CT naming."""
        issues: list[DefineXmlIssue] = []

        # Collect all CodelistDef OIDs actually defined in this document
        defined_codelists: set[str] = set()
        for cl in root.iter(f"{{{_DEF_NS}}}CodelistDef"):
            oid = cl.get("OID", "")
            if oid:
                defined_codelists.add(oid)
        # Also check the standard ODM namespace
        for cl in root.iter(f"{{{_ODM_NS}}}CodeList"):
            oid = cl.get("OID", "")
            if oid:
                defined_codelists.add(oid)

        # Find ItemDefs that reference a codelist
        for item_def in root.iter(f"{{{_ODM_NS}}}ItemDef"):
            var_name = item_def.get("Name", "")
            # Look for CodelistRef child
            cl_ref = item_def.find(f"{{{_ODM_NS}}}CodelistRef")
            if cl_ref is None:
                cl_ref = item_def.find(f"{{{_DEF_NS}}}CodelistRef")
            if cl_ref is None:
                continue
            ref_oid = cl_ref.get("CodelistOID", "")
            if not ref_oid:
                continue

            # If the referenced OID is not defined in this document
            # and does not match the expected CDISC CT pattern (CL.*),
            # flag it.
            if (
                ref_oid not in defined_codelists
                and not _CODELIST_OID_RE.match(ref_oid)
            ):
                # Determine domain from OID pattern IT.{DOMAIN}.{VAR}
                domain = ""
                for ig_def in root.iter(f"{{{_ODM_NS}}}ItemGroupDef"):
                    for item_ref in ig_def.iter(f"{{{_ODM_NS}}}ItemRef"):
                        if item_ref.get("ItemOID", "").endswith(f".{var_name}"):
                            domain = ig_def.get("Name", "").upper()
                            break
                    if domain:
                        break

                issues.append(
                    DefineXmlIssue(
                        issue_type="codelist_ref",
                        domain=domain,
                        variable=var_name,
                        description=(
                            f"CodelistRef OID {ref_oid!r} for variable "
                            f"{var_name} is not defined in this document "
                            "and does not match CDISC CT naming convention "
                            "(expected prefix 'CL.')."
                        ),
                    )
                )

        return issues
