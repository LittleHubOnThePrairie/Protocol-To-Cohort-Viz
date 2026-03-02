"""Tests for DefineXmlValidator (PTCV-23)."""

from __future__ import annotations

import pandas as pd
import pytest

from ptcv.sdtm.validation.define_xml_validator import DefineXmlValidator
from ptcv.sdtm.validation.models import DefineXmlIssue


# ---------------------------------------------------------------------------
# Helpers to build minimal Define-XML bytes
# ---------------------------------------------------------------------------

_ODM_NS = "http://www.cdisc.org/ns/odm/v1.3"
_DEF_NS = "http://www.cdisc.org/ns/def/v2.1"
_XSI_NS = "http://www.w3.org/2001/XMLSchema-instance"


def _build_define_xml(
    domains: dict[str, list[str]],
    codelist_refs: dict[str, str] | None = None,
) -> bytes:
    """Build a minimal valid Define-XML v2.1 document for testing.

    Args:
        domains: Mapping domain name → list of variable names to declare.
        codelist_refs: Mapping variable name → CodelistOID to add.
    """
    try:
        from lxml import etree as ET
        nsmap = {None: _ODM_NS, "def": _DEF_NS, "xsi": _XSI_NS}
        root = ET.Element(f"{{{_ODM_NS}}}ODM", nsmap=nsmap)
    except ImportError:
        import xml.etree.ElementTree as ET
        ET.register_namespace("", _ODM_NS)
        ET.register_namespace("def", _DEF_NS)
        ET.register_namespace("xsi", _XSI_NS)
        root = ET.Element(f"{{{_ODM_NS}}}ODM")

    root.set("ODMVersion", "1.3.2")
    root.set("FileType", "Snapshot")
    root.set("FileOID", "test")
    root.set("CreationDateTime", "2024-01-01T00:00:00")

    study = ET.SubElement(root, f"{{{_ODM_NS}}}Study", attrib={"OID": "NCT001"})
    gv = ET.SubElement(study, f"{{{_ODM_NS}}}GlobalVariables")
    ET.SubElement(gv, f"{{{_ODM_NS}}}StudyName").text = "Test"
    ET.SubElement(gv, f"{{{_ODM_NS}}}StudyDescription").text = "Test"
    ET.SubElement(gv, f"{{{_ODM_NS}}}ProtocolName").text = "NCT001"

    mdv = ET.SubElement(
        study, f"{{{_ODM_NS}}}MetaDataVersion",
        attrib={
            "OID": "MDV.NCT001",
            "Name": "NCT001 Define-XML v2.1",
            f"{{{_DEF_NS}}}DefineVersion": "2.1.0",
        },
    )

    all_vars: dict[str, str] = {}  # OID → Name

    for domain, variables in domains.items():
        igdef = ET.SubElement(
            mdv, f"{{{_ODM_NS}}}ItemGroupDef",
            attrib={
                "OID": f"IG.{domain}",
                "Name": domain,
                "Repeating": "Yes",
                "IsReferenceData": "Yes",
            },
        )
        for i, var in enumerate(variables, start=1):
            oid = f"IT.{domain}.{var}"
            ET.SubElement(
                igdef, f"{{{_ODM_NS}}}ItemRef",
                attrib={"ItemOID": oid, "Mandatory": "No", "OrderNumber": str(i)},
            )
            all_vars[oid] = var

    for oid, var_name in all_vars.items():
        domain = oid.split(".")[1]
        idef = ET.SubElement(
            mdv, f"{{{_ODM_NS}}}ItemDef",
            attrib={
                "OID": oid,
                "Name": var_name,
                "DataType": "text",
                "Length": "40",
            },
        )
        # Add codelist ref if specified
        if codelist_refs and var_name in codelist_refs:
            ET.SubElement(
                idef, f"{{{_ODM_NS}}}CodelistRef",
                attrib={"CodelistOID": codelist_refs[var_name]},
            )

    try:
        return ET.tostring(
            root, encoding="utf-8", xml_declaration=True, pretty_print=True
        )
    except TypeError:
        return ET.tostring(root, encoding="utf-8", xml_declaration=True)


class TestDefineXmlValidatorNoIssues:
    """Valid define.xml + matching XPT → no issues reported."""

    def test_matching_variables_no_issues(self):
        xml_bytes = _build_define_xml({"TS": ["STUDYID", "DOMAIN", "TSSEQ", "TSPARMCD"]})
        df = pd.DataFrame(columns=["STUDYID", "DOMAIN", "TSSEQ", "TSPARMCD"])
        issues = DefineXmlValidator(xml_bytes, {"TS": df}).validate()
        structural = [i for i in issues if i.issue_type == "structural_mismatch"]
        assert structural == []

    def test_returns_list_of_define_xml_issue(self):
        xml_bytes = _build_define_xml({"TS": ["STUDYID", "DOMAIN"]})
        df = pd.DataFrame(columns=["STUDYID", "DOMAIN"])
        issues = DefineXmlValidator(xml_bytes, {"TS": df}).validate()
        assert isinstance(issues, list)
        assert all(isinstance(i, DefineXmlIssue) for i in issues)


class TestStructuralMismatch:
    """Variables in define.xml absent from XPT → structural_mismatch."""

    def test_extra_var_in_define_xml(self):
        # define.xml declares STUDYID, DOMAIN, TSSEQ, TSPARMCD
        # XPT only has STUDYID, DOMAIN → TSSEQ and TSPARMCD are mismatched
        xml_bytes = _build_define_xml(
            {"TS": ["STUDYID", "DOMAIN", "TSSEQ", "TSPARMCD"]}
        )
        df = pd.DataFrame(columns=["STUDYID", "DOMAIN"])  # missing TSSEQ, TSPARMCD
        issues = DefineXmlValidator(xml_bytes, {"TS": df}).validate()
        structural = [i for i in issues if i.issue_type == "structural_mismatch"]
        assert len(structural) >= 2
        missing_vars = {i.variable for i in structural}
        assert "TSSEQ" in missing_vars
        assert "TSPARMCD" in missing_vars

    def test_structural_mismatch_domain_is_correct(self):
        xml_bytes = _build_define_xml({"TV": ["STUDYID", "DOMAIN", "VISITNUM", "EXTRAVAR"]})
        df = pd.DataFrame(columns=["STUDYID", "DOMAIN", "VISITNUM"])
        issues = DefineXmlValidator(xml_bytes, {"TV": df}).validate()
        structural = [i for i in issues if i.issue_type == "structural_mismatch"]
        assert any(i.domain == "TV" and i.variable == "EXTRAVAR" for i in structural)

    def test_no_mismatch_when_dataset_not_loaded(self):
        # define.xml declares TV variables but we only pass TS in datasets
        xml_bytes = _build_define_xml(
            {"TS": ["STUDYID"], "TV": ["STUDYID", "EXTRAVAR"]}
        )
        ts_df = pd.DataFrame(columns=["STUDYID"])
        # TV not passed → no mismatch can be detected for TV (no reference)
        issues = DefineXmlValidator(xml_bytes, {"TS": ts_df}).validate()
        ts_structural = [
            i for i in issues
            if i.issue_type == "structural_mismatch" and i.domain == "TS"
        ]
        assert ts_structural == []


class TestCodelistRefCheck:
    """Codelist references that don't follow CL.* convention are flagged."""

    def test_invalid_codelist_oid_flagged(self):
        xml_bytes = _build_define_xml(
            {"TS": ["STUDYID", "TSPARMCD"]},
            codelist_refs={"TSPARMCD": "BADPREFIX.UNKNOWN"},
        )
        df = pd.DataFrame(columns=["STUDYID", "TSPARMCD"])
        issues = DefineXmlValidator(xml_bytes, {"TS": df}).validate()
        codelist_issues = [i for i in issues if i.issue_type == "codelist_ref"]
        assert codelist_issues

    def test_valid_cdisc_ct_prefix_not_flagged(self):
        xml_bytes = _build_define_xml(
            {"TS": ["STUDYID", "TSPARMCD"]},
            codelist_refs={"TSPARMCD": "CL.C66737.TSPARMCD"},
        )
        df = pd.DataFrame(columns=["STUDYID", "TSPARMCD"])
        issues = DefineXmlValidator(xml_bytes, {"TS": df}).validate()
        codelist_issues = [i for i in issues if i.issue_type == "codelist_ref"]
        assert codelist_issues == []


class TestParseError:
    """Malformed XML returns a parse_error issue."""

    def test_invalid_xml_returns_parse_error(self):
        bad_bytes = b"<not valid xml"
        issues = DefineXmlValidator(bad_bytes, {}).validate()
        assert len(issues) == 1
        assert issues[0].issue_type == "parse_error"

    def test_empty_bytes_returns_parse_error(self):
        issues = DefineXmlValidator(b"", {}).validate()
        assert issues[0].issue_type == "parse_error"


class TestGherkinScenario:
    """PTCV-23 Scenario: Define-XML validated against actual dataset structure."""

    def test_variable_absent_from_xpt_is_reported(self):
        """Variable in define.xml absent from XPT → structural_mismatch."""
        xml_bytes = _build_define_xml(
            {"TS": ["STUDYID", "DOMAIN", "GHOST_VAR"]}
        )
        df = pd.DataFrame(columns=["STUDYID", "DOMAIN"])
        issues = DefineXmlValidator(xml_bytes, {"TS": df}).validate()
        mismatches = [i for i in issues if i.issue_type == "structural_mismatch"]
        assert any(i.variable == "GHOST_VAR" for i in mismatches)

    def test_mismatches_have_description(self, ts_df):
        xml_bytes = _build_define_xml(
            {"TS": list(ts_df.columns) + ["MISSING_VAR"]}
        )
        issues = DefineXmlValidator(xml_bytes, {"TS": ts_df}).validate()
        for issue in issues:
            assert issue.description, "Every issue must have a description"
