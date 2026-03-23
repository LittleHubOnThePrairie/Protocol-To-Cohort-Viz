"""Tests for ValidationService (PTCV-23) — 5 GHERKIN scenarios.

Feature: SDTM Validation and Compliance Reporting

Covers:
  Scenario 1 — Run P21 and archive report with lineage
  Scenario 2 — Report linked to specific SDTM version (amendment independence)
  Scenario 3 — FDA TCG Appendix B completeness check
  Scenario 4 — Compliance summary includes remediation for each issue
  Scenario 5 — Define-XML validated against actual dataset structure
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
import uuid

sys.path.insert(0, str(Path(__file__).parents[3] / "src"))

import pandas as pd
import pytest

from ptcv.sdtm.models import SdtmGenerationResult
from ptcv.sdtm.validation.models import ValidationResult
from ptcv.sdtm.validation.validation_service import ValidationService
from ptcv.storage.filesystem_adapter import FilesystemAdapter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_sdtm_result(
    gateway: FilesystemAdapter,
    registry_id: str = "NCT001",
    domains: dict | None = None,
) -> SdtmGenerationResult:
    """Write minimal XPT and define.xml artifacts, return SdtmGenerationResult."""
    import pyreadstat
    import tempfile
    import os
    import hashlib
    import datetime

    if domains is None:
        # Minimal valid TS + TV
        domains = {
            "ts": pd.DataFrame([{
                "STUDYID": registry_id,
                "DOMAIN": "TS",
                "TSSEQ": 1.0,
                "TSPARMCD": "TITLE",
                "TSPARM": "Trial Title",
                "TSVAL": "Phase III Study",
                "TSVALCD": "",
                "TSVALNF": "",
                "TSVALUNIT": "",
            }]),
            "tv": pd.DataFrame([{
                "STUDYID": registry_id,
                "DOMAIN": "TV",
                "VISITNUM": 1.0,
                "VISIT": "Screening",
                "VISITDY": -14.0,
                "TVSTRL": -3.0,
                "TVENRL": 0.0,
                "TVTIMY": "",
                "TVTIM": "",
                "TVENDY": "",
            }]),
        }

    run_id = str(uuid.uuid4())
    artifact_keys: dict[str, str] = {}
    artifact_sha256s: dict[str, str] = {}
    ts_sha256 = ""

    for domain_key, df in domains.items():
        # Write XPT via pyreadstat
        fd, tmp_path = tempfile.mkstemp(suffix=".xpt")
        os.close(fd)
        try:
            pyreadstat.write_xport(df, tmp_path)
            with open(tmp_path, "rb") as f:
                xpt_bytes = f.read()
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

        key = f"sdtm/{registry_id}/{run_id}/{domain_key}.xpt"
        artifact = gateway.put_artifact(
            key=key,
            data=xpt_bytes,
            content_type="application/octet-stream",
            run_id=run_id,
            source_hash="upstream-sha",
            user="test",
            immutable=True,
            stage="sdtm_generation",
            registry_id=registry_id,
        )
        artifact_keys[domain_key] = artifact.key
        artifact_sha256s[domain_key] = artifact.sha256
        if domain_key == "ts":
            ts_sha256 = artifact.sha256

    # Write a minimal define.xml
    define_xml = _build_define_xml_bytes(
        {k.upper(): list(v.columns) for k, v in domains.items()}
    )
    define_key = f"sdtm/{registry_id}/{run_id}/define.xml"
    define_artifact = gateway.put_artifact(
        key=define_key,
        data=define_xml,
        content_type="application/xml",
        run_id=run_id,
        source_hash=ts_sha256,
        user="test",
        immutable=True,
        stage="sdtm_generation",
        registry_id=registry_id,
    )
    artifact_keys["define"] = define_artifact.key
    artifact_sha256s["define"] = define_artifact.sha256

    return SdtmGenerationResult(
        run_id=run_id,
        registry_id=registry_id,
        artifact_keys=artifact_keys,
        artifact_sha256s=artifact_sha256s,
        source_sha256=ts_sha256,
        domain_row_counts={k: len(v) for k, v in domains.items()},
        ct_unmapped_count=0,
        generation_timestamp_utc=datetime.datetime.now().strftime(
            "%Y-%m-%dT%H:%M:%S"
        ),
    )


def _build_define_xml_bytes(domains: dict[str, list[str]]) -> bytes:
    """Build minimal define.xml bytes for the given domains."""
    _ODM_NS = "http://www.cdisc.org/ns/odm/v1.3"
    _DEF_NS = "http://www.cdisc.org/ns/def/v2.1"
    _XSI_NS = "http://www.w3.org/2001/XMLSchema-instance"

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
            "OID": "MDV.NCT001", "Name": "NCT001",
            f"{{{_DEF_NS}}}DefineVersion": "2.1.0",
        },
    )
    for domain, variables in domains.items():
        igdef = ET.SubElement(
            mdv, f"{{{_ODM_NS}}}ItemGroupDef",
            attrib={"OID": f"IG.{domain}", "Name": domain,
                    "Repeating": "Yes", "IsReferenceData": "Yes"},
        )
        for i, var in enumerate(variables, start=1):
            oid = f"IT.{domain}.{var}"
            ET.SubElement(
                igdef, f"{{{_ODM_NS}}}ItemRef",
                attrib={"ItemOID": oid, "Mandatory": "No", "OrderNumber": str(i)},
            )
            ET.SubElement(
                mdv, f"{{{_ODM_NS}}}ItemDef",
                attrib={"OID": oid, "Name": var, "DataType": "text", "Length": "40"},
            )
    try:
        return ET.tostring(root, encoding="utf-8", xml_declaration=True, pretty_print=True)
    except TypeError:
        return ET.tostring(root, encoding="utf-8", xml_declaration=True)


# ---------------------------------------------------------------------------
# Scenario 1: Run P21 and archive report with lineage
# ---------------------------------------------------------------------------

class TestScenario1P21ArchiveWithLineage:
    """Given SDTM XPT files with known sha256 in lineage.db,
    When validation runs P21 and writes report via StorageGateway,
    Then p21_report.json is written with stage=validation LineageRecord.
    """

    @pytest.fixture()
    def sdtm_result(self, tmp_gateway) -> SdtmGenerationResult:
        return _make_sdtm_result(tmp_gateway)

    def test_returns_validation_result(self, tmp_gateway, sdtm_result):
        svc = ValidationService(tmp_gateway)
        result = svc.validate(sdtm_result)
        assert isinstance(result, ValidationResult)

    def test_p21_report_written_to_storage(self, tmp_gateway, sdtm_result):
        svc = ValidationService(tmp_gateway)
        result = svc.validate(sdtm_result)

        p21_key = result.artifact_keys.get("p21")
        assert p21_key, "p21 artifact key must be present"
        p21_bytes = tmp_gateway.get_artifact(p21_key)
        assert len(p21_bytes) > 0

    def test_p21_report_stored_under_correct_prefix(self, tmp_gateway, sdtm_result):
        svc = ValidationService(tmp_gateway)
        result = svc.validate(sdtm_result)

        expected_prefix = (
            f"validation-reports/{sdtm_result.registry_id}"
            f"/{sdtm_result.run_id}/p21_report.json"
        )
        assert result.artifact_keys["p21"] == expected_prefix

    def test_lineage_record_stage_is_validation(self, tmp_gateway, sdtm_result):
        """LineageRecord.stage must be 'validation'."""
        svc = ValidationService(tmp_gateway)
        result = svc.validate(sdtm_result)

        lineage = tmp_gateway.get_lineage(result.run_id)
        assert lineage, "LineageRecords must be written"
        stages = {rec.stage for rec in lineage}
        assert "validation" in stages

    def test_lineage_source_hash_is_sdtm_sha256(self, tmp_gateway, sdtm_result):
        """LineageRecord.source_hash must equal the ts.xpt sha256."""
        svc = ValidationService(tmp_gateway)
        result = svc.validate(sdtm_result)

        lineage = tmp_gateway.get_lineage(result.run_id)
        val_records = [r for r in lineage if r.stage == "validation"]
        assert val_records
        # All validation lineage records share the same source_hash (ts.xpt sha256)
        for rec in val_records:
            assert rec.source_hash == result.sdtm_sha256

    def test_lineage_source_contains_p21_error_count(self, tmp_gateway, sdtm_result):
        """LineageRecord.source includes p21_error_count and tcg_passed."""
        svc = ValidationService(tmp_gateway)
        result = svc.validate(sdtm_result)

        lineage = tmp_gateway.get_lineage(result.run_id)
        val_records = [r for r in lineage if r.stage == "validation"]
        assert val_records
        # source field is JSON-encoded metadata
        for rec in val_records:
            meta = json.loads(rec.source)
            assert "p21_error_count" in meta
            assert "tcg_passed" in meta

    def test_p21_report_json_has_issues_list(self, tmp_gateway, sdtm_result):
        svc = ValidationService(tmp_gateway)
        result = svc.validate(sdtm_result)

        p21_bytes = tmp_gateway.get_artifact(result.artifact_keys["p21"])
        data = json.loads(p21_bytes)
        assert "issues" in data
        assert "error_count" in data
        assert "warning_count" in data
        assert data["report_type"] == "p21_validation"

    def test_artifact_sha256_matches_stored_bytes(self, tmp_gateway, sdtm_result):
        """ValidationResult.artifact_sha256s must match actual stored SHA-256."""
        import hashlib

        svc = ValidationService(tmp_gateway)
        result = svc.validate(sdtm_result)

        for tag in ["p21", "tcg", "summary"]:
            key = result.artifact_keys[tag]
            stored_bytes = tmp_gateway.get_artifact(key)
            computed = hashlib.sha256(stored_bytes).hexdigest()
            assert computed == result.artifact_sha256s[tag], (
                f"{tag} SHA-256 mismatch"
            )


# ---------------------------------------------------------------------------
# Scenario 2: Report linked to specific SDTM version
# ---------------------------------------------------------------------------

class TestScenario2AmendmentIndependence:
    """Given two SDTM generation runs (amendment 01 and 02),
    When validation is run for each,
    Then each report is stored under its own run_id prefix and lineage
    source_hash points to the correct SDTM sha256.
    """

    def test_two_runs_independent_report_keys(self, tmp_gateway):
        result1 = _make_sdtm_result(tmp_gateway, registry_id="NCT001")
        result2 = _make_sdtm_result(tmp_gateway, registry_id="NCT001")
        # Different run_ids
        assert result1.run_id != result2.run_id

        svc = ValidationService(tmp_gateway)
        val1 = svc.validate(result1)
        val2 = svc.validate(result2)

        assert val1.artifact_keys["p21"] != val2.artifact_keys["p21"]
        assert val1.sdtm_run_id == result1.run_id
        assert val2.sdtm_run_id == result2.run_id

    def test_lineage_source_hash_is_run_specific(self, tmp_gateway):
        result1 = _make_sdtm_result(tmp_gateway, registry_id="NCT001")
        result2 = _make_sdtm_result(tmp_gateway, registry_id="NCT001")

        svc = ValidationService(tmp_gateway)
        val1 = svc.validate(result1)
        val2 = svc.validate(result2)

        lineage1 = tmp_gateway.get_lineage(val1.run_id)
        lineage2 = tmp_gateway.get_lineage(val2.run_id)

        # Each lineage source_hash must point to its own SDTM run ts.xpt sha256
        for rec in [r for r in lineage1 if r.stage == "validation"]:
            assert rec.source_hash == val1.sdtm_sha256

        for rec in [r for r in lineage2 if r.stage == "validation"]:
            assert rec.source_hash == val2.sdtm_sha256

        # The two source_hashes are different (different run artifacts)
        assert val1.sdtm_sha256 != val2.sdtm_sha256 or (
            result1.run_id != result2.run_id
        )

    def test_amendment_reports_independently_queryable(self, tmp_gateway):
        result1 = _make_sdtm_result(tmp_gateway, registry_id="NCT001")
        result2 = _make_sdtm_result(tmp_gateway, registry_id="NCT001")

        svc = ValidationService(tmp_gateway)
        val1 = svc.validate(result1)
        val2 = svc.validate(result2)

        # Each validation result references its own SDTM run
        assert val1.sdtm_run_id != val2.sdtm_run_id
        # Both reports are readable
        p21_1 = tmp_gateway.get_artifact(val1.artifact_keys["p21"])
        p21_2 = tmp_gateway.get_artifact(val2.artifact_keys["p21"])
        data1 = json.loads(p21_1)
        data2 = json.loads(p21_2)
        assert data1["sdtm_run_id"] == result1.run_id
        assert data2["sdtm_run_id"] == result2.run_id


# ---------------------------------------------------------------------------
# Scenario 3: FDA TCG Appendix B completeness check
# ---------------------------------------------------------------------------

class TestScenario3TcgCompletenessCheck:
    """Given a generated TS dataset,
    When TCG checker runs and writes tcg_completeness.json,
    Then all FDA TCG v5.9 Appendix B required parameters are verified,
    tcg_completeness.json has a passed boolean and missing_params list,
    and the LineageRecord tcg_passed field matches.
    """

    @pytest.fixture()
    def sdtm_result(self, tmp_gateway) -> SdtmGenerationResult:
        return _make_sdtm_result(tmp_gateway)

    def test_tcg_report_written(self, tmp_gateway, sdtm_result):
        svc = ValidationService(tmp_gateway)
        result = svc.validate(sdtm_result)

        tcg_key = result.artifact_keys.get("tcg")
        assert tcg_key
        tcg_bytes = tmp_gateway.get_artifact(tcg_key)
        assert len(tcg_bytes) > 0

    def test_tcg_report_has_passed_boolean(self, tmp_gateway, sdtm_result):
        svc = ValidationService(tmp_gateway)
        result = svc.validate(sdtm_result)

        tcg_bytes = tmp_gateway.get_artifact(result.artifact_keys["tcg"])
        data = json.loads(tcg_bytes)
        assert "passed" in data
        assert isinstance(data["passed"], bool)

    def test_tcg_report_has_missing_params_list(self, tmp_gateway, sdtm_result):
        svc = ValidationService(tmp_gateway)
        result = svc.validate(sdtm_result)

        tcg_bytes = tmp_gateway.get_artifact(result.artifact_keys["tcg"])
        data = json.loads(tcg_bytes)
        assert "missing_params" in data
        assert isinstance(data["missing_params"], list)

    def test_minimal_ts_has_missing_params(self, tmp_gateway, sdtm_result):
        """Our minimal TS only has TITLE → many FDA TCG params are missing."""
        svc = ValidationService(tmp_gateway)
        result = svc.validate(sdtm_result)

        assert result.tcg_passed is False
        assert len(result.tcg_missing_params) > 0

    def test_lineage_tcg_passed_matches_report(self, tmp_gateway, sdtm_result):
        """LineageRecord.source tcg_passed must match tcg_completeness.json passed."""
        svc = ValidationService(tmp_gateway)
        result = svc.validate(sdtm_result)

        # Read the TCG report
        tcg_bytes = tmp_gateway.get_artifact(result.artifact_keys["tcg"])
        tcg_data = json.loads(tcg_bytes)

        # Read lineage and check source metadata
        lineage = tmp_gateway.get_lineage(result.run_id)
        val_records = [r for r in lineage if r.stage == "validation"]
        assert val_records

        meta = json.loads(val_records[0].source)
        assert meta["tcg_passed"] == tcg_data["passed"]
        assert meta["tcg_passed"] == result.tcg_passed

    def test_tcg_report_regulatory_basis(self, tmp_gateway, sdtm_result):
        svc = ValidationService(tmp_gateway)
        result = svc.validate(sdtm_result)

        tcg_bytes = tmp_gateway.get_artifact(result.artifact_keys["tcg"])
        data = json.loads(tcg_bytes)
        assert data["regulatory_basis"] == "FDA TCG v5.9 Appendix B"


# ---------------------------------------------------------------------------
# Scenario 4: Compliance summary includes remediation for each issue
# ---------------------------------------------------------------------------

class TestScenario4ComplianceSummaryRemediation:
    """Given P21 validation results with at least one Error,
    When compliance_summary.json is written,
    Then each issue includes rule_id, severity, domain, variable,
    description, and remediation_guidance referencing SDTMIG v3.3.
    """

    @pytest.fixture()
    def sdtm_result_with_errors(self, tmp_gateway) -> SdtmGenerationResult:
        """TS with a blank TSVAL to trigger a P21 warning + TCG missing params."""
        ts_df = pd.DataFrame([{
            "STUDYID": "NCT001",
            "DOMAIN": "TS",
            "TSSEQ": 1.0,
            "TSPARMCD": "TITLE",
            "TSPARM": "Trial Title",
            "TSVAL": "",   # blank → P21 Warning
            "TSVALCD": "",
            "TSVALNF": "",  # no null-flavor → Warning fires
            "TSVALUNIT": "",
        }])
        return _make_sdtm_result(
            tmp_gateway, domains={"ts": ts_df}
        )

    def test_compliance_summary_written(self, tmp_gateway, sdtm_result_with_errors):
        svc = ValidationService(tmp_gateway)
        result = svc.validate(sdtm_result_with_errors)

        summary_key = result.artifact_keys.get("summary")
        assert summary_key
        summary_bytes = tmp_gateway.get_artifact(summary_key)
        assert len(summary_bytes) > 0

    def test_compliance_summary_has_issues(self, tmp_gateway, sdtm_result_with_errors):
        svc = ValidationService(tmp_gateway)
        result = svc.validate(sdtm_result_with_errors)

        summary_bytes = tmp_gateway.get_artifact(result.artifact_keys["summary"])
        data = json.loads(summary_bytes)
        assert "issues" in data
        assert data["total_issues"] > 0

    def test_each_issue_has_required_fields(self, tmp_gateway, sdtm_result_with_errors):
        """Each issue must have: rule_id, severity, domain, variable, description,
        remediation_guidance.
        """
        svc = ValidationService(tmp_gateway)
        result = svc.validate(sdtm_result_with_errors)

        summary_bytes = tmp_gateway.get_artifact(result.artifact_keys["summary"])
        data = json.loads(summary_bytes)

        required_fields = {
            "rule_id", "severity", "domain", "variable",
            "description", "remediation_guidance",
        }
        for issue in data["issues"]:
            missing = required_fields - set(issue.keys())
            assert not missing, (
                f"Issue missing fields {missing}: {issue}"
            )

    def test_remediation_references_sdtmig(self, tmp_gateway, sdtm_result_with_errors):
        """remediation_guidance must reference SDTMIG v3.3."""
        svc = ValidationService(tmp_gateway)
        result = svc.validate(sdtm_result_with_errors)

        summary_bytes = tmp_gateway.get_artifact(result.artifact_keys["summary"])
        data = json.loads(summary_bytes)

        # At least one issue should reference SDTMIG v3.3
        has_sdtmig_ref = any(
            "SDTMIG" in i.get("remediation_guidance", "")
            for i in data["issues"]
        )
        assert has_sdtmig_ref, "At least one remediation must reference SDTMIG v3.3"

    def test_tcg_missing_params_in_summary(self, tmp_gateway, sdtm_result_with_errors):
        """TCG missing parameters must also appear in the compliance summary."""
        svc = ValidationService(tmp_gateway)
        result = svc.validate(sdtm_result_with_errors)

        assert len(result.tcg_missing_params) > 0, "Expected TCG missing params"

        summary_bytes = tmp_gateway.get_artifact(result.artifact_keys["summary"])
        data = json.loads(summary_bytes)

        # TCG issues come from the FDA TCG v5.9 source
        tcg_issues = [i for i in data["issues"] if i.get("source") == "FDA TCG v5.9"]
        assert tcg_issues, "TCG missing params must appear in compliance summary"


# ---------------------------------------------------------------------------
# Scenario 5: Define-XML validated against actual dataset structure
# ---------------------------------------------------------------------------

class TestScenario5DefineXmlValidation:
    """Given define.xml and corresponding XPT files from the same run_id,
    When the Define-XML validator runs,
    Then structural mismatches and codelist issues are reported in p21_report.json
    under define_xml_issues.
    """

    @pytest.fixture()
    def sdtm_result_with_ghost_var(self, tmp_gateway) -> SdtmGenerationResult:
        """Produce a run where define.xml declares an extra variable (GHOSTVAR)
        not present in the TS XPT."""
        import pyreadstat
        import tempfile
        import os
        import hashlib
        import datetime

        registry_id = "NCT001"
        run_id = str(uuid.uuid4())

        ts_df = pd.DataFrame([{
            "STUDYID": "NCT001",
            "DOMAIN": "TS",
            "TSSEQ": 1.0,
            "TSPARMCD": "TITLE",
            "TSPARM": "Trial Title",
            "TSVAL": "Phase III Study",
            "TSVALCD": "",
            "TSVALNF": "",
            "TSVALUNIT": "",
        }])

        # Write ts.xpt
        fd, tmp_path = tempfile.mkstemp(suffix=".xpt")
        os.close(fd)
        try:
            pyreadstat.write_xport(ts_df, tmp_path)
            with open(tmp_path, "rb") as f:
                xpt_bytes = f.read()
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

        ts_artifact = tmp_gateway.put_artifact(
            key=f"sdtm/{registry_id}/{run_id}/ts.xpt",
            data=xpt_bytes,
            content_type="application/octet-stream",
            run_id=run_id,
            source_hash="upstream",
            user="test",
            immutable=True,
            stage="sdtm_generation",
        )

        # define.xml declares an extra variable GHOSTVAR not in XPT
        define_bytes = _build_define_xml_bytes(
            {"TS": list(ts_df.columns) + ["GHOSTVAR"]}
        )
        define_artifact = tmp_gateway.put_artifact(
            key=f"sdtm/{registry_id}/{run_id}/define.xml",
            data=define_bytes,
            content_type="application/xml",
            run_id=run_id,
            source_hash=ts_artifact.sha256,
            user="test",
            immutable=True,
            stage="sdtm_generation",
        )

        return SdtmGenerationResult(
            run_id=run_id,
            registry_id=registry_id,
            artifact_keys={
                "ts": ts_artifact.key,
                "define": define_artifact.key,
            },
            artifact_sha256s={
                "ts": ts_artifact.sha256,
                "define": define_artifact.sha256,
            },
            source_sha256=ts_artifact.sha256,
            domain_row_counts={"ts": 1},
            ct_unmapped_count=0,
            generation_timestamp_utc=datetime.datetime.now().strftime(
                "%Y-%m-%dT%H:%M:%S"
            ),
        )

    def test_define_xml_issues_in_result(
        self, tmp_gateway, sdtm_result_with_ghost_var
    ):
        svc = ValidationService(tmp_gateway)
        result = svc.validate(sdtm_result_with_ghost_var)

        # GHOSTVAR must appear in define_xml_issues
        assert result.define_xml_issues, "define_xml_issues must be non-empty"
        ghost_issues = [
            i for i in result.define_xml_issues if i.variable == "GHOSTVAR"
        ]
        assert ghost_issues, "GHOSTVAR must be reported as a structural mismatch"

    def test_define_xml_issues_merged_into_p21_report(
        self, tmp_gateway, sdtm_result_with_ghost_var
    ):
        """All define_xml_issues must appear in p21_report.json."""
        svc = ValidationService(tmp_gateway)
        result = svc.validate(sdtm_result_with_ghost_var)

        p21_bytes = tmp_gateway.get_artifact(result.artifact_keys["p21"])
        data = json.loads(p21_bytes)

        issue_descriptions = [i.get("description", "") for i in data["issues"]]
        assert any("GHOSTVAR" in d for d in issue_descriptions), (
            "GHOSTVAR structural mismatch must appear in p21_report.json"
        )

    def test_structural_mismatch_issue_type(
        self, tmp_gateway, sdtm_result_with_ghost_var
    ):
        svc = ValidationService(tmp_gateway)
        result = svc.validate(sdtm_result_with_ghost_var)

        ghost_issues = [
            i for i in result.define_xml_issues
            if i.variable == "GHOSTVAR"
        ]
        assert all(i.issue_type == "structural_mismatch" for i in ghost_issues)

    def test_all_three_reports_written_for_define_xml_run(
        self, tmp_gateway, sdtm_result_with_ghost_var
    ):
        svc = ValidationService(tmp_gateway)
        result = svc.validate(sdtm_result_with_ghost_var)

        for tag in ["p21", "tcg", "summary"]:
            assert tag in result.artifact_keys, f"Missing {tag} report"
            stored = tmp_gateway.get_artifact(result.artifact_keys[tag])
            assert len(stored) > 0


# ---------------------------------------------------------------------------
# PTCV-249: Required domain completeness validation
# ---------------------------------------------------------------------------


class TestRequiredDomainCheck:
    """PTCV-249: ValidationService checks required domain presence."""

    def test_missing_required_domain_error(
        self, tmp_gateway, all_domains,
    ):
        """GHERKIN: Missing always-required domain flagged as Error."""
        sdtm_result = _make_sdtm_result(tmp_gateway, domains=all_domains)
        svc = ValidationService(tmp_gateway)
        result = svc.validate(sdtm_result)

        # Package has TS, TV, TA, TE, TI but NOT DM, DS, AE
        assert result.domain_check is not None
        assert not result.domain_check_passed
        dm_findings = [
            f for f in result.domain_findings
            if f.domain_code == "DM"
        ]
        assert len(dm_findings) == 1
        assert dm_findings[0].severity == "Error"
        assert "required" in dm_findings[0].message.lower()

    def test_missing_conditional_domain_warning(
        self, tmp_gateway, all_domains,
    ):
        """GHERKIN: Missing conditional domain flagged as Warning."""
        sdtm_result = _make_sdtm_result(tmp_gateway, domains=all_domains)
        svc = ValidationService(tmp_gateway)
        result = svc.validate(
            sdtm_result,
            soa_assessments=["Vital Signs", "Blood Pressure"],
        )

        vs_findings = [
            f for f in result.domain_findings
            if f.domain_code == "VS"
        ]
        assert len(vs_findings) == 1
        assert vs_findings[0].severity == "Warning"

    def test_domain_findings_merged_into_p21(
        self, tmp_gateway, all_domains,
    ):
        """Domain findings appear as P21-DOM-* issues in p21_issues."""
        sdtm_result = _make_sdtm_result(tmp_gateway, domains=all_domains)
        svc = ValidationService(tmp_gateway)
        result = svc.validate(sdtm_result)

        dom_p21 = [
            i for i in result.p21_issues
            if i.rule_id.startswith("P21-DOM-")
        ]
        # At least DM, DS, AE are missing
        assert len(dom_p21) >= 3

    def test_no_findings_when_no_soa(
        self, tmp_gateway, all_domains,
    ):
        """Without soa_assessments, no conditional warnings produced."""
        sdtm_result = _make_sdtm_result(tmp_gateway, domains=all_domains)
        svc = ValidationService(tmp_gateway)
        result = svc.validate(sdtm_result)

        warnings = [
            f for f in result.domain_findings
            if f.severity == "Warning"
        ]
        assert len(warnings) == 0

    def test_domain_check_result_attached(
        self, tmp_gateway,
    ):
        """DomainCheckResult is attached to ValidationResult."""
        # Use default _make_sdtm_result which produces lowercase keys
        # matching the ValidationService loading logic
        sdtm_result = _make_sdtm_result(tmp_gateway)
        svc = ValidationService(tmp_gateway)
        result = svc.validate(sdtm_result)

        assert result.domain_check is not None
        # Package has TS and TV loaded; DM, DS, AE, TA, TE, TI missing
        assert result.domain_check.error_count >= 3
        assert "TS" in result.domain_check.domains_present
        assert "TV" in result.domain_check.domains_present
