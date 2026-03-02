"""Tests for DefineXmlGenerator."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2] / "src"))

import pytest
import pandas as pd

from ptcv.sdtm.define_xml import DefineXmlGenerator


@pytest.fixture()
def sample_datasets() -> dict[str, pd.DataFrame]:
    """Minimal DataFrames for all 5 trial design domains."""
    return {
        "TS": pd.DataFrame({
            "STUDYID": ["STUDY001"],
            "DOMAIN": ["TS"],
            "TSSEQ": [1.0],
            "TSPARMCD": ["TITLE"],
            "TSPARM": ["Trial Title"],
            "TSVAL": ["A Phase III Study"],
            "TSVALCD": [""],
            "TSVALNF": [""],
            "TSVALUNIT": [""],
        }),
        "TA": pd.DataFrame({
            "STUDYID": ["STUDY001"],
            "DOMAIN": ["TA"],
            "ARMCD": ["ARM01"],
            "ARM": ["Treatment Arm"],
            "TAETORD": [1.0],
            "ETCD": ["TRT"],
            "ELEMENT": ["Treatment"],
            "TABRANCH": [""],
            "TATRANS": [""],
            "EPOCH": ["Treatment"],
        }),
        "TE": pd.DataFrame({
            "STUDYID": ["STUDY001"],
            "DOMAIN": ["TE"],
            "ETCD": ["TRT"],
            "ELEMENT": ["Treatment"],
            "TESTRL": [""],
            "TEENRL": [""],
            "TEDUR": [""],
        }),
        "TV": pd.DataFrame({
            "STUDYID": ["STUDY001"],
            "DOMAIN": ["TV"],
            "VISITNUM": [1.0],
            "VISIT": ["Screening"],
            "VISITDY": [-14.0],
            "TVSTRL": [-3.0],
            "TVENRL": [0.0],
            "TVTIMY": [""],
            "TVTIM": [""],
            "TVENDY": [""],
        }),
        "TI": pd.DataFrame({
            "STUDYID": ["STUDY001"],
            "DOMAIN": ["TI"],
            "IETESTCD": ["IE001I"],
            "IETEST": ["Age 18-65"],
            "IECAT": ["INCLUSION"],
            "IESCAT": [""],
            "TIRL": [""],
            "TIVERS": ["1"],
        }),
    }


class TestDefineXmlGenerator:
    def test_produces_bytes(self, sample_datasets):
        gen = DefineXmlGenerator()
        result = gen.generate(sample_datasets, "STUDY001")
        assert isinstance(result, bytes)
        assert len(result) > 0

    def test_xml_declaration(self, sample_datasets):
        gen = DefineXmlGenerator()
        result = gen.generate(sample_datasets, "STUDY001")
        assert result.startswith(b"<?xml")

    def test_odm_namespace_present(self, sample_datasets):
        gen = DefineXmlGenerator()
        result = gen.generate(sample_datasets, "STUDY001")
        text = result.decode("utf-8")
        assert "cdisc.org/ns/odm" in text

    def test_define_namespace_present(self, sample_datasets):
        gen = DefineXmlGenerator()
        result = gen.generate(sample_datasets, "STUDY001")
        text = result.decode("utf-8")
        assert "cdisc.org/ns/def" in text

    def test_all_domains_referenced(self, sample_datasets):
        """GHERKIN: define.xml references each XPT dataset."""
        gen = DefineXmlGenerator()
        result = gen.generate(sample_datasets, "STUDY001")
        text = result.decode("utf-8")
        for domain in ["TS", "TA", "TE", "TV", "TI"]:
            # Either in ItemGroupDef OID or leaf href
            assert domain in text, f"Domain {domain} missing from Define-XML"

    def test_all_variable_names_present(self, sample_datasets):
        """GHERKIN: all variable names match the actual XPT columns."""
        gen = DefineXmlGenerator()
        result = gen.generate(sample_datasets, "STUDY001")
        text = result.decode("utf-8")
        for domain, df in sample_datasets.items():
            for col in df.columns:
                assert col in text, f"Variable {col} missing from Define-XML"

    def test_xpt_href_present(self, sample_datasets):
        """GHERKIN: define.xml references each XPT filename."""
        gen = DefineXmlGenerator()
        result = gen.generate(sample_datasets, "STUDY001")
        text = result.decode("utf-8")
        for domain in ["ts", "ta", "te", "tv", "ti"]:
            assert f"{domain}.xpt" in text

    def test_source_sha256_annotated(self, sample_datasets):
        """GHERKIN: LineageRecord links define.xml sha256 to source XPT sha256s."""
        gen = DefineXmlGenerator()
        sha256s = {
            "TS": "abc123def456",
            "TV": "deadbeef0000",
        }
        result = gen.generate(sample_datasets, "STUDY001", source_xpt_sha256s=sha256s)
        text = result.decode("utf-8")
        # sha256 prefix should appear in the XML description
        assert "abc123" in text

    def test_study_name_in_output(self, sample_datasets):
        gen = DefineXmlGenerator(study_name="Test Hypertension Study")
        result = gen.generate(sample_datasets, "STUDY001")
        text = result.decode("utf-8")
        assert "Test Hypertension Study" in text

    def test_define_version_2_1(self, sample_datasets):
        gen = DefineXmlGenerator()
        result = gen.generate(sample_datasets, "STUDY001")
        text = result.decode("utf-8")
        assert "2.1.0" in text

    def test_sdtm_version_1_7(self, sample_datasets):
        gen = DefineXmlGenerator()
        result = gen.generate(sample_datasets, "STUDY001")
        text = result.decode("utf-8")
        assert "1.7" in text
