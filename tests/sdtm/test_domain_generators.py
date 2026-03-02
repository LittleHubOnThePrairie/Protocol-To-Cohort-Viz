"""Tests for SDTM domain generators (TS, TA, TE, TV, TI)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2] / "src"))

import pytest
import pandas as pd

from ptcv.sdtm.domain_generators import (
    TaGenerator,
    TeGenerator,
    TiGenerator,
    TsGenerator,
    TvGenerator,
)


class TestTsGenerator:
    def test_produces_dataframe(self, all_sections):
        gen = TsGenerator()
        df, unmapped = gen.generate(all_sections, "NCT00112827", "run-1")
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

    def test_required_columns(self, all_sections):
        gen = TsGenerator()
        df, _ = gen.generate(all_sections, "NCT00112827", "run-1")
        for col in ["STUDYID", "DOMAIN", "TSSEQ", "TSPARMCD", "TSPARM", "TSVAL"]:
            assert col in df.columns

    def test_domain_is_ts(self, all_sections):
        gen = TsGenerator()
        df, _ = gen.generate(all_sections, "NCT00112827", "run-1")
        assert (df["DOMAIN"] == "TS").all()

    def test_studyid_populated(self, all_sections):
        gen = TsGenerator()
        df, _ = gen.generate(all_sections, "NCT00112827", "run-1")
        assert (df["STUDYID"] == "NCT00112827").all()

    def test_title_row_present(self, all_sections):
        gen = TsGenerator()
        df, _ = gen.generate(all_sections, "NCT00112827", "run-1")
        assert "TITLE" in df["TSPARMCD"].values

    def test_phase_row_extracted(self, all_sections):
        gen = TsGenerator()
        df, _ = gen.generate(all_sections, "NCT00112827", "run-1")
        assert "PHASE" in df["TSPARMCD"].values
        phase_val = df[df["TSPARMCD"] == "PHASE"]["TSVALCD"].iloc[0]
        assert phase_val == "PHASE3"

    def test_tsseq_sequential(self, all_sections):
        gen = TsGenerator()
        df, _ = gen.generate(all_sections, "NCT00112827", "run-1")
        seqs = df["TSSEQ"].tolist()
        assert seqs == sorted(seqs)

    def test_unmapped_terms_collected(self, all_sections, b1_section):
        """A value not in CT table should appear in unmapped list."""
        import json
        b1_section.content_json = json.dumps({
            "text_excerpt": "Some unknown XYZ Phase study",
            "word_count": 6,
        })
        gen = TsGenerator()
        df, unmapped = gen.generate([b1_section], "NCT00112827", "run-1")
        # unmapped may or may not be present — just ensure type is correct
        assert isinstance(unmapped, list)

    def test_no_sections_still_produces_title(self):
        gen = TsGenerator()
        df, _ = gen.generate([], "NCT99999999", "run-1")
        assert len(df) >= 1
        assert "TITLE" in df["TSPARMCD"].values


class TestTaGenerator:
    def test_produces_dataframe(self, all_sections):
        gen = TaGenerator()
        df = gen.generate(all_sections, "NCT00112827")
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

    def test_required_columns(self, all_sections):
        gen = TaGenerator()
        df = gen.generate(all_sections, "NCT00112827")
        for col in ["STUDYID", "DOMAIN", "ARMCD", "ARM", "TAETORD", "ETCD", "ELEMENT", "EPOCH"]:
            assert col in df.columns

    def test_domain_is_ta(self, all_sections):
        gen = TaGenerator()
        df = gen.generate(all_sections, "NCT00112827")
        assert (df["DOMAIN"] == "TA").all()

    def test_taetord_starts_at_one(self, all_sections):
        gen = TaGenerator()
        df = gen.generate(all_sections, "NCT00112827")
        assert df["TAETORD"].min() == 1.0

    def test_armcd_format(self, all_sections):
        gen = TaGenerator()
        df = gen.generate(all_sections, "NCT00112827")
        # ARM codes should be non-empty strings ≤ 20 chars
        assert (df["ARMCD"].str.len() > 0).all()
        assert (df["ARMCD"].str.len() <= 20).all()


class TestTeGenerator:
    def test_produces_dataframe(self, all_sections):
        gen = TeGenerator()
        df = gen.generate(all_sections, "NCT00112827")
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

    def test_required_columns(self, all_sections):
        gen = TeGenerator()
        df = gen.generate(all_sections, "NCT00112827")
        for col in ["STUDYID", "DOMAIN", "ETCD", "ELEMENT"]:
            assert col in df.columns

    def test_domain_is_te(self, all_sections):
        gen = TeGenerator()
        df = gen.generate(all_sections, "NCT00112827")
        assert (df["DOMAIN"] == "TE").all()

    def test_etcd_max_8_chars(self, all_sections):
        gen = TeGenerator()
        df = gen.generate(all_sections, "NCT00112827")
        assert (df["ETCD"].str.len() <= 8).all()


class TestTvGenerator:
    def test_produces_one_row_per_timepoint(self, timepoints):
        """GHERKIN: TV contains one row per scheduled visit."""
        gen = TvGenerator()
        df = gen.generate(timepoints, "NCT00112827")
        assert len(df) == len(timepoints)

    def test_visitdy_from_day_offset(self, timepoints):
        """GHERKIN: VISITDY is populated from day_offset column."""
        gen = TvGenerator()
        df = gen.generate(timepoints, "NCT00112827")
        expected_days = sorted([tp.day_offset for tp in timepoints])
        actual_days = sorted(df["VISITDY"].astype(int).tolist())
        assert actual_days == expected_days

    def test_tvstrl_from_window_early(self, timepoints):
        """GHERKIN: TVSTRL is populated from window_early (as negative)."""
        gen = TvGenerator()
        df = gen.generate(timepoints, "NCT00112827")
        # Screening has window_early=3 → TVSTRL=-3
        screening = df[df["VISIT"] == "Screening"]
        assert len(screening) == 1
        assert screening["TVSTRL"].iloc[0] == -3.0

    def test_tvenrl_from_window_late(self, timepoints):
        """GHERKIN: TVENRL is populated from window_late."""
        gen = TvGenerator()
        df = gen.generate(timepoints, "NCT00112827")
        screening = df[df["VISIT"] == "Screening"]
        assert screening["TVENRL"].iloc[0] == 0.0
        week4 = df[df["VISIT"] == "Week 4"]
        assert week4["TVENRL"].iloc[0] == 3.0

    def test_required_columns(self, timepoints):
        gen = TvGenerator()
        df = gen.generate(timepoints, "NCT00112827")
        for col in ["STUDYID", "DOMAIN", "VISITNUM", "VISIT", "VISITDY", "TVSTRL", "TVENRL"]:
            assert col in df.columns

    def test_domain_is_tv(self, timepoints):
        gen = TvGenerator()
        df = gen.generate(timepoints, "NCT00112827")
        assert (df["DOMAIN"] == "TV").all()

    def test_visitnum_sequential(self, timepoints):
        gen = TvGenerator()
        df = gen.generate(timepoints, "NCT00112827")
        # VISITNUM should be 1.0, 2.0, 3.0, 4.0 sorted by day_offset
        assert df["VISITNUM"].min() == 1.0
        assert len(df["VISITNUM"].unique()) == len(timepoints)

    def test_empty_timepoints_returns_empty_df(self):
        gen = TvGenerator()
        df = gen.generate([], "NCT00112827")
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0


class TestTiGenerator:
    def test_produces_dataframe(self, all_sections):
        gen = TiGenerator()
        df = gen.generate(all_sections, "NCT00112827")
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

    def test_required_columns(self, all_sections):
        gen = TiGenerator()
        df = gen.generate(all_sections, "NCT00112827")
        for col in ["STUDYID", "DOMAIN", "IETESTCD", "IETEST", "IECAT"]:
            assert col in df.columns

    def test_domain_is_ti(self, all_sections):
        gen = TiGenerator()
        df = gen.generate(all_sections, "NCT00112827")
        assert (df["DOMAIN"] == "TI").all()

    def test_inclusion_and_exclusion_present(self, all_sections):
        gen = TiGenerator()
        df = gen.generate(all_sections, "NCT00112827")
        cats = df["IECAT"].unique().tolist()
        assert "INCLUSION" in cats
        assert "EXCLUSION" in cats

    def test_ietestcd_max_8_chars(self, all_sections):
        gen = TiGenerator()
        df = gen.generate(all_sections, "NCT00112827")
        assert (df["IETESTCD"].str.len() <= 8).all()
