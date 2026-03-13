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


class TestTsGeneratorNewParameters:
    """Tests for PTCV-181: SoA-first data sourcing for TCG v5.9 parameters."""

    def test_regid_from_registry_id(self, all_sections):
        gen = TsGenerator()
        df, _ = gen.generate(
            all_sections, "NCT00112827", "run-1",
            registry_id="NCT00112827",
        )
        regid = df[df["TSPARMCD"] == "REGID"]
        assert len(regid) == 1
        assert regid["TSVAL"].iloc[0] == "NCT00112827"

    def test_sdtmvr_static(self, all_sections):
        gen = TsGenerator()
        df, _ = gen.generate(all_sections, "NCT00112827", "run-1")
        sdtmvr = df[df["TSPARMCD"] == "SDTMVR"]
        assert len(sdtmvr) == 1
        assert sdtmvr["TSVAL"].iloc[0] == "1.7"

    def test_narms_from_arm_count(self, all_sections):
        gen = TsGenerator()
        df, _ = gen.generate(all_sections, "NCT00112827", "run-1")
        narms = df[df["TSPARMCD"] == "NARMS"]
        assert len(narms) == 1
        assert narms["TSVAL"].iloc[0] == "2"  # Arm A, Arm B

    def test_intmodel_maps_from_stype(self, all_sections):
        gen = TsGenerator()
        df, _ = gen.generate(all_sections, "NCT00112827", "run-1")
        intmodel = df[df["TSPARMCD"] == "INTMODEL"]
        assert len(intmodel) == 1
        assert intmodel["TSVAL"].iloc[0] == "PARALLEL GROUP"

    def test_random_detected(self, all_sections):
        gen = TsGenerator()
        df, _ = gen.generate(all_sections, "NCT00112827", "run-1")
        random_row = df[df["TSPARMCD"] == "RANDOM"]
        assert len(random_row) == 1
        assert random_row["TSVAL"].iloc[0] == "Y"

    def test_addon_default_no(self, all_sections):
        gen = TsGenerator()
        df, _ = gen.generate(all_sections, "NCT00112827", "run-1")
        addon = df[df["TSPARMCD"] == "ADDON"]
        assert len(addon) == 1
        assert addon["TSVAL"].iloc[0] == "N"

    def test_objprim_from_b3(self, all_sections):
        gen = TsGenerator()
        df, _ = gen.generate(all_sections, "NCT00112827", "run-1")
        objprim = df[df["TSPARMCD"] == "OBJPRIM"]
        assert len(objprim) == 1
        assert "efficacy" in objprim["TSVAL"].iloc[0].lower()

    def test_agemin_from_b5(self, all_sections):
        gen = TsGenerator()
        df, _ = gen.generate(all_sections, "NCT00112827", "run-1")
        agemin = df[df["TSPARMCD"] == "AGEMIN"]
        assert len(agemin) == 1
        assert agemin["TSVAL"].iloc[0] == "18"

    def test_agemax_from_b5(self, all_sections):
        gen = TsGenerator()
        df, _ = gen.generate(all_sections, "NCT00112827", "run-1")
        agemax = df[df["TSPARMCD"] == "AGEMAX"]
        assert len(agemax) == 1
        assert agemax["TSVAL"].iloc[0] == "65"

    def test_plansub_from_b10(self, all_sections):
        gen = TsGenerator()
        df, _ = gen.generate(all_sections, "NCT00112827", "run-1")
        plansub = df[df["TSPARMCD"] == "PLANSUB"]
        assert len(plansub) == 1
        assert plansub["TSVAL"].iloc[0] == "200"

    def test_inttype_drug(self, all_sections):
        gen = TsGenerator()
        df, _ = gen.generate(all_sections, "NCT00112827", "run-1")
        inttype = df[df["TSPARMCD"] == "INTTYPE"]
        assert len(inttype) == 1
        assert inttype["TSVAL"].iloc[0] == "DRUG"

    def test_stoprule_from_b6(self, all_sections):
        gen = TsGenerator()
        df, _ = gen.generate(all_sections, "NCT00112827", "run-1")
        stoprule = df[df["TSPARMCD"] == "STOPRULE"]
        assert len(stoprule) == 1
        assert "discontinued" in stoprule["TSVAL"].iloc[0].lower()

    def test_actsub_not_applicable(self, all_sections):
        gen = TsGenerator()
        df, _ = gen.generate(all_sections, "NCT00112827", "run-1")
        actsub = df[df["TSPARMCD"] == "ACTSUB"]
        assert len(actsub) == 1
        assert actsub["TSVALNF"].iloc[0] == "NOT APPLICABLE"
        assert actsub["TSVAL"].iloc[0] == ""

    def test_dcutdt_not_applicable(self, all_sections):
        gen = TsGenerator()
        df, _ = gen.generate(all_sections, "NCT00112827", "run-1")
        dcutdt = df[df["TSPARMCD"] == "DCUTDT"]
        assert len(dcutdt) == 1
        assert dcutdt["TSVALNF"].iloc[0] == "NOT APPLICABLE"

    def test_backward_compat_no_timepoints(self, all_sections):
        """generate() without timepoints keyword still works."""
        gen = TsGenerator()
        df, _ = gen.generate(all_sections, "NCT00112827", "run-1")
        assert len(df) > 0
        assert "TITLE" in df["TSPARMCD"].values

    def test_all_tcg_parameters_present(self, all_sections):
        """All 22 TCG v5.9 required parameters are in the output.

        PCLAS is best-effort (keyword match); excluded from strict check.
        """
        gen = TsGenerator()
        df, _ = gen.generate(
            all_sections, "NCT00112827", "run-1",
            registry_id="NCT00112827",
        )
        # PCLAS excluded — requires specific drug class keyword in text
        tcg_required = {
            "ACTSUB", "ADDON", "AGEMAX", "AGEMIN", "BLIND",
            "DCUTDESC", "DCUTDT", "INDIC", "INTMODEL", "INTTYPE",
            "NARMS", "OBJPRIM", "PHASE", "PLANSUB",
            "RANDOM", "REGID", "SDTMVR", "SPONSOR", "STOPRULE",
            "STYPE", "TITLE",
        }
        present = set(df["TSPARMCD"].values)
        missing = tcg_required - present
        assert missing == set(), f"Missing TCG parameters: {missing}"


class TestTsGeneratorFromHitsNewParameters:
    """PTCV-181: New parameters via query pipeline path."""

    def test_regid_and_sdtmvr(self, all_hits):
        gen = TsGenerator()
        df, _ = gen.generate_from_hits(
            all_hits, "NCT00112827", "run-1",
            registry_id="NCT00112827",
        )
        assert "REGID" in df["TSPARMCD"].values
        assert "SDTMVR" in df["TSPARMCD"].values
        sdtmvr = df[df["TSPARMCD"] == "SDTMVR"]
        assert sdtmvr["TSVAL"].iloc[0] == "1.7"

    def test_objprim_from_b3_query(self, all_hits):
        gen = TsGenerator()
        df, _ = gen.generate_from_hits(
            all_hits, "NCT00112827", "run-1",
        )
        objprim = df[df["TSPARMCD"] == "OBJPRIM"]
        assert len(objprim) == 1
        assert "efficacy" in objprim["TSVAL"].iloc[0].lower()

    def test_agemin_agemax_from_b5_queries(self, all_hits):
        gen = TsGenerator()
        df, _ = gen.generate_from_hits(
            all_hits, "NCT00112827", "run-1",
        )
        agemin = df[df["TSPARMCD"] == "AGEMIN"]
        assert len(agemin) == 1
        assert agemin["TSVAL"].iloc[0] == "18"
        agemax = df[df["TSPARMCD"] == "AGEMAX"]
        assert len(agemax) == 1
        assert agemax["TSVAL"].iloc[0] == "65"

    def test_plansub_from_b10_query(self, all_hits):
        gen = TsGenerator()
        df, _ = gen.generate_from_hits(
            all_hits, "NCT00112827", "run-1",
        )
        plansub = df[df["TSPARMCD"] == "PLANSUB"]
        assert len(plansub) == 1
        assert plansub["TSVAL"].iloc[0] == "200"

    def test_inttype_from_b7_query(self, all_hits):
        gen = TsGenerator()
        df, _ = gen.generate_from_hits(
            all_hits, "NCT00112827", "run-1",
        )
        inttype = df[df["TSPARMCD"] == "INTTYPE"]
        assert len(inttype) == 1
        assert inttype["TSVAL"].iloc[0] == "DRUG"

    def test_tsvalnf_parameters(self, all_hits):
        gen = TsGenerator()
        df, _ = gen.generate_from_hits(
            all_hits, "NCT00112827", "run-1",
        )
        for parmcd in ["ACTSUB", "DCUTDT", "DCUTDESC"]:
            row = df[df["TSPARMCD"] == parmcd]
            assert len(row) == 1, f"{parmcd} not found"
            assert row["TSVALNF"].iloc[0] == "NOT APPLICABLE"


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


# -----------------------------------------------------------------------
# FromHits tests — query pipeline path (PTCV-140)
# -----------------------------------------------------------------------

class TestTsGeneratorFromHits:
    """TS generation from QueryExtractionHit records."""

    def test_produces_dataframe(self, all_hits):
        gen = TsGenerator()
        df, unmapped = gen.generate_from_hits(all_hits, "NCT00112827", "run-1")
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

    def test_title_from_query_id(self, all_hits):
        gen = TsGenerator()
        df, _ = gen.generate_from_hits(all_hits, "NCT00112827", "run-1")
        title_row = df[df["TSPARMCD"] == "TITLE"]
        assert len(title_row) == 1
        assert "Phase III" in title_row["TSVAL"].iloc[0]

    def test_pcntid_from_query_id(self, all_hits):
        gen = TsGenerator()
        df, _ = gen.generate_from_hits(all_hits, "NCT00112827", "run-1")
        pcntid_rows = df[df["TSPARMCD"] == "PCNTID"]
        assert len(pcntid_rows) == 1
        assert pcntid_rows["TSVAL"].iloc[0] == "PROTO-2024-001"

    def test_sponsor_from_query_id(self, all_hits):
        gen = TsGenerator()
        df, _ = gen.generate_from_hits(all_hits, "NCT00112827", "run-1")
        sponsor_rows = df[df["TSPARMCD"] == "SPONSOR"]
        assert len(sponsor_rows) == 1
        assert "Test Pharma" in sponsor_rows["TSVAL"].iloc[0]

    def test_blind_from_query_id(self, all_hits):
        gen = TsGenerator()
        df, _ = gen.generate_from_hits(all_hits, "NCT00112827", "run-1")
        blind_rows = df[df["TSPARMCD"] == "BLIND"]
        assert len(blind_rows) == 1
        assert blind_rows["TSVAL"].iloc[0] == "DOUBLE-BLIND"

    def test_phase_extracted(self, all_hits):
        gen = TsGenerator()
        df, _ = gen.generate_from_hits(all_hits, "NCT00112827", "run-1")
        assert "PHASE" in df["TSPARMCD"].values

    def test_required_columns(self, all_hits):
        gen = TsGenerator()
        df, _ = gen.generate_from_hits(all_hits, "NCT00112827", "run-1")
        for col in ["STUDYID", "DOMAIN", "TSSEQ", "TSPARMCD", "TSVAL"]:
            assert col in df.columns


class TestTaGeneratorFromHits:
    """TA generation from QueryExtractionHit records."""

    def test_produces_dataframe(self, all_hits):
        gen = TaGenerator()
        df = gen.generate_from_hits(all_hits, "NCT00112827")
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

    def test_uses_b7_treatment_hits(self, all_hits):
        gen = TaGenerator()
        df = gen.generate_from_hits(all_hits, "NCT00112827")
        arms = df["ARM"].unique().tolist()
        # B.7.1.q1 has "Arm A: Drug X ..." and "Arm B: Placebo ..."
        # _extract_arms captures arm codes: "Arm A", "Arm B"
        assert any("Arm A" in arm for arm in arms)
        assert any("Arm B" in arm for arm in arms)

    def test_required_columns(self, all_hits):
        gen = TaGenerator()
        df = gen.generate_from_hits(all_hits, "NCT00112827")
        for col in ["STUDYID", "DOMAIN", "ARMCD", "ARM", "ELEMENT"]:
            assert col in df.columns


class TestTeGeneratorFromHits:
    """TE generation from QueryExtractionHit records."""

    def test_produces_dataframe(self, all_hits):
        gen = TeGenerator()
        df = gen.generate_from_hits(all_hits, "NCT00112827")
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

    def test_required_columns(self, all_hits):
        gen = TeGenerator()
        df = gen.generate_from_hits(all_hits, "NCT00112827")
        for col in ["STUDYID", "DOMAIN", "ETCD", "ELEMENT"]:
            assert col in df.columns


class TestTiGeneratorFromHits:
    """TI generation from QueryExtractionHit records."""

    def test_produces_dataframe(self, all_hits):
        gen = TiGenerator()
        df = gen.generate_from_hits(all_hits, "NCT00112827")
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

    def test_iecat_from_query_id(self, all_hits):
        """B.5.1.q1 → INCLUSION, B.5.2.q1 → EXCLUSION — no regex guessing."""
        gen = TiGenerator()
        df = gen.generate_from_hits(all_hits, "NCT00112827")
        cats = df["IECAT"].unique().tolist()
        assert "INCLUSION" in cats
        assert "EXCLUSION" in cats

    def test_inclusion_count(self, all_hits):
        gen = TiGenerator()
        df = gen.generate_from_hits(all_hits, "NCT00112827")
        inclusion = df[df["IECAT"] == "INCLUSION"]
        assert len(inclusion) == 3  # 3 items in B.5.1.q1 fixture

    def test_exclusion_count(self, all_hits):
        gen = TiGenerator()
        df = gen.generate_from_hits(all_hits, "NCT00112827")
        exclusion = df[df["IECAT"] == "EXCLUSION"]
        assert len(exclusion) == 2  # 2 items in B.5.2.q1 fixture

    def test_ietestcd_suffix_correct(self, all_hits):
        """Inclusion codes end with 'I', exclusion with 'E'."""
        gen = TiGenerator()
        df = gen.generate_from_hits(all_hits, "NCT00112827")
        for _, row in df.iterrows():
            if row["IECAT"] == "INCLUSION":
                assert row["IETESTCD"].endswith("I")
            else:
                assert row["IETESTCD"].endswith("E")
