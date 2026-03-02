"""Tests for ICH E6(R3) section classifiers (PTCV-20).

IQ/OQ tests for RuleBasedClassifier only (no API keys required).
RAGClassifier integration tests run separately with real credentials.
"""

from __future__ import annotations

import json
import pytest

from ptcv.ich_parser.classifier import (
    RuleBasedClassifier,
    REVIEW_THRESHOLD,
    _ICH_SECTIONS,
)
from ptcv.ich_parser.models import IchSection


RUN_ID = "test-run-001"
SOURCE_RUN_ID = "source-run-001"
SOURCE_SHA = "abc123"
REGISTRY_ID = "EUCT-2024-000001-01"


class TestRuleBasedClassifier:
    """IQ/OQ tests for RuleBasedClassifier."""

    @pytest.fixture
    def clf(self) -> RuleBasedClassifier:
        return RuleBasedClassifier()

    # ------------------------------------------------------------------
    # Core classification behaviour
    # ------------------------------------------------------------------

    def test_classify_returns_list_of_ich_sections(
        self, clf: RuleBasedClassifier, sample_text: str
    ) -> None:
        sections = clf.classify(
            sample_text, REGISTRY_ID, RUN_ID, SOURCE_RUN_ID, SOURCE_SHA
        )
        assert isinstance(sections, list)
        assert all(isinstance(s, IchSection) for s in sections)

    def test_classify_detects_multiple_sections(
        self, clf: RuleBasedClassifier, sample_text: str
    ) -> None:
        """A realistic multi-section protocol should yield several sections."""
        sections = clf.classify(
            sample_text, REGISTRY_ID, RUN_ID, SOURCE_RUN_ID, SOURCE_SHA
        )
        assert len(sections) >= 5, (
            f"Expected at least 5 sections from sample protocol, got {len(sections)}"
        )

    def test_classify_populates_all_required_fields(
        self, clf: RuleBasedClassifier, sample_text: str
    ) -> None:
        sections = clf.classify(
            sample_text, REGISTRY_ID, RUN_ID, SOURCE_RUN_ID, SOURCE_SHA
        )
        for sec in sections:
            assert sec.run_id == RUN_ID
            assert sec.source_run_id == SOURCE_RUN_ID
            assert sec.source_sha256 == SOURCE_SHA
            assert sec.registry_id == REGISTRY_ID
            assert sec.section_code in _ICH_SECTIONS
            assert sec.section_name
            assert sec.content_json
            # extraction_timestamp_utc is set by IchParser, not classifier
            assert sec.confidence_score is not None

    def test_confidence_score_between_zero_and_one(
        self, clf: RuleBasedClassifier, sample_text: str
    ) -> None:
        """ALCOA+ Accurate: confidence_score must be 0.0–1.0."""
        sections = clf.classify(
            sample_text, REGISTRY_ID, RUN_ID, SOURCE_RUN_ID, SOURCE_SHA
        )
        for sec in sections:
            assert 0.0 <= sec.confidence_score <= 1.0, (
                f"confidence_score {sec.confidence_score} out of range "
                f"for section {sec.section_code}"
            )

    def test_review_required_set_for_low_confidence(
        self, clf: RuleBasedClassifier, sample_text: str
    ) -> None:
        """Sections below REVIEW_THRESHOLD must have review_required=True."""
        sections = clf.classify(
            sample_text, REGISTRY_ID, RUN_ID, SOURCE_RUN_ID, SOURCE_SHA
        )
        for sec in sections:
            if sec.confidence_score < REVIEW_THRESHOLD:
                assert sec.review_required is True, (
                    f"review_required should be True for confidence "
                    f"{sec.confidence_score} (section {sec.section_code})"
                )
            else:
                assert sec.review_required is False

    def test_no_duplicate_section_codes(
        self, clf: RuleBasedClassifier, sample_text: str
    ) -> None:
        """Deduplication: only one result per section_code."""
        sections = clf.classify(
            sample_text, REGISTRY_ID, RUN_ID, SOURCE_RUN_ID, SOURCE_SHA
        )
        codes = [s.section_code for s in sections]
        assert len(codes) == len(set(codes)), f"Duplicate section codes: {codes}"

    def test_content_json_is_valid_json(
        self, clf: RuleBasedClassifier, sample_text: str
    ) -> None:
        sections = clf.classify(
            sample_text, REGISTRY_ID, RUN_ID, SOURCE_RUN_ID, SOURCE_SHA
        )
        for sec in sections:
            parsed = json.loads(sec.content_json)
            assert isinstance(parsed, dict)
            assert "text_excerpt" in parsed
            assert "word_count" in parsed

    # ------------------------------------------------------------------
    # Specific section detection
    # ------------------------------------------------------------------

    def test_detects_objectives_section(
        self, clf: RuleBasedClassifier, sample_text: str
    ) -> None:
        sections = clf.classify(
            sample_text, REGISTRY_ID, RUN_ID, SOURCE_RUN_ID, SOURCE_SHA
        )
        codes = [s.section_code for s in sections]
        assert "B.3" in codes, f"B.3 Objectives not detected. Found: {codes}"

    def test_detects_safety_section(
        self, clf: RuleBasedClassifier, sample_text: str
    ) -> None:
        sections = clf.classify(
            sample_text, REGISTRY_ID, RUN_ID, SOURCE_RUN_ID, SOURCE_SHA
        )
        codes = [s.section_code for s in sections]
        assert "B.9" in codes, f"B.9 Safety not detected. Found: {codes}"

    def test_detects_statistics_section(
        self, clf: RuleBasedClassifier, sample_text: str
    ) -> None:
        sections = clf.classify(
            sample_text, REGISTRY_ID, RUN_ID, SOURCE_RUN_ID, SOURCE_SHA
        )
        codes = [s.section_code for s in sections]
        assert "B.10" in codes, f"B.10 Statistics not detected. Found: {codes}"

    # ------------------------------------------------------------------
    # Legacy fallback
    # ------------------------------------------------------------------

    def test_empty_text_returns_empty_list(
        self, clf: RuleBasedClassifier
    ) -> None:
        sections = clf.classify(
            "     ", REGISTRY_ID, RUN_ID, SOURCE_RUN_ID, SOURCE_SHA
        )
        assert sections == []

    def test_unstructured_text_may_return_low_confidence(
        self, clf: RuleBasedClassifier
    ) -> None:
        """Non-standard text should yield low confidence scores."""
        garbled = "Lorem ipsum dolor sit amet consectetur adipiscing elit " * 20
        sections = clf.classify(
            garbled, REGISTRY_ID, RUN_ID, SOURCE_RUN_ID, SOURCE_SHA
        )
        if sections:
            for sec in sections:
                assert sec.confidence_score <= 0.5

    # ------------------------------------------------------------------
    # Block splitting
    # ------------------------------------------------------------------

    def test_split_into_blocks_by_heading(
        self, clf: RuleBasedClassifier
    ) -> None:
        text = "1. TITLE\nsome content\n2. BACKGROUND\nmore content"
        blocks = clf._split_into_blocks(text)
        assert len(blocks) >= 2

    def test_single_block_when_no_headings(
        self, clf: RuleBasedClassifier
    ) -> None:
        text = "no headings here just plain text"
        blocks = clf._split_into_blocks(text)
        assert len(blocks) == 1
        assert blocks[0] == text
