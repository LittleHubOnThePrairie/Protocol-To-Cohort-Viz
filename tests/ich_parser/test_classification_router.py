"""Tests for ClassificationRouter — PTCV-161.

Covers all 5 GHERKIN scenarios from PTCV-161 acceptance criteria:
  1. High-confidence local acceptance
  2. Low-confidence Sonnet routing
  3. Audit logging (immutable RoutingDecision records)
  4. Routing stats (local_pct, sonnet_pct, agreements)
  5. Graceful fallback (no API key → all local)
Plus:
  6. JSONL judgement storage
  7. Configurable threshold
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parents[2] / "src"))

from ptcv.ich_parser.classification_router import (
    CascadeResult,
    ClassificationRouter,
    LocalCandidate,
    RoutingDecision,
    RoutingStats,
    SonnetJudgement,
)
from ptcv.ich_parser.classifier import RuleBasedClassifier
from ptcv.ich_parser.models import IchSection


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------


def _make_section(
    code: str = "B.4",
    name: str = "Trial Design",
    confidence: float = 0.50,
    text: str = "Study design randomisation blinding",
) -> IchSection:
    return IchSection(
        run_id="run-1",
        source_run_id="src-1",
        source_sha256="abc123",
        registry_id="NCT00000001",
        section_code=code,
        section_name=name,
        content_json="{}",
        confidence_score=confidence,
        review_required=confidence < 0.70,
        legacy_format=False,
        content_text=text,
    )


def _make_sonnet_response(
    section_code: str = "B.4",
    reasoning: str = "Trial design section",
) -> MagicMock:
    """Build a mock Anthropic API response for cascade Sonnet call."""
    content_block = MagicMock()
    content_block.text = json.dumps({
        "section_code": section_code,
        "reasoning": reasoning,
    })
    response = MagicMock()
    response.content = [content_block]
    response.usage = MagicMock()
    response.usage.input_tokens = 150
    response.usage.output_tokens = 30
    return response


@pytest.fixture()
def mock_classifier():
    """A mock classifier that returns controllable sections."""
    cls = MagicMock(spec=RuleBasedClassifier)
    cls._split_into_blocks = MagicMock(return_value=["block text"])
    cls._score_block_topk = MagicMock(return_value=[
        ("B.4", 0.50, {"text_excerpt": "block text", "word_count": 5}),
        ("B.5", 0.30, {"text_excerpt": "block text", "word_count": 5}),
        ("B.7", 0.10, {"text_excerpt": "block text", "word_count": 5}),
    ])
    return cls


@pytest.fixture()
def mock_gateway():
    """A mock StorageGateway."""
    gw = MagicMock()
    gw.put_artifact = MagicMock()
    return gw


# ---------------------------------------------------------------------------
# 1. High-confidence sections stay local
# ---------------------------------------------------------------------------


class TestHighConfidenceLocal:
    """PTCV-161 Scenario: High-confidence sections stay local."""

    def test_section_above_threshold_not_sent_to_sonnet(
        self, mock_classifier, mock_gateway
    ):
        """Section >= 0.70 stays local, Sonnet never called."""
        mock_classifier.classify.return_value = [
            _make_section("B.1", "General Information", 0.85),
        ]

        router = ClassificationRouter(
            classifier=mock_classifier,
            gateway=mock_gateway,
            confidence_threshold=0.70,
        )
        # Ensure Sonnet is not used
        router._use_sonnet = False

        result = router.classify(
            text_blocks=[{"page_number": 1, "text": "General info"}],
            registry_id="NCT00000001",
            run_id="run-1",
            source_run_id="src-1",
            source_sha256="abc",
        )

        assert len(result.decisions) == 1
        assert result.decisions[0].route == "local"
        assert result.decisions[0].final_section_code == "B.1"
        assert result.decisions[0].sonnet_judgement is None

    def test_high_confidence_preserves_local_score(
        self, mock_classifier, mock_gateway
    ):
        mock_classifier.classify.return_value = [
            _make_section("B.3", "Trial Objectives", 0.92),
        ]

        router = ClassificationRouter(
            classifier=mock_classifier,
            gateway=mock_gateway,
            confidence_threshold=0.70,
        )
        router._use_sonnet = False

        result = router.classify(
            text_blocks=[{"page_number": 1, "text": "Objectives"}],
            registry_id="NCT00000001",
            run_id="run-1",
            source_run_id="src-1",
            source_sha256="abc",
        )

        assert result.decisions[0].final_confidence == 0.92


# ---------------------------------------------------------------------------
# 2. Low-confidence sections route to Sonnet
# ---------------------------------------------------------------------------


class TestLowConfidenceSonnet:
    """PTCV-161 Scenario: Low-confidence sections route to Sonnet."""

    def test_section_below_threshold_routes_to_sonnet(
        self, mock_classifier, mock_gateway
    ):
        """Section < 0.70 is sent to Sonnet; judgement accepted."""
        mock_classifier.classify.return_value = [
            _make_section("B.4", "Trial Design", 0.45),
        ]

        router = ClassificationRouter(
            classifier=mock_classifier,
            gateway=mock_gateway,
            confidence_threshold=0.70,
        )
        router._use_sonnet = True
        router._api_key = "test-key"

        mock_response = _make_sonnet_response("B.5", "Eligibility text")
        with patch.object(
            router, "_get_sonnet_client"
        ) as mock_get_client:
            mock_client = MagicMock()
            mock_client.messages.create.return_value = mock_response
            mock_get_client.return_value = mock_client

            result = router.classify(
                text_blocks=[{"page_number": 1, "text": "inclusion"}],
                registry_id="NCT00000001",
                run_id="run-1",
                source_run_id="src-1",
                source_sha256="abc",
            )

        assert len(result.decisions) == 1
        d = result.decisions[0]
        assert d.route == "sonnet"
        assert d.sonnet_judgement is not None
        assert d.sonnet_judgement.section_code == "B.5"
        assert d.sonnet_judgement.reasoning == "Eligibility text"
        assert d.final_section_code == "B.5"

    def test_sonnet_judgement_has_token_counts(
        self, mock_classifier, mock_gateway
    ):
        mock_classifier.classify.return_value = [
            _make_section("B.4", "Trial Design", 0.40),
        ]

        router = ClassificationRouter(
            classifier=mock_classifier,
            gateway=mock_gateway,
            confidence_threshold=0.70,
        )
        router._use_sonnet = True
        router._api_key = "test-key"

        mock_response = _make_sonnet_response("B.4", "Design section")
        with patch.object(
            router, "_get_sonnet_client"
        ) as mock_get_client:
            mock_client = MagicMock()
            mock_client.messages.create.return_value = mock_response
            mock_get_client.return_value = mock_client

            result = router.classify(
                text_blocks=[{"page_number": 1, "text": "design"}],
                registry_id="NCT00000001",
                run_id="run-1",
                source_run_id="src-1",
                source_sha256="abc",
            )

        j = result.decisions[0].sonnet_judgement
        assert j.input_tokens == 150
        assert j.output_tokens == 30

    def test_top3_candidates_passed_to_sonnet(
        self, mock_classifier, mock_gateway
    ):
        """Verify local top-3 candidates are in the routing decision."""
        mock_classifier.classify.return_value = [
            _make_section("B.4", "Trial Design", 0.50),
        ]

        router = ClassificationRouter(
            classifier=mock_classifier,
            gateway=mock_gateway,
            confidence_threshold=0.70,
        )
        router._use_sonnet = True
        router._api_key = "test-key"

        mock_response = _make_sonnet_response("B.4", "Design")
        with patch.object(
            router, "_get_sonnet_client"
        ) as mock_get_client:
            mock_client = MagicMock()
            mock_client.messages.create.return_value = mock_response
            mock_get_client.return_value = mock_client

            result = router.classify(
                text_blocks=[{"page_number": 1, "text": "design"}],
                registry_id="NCT00000001",
                run_id="run-1",
                source_run_id="src-1",
                source_sha256="abc",
            )

        d = result.decisions[0]
        assert len(d.local_candidates) == 3
        assert d.local_candidates[0].section_code == "B.4"
        assert d.local_candidates[1].section_code == "B.5"
        assert d.local_candidates[2].section_code == "B.7"


# ---------------------------------------------------------------------------
# 3. Audit logging — immutable RoutingDecision records
# ---------------------------------------------------------------------------


class TestAuditLogging:
    """PTCV-161 Scenario: Every decision is logged with required fields."""

    def test_routing_decision_has_all_required_fields(
        self, mock_classifier, mock_gateway
    ):
        mock_classifier.classify.return_value = [
            _make_section("B.1", "General Information", 0.90),
        ]

        router = ClassificationRouter(
            classifier=mock_classifier,
            gateway=mock_gateway,
        )
        router._use_sonnet = False

        result = router.classify(
            text_blocks=[{"page_number": 1, "text": "General info"}],
            registry_id="NCT00000001",
            run_id="run-1",
            source_run_id="src-1",
            source_sha256="abc",
        )

        d = result.decisions[0]
        assert isinstance(d.block_index, int)
        assert isinstance(d.content_hash, str)
        assert len(d.content_hash) == 64  # SHA-256 hex
        assert d.route in ("local", "sonnet")
        assert isinstance(d.threshold_used, float)
        assert isinstance(d.final_section_code, str)
        assert isinstance(d.final_confidence, float)
        assert isinstance(d.local_candidates, list)

    def test_routing_decision_is_frozen(self):
        """RoutingDecision should be immutable (frozen dataclass)."""
        d = RoutingDecision(
            block_index=0,
            content_hash="a" * 64,
            local_candidates=[],
            route="local",
            threshold_used=0.70,
            final_section_code="B.1",
            final_confidence=0.90,
        )
        with pytest.raises(AttributeError):
            d.route = "sonnet"  # type: ignore[misc]

    def test_content_hash_is_sha256_of_text(
        self, mock_classifier, mock_gateway
    ):
        text = "Study design randomisation blinding"
        expected_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()

        mock_classifier.classify.return_value = [
            _make_section("B.4", "Trial Design", 0.90, text=text),
        ]

        router = ClassificationRouter(
            classifier=mock_classifier,
            gateway=mock_gateway,
        )
        router._use_sonnet = False

        result = router.classify(
            text_blocks=[{"page_number": 1, "text": text}],
            registry_id="NCT00000001",
            run_id="run-1",
            source_run_id="src-1",
            source_sha256="abc",
        )

        assert result.decisions[0].content_hash == expected_hash


# ---------------------------------------------------------------------------
# 4. Routing stats
# ---------------------------------------------------------------------------


class TestRoutingStats:
    """PTCV-161 Scenario: Routing stats tracking."""

    def test_all_local_stats(self, mock_classifier, mock_gateway):
        """All sections above threshold → 100% local."""
        mock_classifier.classify.return_value = [
            _make_section("B.1", "General", 0.85),
            _make_section("B.3", "Objectives", 0.92),
        ]

        router = ClassificationRouter(
            classifier=mock_classifier,
            gateway=mock_gateway,
            confidence_threshold=0.70,
        )
        router._use_sonnet = False

        result = router.classify(
            text_blocks=[{"page_number": 1, "text": "text"}],
            registry_id="NCT00000001",
            run_id="run-1",
            source_run_id="src-1",
            source_sha256="abc",
        )

        assert result.stats.total_sections == 2
        assert result.stats.local_count == 2
        assert result.stats.sonnet_count == 0
        assert result.stats.local_pct == 1.0
        assert result.stats.sonnet_pct == 0.0

    def test_mixed_routing_stats(self, mock_classifier, mock_gateway):
        """Mix of local and sonnet routing."""
        mock_classifier.classify.return_value = [
            _make_section("B.1", "General", 0.85),
            _make_section("B.4", "Trial Design", 0.40),
        ]

        router = ClassificationRouter(
            classifier=mock_classifier,
            gateway=mock_gateway,
            confidence_threshold=0.70,
        )
        router._use_sonnet = True
        router._api_key = "test-key"

        # Sonnet agrees with local top-1
        mock_response = _make_sonnet_response("B.4", "Correct")
        with patch.object(
            router, "_get_sonnet_client"
        ) as mock_get_client:
            mock_client = MagicMock()
            mock_client.messages.create.return_value = mock_response
            mock_get_client.return_value = mock_client

            result = router.classify(
                text_blocks=[{"page_number": 1, "text": "text"}],
                registry_id="NCT00000001",
                run_id="run-1",
                source_run_id="src-1",
                source_sha256="abc",
            )

        assert result.stats.total_sections == 2
        assert result.stats.local_count == 1
        assert result.stats.sonnet_count == 1
        assert result.stats.local_pct == 0.5
        assert result.stats.sonnet_pct == 0.5

    def test_agreement_disagreement_tracking(
        self, mock_classifier, mock_gateway
    ):
        """Sonnet agrees → agreement count incremented."""
        mock_classifier.classify.return_value = [
            _make_section("B.4", "Trial Design", 0.40),
        ]

        router = ClassificationRouter(
            classifier=mock_classifier,
            gateway=mock_gateway,
            confidence_threshold=0.70,
        )
        router._use_sonnet = True
        router._api_key = "test-key"

        # Sonnet disagrees — picks B.5 instead of B.4
        mock_response = _make_sonnet_response("B.5", "Eligibility")
        with patch.object(
            router, "_get_sonnet_client"
        ) as mock_get_client:
            mock_client = MagicMock()
            mock_client.messages.create.return_value = mock_response
            mock_get_client.return_value = mock_client

            result = router.classify(
                text_blocks=[{"page_number": 1, "text": "text"}],
                registry_id="NCT00000001",
                run_id="run-1",
                source_run_id="src-1",
                source_sha256="abc",
            )

        assert result.stats.disagreements == 1
        assert result.stats.agreements == 0


# ---------------------------------------------------------------------------
# 5. Graceful fallback (no API key)
# ---------------------------------------------------------------------------


class TestGracefulFallback:
    """PTCV-161 Scenario: No API key → all sections stay local."""

    def test_no_api_key_all_local(self, mock_classifier, mock_gateway):
        mock_classifier.classify.return_value = [
            _make_section("B.4", "Trial Design", 0.30),
            _make_section("B.5", "Selection", 0.25),
        ]

        with patch.dict("os.environ", {}, clear=False):
            # Remove ANTHROPIC_API_KEY if present
            import os
            old_key = os.environ.pop("ANTHROPIC_API_KEY", None)
            try:
                router = ClassificationRouter(
                    classifier=mock_classifier,
                    gateway=mock_gateway,
                    confidence_threshold=0.70,
                )

                result = router.classify(
                    text_blocks=[
                        {"page_number": 1, "text": "text"},
                    ],
                    registry_id="NCT00000001",
                    run_id="run-1",
                    source_run_id="src-1",
                    source_sha256="abc",
                )
            finally:
                if old_key:
                    os.environ["ANTHROPIC_API_KEY"] = old_key

        # All sections stay local despite low confidence
        assert result.stats.local_count == 2
        assert result.stats.sonnet_count == 0
        assert all(d.route == "local" for d in result.decisions)

    def test_deterministic_path_guaranteed(
        self, mock_classifier, mock_gateway
    ):
        """Without API key, output is fully deterministic."""
        mock_classifier.classify.return_value = [
            _make_section("B.1", "General", 0.50),
        ]

        import os
        old_key = os.environ.pop("ANTHROPIC_API_KEY", None)
        try:
            router = ClassificationRouter(
                classifier=mock_classifier,
                gateway=mock_gateway,
            )
            r1 = router.classify(
                text_blocks=[{"page_number": 1, "text": "t"}],
                registry_id="NCT00000001",
                run_id="run-1",
                source_run_id="src-1",
                source_sha256="abc",
            )
            r2 = router.classify(
                text_blocks=[{"page_number": 1, "text": "t"}],
                registry_id="NCT00000001",
                run_id="run-1",
                source_run_id="src-1",
                source_sha256="abc",
            )
        finally:
            if old_key:
                os.environ["ANTHROPIC_API_KEY"] = old_key

        assert r1.decisions[0].route == r2.decisions[0].route
        assert (
            r1.decisions[0].final_section_code
            == r2.decisions[0].final_section_code
        )


# ---------------------------------------------------------------------------
# 6. JSONL judgement storage
# ---------------------------------------------------------------------------


class TestJudgementStorage:
    """PTCV-161: Sonnet judgements stored as JSONL for training."""

    def test_jsonl_written_when_sonnet_used(
        self, mock_classifier, mock_gateway
    ):
        mock_classifier.classify.return_value = [
            _make_section("B.4", "Trial Design", 0.40),
        ]

        router = ClassificationRouter(
            classifier=mock_classifier,
            gateway=mock_gateway,
            confidence_threshold=0.70,
        )
        router._use_sonnet = True
        router._api_key = "test-key"

        mock_response = _make_sonnet_response("B.5", "Eligibility")
        with patch.object(
            router, "_get_sonnet_client"
        ) as mock_get_client:
            mock_client = MagicMock()
            mock_client.messages.create.return_value = mock_response
            mock_get_client.return_value = mock_client

            result = router.classify(
                text_blocks=[{"page_number": 1, "text": "text"}],
                registry_id="NCT00000001",
                run_id="run-1",
                source_run_id="src-1",
                source_sha256="abc",
            )

        assert result.judgement_artifact_key.endswith(
            "sonnet_judgements.jsonl"
        )
        assert len(result.judgement_artifact_sha256) == 64
        mock_gateway.put_artifact.assert_called()

    def test_jsonl_not_written_when_all_local(
        self, mock_classifier, mock_gateway
    ):
        mock_classifier.classify.return_value = [
            _make_section("B.1", "General", 0.90),
        ]

        router = ClassificationRouter(
            classifier=mock_classifier,
            gateway=mock_gateway,
        )
        router._use_sonnet = False

        result = router.classify(
            text_blocks=[{"page_number": 1, "text": "text"}],
            registry_id="NCT00000001",
            run_id="run-1",
            source_run_id="src-1",
            source_sha256="abc",
        )

        # No Sonnet calls → no JSONL
        assert result.judgement_artifact_key == ""
        assert result.judgement_artifact_sha256 == ""

    def test_jsonl_artifact_key_format(
        self, mock_classifier, mock_gateway
    ):
        mock_classifier.classify.return_value = [
            _make_section("B.4", "Trial Design", 0.40),
        ]

        router = ClassificationRouter(
            classifier=mock_classifier,
            gateway=mock_gateway,
        )
        router._use_sonnet = True
        router._api_key = "test-key"

        mock_response = _make_sonnet_response("B.4", "Design")
        with patch.object(
            router, "_get_sonnet_client"
        ) as mock_get_client:
            mock_client = MagicMock()
            mock_client.messages.create.return_value = mock_response
            mock_get_client.return_value = mock_client

            result = router.classify(
                text_blocks=[{"page_number": 1, "text": "text"}],
                registry_id="NCT00000001",
                run_id="cascade-run-42",
                source_run_id="src-1",
                source_sha256="abc",
            )

        assert result.judgement_artifact_key == (
            "cascade/cascade-run-42/sonnet_judgements.jsonl"
        )


# ---------------------------------------------------------------------------
# 7. Configurable threshold
# ---------------------------------------------------------------------------


class TestConfigurableThreshold:
    """PTCV-161: Threshold can be customised."""

    def test_custom_threshold_respected(
        self, mock_classifier, mock_gateway
    ):
        """With threshold=0.50, a 0.60 section stays local."""
        mock_classifier.classify.return_value = [
            _make_section("B.4", "Trial Design", 0.60),
        ]

        router = ClassificationRouter(
            classifier=mock_classifier,
            gateway=mock_gateway,
            confidence_threshold=0.50,
        )
        router._use_sonnet = False

        result = router.classify(
            text_blocks=[{"page_number": 1, "text": "text"}],
            registry_id="NCT00000001",
            run_id="run-1",
            source_run_id="src-1",
            source_sha256="abc",
        )

        assert result.decisions[0].route == "local"

    def test_default_threshold_from_yaml(self, mock_gateway):
        """Default threshold loads from YAML config."""
        router = ClassificationRouter(gateway=mock_gateway)
        assert router._threshold == 0.70


# ---------------------------------------------------------------------------
# CascadeResult structure
# ---------------------------------------------------------------------------


class TestCascadeResult:
    """Verify CascadeResult output structure."""

    def test_cascade_result_has_run_id(
        self, mock_classifier, mock_gateway
    ):
        mock_classifier.classify.return_value = [
            _make_section("B.1", "General", 0.90),
        ]

        router = ClassificationRouter(
            classifier=mock_classifier,
            gateway=mock_gateway,
        )
        router._use_sonnet = False

        result = router.classify(
            text_blocks=[{"page_number": 1, "text": "text"}],
            registry_id="NCT00000001",
            run_id="my-run-id",
            source_run_id="src-1",
            source_sha256="abc",
        )

        assert result.run_id == "my-run-id"
        assert result.registry_id == "NCT00000001"
        assert isinstance(result.sections, list)
        assert isinstance(result.stats, RoutingStats)

    def test_final_sections_updated_by_sonnet(
        self, mock_classifier, mock_gateway
    ):
        """When Sonnet reclassifies, the IchSection is updated."""
        mock_classifier.classify.return_value = [
            _make_section("B.4", "Trial Design", 0.40),
        ]

        router = ClassificationRouter(
            classifier=mock_classifier,
            gateway=mock_gateway,
        )
        router._use_sonnet = True
        router._api_key = "test-key"

        mock_response = _make_sonnet_response("B.5", "Eligibility")
        with patch.object(
            router, "_get_sonnet_client"
        ) as mock_get_client:
            mock_client = MagicMock()
            mock_client.messages.create.return_value = mock_response
            mock_get_client.return_value = mock_client

            result = router.classify(
                text_blocks=[{"page_number": 1, "text": "text"}],
                registry_id="NCT00000001",
                run_id="run-1",
                source_run_id="src-1",
                source_sha256="abc",
            )

        assert result.sections[0].section_code == "B.5"


# ---------------------------------------------------------------------------
# Data model unit tests
# ---------------------------------------------------------------------------


class TestDataModels:
    def test_local_candidate_frozen(self):
        c = LocalCandidate("B.1", "General", 0.90)
        with pytest.raises(AttributeError):
            c.section_code = "B.2"  # type: ignore[misc]

    def test_sonnet_judgement_frozen(self):
        j = SonnetJudgement("B.4", "Design section", 100, 30)
        with pytest.raises(AttributeError):
            j.section_code = "B.5"  # type: ignore[misc]

    def test_sonnet_judgement_defaults(self):
        j = SonnetJudgement("B.4", "reason")
        assert j.input_tokens == 0
        assert j.output_tokens == 0

    def test_routing_stats_defaults(self):
        s = RoutingStats()
        assert s.total_sections == 0
        assert s.agreements == 0

    def test_routing_decision_rag_exemplar_count_default(self):
        """rag_exemplar_count defaults to 0 for backward compat."""
        d = RoutingDecision(
            block_index=0,
            content_hash="a" * 64,
            local_candidates=[],
            route="local",
            threshold_used=0.70,
            final_section_code="B.1",
            final_confidence=0.90,
        )
        assert d.rag_exemplar_count == 0


# ---------------------------------------------------------------------------
# 8. RAG context injection (PTCV-162)
# ---------------------------------------------------------------------------


class _FakeRagExemplar:
    """Lightweight stand-in for RagExemplar in tests."""

    def __init__(
        self,
        section_code: str = "B.5",
        section_name: str = "Selection",
        registry_id: str = "NCT00000002",
        confidence_score: float = 0.92,
        content_text: str = "Inclusion criteria for patients",
        similarity_score: float = 0.88,
    ):
        self.section_code = section_code
        self.section_name = section_name
        self.registry_id = registry_id
        self.confidence_score = confidence_score
        self.content_text = content_text
        self.similarity_score = similarity_score


class TestRagContextInjection:
    """PTCV-162: RAG context is injected into Sonnet prompt."""

    def test_rag_exemplars_included_in_prompt(
        self, mock_classifier, mock_gateway
    ):
        """When RAG index returns hits, they appear in the Sonnet prompt."""
        mock_classifier.classify.return_value = [
            _make_section("B.4", "Trial Design", 0.40),
        ]

        mock_rag_index = MagicMock()
        mock_rag_index.query.return_value = [
            _FakeRagExemplar("B.5", "Selection", "NCT00000099",
                             0.92, "Prior eligibility text", 0.88),
            _FakeRagExemplar("B.5", "Selection", "NCT00000100",
                             0.90, "Another eligibility section", 0.85),
        ]

        router = ClassificationRouter(
            classifier=mock_classifier,
            gateway=mock_gateway,
            confidence_threshold=0.70,
            rag_index=mock_rag_index,
        )
        router._use_sonnet = True
        router._api_key = "test-key"

        mock_response = _make_sonnet_response("B.5", "Eligibility")
        with patch.object(
            router, "_get_sonnet_client"
        ) as mock_get_client:
            mock_client = MagicMock()
            mock_client.messages.create.return_value = mock_response
            mock_get_client.return_value = mock_client

            result = router.classify(
                text_blocks=[{"page_number": 1, "text": "inclusion"}],
                registry_id="NCT00000001",
                run_id="run-1",
                source_run_id="src-1",
                source_sha256="abc",
            )

            # Verify prompt contains RAG context
            call_args = mock_client.messages.create.call_args
            prompt_text = call_args.kwargs["messages"][0]["content"]
            assert "NCT00000099" in prompt_text
            assert "NCT00000100" in prompt_text
            assert "Prior eligibility text" in prompt_text
            assert "similarity: 0.88" in prompt_text

        # RAG count tracked in decision
        assert result.decisions[0].rag_exemplar_count == 2

    def test_no_rag_index_prompt_has_placeholder(
        self, mock_classifier, mock_gateway
    ):
        """When rag_index is None, prompt has placeholder text."""
        mock_classifier.classify.return_value = [
            _make_section("B.4", "Trial Design", 0.40),
        ]

        router = ClassificationRouter(
            classifier=mock_classifier,
            gateway=mock_gateway,
            confidence_threshold=0.70,
            rag_index=None,
        )
        router._use_sonnet = True
        router._api_key = "test-key"

        mock_response = _make_sonnet_response("B.4", "Design")
        with patch.object(
            router, "_get_sonnet_client"
        ) as mock_get_client:
            mock_client = MagicMock()
            mock_client.messages.create.return_value = mock_response
            mock_get_client.return_value = mock_client

            router.classify(
                text_blocks=[{"page_number": 1, "text": "design"}],
                registry_id="NCT00000001",
                run_id="run-1",
                source_run_id="src-1",
                source_sha256="abc",
            )

            call_args = mock_client.messages.create.call_args
            prompt_text = call_args.kwargs["messages"][0]["content"]
            assert "(no prior examples available)" in prompt_text

    def test_rag_query_failure_gracefully_skipped(
        self, mock_classifier, mock_gateway
    ):
        """When rag_index.query() raises, Sonnet is still called."""
        mock_classifier.classify.return_value = [
            _make_section("B.4", "Trial Design", 0.40),
        ]

        mock_rag_index = MagicMock()
        mock_rag_index.query.side_effect = RuntimeError("index error")

        router = ClassificationRouter(
            classifier=mock_classifier,
            gateway=mock_gateway,
            confidence_threshold=0.70,
            rag_index=mock_rag_index,
        )
        router._use_sonnet = True
        router._api_key = "test-key"

        mock_response = _make_sonnet_response("B.4", "Design")
        with patch.object(
            router, "_get_sonnet_client"
        ) as mock_get_client:
            mock_client = MagicMock()
            mock_client.messages.create.return_value = mock_response
            mock_get_client.return_value = mock_client

            result = router.classify(
                text_blocks=[{"page_number": 1, "text": "design"}],
                registry_id="NCT00000001",
                run_id="run-1",
                source_run_id="src-1",
                source_sha256="abc",
            )

        # Sonnet still called despite RAG failure
        assert result.decisions[0].route == "sonnet"
        assert result.decisions[0].rag_exemplar_count == 0

    def test_rag_context_with_confidence_boost_scenario(
        self, mock_classifier, mock_gateway
    ):
        """GHERKIN Scenario 2: Low-confidence section with 3 RAG hits
        all classified as B.5, Sonnet receives RAG context and
        chooses B.5."""
        mock_classifier.classify.return_value = [
            _make_section("B.4", "Trial Design", 0.55),
        ]

        # All 3 RAG hits point to B.5
        mock_rag_index = MagicMock()
        mock_rag_index.query.return_value = [
            _FakeRagExemplar("B.5", "Selection", "NCT00000010",
                             0.95, "Inclusion/exclusion criteria", 0.91),
            _FakeRagExemplar("B.5", "Selection", "NCT00000020",
                             0.92, "Patient eligibility", 0.87),
            _FakeRagExemplar("B.5", "Selection", "NCT00000030",
                             0.88, "Screening requirements", 0.84),
        ]

        router = ClassificationRouter(
            classifier=mock_classifier,
            gateway=mock_gateway,
            confidence_threshold=0.70,
            rag_index=mock_rag_index,
        )
        router._use_sonnet = True
        router._api_key = "test-key"

        # Sonnet sees the RAG context and agrees with B.5
        mock_response = _make_sonnet_response(
            "B.5",
            "RAG context strongly supports B.5 classification",
        )
        with patch.object(
            router, "_get_sonnet_client"
        ) as mock_get_client:
            mock_client = MagicMock()
            mock_client.messages.create.return_value = mock_response
            mock_get_client.return_value = mock_client

            result = router.classify(
                text_blocks=[
                    {"page_number": 1, "text": "eligibility criteria"},
                ],
                registry_id="NCT00000001",
                run_id="run-1",
                source_run_id="src-1",
                source_sha256="abc",
            )

        d = result.decisions[0]
        assert d.route == "sonnet"
        assert d.final_section_code == "B.5"
        assert d.rag_exemplar_count == 3

        # Verify all 3 RAG exemplars were in the prompt
        call_args = mock_client.messages.create.call_args
        prompt_text = call_args.kwargs["messages"][0]["content"]
        assert "NCT00000010" in prompt_text
        assert "NCT00000020" in prompt_text
        assert "NCT00000030" in prompt_text

    def test_high_confidence_skips_rag(
        self, mock_classifier, mock_gateway
    ):
        """High-confidence sections stay local — RAG never queried."""
        mock_classifier.classify.return_value = [
            _make_section("B.1", "General", 0.90),
        ]

        mock_rag_index = MagicMock()

        router = ClassificationRouter(
            classifier=mock_classifier,
            gateway=mock_gateway,
            confidence_threshold=0.70,
            rag_index=mock_rag_index,
        )
        router._use_sonnet = False

        result = router.classify(
            text_blocks=[{"page_number": 1, "text": "general"}],
            registry_id="NCT00000001",
            run_id="run-1",
            source_run_id="src-1",
            source_sha256="abc",
        )

        # High confidence → local → RAG not queried
        mock_rag_index.query.assert_not_called()
        assert result.decisions[0].route == "local"
        assert result.decisions[0].rag_exemplar_count == 0
