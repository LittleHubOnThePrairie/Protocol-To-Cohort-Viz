"""Tests for PTCV-230: Registry metadata injection into QueryExtractor routes.

Verifies that _inject_registry_routes() and _resolve_nct_id() correctly
inject ClinicalTrials.gov registry content into query extraction routes.
"""

import pytest
from unittest.mock import MagicMock, patch

from ptcv.ich_parser.query_extractor import QueryExtractor
from ptcv.ich_parser.section_matcher import MatchConfidence


def _make_protocol_index(
    source_path: str = "",
    full_text: str = "",
) -> MagicMock:
    """Create a mock ProtocolIndex."""
    pi = MagicMock()
    pi.source_path = source_path
    pi.full_text = full_text
    pi.section_headers = []
    pi.content_spans = {}
    return pi


def _make_mapped_section(
    section_code: str,
    section_name: str,
    content_text: str,
    quality_rating: float = 0.9,
) -> MagicMock:
    """Create a mock MappedRegistrySection."""
    s = MagicMock()
    s.section_code = section_code
    s.section_name = section_name
    s.content_text = content_text
    s.quality_rating = quality_rating
    return s


class TestResolveNctId:
    """Tests for QueryExtractor._resolve_nct_id()."""

    def test_from_filename(self):
        """NCT ID extracted from source_path filename."""
        pi = _make_protocol_index(
            source_path="/data/protocols/NCT05376319_protocol.pdf"
        )
        assert QueryExtractor._resolve_nct_id(pi) == "NCT05376319"

    def test_from_front_matter(self):
        """NCT ID extracted from full_text front-matter."""
        pi = _make_protocol_index(
            source_path="/data/protocols/some_protocol.pdf",
            full_text="Protocol Title\nClinicalTrials.gov: NCT01512251\nVersion 3.0",
        )
        assert QueryExtractor._resolve_nct_id(pi) == "NCT01512251"

    def test_filename_takes_priority(self):
        """Filename NCT ID preferred over front-matter."""
        pi = _make_protocol_index(
            source_path="NCT11111111.pdf",
            full_text="Registry: NCT22222222",
        )
        assert QueryExtractor._resolve_nct_id(pi) == "NCT11111111"

    def test_no_nct_id(self):
        """Returns None when no NCT ID found."""
        pi = _make_protocol_index(
            source_path="/data/protocol.pdf",
            full_text="A protocol without any registry reference.",
        )
        assert QueryExtractor._resolve_nct_id(pi) is None

    def test_empty_protocol_index(self):
        """Returns None for empty protocol index."""
        pi = _make_protocol_index(source_path="", full_text="")
        assert QueryExtractor._resolve_nct_id(pi) is None


class TestInjectRegistryRoutes:
    """Tests for QueryExtractor._inject_registry_routes()."""

    def _patch_registry(
        self, metadata: dict, mapped_sections: list,
    ):
        """Context manager patching fetcher and mapper."""
        mock_fetcher_cls = MagicMock()
        mock_fetcher = MagicMock()
        mock_fetcher.fetch.return_value = metadata
        mock_fetcher_cls.return_value = mock_fetcher

        mock_mapper_cls = MagicMock()
        mock_mapper = MagicMock()
        mock_mapper.map.return_value = mapped_sections
        mock_mapper_cls.return_value = mock_mapper

        return patch.multiple(
            "ptcv.ich_parser.query_extractor",
            **{},  # empty — we patch via import side_effect below
        ), mock_fetcher_cls, mock_mapper_cls

    @patch("ptcv.registry.metadata_fetcher.RegistryMetadataFetcher")
    @patch("ptcv.registry.ich_mapper.MetadataToIchMapper")
    def test_fills_missing_route(self, MockMapper, MockFetcher):
        """Registry creates route when no PDF route exists."""
        MockFetcher.return_value.fetch.return_value = {
            "protocolSection": {"identificationModule": {}}
        }
        MockMapper.return_value.map.return_value = [
            _make_mapped_section("B.3", "Trial Objectives",
                                 "Official Title: Test Trial"),
        ]

        routes: dict[str, tuple[str, str, MatchConfidence]] = {}
        pi = _make_protocol_index(
            source_path="NCT05376319.pdf"
        )

        QueryExtractor._inject_registry_routes(routes, pi)

        assert "B.3" in routes
        content, src, conf = routes["B.3"]
        assert "[REGISTRY" in content
        assert "Official Title: Test Trial" in content
        assert "registry:NCT05376319" in src
        assert conf == MatchConfidence.REVIEW

    @patch("ptcv.registry.metadata_fetcher.RegistryMetadataFetcher")
    @patch("ptcv.registry.ich_mapper.MetadataToIchMapper")
    def test_supplements_low_confidence_route(
        self, MockMapper, MockFetcher,
    ):
        """Registry appended to LOW confidence PDF route."""
        MockFetcher.return_value.fetch.return_value = {
            "protocolSection": {}
        }
        MockMapper.return_value.map.return_value = [
            _make_mapped_section("B.4", "Trial Design",
                                 "Phase: PHASE2, Randomized"),
        ]

        routes: dict[str, tuple[str, str, MatchConfidence]] = {
            "B.4": (
                "Some PDF text about randomization procedures.",
                "6.1",
                MatchConfidence.LOW,
            ),
        }
        pi = _make_protocol_index(
            source_path="NCT05376319.pdf"
        )

        QueryExtractor._inject_registry_routes(routes, pi)

        content, src, conf = routes["B.4"]
        assert "Some PDF text" in content
        assert "[REGISTRY" in content
        assert "Phase: PHASE2" in content
        assert "registry:NCT05376319" in src
        assert conf == MatchConfidence.REVIEW

    @patch("ptcv.registry.metadata_fetcher.RegistryMetadataFetcher")
    @patch("ptcv.registry.ich_mapper.MetadataToIchMapper")
    def test_skips_high_confidence_route(
        self, MockMapper, MockFetcher,
    ):
        """Registry NOT injected when PDF route is HIGH confidence."""
        MockFetcher.return_value.fetch.return_value = {
            "protocolSection": {}
        }
        MockMapper.return_value.map.return_value = [
            _make_mapped_section("B.5", "Selection of Subjects",
                                 "Registry eligibility criteria"),
        ]

        original_content = "Detailed PDF eligibility section."
        routes: dict[str, tuple[str, str, MatchConfidence]] = {
            "B.5": (
                original_content,
                "5.1",
                MatchConfidence.HIGH,
            ),
        }
        pi = _make_protocol_index(
            source_path="NCT05376319.pdf"
        )

        QueryExtractor._inject_registry_routes(routes, pi)

        content, src, conf = routes["B.5"]
        assert content == original_content
        assert "registry" not in src
        assert conf == MatchConfidence.HIGH

    @patch("ptcv.registry.metadata_fetcher.RegistryMetadataFetcher")
    @patch("ptcv.registry.ich_mapper.MetadataToIchMapper")
    def test_skips_review_confidence_route(
        self, MockMapper, MockFetcher,
    ):
        """Registry NOT injected when PDF route is REVIEW confidence."""
        MockFetcher.return_value.fetch.return_value = {
            "protocolSection": {}
        }
        MockMapper.return_value.map.return_value = [
            _make_mapped_section("B.1", "General Information",
                                 "Sponsor: Mayo Clinic"),
        ]

        routes: dict[str, tuple[str, str, MatchConfidence]] = {
            "B.1": (
                "Front-matter with title and sponsor.",
                "front_matter",
                MatchConfidence.REVIEW,
            ),
        }
        pi = _make_protocol_index(
            source_path="NCT05376319.pdf"
        )

        QueryExtractor._inject_registry_routes(routes, pi)

        content, _, conf = routes["B.1"]
        assert "Sponsor: Mayo Clinic" not in content
        assert conf == MatchConfidence.REVIEW

    def test_no_nct_id_skips(self):
        """No injection when NCT ID cannot be resolved."""
        routes: dict[str, tuple[str, str, MatchConfidence]] = {}
        pi = _make_protocol_index(
            source_path="unknown_protocol.pdf",
            full_text="No registry reference here.",
        )

        QueryExtractor._inject_registry_routes(routes, pi)

        assert routes == {}

    @patch("ptcv.registry.metadata_fetcher.RegistryMetadataFetcher")
    @patch("ptcv.registry.ich_mapper.MetadataToIchMapper")
    def test_no_metadata_skips(self, MockMapper, MockFetcher):
        """No injection when fetcher returns None."""
        MockFetcher.return_value.fetch.return_value = None

        routes: dict[str, tuple[str, str, MatchConfidence]] = {}
        pi = _make_protocol_index(
            source_path="NCT05376319.pdf"
        )

        QueryExtractor._inject_registry_routes(routes, pi)

        assert routes == {}
        MockMapper.return_value.map.assert_not_called()

    @patch("ptcv.registry.metadata_fetcher.RegistryMetadataFetcher")
    @patch("ptcv.registry.ich_mapper.MetadataToIchMapper")
    def test_multiple_sections_injected(
        self, MockMapper, MockFetcher,
    ):
        """Multiple registry sections injected at once."""
        MockFetcher.return_value.fetch.return_value = {
            "protocolSection": {}
        }
        MockMapper.return_value.map.return_value = [
            _make_mapped_section("B.1", "General", "Sponsor: Mayo"),
            _make_mapped_section("B.3", "Objectives", "Title: Study X"),
            _make_mapped_section("B.8", "Efficacy", "Primary: Remission"),
        ]

        routes: dict[str, tuple[str, str, MatchConfidence]] = {}
        pi = _make_protocol_index(
            source_path="NCT05376319.pdf"
        )

        QueryExtractor._inject_registry_routes(routes, pi)

        assert "B.1" in routes
        assert "B.3" in routes
        assert "B.8" in routes
        assert "Sponsor: Mayo" in routes["B.1"][0]
        assert "Title: Study X" in routes["B.3"][0]
        assert "Primary: Remission" in routes["B.8"][0]

    @patch("ptcv.registry.metadata_fetcher.RegistryMetadataFetcher")
    @patch("ptcv.registry.ich_mapper.MetadataToIchMapper")
    def test_provenance_marker_present(
        self, MockMapper, MockFetcher,
    ):
        """Registry content tagged with [REGISTRY] provenance."""
        MockFetcher.return_value.fetch.return_value = {
            "protocolSection": {}
        }
        MockMapper.return_value.map.return_value = [
            _make_mapped_section("B.8", "Efficacy",
                                 "Primary: Complete Remission"),
        ]

        routes: dict[str, tuple[str, str, MatchConfidence]] = {}
        pi = _make_protocol_index(
            source_path="NCT05376319.pdf"
        )

        QueryExtractor._inject_registry_routes(routes, pi)

        content = routes["B.8"][0]
        assert "[REGISTRY" in content
        assert "NCT05376319" in content
