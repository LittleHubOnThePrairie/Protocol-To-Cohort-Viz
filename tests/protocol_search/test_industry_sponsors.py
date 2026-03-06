"""Tests for PTCV-39 industry sponsor lookup table.

Qualification phase: OQ (operational qualification)
Regulatory requirement: PTCV-39 — ICH E6(R3)-aligned sampling strategy.
Risk tier: LOW — static reference data validation.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[2] / "src"))

from ptcv.protocol_search.industry_sponsors import (
    ALIAS_TO_CANONICAL,
    INDUSTRY_SPONSORS,
    canonicalise_sponsor,
    is_known_industry_sponsor,
)


class TestIndustrySponsorLookupTable:
    """PTCV-39 Scenario: Industry sponsor lookup table is created."""

    def test_contains_at_least_50_sponsors(self):
        """Lookup table has >= 50 canonical sponsor entries."""
        assert len(INDUSTRY_SPONSORS) >= 50

    def test_every_entry_has_at_least_one_alias(self):
        """Each canonical sponsor maps to a non-empty alias list."""
        for canonical, aliases in INDUSTRY_SPONSORS.items():
            assert len(aliases) >= 1, f"{canonical} has no aliases"

    def test_canonical_name_is_in_alias_list(self):
        """The canonical name itself should appear in its alias list."""
        for canonical, aliases in INDUSTRY_SPONSORS.items():
            alias_lower = {a.lower() for a in aliases}
            assert canonical.lower() in alias_lower, (
                f"Canonical name '{canonical}' not in its own alias list"
            )

    def test_no_duplicate_aliases_across_sponsors(self):
        """Aliases must not appear under multiple canonical names."""
        seen: dict[str, str] = {}
        for canonical, aliases in INDUSTRY_SPONSORS.items():
            for alias in aliases:
                key = alias.lower()
                if key in seen:
                    pytest.fail(
                        f"Alias '{alias}' appears under both "
                        f"'{seen[key]}' and '{canonical}'"
                    )
                seen[key] = canonical

    def test_major_pharma_companies_present(self):
        """Top pharma companies are represented in the lookup."""
        expected = [
            "Pfizer", "Novartis", "Roche", "AstraZeneca",
            "Merck & Co.", "Sanofi", "AbbVie", "GSK",
            "Eli Lilly", "Bristol-Myers Squibb", "Amgen",
            "Gilead Sciences", "Novo Nordisk", "Takeda",
            "Johnson & Johnson", "Bayer",
        ]
        for name in expected:
            assert name in INDUSTRY_SPONSORS, f"{name} missing from lookup"


class TestIsKnownIndustrySponsor:
    """Unit tests for is_known_industry_sponsor()."""

    def test_exact_match(self):
        """Exact alias match returns True."""
        assert is_known_industry_sponsor("Pfizer") is True

    def test_case_insensitive_match(self):
        """Match is case-insensitive."""
        assert is_known_industry_sponsor("pfizer") is True
        assert is_known_industry_sponsor("PFIZER") is True
        assert is_known_industry_sponsor("AstraZeneca") is True
        assert is_known_industry_sponsor("astrazeneca") is True

    def test_alias_match(self):
        """Non-canonical alias returns True."""
        assert is_known_industry_sponsor("Genentech") is True
        assert is_known_industry_sponsor("Genentech, Inc.") is True
        assert is_known_industry_sponsor("Janssen") is True
        assert is_known_industry_sponsor("Wyeth") is True

    def test_unknown_sponsor_returns_false(self):
        """Unknown name returns False."""
        assert is_known_industry_sponsor("Unknown Sponsor Corp") is False
        assert is_known_industry_sponsor("City of Hope Medical Center") is False
        assert is_known_industry_sponsor("") is False


class TestCanonicaliseSponsor:
    """Unit tests for canonicalise_sponsor()."""

    def test_canonical_name_returned(self):
        """Exact canonical match returns itself."""
        assert canonicalise_sponsor("Pfizer") == "Pfizer"

    def test_alias_resolves_to_canonical(self):
        """Alias resolves to the canonical company name."""
        assert canonicalise_sponsor("Genentech, Inc.") == "Roche"
        assert canonicalise_sponsor("Janssen") == "Johnson & Johnson"
        assert canonicalise_sponsor("Wyeth") == "Pfizer"
        assert canonicalise_sponsor("Celgene") == "Bristol-Myers Squibb"

    def test_case_insensitive(self):
        """Lookup is case-insensitive."""
        assert canonicalise_sponsor("genentech, inc.") == "Roche"

    def test_unknown_returns_none(self):
        """Unknown sponsor returns None."""
        assert canonicalise_sponsor("Random Research Lab") is None
        assert canonicalise_sponsor("") is None


class TestAliasToCanonical:
    """Structural integrity of the reverse-lookup dict."""

    def test_alias_count_matches_all_aliases(self):
        """ALIAS_TO_CANONICAL has one entry per unique alias."""
        total_aliases = sum(len(a) for a in INDUSTRY_SPONSORS.values())
        assert len(ALIAS_TO_CANONICAL) == total_aliases

    def test_all_values_are_canonical_names(self):
        """Every value in the reverse map is a key in INDUSTRY_SPONSORS."""
        for alias, canonical in ALIAS_TO_CANONICAL.items():
            assert canonical in INDUSTRY_SPONSORS, (
                f"Reverse map value '{canonical}' for alias '{alias}' "
                f"is not a canonical name"
            )
