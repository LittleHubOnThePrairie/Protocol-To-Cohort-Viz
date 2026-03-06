"""Industry sponsor lookup table for ClinicalTrials.gov filtering.

Provides a canonical mapping of major pharmaceutical and biotechnology
companies to their known aliases on ClinicalTrials.gov.  Used for
secondary validation when the API's ``LeadSponsorClass`` field is
ambiguous or when enriching metadata with canonical sponsor names.

The primary filtering mechanism is the ``AREA[LeadSponsorClass]INDUSTRY``
advanced filter on the ClinicalTrials.gov v2 API, which natively
classifies sponsors.  This lookup supplements that with alias resolution.

Implements PTCV-39: Sampling strategy for ICH E6(R3)-compliant protocols.

Risk tier: LOW — static reference data, no PHI.
"""

# ---------------------------------------------------------------------------
# Industry sponsor lookup table
#
# Canonical company name → list of known aliases as they appear in the
# ClinicalTrials.gov ``LeadSponsorName`` field.
#
# Follows the project data pattern established by:
#   _PHASE_CODES      (protocol_search/eu_ctr_service.py)
#   _CONDITION_AREA_CODES (protocol_search/eu_ctr_service.py)
#   _VISIT_SYNONYMS   (soa_extractor/resolver.py)
#   _CT_PHASE          (sdtm/ct_normalizer.py)
# ---------------------------------------------------------------------------

INDUSTRY_SPONSORS: dict[str, list[str]] = {
    # --- Top 20 Global Pharma (by revenue) ---
    "Pfizer": [
        "Pfizer", "Pfizer Inc.", "Pfizer Inc",
        "Pfizer's Upjohn", "Wyeth", "Hospira",
    ],
    "Johnson & Johnson": [
        "Johnson & Johnson", "Janssen", "Janssen Research & Development, LLC",
        "Janssen Pharmaceuticals", "Janssen Sciences Ireland UC",
        "Janssen-Cilag International NV", "Janssen Biotech, Inc.",
    ],
    "Roche": [
        "Roche", "Hoffmann-La Roche", "F. Hoffmann-La Roche",
        "Genentech, Inc.", "Genentech",
    ],
    "Novartis": [
        "Novartis", "Novartis Pharmaceuticals", "Novartis AG",
        "Novartis Pharmaceuticals Corporation",
        "Sandoz", "Sandoz International GmbH",
    ],
    "Merck & Co.": [
        "Merck & Co.", "Merck Sharp & Dohme LLC",
        "Merck Sharp & Dohme Corp.", "Merck & Co., Inc.", "MSD",
    ],
    "AbbVie": [
        "AbbVie", "AbbVie Inc.", "Allergan",
        "Allergan, Inc.", "Allergan plc",
    ],
    "Sanofi": [
        "Sanofi", "Sanofi-Aventis", "Sanofi Pasteur",
        "Sanofi Pasteur, a Sanofi Company", "Regeneron Pharmaceuticals",
    ],
    "AstraZeneca": [
        "AstraZeneca", "AstraZeneca AB",
        "MedImmune LLC", "MedImmune",
        "Alexion Pharmaceuticals", "Alexion, AstraZeneca Rare Disease",
    ],
    "GSK": [
        "GlaxoSmithKline", "GSK", "GlaxoSmithKline plc",
        "ViiV Healthcare",
    ],
    "Eli Lilly": [
        "Eli Lilly and Company", "Eli Lilly", "Lilly",
    ],
    "Bristol-Myers Squibb": [
        "Bristol-Myers Squibb", "Bristol Myers Squibb",
        "Celgene", "Celgene Corporation",
    ],
    "Amgen": [
        "Amgen", "Amgen Inc.",
    ],
    "Gilead Sciences": [
        "Gilead Sciences", "Gilead", "Kite, A Gilead Company",
        "Kite Pharma",
    ],
    "Bayer": [
        "Bayer", "Bayer AG", "Bayer HealthCare AG",
        "Bayer CropScience AG",
    ],
    "Novo Nordisk": [
        "Novo Nordisk", "Novo Nordisk A/S",
    ],
    "Takeda": [
        "Takeda", "Takeda Pharmaceutical Company Limited",
        "Takeda Development Center Americas, Inc.",
        "Shire", "Shire Human Genetic Therapies, Inc.",
    ],
    "Boehringer Ingelheim": [
        "Boehringer Ingelheim", "Boehringer Ingelheim Pharmaceuticals",
    ],
    "Merck KGaA": [
        "Merck KGaA", "Merck KGaA, Darmstadt, Germany",
        "Merck Healthcare KGaA, Darmstadt, Germany, an affiliate of Merck KGaA, Darmstadt, Germany",
        "EMD Serono", "EMD Serono Research & Development Institute, Inc.",
    ],
    "Biogen": [
        "Biogen", "Biogen Idec",
    ],
    "Moderna": [
        "ModernaTX, Inc.", "Moderna",
    ],
    # --- Large Biotech ---
    "Regeneron": [
        "Regeneron Pharmaceuticals, Inc.", "Regeneron",
    ],
    "Vertex Pharmaceuticals": [
        "Vertex Pharmaceuticals Incorporated", "Vertex Pharmaceuticals",
    ],
    "Seagen": [
        "Seagen", "Seagen Inc.", "Seattle Genetics, Inc.",
    ],
    "BioMarin": [
        "BioMarin Pharmaceutical", "BioMarin",
    ],
    "Incyte": [
        "Incyte Corporation", "Incyte",
    ],
    "Jazz Pharmaceuticals": [
        "Jazz Pharmaceuticals", "Jazz Pharmaceuticals, Inc.",
    ],
    "Alcon": [
        "Alcon Research", "Alcon",
    ],
    "Daiichi Sankyo": [
        "Daiichi Sankyo", "Daiichi Sankyo, Inc.",
        "Daiichi Sankyo Co., Ltd.",
    ],
    "Astellas": [
        "Astellas Pharma Inc", "Astellas Pharma Global Development, Inc.",
        "Astellas",
    ],
    "UCB": [
        "UCB Pharma", "UCB Biopharma S.P.R.L.", "UCB",
    ],
    "CSL Behring": [
        "CSL Behring", "CSL",
    ],
    "Eisai": [
        "Eisai Inc.", "Eisai Co., Ltd.", "Eisai",
    ],
    "Ipsen": [
        "Ipsen", "Ipsen Biopharmaceuticals, Inc.",
    ],
    "Otsuka": [
        "Otsuka Pharmaceutical Co., Ltd.",
        "Otsuka Pharmaceutical Development & Commercialization, Inc.",
        "Otsuka",
    ],
    "Lundbeck": [
        "H. Lundbeck A/S", "Lundbeck",
    ],
    "Servier": [
        "Servier", "Les Laboratoires Servier",
    ],
    "Teva": [
        "Teva Pharmaceutical Industries", "Teva",
        "Teva Branded Pharmaceutical Products R&D, Inc.",
    ],
    "Viatris": [
        "Viatris", "Viatris Inc.", "Mylan", "Mylan Inc.",
    ],
    "Sun Pharma": [
        "Sun Pharmaceutical Industries", "Sun Pharma",
    ],
    "Bausch Health": [
        "Bausch Health Americas, Inc.", "Valeant Pharmaceuticals",
        "Bausch Health",
    ],
    "Organon": [
        "Organon", "Organon and Co.",
    ],
    "Haleon": [
        "Haleon", "HALEON Group",
    ],
    "Sumitomo Pharma": [
        "Sumitomo Pharma America, Inc.", "Sunovion Pharmaceuticals Inc.",
        "Sumitomo Pharma",
    ],
    "Ferring": [
        "Ferring Pharmaceuticals", "Ferring",
    ],
    "Almirall": [
        "Almirall, S.A.", "Almirall",
    ],
    "LEO Pharma": [
        "LEO Pharma", "LEO Pharma A/S",
    ],
    "Hikma": [
        "Hikma Pharmaceuticals PLC", "Hikma",
    ],
    "Recordati": [
        "Recordati", "Recordati Rare Diseases",
    ],
    "Menarini": [
        "Menarini", "Menarini Group",
        "A. Menarini Industrie Farmaceutiche Riunite s.r.l.",
    ],
    "Chiesi": [
        "Chiesi Farmaceutici S.p.A.", "Chiesi",
        "Chiesi Poland Sp. z o.o.",
    ],
    "Helsinn": [
        "Helsinn Healthcare SA", "Helsinn",
    ],
    # --- Emerging / Specialty Biotech ---
    "Alnylam": [
        "Alnylam Pharmaceuticals", "Alnylam",
    ],
    "BioNTech": [
        "BioNTech SE", "BioNTech",
    ],
    "Blueprint Medicines": [
        "Blueprint Medicines Corporation", "Blueprint Medicines",
    ],
    "Ultragenyx": [
        "Ultragenyx Pharmaceutical Inc", "Ultragenyx",
    ],
    "Neurocrine Biosciences": [
        "Neurocrine Biosciences", "Neurocrine",
    ],
    "Exact Sciences": [
        "Exact Sciences Corporation", "Exact Sciences",
    ],
    "Ionis": [
        "Ionis Pharmaceuticals, Inc.", "Ionis",
    ],
}

# Flat set of all known aliases for O(1) membership testing.
_ALL_SPONSOR_ALIASES: set[str] = set()
for _aliases in INDUSTRY_SPONSORS.values():
    _ALL_SPONSOR_ALIASES.update(a.lower() for a in _aliases)

# Reverse lookup: lowercased alias → canonical name.
ALIAS_TO_CANONICAL: dict[str, str] = {}
for _canonical, _aliases in INDUSTRY_SPONSORS.items():
    for _alias in _aliases:
        ALIAS_TO_CANONICAL[_alias.lower()] = _canonical


def is_known_industry_sponsor(sponsor_name: str) -> bool:
    """Check whether a sponsor name matches any known industry alias.

    Case-insensitive comparison against all aliases in
    ``INDUSTRY_SPONSORS``.

    Args:
        sponsor_name: Sponsor name as it appears on ClinicalTrials.gov.

    Returns:
        True if the name matches a known industry sponsor alias.
    """
    return sponsor_name.lower() in _ALL_SPONSOR_ALIASES


def canonicalise_sponsor(sponsor_name: str) -> str | None:
    """Return the canonical company name for a sponsor alias.

    Args:
        sponsor_name: Sponsor name as it appears on ClinicalTrials.gov.

    Returns:
        Canonical name (e.g., "Pfizer") or None if not recognised.
    """
    return ALIAS_TO_CANONICAL.get(sponsor_name.lower())
