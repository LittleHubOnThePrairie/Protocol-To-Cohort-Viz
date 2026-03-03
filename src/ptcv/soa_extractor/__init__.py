"""PTCV SoA Extractor — Schedule of Activities to CDISC USDM v4.0 pipeline.

Public API for PTCV-21: Schedule of Activities Extractor and USDM Mapper.

Main entry point: SoaExtractor.extract() accepts a list of IchSection
objects from PTCV-20 and returns an ExtractResult with artifact keys and
entity counts.

Typical usage::

    from ptcv.ich_parser import IchParser
    from ptcv.soa_extractor import SoaExtractor

    parse_result = IchParser().parse(protocol_text, registry_id="NCT00112827")
    # ... retrieve sections from stored Parquet ...
    extractor = SoaExtractor()
    result = extractor.extract(
        sections=sections,
        registry_id="NCT00112827",
        source_run_id=parse_result.run_id,
        source_sha256=parse_result.artifact_sha256,
    )
"""

from .extractor import SoaExtractor
from .mapper import UsdmMapper
from .models import (
    ExtractResult,
    RawSoaTable,
    SynonymMapping,
    UsdmActivity,
    UsdmEpoch,
    UsdmScheduledInstance,
    UsdmTimepoint,
)
from .parser import SoaTableParser
from .resolver import SynonymResolver
from .table_discovery import TableDiscovery
from .writer import UsdmParquetWriter

__all__ = [
    "SoaExtractor",
    "UsdmMapper",
    "SoaTableParser",
    "SynonymResolver",
    "TableDiscovery",
    "UsdmParquetWriter",
    "ExtractResult",
    "RawSoaTable",
    "SynonymMapping",
    "UsdmActivity",
    "UsdmEpoch",
    "UsdmScheduledInstance",
    "UsdmTimepoint",
]
