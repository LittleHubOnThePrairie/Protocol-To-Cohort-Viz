"""PTCV SDTM Trial Design Domain Generator and Validator (PTCV-22, PTCV-23).

Generates CDISC SDTM v1.7 trial design datasets (TS, TA, TE, TV, TI)
and Define-XML v2.1 from ICH E6(R3) sections and USDM SoA timepoints.
All artifacts are written to StorageGateway with immutable=True (WORM).

Validation (PTCV-23): ValidationService runs Pinnacle 21 Community rules,
FDA TCG v5.9 Appendix B checks, and Define-XML structural validation.
"""

from .ct_normalizer import CtLookupResult, CtNormalizer
from .define_xml import DefineXmlGenerator
from .domain_generators import (
    TaGenerator,
    TeGenerator,
    TiGenerator,
    TsGenerator,
    TvGenerator,
)
from .models import CtReviewQueueEntry, SdtmGenerationResult, SoaCellMetadata
from .review_queue import CtReviewQueue
from .sdtm_service import SdtmService
from .validation import (
    DefineXmlIssue,
    DefineXmlValidator,
    P21Issue,
    P21Validator,
    ScheduleIssue,
    ScheduleValidationReport,
    TcgChecker,
    TcgParameter,
    ValidationResult,
    ValidationService,
    VisitScheduleValidator,
)

__all__ = [
    "CtLookupResult",
    "CtNormalizer",
    "CtReviewQueue",
    "CtReviewQueueEntry",
    "DefineXmlGenerator",
    "DefineXmlIssue",
    "DefineXmlValidator",
    "P21Issue",
    "P21Validator",
    "ScheduleIssue",
    "ScheduleValidationReport",
    "SdtmGenerationResult",
    "SdtmService",
    "SoaCellMetadata",
    "TaGenerator",
    "TcgChecker",
    "TcgParameter",
    "TeGenerator",
    "TiGenerator",
    "TsGenerator",
    "TvGenerator",
    "ValidationResult",
    "ValidationService",
    "VisitScheduleValidator",
]
