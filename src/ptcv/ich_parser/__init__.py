"""ICH E6(R3) Section Parser package.

Public API:

    IchParser            — Main service: classify → store → lineage
    IchSection           — Data model for one classified section
    SectionClassifier    — Abstract base for classification backends
    RuleBasedClassifier  — Regex/keyword classifier (no external deps)
    RAGClassifier        — Cohere embeddings + Claude Sonnet classifier
    ReviewQueue          — SQLite append-only human review queue
    FormatDetector       — Pre-parser format detection gate (PTCV-31)
    ProtocolFormat       — Enum: ICH_E6R3, ICH_M11, CTD, FDA_IND, UNKNOWN
    FormatDetectionResult — Result of format detection
    M11ProtocolParser    — ICH M11 CeSHarP machine-readable parser (PTCV-56)
    ProtocolIndex        — Navigable document index (TOC + headers) (PTCV-89)
    extract_protocol_index — Extract ProtocolIndex from PDF (PTCV-89)
    QueryExtractor       — Query-driven extraction engine (PTCV-91)
    QueryExtraction      — Single extracted answer (PTCV-91)
    ExtractionGap        — Unanswered query (PTCV-91)
    ExtractionResult     — Aggregate extraction result (PTCV-91)
    assemble_template    — Template assembler: hits → Appendix B doc (PTCV-92)
    AssembledProtocol    — Assembled protocol output (PTCV-92)
    run_benchmark        — Benchmark entry point (PTCV-93)
    BenchmarkReport      — Aggregate benchmark report (PTCV-93)
    RefinementStore      — Iterative refinement feedback store (PTCV-94)
"""

from .models import IchSection, ReviewQueueEntry
from .classifier import SectionClassifier, RuleBasedClassifier, RAGClassifier
from .fidelity_checker import FidelityChecker
from .format_detector import FormatDetector, FormatDetectionResult, ProtocolFormat
from .m11_parser import M11ProtocolParser
from .query_extractor import (
    ExtractionGap,
    ExtractionResult,
    QueryExtraction,
    QueryExtractor,
)
from .query_schema import AppendixBQuery, load_query_schema
from .review_queue import ReviewQueue
from .section_matcher import (
    MatchConfidence,
    MatchResult,
    SectionMatch,
    SectionMatcher,
    SectionMapping,
)
from .benchmark import BenchmarkReport, run_benchmark
from .refinement_store import RefinementStore
from .template_assembler import AssembledProtocol, assemble_template
from .toc_extractor import ProtocolIndex, extract_protocol_index
from .parser import IchParser

__all__ = [
    "AppendixBQuery",
    "AssembledProtocol",
    "BenchmarkReport",
    "ExtractionGap",
    "ExtractionResult",
    "FidelityChecker",
    "IchParser",
    "IchSection",
    "ReviewQueueEntry",
    "SectionClassifier",
    "RuleBasedClassifier",
    "RAGClassifier",
    "ReviewQueue",
    "FormatDetector",
    "FormatDetectionResult",
    "ProtocolFormat",
    "MatchConfidence",
    "MatchResult",
    "M11ProtocolParser",
    "ProtocolIndex",
    "QueryExtraction",
    "QueryExtractor",
    "SectionMatch",
    "SectionMatcher",
    "SectionMapping",
    "assemble_template",
    "extract_protocol_index",
    "load_query_schema",
    "RefinementStore",
    "run_benchmark",
]
