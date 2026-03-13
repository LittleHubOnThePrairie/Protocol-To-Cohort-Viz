"""PTCV Pipeline orchestration package (PTCV-24, PTCV-59, PTCV-60).

Public API:
    PipelineOrchestrator  — wires all 7 pipeline stages
    PipelineResult        — result data model
    StageCheckpoint       — per-stage lineage checkpoint
    LineageChainVerification — lineage chain verification result
    create_gateway        — factory for StorageGateway backends
    PIPELINE_STAGES       — ordered list of stage names
    AmendmentChange       — single categorised protocol change
    AmendmentDiff         — diff result between two protocol versions
    AmendmentDiffEngine   — compares MockSdtmDataset objects
    ProtocolVersion       — version metadata for ancestry tracking
    VersionAncestry       — chronological version chain
"""

from .degradation import (
    ClassificationLevel,
    ExtractionLevel,
    PipelineCapabilities,
    detect_capabilities,
    select_classification_level,
    select_extraction_level,
)
from .amendment_diff import (
    AmendmentChange,
    AmendmentDiff,
    AmendmentDiffEngine,
    ProtocolVersion,
    VersionAncestry,
)
from .gateway_factory import create_gateway
from .models import (
    LineageChainVerification,
    PipelineResult,
    StageCheckpoint,
    PIPELINE_STAGES,
)
from .orchestrator import PipelineOrchestrator

__all__ = [
    "AmendmentChange",
    "AmendmentDiff",
    "AmendmentDiffEngine",
    "ClassificationLevel",
    "ExtractionLevel",
    "LineageChainVerification",
    "PIPELINE_STAGES",
    "PipelineCapabilities",
    "PipelineOrchestrator",
    "PipelineResult",
    "ProtocolVersion",
    "StageCheckpoint",
    "VersionAncestry",
    "create_gateway",
    "detect_capabilities",
    "select_classification_level",
    "select_extraction_level",
]
