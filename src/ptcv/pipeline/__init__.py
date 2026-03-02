"""PTCV Pipeline orchestration package (PTCV-24).

Public API:
    PipelineOrchestrator  — wires all 6 pipeline stages
    PipelineResult        — result data model
    StageCheckpoint       — per-stage lineage checkpoint
    LineageChainVerification — lineage chain verification result
    create_gateway        — factory for StorageGateway backends
    PIPELINE_STAGES       — ordered list of stage names
"""

from .gateway_factory import create_gateway
from .models import (
    LineageChainVerification,
    PipelineResult,
    StageCheckpoint,
    PIPELINE_STAGES,
)
from .orchestrator import PipelineOrchestrator

__all__ = [
    "PipelineOrchestrator",
    "PipelineResult",
    "StageCheckpoint",
    "LineageChainVerification",
    "create_gateway",
    "PIPELINE_STAGES",
]
