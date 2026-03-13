"""Query pipeline artifact persistence (PTCV-126).

Saves query pipeline results to disk via ``FilesystemAdapter`` so that:
1. Artifacts survive Streamlit session restarts (ALCOA++ Enduring).
2. Each artifact is SHA-256 hashed and recorded in ``lineage.db``
   (ALCOA++ Attributable + Contemporaneous).
3. The SoA tab can load ``AssembledProtocol`` from disk when
   ``st.session_state`` is empty.

Artifact layout under ``<data_root>/``::

    query-pipeline/<run_id>/assembled_protocol.json
    query-pipeline/<run_id>/match_result.json
    query-pipeline/<run_id>/coverage.json
    query-pipeline/<run_id>/stage_timings.json

A manifest (``query-pipeline/<file_sha>/manifest.json``) maps
``file_sha`` → latest ``run_id`` for retrieval after restart.

Risk tier: LOW — persistence adapter, no destructive side effects.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import uuid
from pathlib import Path
from typing import Any, Optional

from ..ich_parser.template_assembler import AssembledProtocol
from ..storage.gateway import StorageGateway

logger = logging.getLogger(__name__)

_STAGE = "query_pipeline"
_USER = "streamlit-query-pipeline"


def save_query_artifacts(
    gateway: StorageGateway,
    file_sha: str,
    result: dict[str, Any],
    registry_id: str = "",
) -> str:
    """Persist query pipeline result dict to storage with lineage.

    Args:
        gateway: Initialised ``StorageGateway`` (typically
            ``FilesystemAdapter``).
        file_sha: SHA-256 hex digest of the source PDF.
        result: Dict returned by ``run_query_pipeline()``.
        registry_id: Trial registry identifier for lineage metadata.

    Returns:
        The ``run_id`` assigned to this persistence batch.
    """
    run_id = str(uuid.uuid4())

    # 1. Assembled protocol (the primary artifact)
    assembled: Optional[AssembledProtocol] = result.get("assembled")
    if assembled is not None:
        assembled_dict = assembled.to_dict()
        assembled_bytes = json.dumps(
            assembled_dict, ensure_ascii=False, default=str,
        ).encode("utf-8")
        gateway.put_artifact(
            key=f"query-pipeline/{run_id}/assembled_protocol.json",
            data=assembled_bytes,
            content_type="application/json",
            run_id=run_id,
            source_hash=file_sha,
            user=_USER,
            stage=_STAGE,
            registry_id=registry_id or None,
        )
        logger.info(
            "Saved assembled_protocol.json (%d bytes, run=%s)",
            len(assembled_bytes), run_id,
        )

    # 2. Coverage report
    coverage = result.get("coverage")
    if coverage is not None:
        if dataclasses.is_dataclass(coverage) and not isinstance(
            coverage, type,
        ):
            coverage_dict = dataclasses.asdict(coverage)
        else:
            coverage_dict = coverage
        coverage_bytes = json.dumps(
            coverage_dict, ensure_ascii=False, default=str,
        ).encode("utf-8")
        gateway.put_artifact(
            key=f"query-pipeline/{run_id}/coverage.json",
            data=coverage_bytes,
            content_type="application/json",
            run_id=run_id,
            source_hash=file_sha,
            user=_USER,
            stage=_STAGE,
            registry_id=registry_id or None,
        )

    # 3. Stage timings
    timings = result.get("stage_timings")
    if timings:
        timings_bytes = json.dumps(
            timings, ensure_ascii=False,
        ).encode("utf-8")
        gateway.put_artifact(
            key=f"query-pipeline/{run_id}/stage_timings.json",
            data=timings_bytes,
            content_type="application/json",
            run_id=run_id,
            source_hash=file_sha,
            user=_USER,
            stage=_STAGE,
            registry_id=registry_id or None,
        )

    # 4. Assembled markdown (human-readable artefact)
    markdown = result.get("assembled_markdown")
    if markdown:
        md_bytes = markdown.encode("utf-8")
        gateway.put_artifact(
            key=f"query-pipeline/{run_id}/assembled_protocol.md",
            data=md_bytes,
            content_type="text/markdown",
            run_id=run_id,
            source_hash=file_sha,
            user=_USER,
            stage=_STAGE,
            registry_id=registry_id or None,
        )

    # 5. Write manifest mapping file_sha → run_id.
    manifest = {
        "file_sha": file_sha,
        "run_id": run_id,
        "registry_id": registry_id,
    }
    manifest_bytes = json.dumps(manifest).encode("utf-8")
    gateway.put_artifact(
        key=f"query-pipeline/{file_sha}/manifest.json",
        data=manifest_bytes,
        content_type="application/json",
        run_id=run_id,
        source_hash=file_sha,
        user=_USER,
        stage=_STAGE,
        registry_id=registry_id or None,
    )

    logger.info(
        "Query pipeline artifacts saved: run_id=%s, file_sha=%s",
        run_id, file_sha,
    )
    return run_id


def load_assembled_protocol(
    gateway: StorageGateway,
    file_sha: str,
) -> Optional[AssembledProtocol]:
    """Load the latest AssembledProtocol from disk for a given PDF.

    Reads the manifest for ``file_sha`` to find the latest ``run_id``,
    then loads and deserialises ``assembled_protocol.json``.

    Args:
        gateway: Initialised ``StorageGateway``.
        file_sha: SHA-256 hex digest of the source PDF.

    Returns:
        ``AssembledProtocol`` if found on disk, ``None`` otherwise.
    """
    manifest_key = f"query-pipeline/{file_sha}/manifest.json"
    try:
        manifest_bytes = gateway.get_artifact(manifest_key)
    except FileNotFoundError:
        return None

    try:
        manifest = json.loads(manifest_bytes.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        logger.warning("Failed to parse manifest %s: %s", manifest_key, exc)
        return None

    run_id = manifest.get("run_id", "")
    if not run_id:
        return None

    assembled_key = f"query-pipeline/{run_id}/assembled_protocol.json"
    try:
        assembled_bytes = gateway.get_artifact(assembled_key)
    except FileNotFoundError:
        logger.warning(
            "Manifest exists but assembled_protocol.json missing: %s",
            assembled_key,
        )
        return None

    try:
        assembled_dict = json.loads(assembled_bytes.decode("utf-8"))
        return AssembledProtocol.from_dict(assembled_dict)
    except (json.JSONDecodeError, UnicodeDecodeError, KeyError) as exc:
        logger.warning(
            "Failed to deserialise assembled_protocol.json: %s", exc,
        )
        return None


def has_query_artifacts(
    gateway: StorageGateway,
    file_sha: str,
) -> bool:
    """Check whether query pipeline artifacts exist on disk.

    Args:
        gateway: Initialised ``StorageGateway``.
        file_sha: SHA-256 hex digest of the source PDF.

    Returns:
        ``True`` if a manifest and assembled protocol exist on disk.
    """
    return load_assembled_protocol(gateway, file_sha) is not None
