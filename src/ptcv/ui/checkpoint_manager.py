"""Pipeline checkpoint manager for session resilience (PTCV-83).

Persists pipeline stage results to disk so users can recover from
browser disconnects or Streamlit session timeouts. Checkpoints are
keyed by protocol file SHA-256 and stored as JSON alongside the
existing FilesystemAdapter storage tree.

Checkpoint directory: ``<data_root>/checkpoints/<file_sha>/``
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Stage keys that produce cacheable results in app.py
_CACHE_STAGES = ("parse_cache", "soa_cache", "fidelity_cache", "sdtm_cache")

# Keys in parse_cache that reference storage artifacts (not serialisable)
_SKIP_KEYS = ("extracted_tables", "text_block_dicts")


def _checkpoint_dir(data_root: Path, file_sha: str) -> Path:
    """Return the checkpoint directory for a protocol file."""
    return data_root / "checkpoints" / file_sha


def _sanitise_for_json(data: dict[str, Any]) -> dict[str, Any]:
    """Remove non-serialisable keys from a cache dict."""
    return {
        k: v for k, v in data.items()
        if k not in _SKIP_KEYS
    }


def save_checkpoint(
    data_root: Path,
    file_sha: str,
    cache_key: str,
    cache_data: dict[str, Any],
) -> Path:
    """Save a pipeline stage result to a checkpoint file.

    Args:
        data_root: PTCV data root directory.
        file_sha: SHA-256 of the protocol PDF.
        cache_key: Cache identifier (e.g. "parse_cache").
        cache_data: Stage result dict to persist.

    Returns:
        Path to the written checkpoint file.
    """
    cp_dir = _checkpoint_dir(data_root, file_sha)
    cp_dir.mkdir(parents=True, exist_ok=True)
    cp_path = cp_dir / f"{cache_key}.json"
    serialisable = _sanitise_for_json(cache_data)
    cp_path.write_text(
        json.dumps(serialisable, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    logger.info("Checkpoint saved: %s", cp_path)
    return cp_path


def load_checkpoints(
    data_root: Path,
    file_sha: str,
) -> dict[str, dict[str, Any]]:
    """Load all available checkpoints for a protocol file.

    Args:
        data_root: PTCV data root directory.
        file_sha: SHA-256 of the protocol PDF.

    Returns:
        Dict mapping cache_key → cached result dict. Empty if no
        checkpoints exist.
    """
    cp_dir = _checkpoint_dir(data_root, file_sha)
    if not cp_dir.is_dir():
        return {}

    result: dict[str, dict[str, Any]] = {}
    for cache_key in _CACHE_STAGES:
        cp_path = cp_dir / f"{cache_key}.json"
        if cp_path.exists():
            try:
                data = json.loads(
                    cp_path.read_text(encoding="utf-8"),
                )
                result[cache_key] = data
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning(
                    "Failed to load checkpoint %s: %s",
                    cp_path, exc,
                )
    return result


def get_checkpoint_summary(
    data_root: Path,
    file_sha: str,
) -> list[str]:
    """Return list of completed stage names from checkpoints.

    Args:
        data_root: PTCV data root directory.
        file_sha: SHA-256 of the protocol PDF.

    Returns:
        List of cache_key names that have valid checkpoints.
    """
    cp_dir = _checkpoint_dir(data_root, file_sha)
    if not cp_dir.is_dir():
        return []
    return [
        cache_key
        for cache_key in _CACHE_STAGES
        if (cp_dir / f"{cache_key}.json").exists()
    ]


def clear_checkpoints(
    data_root: Path,
    file_sha: str,
) -> int:
    """Delete all checkpoints for a protocol file.

    Args:
        data_root: PTCV data root directory.
        file_sha: SHA-256 of the protocol PDF.

    Returns:
        Number of checkpoint files deleted.
    """
    cp_dir = _checkpoint_dir(data_root, file_sha)
    if not cp_dir.is_dir():
        return 0
    count = 0
    for f in cp_dir.iterdir():
        if f.suffix == ".json":
            f.unlink()
            count += 1
    # Remove empty directory
    try:
        cp_dir.rmdir()
    except OSError:
        pass
    logger.info("Cleared %d checkpoints for %s", count, file_sha)
    return count


def has_checkpoints(data_root: Path, file_sha: str) -> bool:
    """Check if any checkpoints exist for a protocol file."""
    return bool(get_checkpoint_summary(data_root, file_sha))


# Mapping from cache_key to human-readable stage label
_STAGE_LABELS = {
    "parse_cache": "Extraction + Retemplating",
    "soa_cache": "SoA Extraction",
    "fidelity_cache": "Fidelity Check",
    "sdtm_cache": "SDTM Generation",
}

# Mapping from cache_key to the next stage label for resume
_NEXT_STAGE = {
    "parse_cache": "SoA Extraction",
    "soa_cache": "Fidelity Check",
    "fidelity_cache": "SDTM Generation",
    "sdtm_cache": "Validation (complete)",
}


def get_resume_label(data_root: Path, file_sha: str) -> str:
    """Get a human-readable label for the resume point.

    Args:
        data_root: PTCV data root directory.
        file_sha: SHA-256 of the protocol PDF.

    Returns:
        Label like "Resume from SoA Extraction" or empty string.
    """
    completed = get_checkpoint_summary(data_root, file_sha)
    if not completed:
        return ""
    last = completed[-1]
    next_label = _NEXT_STAGE.get(last, "")
    if next_label:
        return f"Resume from {next_label}"
    return "Resume pipeline"
