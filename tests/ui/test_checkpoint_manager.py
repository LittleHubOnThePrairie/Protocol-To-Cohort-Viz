"""Unit tests for checkpoint manager (PTCV-83).

Tests save, load, clear, and resume label logic using
a temporary directory as data root.
"""

import json
from pathlib import Path

import pytest

from ptcv.ui.checkpoint_manager import (
    _CACHE_STAGES,
    _SKIP_KEYS,
    clear_checkpoints,
    get_checkpoint_summary,
    get_resume_label,
    has_checkpoints,
    load_checkpoints,
    save_checkpoint,
)


@pytest.fixture()
def data_root(tmp_path: Path) -> Path:
    """Provide a temporary data root directory."""
    return tmp_path


FILE_SHA = "abc123def456"


class TestSaveCheckpoint:
    """Tests for save_checkpoint()."""

    def test_creates_file(self, data_root: Path) -> None:
        path = save_checkpoint(
            data_root, FILE_SHA, "parse_cache", {"key": "val"},
        )
        assert path.exists()
        assert path.name == "parse_cache.json"

    def test_file_contains_json(self, data_root: Path) -> None:
        save_checkpoint(
            data_root, FILE_SHA, "parse_cache", {"foo": 42},
        )
        cp_dir = data_root / "checkpoints" / FILE_SHA
        data = json.loads(
            (cp_dir / "parse_cache.json").read_text(encoding="utf-8"),
        )
        assert data["foo"] == 42

    def test_skips_non_serialisable_keys(
        self, data_root: Path,
    ) -> None:
        payload = {
            "artifact_key": "test",
            "text_block_dicts": [{"page": 1}],
            "extracted_tables": ["table"],
        }
        save_checkpoint(
            data_root, FILE_SHA, "parse_cache", payload,
        )
        cp_dir = data_root / "checkpoints" / FILE_SHA
        data = json.loads(
            (cp_dir / "parse_cache.json").read_text(encoding="utf-8"),
        )
        assert "artifact_key" in data
        for skip_key in _SKIP_KEYS:
            assert skip_key not in data

    def test_creates_nested_dirs(self, data_root: Path) -> None:
        save_checkpoint(
            data_root, FILE_SHA, "soa_cache", {},
        )
        assert (data_root / "checkpoints" / FILE_SHA).is_dir()

    def test_overwrites_existing(self, data_root: Path) -> None:
        save_checkpoint(
            data_root, FILE_SHA, "parse_cache", {"v": 1},
        )
        save_checkpoint(
            data_root, FILE_SHA, "parse_cache", {"v": 2},
        )
        cp_dir = data_root / "checkpoints" / FILE_SHA
        data = json.loads(
            (cp_dir / "parse_cache.json").read_text(encoding="utf-8"),
        )
        assert data["v"] == 2


class TestLoadCheckpoints:
    """Tests for load_checkpoints()."""

    def test_empty_when_no_dir(self, data_root: Path) -> None:
        result = load_checkpoints(data_root, "nonexistent")
        assert result == {}

    def test_loads_saved_data(self, data_root: Path) -> None:
        save_checkpoint(
            data_root, FILE_SHA, "parse_cache", {"a": 1},
        )
        save_checkpoint(
            data_root, FILE_SHA, "soa_cache", {"b": 2},
        )
        result = load_checkpoints(data_root, FILE_SHA)
        assert "parse_cache" in result
        assert result["parse_cache"]["a"] == 1
        assert "soa_cache" in result
        assert result["soa_cache"]["b"] == 2

    def test_ignores_non_stage_files(self, data_root: Path) -> None:
        cp_dir = data_root / "checkpoints" / FILE_SHA
        cp_dir.mkdir(parents=True)
        (cp_dir / "random.json").write_text("{}")
        result = load_checkpoints(data_root, FILE_SHA)
        assert "random" not in result

    def test_handles_corrupt_json(self, data_root: Path) -> None:
        cp_dir = data_root / "checkpoints" / FILE_SHA
        cp_dir.mkdir(parents=True)
        (cp_dir / "parse_cache.json").write_text("not json!")
        result = load_checkpoints(data_root, FILE_SHA)
        assert "parse_cache" not in result


class TestGetCheckpointSummary:
    """Tests for get_checkpoint_summary()."""

    def test_empty_no_checkpoints(self, data_root: Path) -> None:
        assert get_checkpoint_summary(data_root, FILE_SHA) == []

    def test_returns_saved_stages(self, data_root: Path) -> None:
        save_checkpoint(
            data_root, FILE_SHA, "parse_cache", {},
        )
        save_checkpoint(
            data_root, FILE_SHA, "soa_cache", {},
        )
        summary = get_checkpoint_summary(data_root, FILE_SHA)
        assert "parse_cache" in summary
        assert "soa_cache" in summary

    def test_order_matches_cache_stages(
        self, data_root: Path,
    ) -> None:
        for ck in _CACHE_STAGES:
            save_checkpoint(data_root, FILE_SHA, ck, {})
        summary = get_checkpoint_summary(data_root, FILE_SHA)
        assert summary == list(_CACHE_STAGES)


class TestClearCheckpoints:
    """Tests for clear_checkpoints()."""

    def test_returns_zero_no_checkpoints(
        self, data_root: Path,
    ) -> None:
        assert clear_checkpoints(data_root, "nonexistent") == 0

    def test_deletes_files(self, data_root: Path) -> None:
        save_checkpoint(
            data_root, FILE_SHA, "parse_cache", {},
        )
        save_checkpoint(
            data_root, FILE_SHA, "soa_cache", {},
        )
        count = clear_checkpoints(data_root, FILE_SHA)
        assert count == 2
        assert not has_checkpoints(data_root, FILE_SHA)

    def test_removes_directory(self, data_root: Path) -> None:
        save_checkpoint(
            data_root, FILE_SHA, "parse_cache", {},
        )
        clear_checkpoints(data_root, FILE_SHA)
        cp_dir = data_root / "checkpoints" / FILE_SHA
        assert not cp_dir.exists()


class TestHasCheckpoints:
    """Tests for has_checkpoints()."""

    def test_false_when_empty(self, data_root: Path) -> None:
        assert not has_checkpoints(data_root, FILE_SHA)

    def test_true_after_save(self, data_root: Path) -> None:
        save_checkpoint(
            data_root, FILE_SHA, "parse_cache", {},
        )
        assert has_checkpoints(data_root, FILE_SHA)


class TestGetResumeLabel:
    """Tests for get_resume_label()."""

    def test_empty_no_checkpoints(self, data_root: Path) -> None:
        assert get_resume_label(data_root, FILE_SHA) == ""

    def test_after_parse_suggests_soa(self, data_root: Path) -> None:
        save_checkpoint(
            data_root, FILE_SHA, "parse_cache", {},
        )
        label = get_resume_label(data_root, FILE_SHA)
        assert "SoA" in label

    def test_after_soa_suggests_fidelity(
        self, data_root: Path,
    ) -> None:
        save_checkpoint(
            data_root, FILE_SHA, "parse_cache", {},
        )
        save_checkpoint(
            data_root, FILE_SHA, "soa_cache", {},
        )
        label = get_resume_label(data_root, FILE_SHA)
        assert "Fidelity" in label

    def test_after_all_stages(self, data_root: Path) -> None:
        for ck in _CACHE_STAGES:
            save_checkpoint(data_root, FILE_SHA, ck, {})
        label = get_resume_label(data_root, FILE_SHA)
        assert "complete" in label.lower() or "Validation" in label
