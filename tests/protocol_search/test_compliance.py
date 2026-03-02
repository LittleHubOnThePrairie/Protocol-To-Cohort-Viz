"""Tests for PTCV compliance modules: AuditLogger and DataIntegrityGuard.

Qualification phase: IQ (installation qualification)
Regulatory requirement: 21 CFR 11.10(e) audit trail, ALCOA+ Consistent
  (SHA-256 integrity verification).
Risk tier: MEDIUM
"""

import hashlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[2] / "src"))

from ptcv.compliance.audit import AuditAction, AuditLogger
from ptcv.compliance.integrity import DataIntegrityGuard


class TestAuditLogger:
    """IQ: AuditLogger writes append-only JSON Lines audit trail."""

    def test_log_writes_jsonl_entry(self, tmp_path):
        """21 CFR 11.10(e): Each log() call appends a JSON line."""
        logger = AuditLogger(module="test", log_path=tmp_path / "audit.jsonl")
        logger.log(
            action=AuditAction.DOWNLOAD,
            record_id="NCT12345678",
            user_id="tester",
            reason="protocol_ingestion",
        )

        lines = (tmp_path / "audit.jsonl").read_text(encoding="utf-8").splitlines()
        assert len(lines) == 1
        entry = json.loads(lines[0])
        assert entry["action"] == "DOWNLOAD"
        assert entry["record_id"] == "NCT12345678"
        assert entry["user_id"] == "tester"
        assert entry["reason"] == "protocol_ingestion"

    def test_log_appends_multiple_entries(self, tmp_path):
        """Repeated log() calls append without overwriting prior entries."""
        logger = AuditLogger(module="test", log_path=tmp_path / "audit.jsonl")
        logger.log(action=AuditAction.SEARCH, record_id="EU-CTR", user_id="u1", reason="r1")
        logger.log(action=AuditAction.DOWNLOAD, record_id="NCT00000001", user_id="u2", reason="r2")

        lines = (tmp_path / "audit.jsonl").read_text(encoding="utf-8").splitlines()
        assert len(lines) == 2

    def test_log_requires_non_empty_reason(self, tmp_path):
        """Audit entry without reason raises ValueError (mandatory WHY field)."""
        logger = AuditLogger(module="test", log_path=tmp_path / "audit.jsonl")
        with pytest.raises(ValueError):
            logger.log(
                action=AuditAction.DOWNLOAD,
                record_id="NCT12345678",
                user_id="tester",
                reason="",
            )

    def test_log_includes_timestamp(self, tmp_path):
        """ALCOA+ Contemporaneous: every entry has a timestamp."""
        logger = AuditLogger(module="test", log_path=tmp_path / "audit.jsonl")
        logger.log(
            action=AuditAction.READ,
            record_id="doc-001",
            user_id="reader",
            reason="review",
        )
        entry = json.loads(
            (tmp_path / "audit.jsonl").read_text(encoding="utf-8").strip()
        )
        assert "timestamp_utc" in entry
        assert entry["timestamp_utc"]

    def test_log_after_dict_included(self, tmp_path):
        """Optional 'after' dict is persisted in the audit entry."""
        logger = AuditLogger(module="test", log_path=tmp_path / "audit.jsonl")
        logger.log(
            action=AuditAction.DOWNLOAD,
            record_id="NCT12345678",
            user_id="tester",
            reason="ingestion",
            after={"file_hash_sha256": "abc123", "format": "PDF"},
        )
        entry = json.loads(
            (tmp_path / "audit.jsonl").read_text(encoding="utf-8").strip()
        )
        assert entry["after"]["file_hash_sha256"] == "abc123"


class TestAuditAction:
    """IQ: AuditAction enum covers required operation types."""

    def test_required_actions_present(self):
        """All 21 CFR 11 mandatory action types are present."""
        required = {"CREATE", "MODIFY", "DELETE", "READ", "DOWNLOAD", "SEARCH"}
        actual = {a.name for a in AuditAction}
        assert required.issubset(actual)


class TestDataIntegrityGuard:
    """IQ: DataIntegrityGuard computes and verifies SHA-256 checksums (ALCOA+)."""

    def test_checkpoint_returns_sha256_hex(self):
        """checkpoint() returns 64-char hex SHA-256 digest."""
        guard = DataIntegrityGuard()
        data = b"protocol content bytes"
        digest = guard.checkpoint(data, stage="test_stage")
        assert isinstance(digest, str)
        assert len(digest) == 64
        assert all(c in "0123456789abcdef" for c in digest)

    def test_checkpoint_accepts_string(self):
        """checkpoint() accepts a UTF-8 string as well as bytes."""
        guard = DataIntegrityGuard()
        digest = guard.checkpoint("hello world", stage="str_stage")
        assert len(digest) == 64

    def test_verify_correct_hash_returns_true(self):
        """verify() returns True when hash matches data."""
        guard = DataIntegrityGuard()
        data = b"some protocol bytes"
        digest = guard.checkpoint(data, stage="verify_stage")
        assert guard.verify(data, digest) is True

    def test_verify_tampered_data_returns_false(self):
        """ALCOA+ Consistent: verify() returns False when data is modified."""
        guard = DataIntegrityGuard()
        data = b"original bytes"
        digest = guard.checkpoint(data, stage="tamper_stage")
        tampered = b"modified bytes"
        assert guard.verify(tampered, digest) is False

    def test_verify_wrong_hash_returns_false(self):
        """verify() returns False when expected hash is incorrect."""
        guard = DataIntegrityGuard()
        data = b"some data"
        assert guard.verify(data, "a" * 64) is False

    def test_checkpoint_deterministic(self):
        """Same data produces same hash across calls."""
        guard = DataIntegrityGuard()
        data = b"deterministic content"
        h1 = guard.checkpoint(data, stage="s")
        h2 = guard.checkpoint(data, stage="s")
        assert h1 == h2
