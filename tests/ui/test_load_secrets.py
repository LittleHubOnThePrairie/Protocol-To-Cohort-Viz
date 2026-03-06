"""Unit tests for _load_secrets (PTCV-104).

Tests the auto-load secrets function without requiring the Streamlit runtime.
"""

from __future__ import annotations

import os
import textwrap
from pathlib import Path
from unittest.mock import patch

from ptcv.ui.app import _load_secrets


class TestLoadSecrets:
    """Tests for _load_secrets()."""

    def test_loads_from_secrets_file(self, tmp_path: Path) -> None:
        """Secrets file is read and env vars are set."""
        secrets = tmp_path / ".secrets"
        secrets.write_text("MY_TEST_KEY=test_value_123\n")

        env = os.environ.copy()
        env.pop("MY_TEST_KEY", None)
        env.pop("ANTHROPIC_API_KEY", None)

        with patch("ptcv.ui.app._PROJECT_ROOT", tmp_path), \
             patch.dict(os.environ, env, clear=True):
            warnings = _load_secrets()
            assert os.environ.get("MY_TEST_KEY") == "test_value_123"
            assert any("ANTHROPIC_API_KEY" in w for w in warnings)

    def test_env_takes_precedence(self, tmp_path: Path) -> None:
        """Existing env vars are NOT overwritten by .secrets file."""
        secrets = tmp_path / ".secrets"
        secrets.write_text("MY_PRIO_KEY=from_file\n")

        env = os.environ.copy()
        env["MY_PRIO_KEY"] = "from_env"

        with patch("ptcv.ui.app._PROJECT_ROOT", tmp_path), \
             patch.dict(os.environ, env, clear=True):
            _load_secrets()
            assert os.environ["MY_PRIO_KEY"] == "from_env"

    def test_skips_comments_and_blanks(self, tmp_path: Path) -> None:
        """Comments and blank lines are ignored."""
        secrets = tmp_path / ".secrets"
        secrets.write_text(textwrap.dedent("""\
            # This is a comment
            ANTHROPIC_API_KEY=sk-test-key

            # Another comment
            BLANK_LINE_ABOVE=yes
        """))

        env = os.environ.copy()
        env.pop("ANTHROPIC_API_KEY", None)
        env.pop("BLANK_LINE_ABOVE", None)

        with patch("ptcv.ui.app._PROJECT_ROOT", tmp_path), \
             patch.dict(os.environ, env, clear=True):
            warnings = _load_secrets()
            assert os.environ.get("ANTHROPIC_API_KEY") == "sk-test-key"
            assert os.environ.get("BLANK_LINE_ABOVE") == "yes"
            assert not any("ANTHROPIC_API_KEY" in w for w in warnings)

    def test_missing_secrets_file_warns(self, tmp_path: Path) -> None:
        """Warning returned when .secrets file doesn't exist and key missing."""
        env = os.environ.copy()
        env.pop("ANTHROPIC_API_KEY", None)

        with patch("ptcv.ui.app._PROJECT_ROOT", tmp_path), \
             patch.dict(os.environ, env, clear=True):
            warnings = _load_secrets()
            assert len(warnings) == 1
            assert ".secrets" in warnings[0]

    def test_no_warning_when_key_in_env(self, tmp_path: Path) -> None:
        """No warning if ANTHROPIC_API_KEY is already set, even without file."""
        env = os.environ.copy()
        env["ANTHROPIC_API_KEY"] = "sk-already-set"

        with patch("ptcv.ui.app._PROJECT_ROOT", tmp_path), \
             patch.dict(os.environ, env, clear=True):
            warnings = _load_secrets()
            assert not warnings

    def test_skips_empty_values(self, tmp_path: Path) -> None:
        """Lines with empty values are skipped."""
        secrets = tmp_path / ".secrets"
        secrets.write_text("EMPTY_VAL=\nANTHROPIC_API_KEY=sk-good\n")

        env = os.environ.copy()
        env.pop("EMPTY_VAL", None)
        env.pop("ANTHROPIC_API_KEY", None)

        with patch("ptcv.ui.app._PROJECT_ROOT", tmp_path), \
             patch.dict(os.environ, env, clear=True):
            _load_secrets()
            assert "EMPTY_VAL" not in os.environ
            assert os.environ.get("ANTHROPIC_API_KEY") == "sk-good"

    def test_value_with_equals_sign(self, tmp_path: Path) -> None:
        """Values containing '=' are preserved correctly."""
        secrets = tmp_path / ".secrets"
        secrets.write_text("MY_TOKEN=abc=def=ghi\n")

        env = os.environ.copy()
        env.pop("MY_TOKEN", None)

        with patch("ptcv.ui.app._PROJECT_ROOT", tmp_path), \
             patch.dict(os.environ, env, clear=True):
            _load_secrets()
            assert os.environ.get("MY_TOKEN") == "abc=def=ghi"
