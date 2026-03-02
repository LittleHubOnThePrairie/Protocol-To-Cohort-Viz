"""Audit Trail — 21 CFR 11.10(e), ALCOA+ Attributable/Contemporaneous.

Append-only audit log. Every data mutation (CREATE, MODIFY, DELETE,
DOWNLOAD) MUST have WHO/WHAT/WHEN/WHY. No mechanism exists to disable
logging — this constraint is intentional and required by regulation.

Risk tier: HIGH — audit records are themselves regulated records.
"""

import json
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Optional


_DEFAULT_LOG_PATH = Path("C:/Dev/PTCV/data/audit/audit.jsonl")


class AuditAction(str, Enum):
    """Regulated data operations subject to audit trail requirements.

    [21 CFR 11.10(e)] Computer-generated, time-stamped audit trails
    must record operator entries and actions that create, modify, or
    delete electronic records.
    """

    CREATE = "CREATE"
    MODIFY = "MODIFY"
    DELETE = "DELETE"
    READ = "READ"
    DOWNLOAD = "DOWNLOAD"
    SEARCH = "SEARCH"


class AuditLogger:
    """Append-only audit trail logger.

    Implements the WHO/WHAT/WHEN/WHY audit pattern required by
    21 CFR 11.10(e) and ALCOA+ Attributable/Contemporaneous
    principles. Log entries are written as JSON Lines to an
    append-only file. There is no mechanism to disable logging.

    Args:
        module: Logical module name (e.g., "protocol_download").
        log_path: Override default audit log path. Only for testing.
    """

    def __init__(
        self,
        module: str,
        log_path: Optional[Path] = None,
    ) -> None:
        self._module = module
        self._log_path = log_path or _DEFAULT_LOG_PATH
        self._log_path.parent.mkdir(parents=True, exist_ok=True)

    def log(
        self,
        action: AuditAction,
        record_id: str,
        user_id: str,
        reason: str,
        before: Optional[dict[str, Any]] = None,
        after: Optional[dict[str, Any]] = None,
        extra: Optional[dict[str, Any]] = None,
    ) -> None:
        """Append an audit entry to the log. Cannot be disabled.

        Args:
            action: The type of regulated operation.
            record_id: Identifier of the record being acted upon.
            user_id: Identity of the actor (no shared accounts).
            reason: Mandatory justification — must be non-empty.
            before: State before modification (for MODIFY/DELETE).
            after: State after modification (for CREATE/MODIFY/DOWNLOAD).
            extra: Optional additional context (no PHI allowed).

        Raises:
            ValueError: If reason is empty (regulation requires it).
        """
        if not reason.strip():
            raise ValueError(
                "Audit log reason is mandatory per 21 CFR 11.10(e). "
                "Provide a non-empty justification string."
            )

        entry: dict[str, Any] = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "module": self._module,
            "action": action.value,
            "record_id": record_id,
            "user_id": user_id,
            "reason": reason,
        }
        if before is not None:
            entry["before"] = before
        if after is not None:
            entry["after"] = after
        if extra:
            entry["extra"] = extra

        with open(self._log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
