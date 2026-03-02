"""PTCV Compliance Module.

Provides audit trail, data integrity, and access control primitives
required by 21 CFR Part 11, ICH E6(R3), and ALCOA+ for all regulated
data operations.

Risk tier: HIGH — audit records are regulated data themselves.
"""

from .audit import AuditAction, AuditLogger
from .integrity import DataIntegrityGuard

__all__ = ["AuditAction", "AuditLogger", "DataIntegrityGuard"]
