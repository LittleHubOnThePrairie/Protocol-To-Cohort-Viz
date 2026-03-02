"""ICH E6(R3) Section Parser package.

Public API:

    IchParser            — Main service: classify → store → lineage
    IchSection           — Data model for one classified section
    SectionClassifier    — Abstract base for classification backends
    RuleBasedClassifier  — Regex/keyword classifier (no external deps)
    RAGClassifier        — Cohere embeddings + Claude Sonnet classifier
    ReviewQueue          — SQLite append-only human review queue
"""

from .models import IchSection, ReviewQueueEntry
from .classifier import SectionClassifier, RuleBasedClassifier, RAGClassifier
from .review_queue import ReviewQueue
from .parser import IchParser

__all__ = [
    "IchParser",
    "IchSection",
    "ReviewQueueEntry",
    "SectionClassifier",
    "RuleBasedClassifier",
    "RAGClassifier",
    "ReviewQueue",
]
