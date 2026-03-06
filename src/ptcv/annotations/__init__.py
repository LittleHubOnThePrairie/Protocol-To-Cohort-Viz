"""Human-in-the-loop annotation package for classifier outputs (PTCV-40).

Public API:

    AnnotationRecord     - Single reviewer annotation for one section
    AnnotationSession    - Collection of annotations for one protocol run
    AnnotationService    - CRUD + persistence via StorageGateway
"""

from .models import AnnotationRecord, AnnotationSession
from .service import AnnotationService

__all__ = [
    "AnnotationRecord",
    "AnnotationSession",
    "AnnotationService",
]
