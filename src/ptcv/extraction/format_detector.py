"""Protocol file format detector for PTCV-19.

Detects whether a protocol file is PDF, CTR-XML (CDISC ODM extension),
Word (.docx), or unknown — using magic bytes and filename extension.

Risk tier: LOW — stateless, no I/O side effects.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path


class ProtocolFormat(str, Enum):
    """Supported protocol file formats.

    Values:
        PDF: Portable Document Format.
        CTR_XML: CDISC ODM extension (EU CTR format).
        WORD: Microsoft Word Open XML (.docx).
        UNKNOWN: Unrecognised format.
    """

    PDF = "pdf"
    CTR_XML = "ctr-xml"
    WORD = "word"
    UNKNOWN = "unknown"


# PDF magic bytes (%PDF)
_PDF_MAGIC = b"%PDF"

# DOCX is a ZIP archive — magic bytes: PK\x03\x04
_ZIP_MAGIC = b"PK\x03\x04"

# XML markers that indicate a CDISC ODM / EU-CTR file
_ODM_MARKERS = (b"<ODM", b"<ClinicalData", b"<Study")


class FormatDetector:
    """Detects the format of a protocol file.

    Format detection order:
    1. Check first bytes for PDF magic ``%PDF``.
    2. Check first bytes for XML markers (``<ODM``, ``<ClinicalData``).
    3. Check first bytes for ZIP magic (DOCX is ZIP).
    4. Fall back to filename extension if bytes are ambiguous.
    5. Return UNKNOWN if all checks fail.
    """

    def detect_from_bytes(
        self, data: bytes, filename: str = ""
    ) -> ProtocolFormat:
        """Detect format from raw bytes, using filename as a hint.

        Args:
            data: File contents (reads first ~512 bytes for magic).
            filename: Original filename — used as fallback hint when
                magic bytes are inconclusive. May be empty.

        Returns:
            Detected ProtocolFormat value.
        """
        header = data[:512]

        if header.startswith(_PDF_MAGIC):
            return ProtocolFormat.PDF

        # XML-based formats (CTR-XML / ODM)
        header_stripped = header.lstrip(b"\xef\xbb\xbf")  # strip BOM
        if header_stripped.lstrip().startswith(b"<?xml") or any(
            marker in header for marker in _ODM_MARKERS
        ):
            return ProtocolFormat.CTR_XML

        # DOCX = ZIP with word/ content-type marker
        if header.startswith(_ZIP_MAGIC):
            return ProtocolFormat.WORD

        # Fall back to filename extension
        return self._detect_from_extension(filename)

    def detect_from_path(self, path: Path) -> ProtocolFormat:
        """Detect format by reading the file at *path*.

        Args:
            path: Path to the protocol file.

        Returns:
            Detected ProtocolFormat value.

        Raises:
            FileNotFoundError: If the file does not exist.
        """
        if not path.exists():
            raise FileNotFoundError(f"Protocol file not found: {path}")
        data = path.read_bytes()
        return self.detect_from_bytes(data, filename=path.name)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_from_extension(filename: str) -> ProtocolFormat:
        """Return ProtocolFormat inferred from filename extension.

        Args:
            filename: Original filename string (may be empty).

        Returns:
            ProtocolFormat, or UNKNOWN if extension is not recognised.
        """
        ext = Path(filename).suffix.lower()
        mapping = {
            ".pdf": ProtocolFormat.PDF,
            ".xml": ProtocolFormat.CTR_XML,
            ".docx": ProtocolFormat.WORD,
            ".doc": ProtocolFormat.WORD,
        }
        return mapping.get(ext, ProtocolFormat.UNKNOWN)
