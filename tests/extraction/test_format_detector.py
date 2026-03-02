"""Tests for ptcv.extraction.format_detector.

Qualification phase: IQ/OQ
Risk tier: LOW (no I/O side effects)
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[2] / "src"))

from ptcv.extraction.format_detector import FormatDetector, ProtocolFormat


class TestProtocolFormat:
    def test_values(self):
        assert ProtocolFormat.PDF.value == "pdf"
        assert ProtocolFormat.CTR_XML.value == "ctr-xml"
        assert ProtocolFormat.WORD.value == "word"
        assert ProtocolFormat.UNKNOWN.value == "unknown"

    def test_is_string_enum(self):
        assert isinstance(ProtocolFormat.PDF, str)


class TestFormatDetectorFromBytes:
    def setup_method(self):
        self.detector = FormatDetector()

    def test_detect_pdf_magic(self, minimal_pdf_bytes):
        fmt = self.detector.detect_from_bytes(minimal_pdf_bytes)
        assert fmt == ProtocolFormat.PDF

    def test_detect_ctr_xml_odm_tag(self, ctr_xml_bytes):
        fmt = self.detector.detect_from_bytes(ctr_xml_bytes)
        assert fmt == ProtocolFormat.CTR_XML

    def test_detect_ctr_xml_plain_odm_marker(self):
        xml = b"<ODM xmlns='http://www.cdisc.org/ns/odm/v1.3'><Study/></ODM>"
        fmt = self.detector.detect_from_bytes(xml)
        assert fmt == ProtocolFormat.CTR_XML

    def test_detect_xml_via_declaration(self):
        xml = b"<?xml version='1.0'?><ODM/>"
        fmt = self.detector.detect_from_bytes(xml)
        assert fmt == ProtocolFormat.CTR_XML

    def test_detect_word_zip_magic(self):
        # DOCX starts with PK\x03\x04
        docx_like = b"PK\x03\x04some content here"
        fmt = self.detector.detect_from_bytes(docx_like)
        assert fmt == ProtocolFormat.WORD

    def test_detect_unknown_empty(self):
        fmt = self.detector.detect_from_bytes(b"")
        assert fmt == ProtocolFormat.UNKNOWN

    def test_detect_unknown_garbage(self):
        fmt = self.detector.detect_from_bytes(b"\x00\x01\x02\x03")
        assert fmt == ProtocolFormat.UNKNOWN

    def test_filename_hint_pdf(self):
        fmt = self.detector.detect_from_bytes(b"\x00\x00", filename="protocol.pdf")
        assert fmt == ProtocolFormat.PDF

    def test_filename_hint_xml(self):
        fmt = self.detector.detect_from_bytes(b"\x00\x00", filename="trial.xml")
        assert fmt == ProtocolFormat.CTR_XML

    def test_filename_hint_docx(self):
        fmt = self.detector.detect_from_bytes(b"\x00\x00", filename="protocol.docx")
        assert fmt == ProtocolFormat.WORD

    def test_magic_takes_priority_over_extension(self):
        # PDF magic bytes but .xml extension — magic wins
        fmt = self.detector.detect_from_bytes(
            b"%PDF-1.4 content", filename="mislabelled.xml"
        )
        assert fmt == ProtocolFormat.PDF

    def test_detect_xml_with_bom(self):
        xml_bom = b"\xef\xbb\xbf<?xml version='1.0'?><ODM/>"
        fmt = self.detector.detect_from_bytes(xml_bom)
        assert fmt == ProtocolFormat.CTR_XML


class TestFormatDetectorFromPath:
    def setup_method(self):
        self.detector = FormatDetector()

    def test_detect_from_pdf_path(self, tmp_path, minimal_pdf_bytes):
        p = tmp_path / "test.pdf"
        p.write_bytes(minimal_pdf_bytes)
        assert self.detector.detect_from_path(p) == ProtocolFormat.PDF

    def test_detect_from_xml_path(self, tmp_path, ctr_xml_bytes):
        p = tmp_path / "trial.xml"
        p.write_bytes(ctr_xml_bytes)
        assert self.detector.detect_from_path(p) == ProtocolFormat.CTR_XML

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            self.detector.detect_from_path(tmp_path / "no_such_file.pdf")
