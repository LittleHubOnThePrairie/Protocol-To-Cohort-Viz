# Jira Changelog

This file tracks Jira issue status changes made by Claude Code.
Use "Post summary of all Jira changes to Confluence" to sync to Confluence.

---

## [2026-03-02T22:16:57Z] PTCV-22: Implement CDISC SDTM Trial Design Domain Generator

**Status Change:** In Progress -> Done
**Actor:** Claude Code

**Summary:**
Implement CDISC SDTM Trial Design Domain Generator

**Files Modified:**
- C:/Dev/PTCV/src/ptcv/sdtm/__init__.py
- C:/Dev/PTCV/src/ptcv/sdtm/ct_normalizer.py
- C:/Dev/PTCV/src/ptcv/sdtm/define_xml.py
- C:/Dev/PTCV/src/ptcv/sdtm/domain_generators.py
- C:/Dev/PTCV/src/ptcv/sdtm/models.py
- C:/Dev/PTCV/src/ptcv/sdtm/review_queue.py
- C:/Dev/PTCV/src/ptcv/sdtm/sdtm_service.py
- C:/Dev/PTCV/tests/sdtm/__init__.py
- C:/Dev/PTCV/tests/sdtm/conftest.py
- C:/Dev/PTCV/tests/sdtm/test_ct_normalizer.py
- C:/Dev/PTCV/tests/sdtm/test_define_xml.py
- C:/Dev/PTCV/tests/sdtm/test_domain_generators.py
- C:/Dev/PTCV/tests/sdtm/test_sdtm_service.py

**Test Coverage:**
6 test file(s) found
Branch: `master` | Commits: 28700ea, fc72881, a7eb13d

---
## [2026-03-02T20:38:43Z] PTCV-29: Harden local PDF datastore with StorageGateway and MinIO WORM

**Status Change:** In Progress -> Done
**Actor:** Claude Code

**Summary:**
Harden local PDF datastore with StorageGateway and MinIO WORM

**Files Modified:**
- C:/Dev/PTCV/scripts/download_sample.py
- C:/Dev/PTCV/src/ptcv/protocol_search/clinicaltrials_service.py
- C:/Dev/PTCV/src/ptcv/protocol_search/eu_ctr_service.py
- C:/Dev/PTCV/src/ptcv/protocol_search/filestore.py
- C:/Dev/PTCV/tests/protocol_search/conftest.py
- C:/Dev/PTCV/tests/protocol_search/test_clinicaltrials_service.py
- C:/Dev/PTCV/tests/protocol_search/test_eu_ctr_service.py
- C:/Dev/PTCV/tests/protocol_search/test_filestore.py
- C:/Dev/PTCV/tests/test_download_sample.py
- C:/Dev/PTCV/src/ptcv/storage/__init__.py
- C:/Dev/PTCV/src/ptcv/storage/_lineage_db.py
- C:/Dev/PTCV/src/ptcv/storage/filesystem_adapter.py
- C:/Dev/PTCV/src/ptcv/storage/gateway.py
- C:/Dev/PTCV/src/ptcv/storage/local_adapter.py
- C:/Dev/PTCV/src/ptcv/storage/models.py
- C:/Dev/PTCV/tests/storage/__init__.py
- C:/Dev/PTCV/tests/storage/test_filesystem_adapter.py
- C:/Dev/PTCV/tests/storage/test_gateway_models.py
- C:/Dev/PTCV/tests/storage/test_lineage_db.py
- C:/Dev/PTCV/tests/storage/test_local_adapter.py
- C:/Dev/PTCV/tests/smoke/test_storage_smoke.py

**Test Coverage:**
9 test file(s) found
Branch: `master` | Commits: fc72881, a7eb13d, b32c138

---

## [2026-03-02T21:45:00Z] PTCV-19: Implement Protocol Format Detection and Text/Table Extraction

**Status Change:** To Do -> In Progress
**Actor:** Claude Code

**Summary:**
Implement protocol extraction layer: format detection (PDF/CTR-XML/Word), PDF cascade
(pdfplumber → Camelot → tabula), CTR-XML native parser, multi-page SoA table
reconstruction, Parquet output via StorageGateway, append-only lineage records.

**Files Modified:**
- src/ptcv/extraction/__init__.py
- src/ptcv/extraction/models.py
- src/ptcv/extraction/format_detector.py
- src/ptcv/extraction/pdf_extractor.py
- src/ptcv/extraction/ctr_xml_extractor.py
- src/ptcv/extraction/parquet_writer.py
- src/ptcv/extraction/extraction_service.py
- tests/extraction/__init__.py
- tests/extraction/conftest.py
- tests/extraction/test_format_detector.py
- tests/extraction/test_pdf_extractor.py
- tests/extraction/test_ctr_xml_extractor.py
- tests/extraction/test_parquet_writer.py
- tests/extraction/test_extraction_service.py
- requirements.txt

**Test Coverage:**
273 passed (70 new extraction tests covering all 5 GHERKIN scenarios)
mypy: no issues found in 7 source files
Branch: `master`

---

## 2026-03-02T00:00:00Z — PTCV-19 Done

- **Issue:** PTCV-19: Protocol Format Detection and Text/Table Extraction
- **Status:** In Progress → Done
- **Actor:** Claude Code
- **Labels:** PRD-22
- **Files modified:** 16 (src/ptcv/extraction/ × 7, tests/extraction/ × 7, requirements.txt, jira_changelog.md)
- **Summary:** 273/273 tests passing, mypy clean. Transition allowed with E2E advisory warning.

## 2026-03-02T22:02:00Z — PTCV-22 In Progress

- **Issue:** PTCV-22: Implement CDISC SDTM Trial Design Domain Generator
- **Status:** To Do → In Progress
- **Actor:** Claude Code
- **Summary:** Implementing TS/TA/TE/TV/TI XPT + Define-XML v2.1 via StorageGateway WORM
