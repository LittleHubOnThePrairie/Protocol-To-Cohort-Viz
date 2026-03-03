# Jira Changelog

This file tracks Jira issue status changes made by Claude Code.
Use "Post summary of all Jira changes to Confluence" to sync to Confluence.

---

## [2026-03-03T17:37:31Z] PTCV-42: Enhanced protocol file browser: group by therapeutic area, display titles, order by quality

**Status Change:** In Progress -> Done
**Actor:** Claude Code

**Summary:**
Enhanced protocol file browser: group by therapeutic area, display titles, order by quality

**Files Modified:**
- C:/Dev/PTCV/src/ptcv/ui/components/file_browser.py
- C:/Dev/PTCV/src/ptcv/ui/components/protocol_catalog.py
- C:/Dev/PTCV/scripts/batch_parse.py
- C:/Dev/PTCV/tests/ui/test_protocol_catalog.py
- C:/Dev/PTCV/tests/scripts/__init__.py
- C:/Dev/PTCV/tests/scripts/test_batch_parse.py

**Test Coverage:**
3 test file(s) found
Branch: `master` | Commits: 28700ea, fc72881, a7eb13d

---
## [2026-03-03T14:01:39Z] PTCV-36: Streamlit: mock SDTM generation with protocol-to-SDTM lineage visualization

**Status Change:** In Progress -> Done
**Actor:** Claude Code

**Summary:**
Streamlit: mock SDTM generation with protocol-to-SDTM lineage visualization

**Files Modified:**
- C:/Dev/PTCV/src/ptcv/ui/components/sdtm_viewer.py
- C:/Dev/PTCV/src/ptcv/ui/app.py
- C:/Dev/PTCV/tests/ui/test_sdtm_viewer.py

**Test Coverage:**
2 test file(s) found
Branch: `master` | Commits: 28700ea, fc72881, a7eb13d

---
## [2026-03-03T13:45:59Z] PTCV-38: Blended Camelot + Table Transformer table discovery for non-standard SoA formats

**Status Change:** In Progress -> Done
**Actor:** Claude Code

**Summary:**
Blended Camelot + Table Transformer table discovery for non-standard SoA formats

**Files Modified:**
- C:/Dev/PTCV/src/ptcv/soa_extractor/parser.py
- C:/Dev/PTCV/src/ptcv/soa_extractor/table_discovery.py
- C:/Dev/PTCV/src/ptcv/soa_extractor/extractor.py
- C:/Dev/PTCV/src/ptcv/soa_extractor/__init__.py
- C:/Dev/PTCV/src/ptcv/ich_parser/classifier.py
- C:/Dev/PTCV/tests/soa_extractor/test_table_discovery.py

**Test Coverage:**
4 test file(s) found
Branch: `master` | Commits: 28700ea, fc72881, a7eb13d

---
## [2026-03-03T01:28:00Z] PTCV-34: Streamlit: multi-swimlane schedule of visits timeline visualization

**Status Change:** In Progress -> Done
**Actor:** Claude Code

**Summary:**
Streamlit: multi-swimlane schedule of visits timeline visualization

**Files Modified:**
- C:/Dev/PTCV/src/ptcv/ui/components/schedule_of_visits.py
- C:/Dev/PTCV/src/ptcv/ui/app.py
- C:/Dev/PTCV/src/ptcv/soa_extractor/writer.py
- C:/Dev/PTCV/tests/ui/test_schedule_of_visits.py
- C:/Dev/PTCV/tests/soa_extractor/test_writer.py

**Test Coverage:**
3 test file(s) found
Branch: `master` | Commits: 28700ea, fc72881, a7eb13d

---
## [2026-03-03T01:21:04Z] PTCV-35: Streamlit: regenerate and display protocol reformatted in ICH E6(R3) structure

**Status Change:** In Progress -> Done
**Actor:** Claude Code

**Summary:**
Streamlit: regenerate and display protocol reformatted in ICH E6(R3) structure

**Files Modified:**
- C:/Dev/PTCV/src/ptcv/ui/app.py
- C:/Dev/PTCV/src/ptcv/ui/components/ich_regenerator.py
- C:/Dev/PTCV/tests/ui/test_ich_regenerator.py
- C:/Dev/PTCV/tests/ui/test_app.py

**Test Coverage:**
2 test file(s) found
Branch: `master` | Commits: 28700ea, fc72881, a7eb13d

---
## [2026-03-03T01:12:39Z] PTCV-31: Pre-parser format detector to gate non-ICH protocols before ICH section classification

**Status Change:** In Progress -> Done
**Actor:** Claude Code

**Summary:**
Pre-parser format detector to gate non-ICH protocols before ICH section classification

**Files Modified:**
- C:/Dev/PTCV/src/ptcv/ich_parser/format_detector.py
- C:/Dev/PTCV/src/ptcv/ich_parser/parser.py
- C:/Dev/PTCV/src/ptcv/ich_parser/__init__.py
- C:/Dev/PTCV/src/ptcv/pipeline/orchestrator.py
- C:/Dev/PTCV/tests/ich_parser/test_format_detector.py

**Test Coverage:**
3 test file(s) found
Branch: `master` | Commits: 28700ea, fc72881, a7eb13d

---
## [2026-03-03T01:09:39Z] PTCV-33: Streamlit: file browser and ICH compliance flag for parsed PDF

**Status Change:** In Progress -> Done
**Actor:** Claude Code

**Summary:**
Streamlit: file browser and ICH compliance flag for parsed PDF

**Files Modified:**
- C:/Dev/PTCV/requirements.txt
- C:/Dev/PTCV/src/ptcv/ui/__init__.py
- C:/Dev/PTCV/src/ptcv/ui/app.py
- C:/Dev/PTCV/src/ptcv/ui/components/__init__.py
- C:/Dev/PTCV/src/ptcv/ui/components/file_browser.py
- C:/Dev/PTCV/tests/ui/__init__.py
- C:/Dev/PTCV/tests/ui/conftest.py
- C:/Dev/PTCV/tests/ui/test_app.py
- C:/Dev/PTCV/tests/ui/test_file_browser.py

**Test Coverage:**
2 test file(s) found
Branch: `master` | Commits: 28700ea, fc72881, a7eb13d

---
## [2026-03-02T23:44:34Z] PTCV-30: Detect non-ICH protocol format and surface verdict in ParseResult and compliance reports

**Status Change:** In Progress -> Done
**Actor:** Claude Code

**Summary:**
Detect non-ICH protocol format and surface verdict in ParseResult and compliance reports

**Files Modified:**
- C:/Dev/PTCV/src/ptcv/ich_parser/parser.py
- C:/Dev/PTCV/src/ptcv/pipeline/orchestrator.py
- C:/Dev/PTCV/src/ptcv/sdtm/validation/validation_service.py
- C:/Dev/PTCV/tests/ich_parser/test_format_verdict.py

**Test Coverage:**
3 test file(s) found
Branch: `master` | Commits: 28700ea, fc72881, a7eb13d

---
## [2026-03-02T23:24:46Z] PTCV-24: Implement End-to-End Pipeline Orchestration and Integration Tests

**Status Change:** In Progress -> Done
**Actor:** Claude Code

**Summary:**
Implement End-to-End Pipeline Orchestration and Integration Tests

**Files Modified:**
- C:/Dev/PTCV/src/ptcv/pipeline/__init__.py
- C:/Dev/PTCV/src/ptcv/pipeline/gateway_factory.py
- C:/Dev/PTCV/src/ptcv/pipeline/models.py
- C:/Dev/PTCV/src/ptcv/pipeline/orchestrator.py
- C:/Dev/PTCV/tests/pipeline/__init__.py
- C:/Dev/PTCV/tests/pipeline/conftest.py
- C:/Dev/PTCV/tests/pipeline/test_gateway_factory.py
- C:/Dev/PTCV/tests/pipeline/test_models.py
- C:/Dev/PTCV/tests/pipeline/test_orchestrator.py

**Test Coverage:**
3 test file(s) found
Branch: `master` | Commits: 28700ea, fc72881, a7eb13d

---
## [2026-03-02T22:37:03Z] PTCV-23: Implement SDTM Validation and Compliance Reporting

**Status Change:** In Progress -> Done
**Actor:** Claude Code

**Summary:**
Implement SDTM Validation and Compliance Reporting

**Files Modified:**
- C:/Dev/PTCV/src/ptcv/sdtm/__init__.py
- C:/Dev/PTCV/src/ptcv/sdtm/validation/__init__.py
- C:/Dev/PTCV/src/ptcv/sdtm/validation/define_xml_validator.py
- C:/Dev/PTCV/src/ptcv/sdtm/validation/models.py
- C:/Dev/PTCV/src/ptcv/sdtm/validation/p21_validator.py
- C:/Dev/PTCV/src/ptcv/sdtm/validation/tcg_checker.py
- C:/Dev/PTCV/src/ptcv/sdtm/validation/validation_service.py
- C:/Dev/PTCV/tests/sdtm/validation/__init__.py
- C:/Dev/PTCV/tests/sdtm/validation/conftest.py
- C:/Dev/PTCV/tests/sdtm/validation/test_define_xml_validator.py
- C:/Dev/PTCV/tests/sdtm/validation/test_p21_validator.py
- C:/Dev/PTCV/tests/sdtm/validation/test_tcg_checker.py
- C:/Dev/PTCV/tests/sdtm/validation/test_validation_service.py

**Test Coverage:**
5 test file(s) found
Branch: `master` | Commits: 28700ea, fc72881, a7eb13d

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
