# Jira Changelog

This file tracks Jira issue status changes made by Claude Code.
Use "Post summary of all Jira changes to Confluence" to sync to Confluence.

---

## [2026-03-05T17:07:22Z] PTCV-89: Extract Protocol TOC and Section Headers as Navigable Index

**Status Change:** In Progress -> Done
**Actor:** Claude Code
**PRD:** PRD-1

**Summary:**
Extract Protocol TOC and Section Headers as Navigable Index

**Files Modified:**
- C:/Dev/PTCV/src/ptcv/ich_parser/toc_extractor.py
- C:/Dev/PTCV/src/ptcv/ich_parser/__init__.py
- C:/Dev/PTCV/tests/ich_parser/test_toc_extractor.py

**Test Coverage:**
1 test file(s) found
Branch: `master` | Commits: 28700ea, fc72881, a7eb13d

---
## [2026-03-05T17:05:13Z] PTCV-90: Semantic Section Matcher: Map Protocol Headers to Appendix B Sections

**Status Change:** In Progress -> Done
**Actor:** Claude Code

**Summary:**
Semantic Section Matcher: Map Protocol Headers to Appendix B Sections

**Files Modified:**
- C:/Dev/PTCV/src/ptcv/ich_parser/section_matcher.py
- C:/Dev/PTCV/src/ptcv/ich_parser/__init__.py
- C:/Dev/PTCV/tests/ich_parser/test_section_matcher.py

**Test Coverage:**
1 test file(s) found
Branch: `master` | Commits: 28700ea, fc72881, a7eb13d

---
## [2026-03-05T16:19:57Z] PTCV-88: Decompose ICH E6(R3) Appendix B into Structured Query Schema

**Status Change:** In Progress -> Done
**Actor:** Claude Code
**PRD:** PRD-1

**Summary:**
Decompose ICH E6(R3) Appendix B into Structured Query Schema

**Files Modified:**
- C:/Dev/PTCV/data/templates/appendix_b_queries.yaml
- C:/Dev/PTCV/src/ptcv/ich_parser/query_schema.py
- C:/Dev/PTCV/src/ptcv/ich_parser/__init__.py
- C:/Dev/PTCV/tests/ich_parser/test_query_schema.py

**Test Coverage:**
1 test file(s) found
Branch: `master` | Commits: 28700ea, fc72881, a7eb13d

---
## [2026-03-05T13:21:12Z] PTCV-86: Bug: st_code_diff() called with wrong keyword arguments crashes diff view

**Status Change:** In Progress -> Done
**Actor:** Claude Code

**Summary:**
Bug: st_code_diff() called with wrong keyword arguments crashes diff view

**Files Modified:**
- C:/Dev/PTCV/src/ptcv/ui/app.py

**Test Coverage:**
1 test file(s) found
Branch: `master` | Commits: 28700ea, fc72881, a7eb13d

---
## [2026-03-04T22:08:50Z] PTCV-85: Find industry-sponsored, multi-indication, dose-ranging trials with full protocols on ClinicalTrials.gov

**Status Change:** In Progress -> Done
**Actor:** Claude Code

**Summary:**
Find industry-sponsored, multi-indication, dose-ranging trials with full protocols on ClinicalTrials.gov

**Files Modified:**
- C:/Dev/PTCV/src/ptcv/protocol_search/trial_curator.py
- C:/Dev/PTCV/src/ptcv/protocol_search/__init__.py
- C:/Dev/PTCV/tests/protocol_search/test_trial_curator.py

**Test Coverage:**
1 test file(s) found
Branch: `master` | Commits: 28700ea, fc72881, a7eb13d

---
## [2026-03-04T22:02:57Z] PTCV-84: Add Review Queue UI for low-confidence synonym mappings and ICH classifications

**Status Change:** In Progress -> Done
**Actor:** Claude Code

**Summary:**
Add Review Queue UI for low-confidence synonym mappings and ICH classifications

**Files Modified:**
- C:/Dev/PTCV/src/ptcv/ich_parser/review_queue.py
- C:/Dev/PTCV/src/ptcv/ui/app.py
- C:/Dev/PTCV/src/ptcv/ui/components/review_queue_viewer.py
- C:/Dev/PTCV/tests/ich_parser/test_review_decisions.py
- C:/Dev/PTCV/tests/ui/test_review_queue_viewer.py

**Test Coverage:**
3 test file(s) found
Branch: `master` | Commits: 28700ea, fc72881, a7eb13d

---
## [2026-03-04T21:54:08Z] PTCV-83: Add checkpoint resume for long-running pipelines after session disconnect

**Status Change:** In Progress -> Done
**Actor:** Claude Code

**Summary:**
Add checkpoint resume for long-running pipelines after session disconnect

**Files Modified:**
- C:/Dev/PTCV/src/ptcv/ui/checkpoint_manager.py
- C:/Dev/PTCV/src/ptcv/ui/app.py
- C:/Dev/PTCV/tests/ui/test_checkpoint_manager.py

**Test Coverage:**
2 test file(s) found
Branch: `master` | Commits: 28700ea, fc72881, a7eb13d

---
## [2026-03-04T21:50:06Z] PTCV-82: Add pipeline stage progress visualization using st.status() containers

**Status Change:** In Progress -> Done
**Actor:** Claude Code

**Summary:**
Add pipeline stage progress visualization using st.status() containers

**Files Modified:**
- C:/Dev/PTCV/src/ptcv/ui/app.py

**Test Coverage:**
1 test file(s) found
Branch: `master` | Commits: 28700ea, fc72881, a7eb13d

---
## [2026-03-04T21:47:09Z] PTCV-81: Add annotation hover provenance linking retemplated sections to source pages

**Status Change:** In Progress -> Done
**Actor:** Claude Code

**Summary:**
Add annotation hover provenance linking retemplated sections to source pages

**Files Modified:**
- C:/Dev/PTCV/src/ptcv/ui/app.py
- C:/Dev/PTCV/src/ptcv/ui/components/provenance_renderer.py
- C:/Dev/PTCV/tests/ui/test_provenance_renderer.py

**Test Coverage:**
2 test file(s) found
Branch: `master` | Commits: 28700ea, fc72881, a7eb13d

---
## [2026-03-04T21:33:30Z] PTCV-80: Enhanced side-by-side comparison with synchronized scrolling and section alignment

**Status Change:** In Progress -> Done
**Actor:** Claude Code

**Summary:**
Enhanced side-by-side comparison with synchronized scrolling and section alignment

**Files Modified:**
- C:/Dev/PTCV/src/ptcv/ui/app.py
- C:/Dev/PTCV/src/ptcv/ui/components/section_align.py
- C:/Dev/PTCV/tests/ui/test_section_align.py

**Test Coverage:**
2 test file(s) found
Branch: `master` | Commits: 28700ea, fc72881, a7eb13d

---
## [2026-03-04T21:28:14Z] PTCV-79: Add side-by-side original vs retemplated protocol view using streamlit-code-diff

**Status Change:** In Progress -> Done
**Actor:** Claude Code

**Summary:**
Add side-by-side original vs retemplated protocol view using streamlit-code-diff

**Files Modified:**
- C:/Dev/PTCV/requirements.txt
- C:/Dev/PTCV/src/ptcv/ui/app.py
- C:/Dev/PTCV/src/ptcv/ui/components/protocol_diff.py
- C:/Dev/PTCV/tests/ui/test_protocol_diff.py

**Test Coverage:**
2 test file(s) found
Branch: `master` | Commits: 28700ea, fc72881, a7eb13d

---
## [2026-03-04T21:09:22Z] PTCV-78: Add model cost-tier selector (Opus/Sonnet) to retemplating workflow

**Status Change:** In Progress -> Done
**Actor:** Claude Code

**Summary:**
Add model cost-tier selector (Opus/Sonnet) to retemplating workflow

**Files Modified:**
- C:/Dev/PTCV/src/ptcv/ui/app.py
- C:/Dev/PTCV/tests/ui/test_model_tier.py

**Test Coverage:**
1 test file(s) found
Branch: `master` | Commits: 28700ea, fc72881, a7eb13d

---
## [2026-03-04T20:47:47Z] PTCV-77: Replace workflow selectbox with multi-select checkbox grid and dependency auto-enablement

**Status Change:** In Progress -> Done
**Actor:** Claude Code

**Summary:**
Replace workflow selectbox with multi-select checkbox grid and dependency auto-enablement

**Files Modified:**
- C:/Dev/PTCV/src/ptcv/ui/pipeline_stages.py
- C:/Dev/PTCV/src/ptcv/ui/app.py
- C:/Dev/PTCV/tests/ui/test_pipeline_stages.py

**Test Coverage:**
2 test file(s) found
Branch: `master` | Commits: 28700ea, fc72881, a7eb13d

---
## [2026-03-04T20:40:27Z] PTCV-76: Create SoVPlotData filtered view for schedule_of_visits.py visualization

**Status Change:** In Progress -> Done
**Actor:** Claude Code

**Summary:**
Create SoVPlotData filtered view for schedule_of_visits.py visualization

**Files Modified:**
- C:/Dev/PTCV/src/ptcv/ui/components/schedule_of_visits.py
- C:/Dev/PTCV/src/ptcv/ui/app.py
- C:/Dev/PTCV/tests/ui/test_sov_plot_data.py

**Test Coverage:**
2 test file(s) found
Branch: `master` | Commits: 28700ea, fc72881, a7eb13d

---
## [2026-03-04T20:17:19Z] PTCV-74: SoA extractor produces timepoints from non-visit boilerplate text

**Status Change:** In Progress -> Done
**Actor:** Claude Code

**Summary:**
SoA extractor produces timepoints from non-visit boilerplate text

**Files Modified:**
- C:/Dev/PTCV/src/ptcv/extraction/extraction_service.py
- C:/Dev/PTCV/src/ptcv/extraction/format_detector.py
- C:/Dev/PTCV/src/ptcv/extraction/models.py
- C:/Dev/PTCV/src/ptcv/extraction/pdf_extractor.py
- C:/Dev/PTCV/src/ptcv/pipeline/orchestrator.py
- C:/Dev/PTCV/src/ptcv/protocol_search/__init__.py
- C:/Dev/PTCV/src/ptcv/protocol_search/clinicaltrials_service.py
- C:/Dev/PTCV/src/ptcv/sdtm/__init__.py
- C:/Dev/PTCV/src/ptcv/sdtm/models.py
- C:/Dev/PTCV/src/ptcv/soa_extractor/mapper.py
- C:/Dev/PTCV/src/ptcv/ui/app.py
- C:/Dev/PTCV/src/ptcv/ui/components/ich_regenerator.py
- C:/Dev/PTCV/src/ptcv/ui/components/schedule_of_visits.py
- C:/Dev/PTCV/tests/extraction/test_pdf_extractor.py
- C:/Dev/PTCV/tests/ich_parser/test_parser.py
- C:/Dev/PTCV/tests/soa_extractor/test_mapper.py
- C:/Dev/PTCV/tests/ui/test_schedule_of_visits.py

**Test Coverage:**
11 test file(s) found
Branch: `master` | Commits: 28700ea, fc72881, a7eb13d

---
## [2026-03-04T20:17:18Z] PTCV-73: SoV chart x-axis flooded with duplicate Screening/Day 7 timepoints

**Status Change:** In Progress -> Done
**Actor:** Claude Code

**Summary:**
SoV chart x-axis flooded with duplicate Screening/Day 7 timepoints

**Files Modified:**
- C:/Dev/PTCV/src/ptcv/extraction/extraction_service.py
- C:/Dev/PTCV/src/ptcv/extraction/format_detector.py
- C:/Dev/PTCV/src/ptcv/extraction/models.py
- C:/Dev/PTCV/src/ptcv/extraction/pdf_extractor.py
- C:/Dev/PTCV/src/ptcv/pipeline/orchestrator.py
- C:/Dev/PTCV/src/ptcv/protocol_search/__init__.py
- C:/Dev/PTCV/src/ptcv/protocol_search/clinicaltrials_service.py
- C:/Dev/PTCV/src/ptcv/sdtm/__init__.py
- C:/Dev/PTCV/src/ptcv/sdtm/models.py
- C:/Dev/PTCV/src/ptcv/soa_extractor/mapper.py
- C:/Dev/PTCV/src/ptcv/ui/app.py
- C:/Dev/PTCV/src/ptcv/ui/components/ich_regenerator.py
- C:/Dev/PTCV/src/ptcv/ui/components/schedule_of_visits.py
- C:/Dev/PTCV/tests/extraction/test_pdf_extractor.py
- C:/Dev/PTCV/tests/ich_parser/test_parser.py
- C:/Dev/PTCV/tests/soa_extractor/test_mapper.py
- C:/Dev/PTCV/tests/ui/test_schedule_of_visits.py

**Test Coverage:**
11 test file(s) found
Branch: `master` | Commits: 28700ea, fc72881, a7eb13d

---
## [2026-03-04T20:17:18Z] PTCV-72: SoV timeline x-axis missing day-offset schedule labels

**Status Change:** In Progress -> Done
**Actor:** Claude Code

**Summary:**
SoV timeline x-axis missing day-offset schedule labels

**Files Modified:**
- C:/Dev/PTCV/src/ptcv/extraction/extraction_service.py
- C:/Dev/PTCV/src/ptcv/extraction/format_detector.py
- C:/Dev/PTCV/src/ptcv/extraction/models.py
- C:/Dev/PTCV/src/ptcv/extraction/pdf_extractor.py
- C:/Dev/PTCV/src/ptcv/pipeline/orchestrator.py
- C:/Dev/PTCV/src/ptcv/protocol_search/__init__.py
- C:/Dev/PTCV/src/ptcv/protocol_search/clinicaltrials_service.py
- C:/Dev/PTCV/src/ptcv/sdtm/__init__.py
- C:/Dev/PTCV/src/ptcv/sdtm/models.py
- C:/Dev/PTCV/src/ptcv/soa_extractor/mapper.py
- C:/Dev/PTCV/src/ptcv/ui/app.py
- C:/Dev/PTCV/src/ptcv/ui/components/ich_regenerator.py
- C:/Dev/PTCV/src/ptcv/ui/components/schedule_of_visits.py
- C:/Dev/PTCV/tests/extraction/test_pdf_extractor.py
- C:/Dev/PTCV/tests/ich_parser/test_parser.py
- C:/Dev/PTCV/tests/soa_extractor/test_mapper.py
- C:/Dev/PTCV/tests/ui/test_schedule_of_visits.py

**Test Coverage:**
11 test file(s) found
Branch: `master` | Commits: 28700ea, fc72881, a7eb13d

---
## [2026-03-04T20:17:18Z] PTCV-71: Add Intervention swimlane and fix specimen activity classification in SoV timeline

**Status Change:** In Progress -> Done
**Actor:** Claude Code

**Summary:**
Add Intervention swimlane and fix specimen activity classification in SoV timeline

**Files Modified:**
- C:/Dev/PTCV/src/ptcv/extraction/extraction_service.py
- C:/Dev/PTCV/src/ptcv/extraction/format_detector.py
- C:/Dev/PTCV/src/ptcv/extraction/models.py
- C:/Dev/PTCV/src/ptcv/extraction/pdf_extractor.py
- C:/Dev/PTCV/src/ptcv/pipeline/orchestrator.py
- C:/Dev/PTCV/src/ptcv/protocol_search/__init__.py
- C:/Dev/PTCV/src/ptcv/protocol_search/clinicaltrials_service.py
- C:/Dev/PTCV/src/ptcv/sdtm/__init__.py
- C:/Dev/PTCV/src/ptcv/sdtm/models.py
- C:/Dev/PTCV/src/ptcv/soa_extractor/mapper.py
- C:/Dev/PTCV/src/ptcv/ui/app.py
- C:/Dev/PTCV/src/ptcv/ui/components/ich_regenerator.py
- C:/Dev/PTCV/src/ptcv/ui/components/schedule_of_visits.py
- C:/Dev/PTCV/tests/extraction/test_pdf_extractor.py
- C:/Dev/PTCV/tests/ich_parser/test_parser.py
- C:/Dev/PTCV/tests/soa_extractor/test_mapper.py
- C:/Dev/PTCV/tests/ui/test_schedule_of_visits.py

**Test Coverage:**
11 test file(s) found
Branch: `master` | Commits: 28700ea, fc72881, a7eb13d

---
## [2026-03-04T20:17:18Z] PTCV-70: SoV PNG download crashes app

**Status Change:** In Progress -> Done
**Actor:** Claude Code

**Summary:**
SoV PNG download crashes app

**Files Modified:**
- C:/Dev/PTCV/src/ptcv/extraction/extraction_service.py
- C:/Dev/PTCV/src/ptcv/extraction/format_detector.py
- C:/Dev/PTCV/src/ptcv/extraction/models.py
- C:/Dev/PTCV/src/ptcv/extraction/pdf_extractor.py
- C:/Dev/PTCV/src/ptcv/pipeline/orchestrator.py
- C:/Dev/PTCV/src/ptcv/protocol_search/__init__.py
- C:/Dev/PTCV/src/ptcv/protocol_search/clinicaltrials_service.py
- C:/Dev/PTCV/src/ptcv/sdtm/__init__.py
- C:/Dev/PTCV/src/ptcv/sdtm/models.py
- C:/Dev/PTCV/src/ptcv/soa_extractor/mapper.py
- C:/Dev/PTCV/src/ptcv/ui/app.py
- C:/Dev/PTCV/src/ptcv/ui/components/ich_regenerator.py
- C:/Dev/PTCV/src/ptcv/ui/components/schedule_of_visits.py
- C:/Dev/PTCV/tests/extraction/test_pdf_extractor.py
- C:/Dev/PTCV/tests/ich_parser/test_parser.py
- C:/Dev/PTCV/tests/soa_extractor/test_mapper.py
- C:/Dev/PTCV/tests/ui/test_schedule_of_visits.py

**Test Coverage:**
11 test file(s) found
Branch: `master` | Commits: 28700ea, fc72881, a7eb13d

---
## [2026-03-04T20:17:18Z] PTCV-69: Move review thresholds, SoA patterns, and boilerplate regexes to YAML configuration

**Status Change:** In Progress -> Done
**Actor:** Claude Code

**Summary:**
Move review thresholds, SoA patterns, and boilerplate regexes to YAML configuration

**Files Modified:**
- C:/Dev/PTCV/src/ptcv/extraction/extraction_service.py
- C:/Dev/PTCV/src/ptcv/extraction/format_detector.py
- C:/Dev/PTCV/src/ptcv/extraction/models.py
- C:/Dev/PTCV/src/ptcv/extraction/pdf_extractor.py
- C:/Dev/PTCV/src/ptcv/pipeline/orchestrator.py
- C:/Dev/PTCV/src/ptcv/protocol_search/__init__.py
- C:/Dev/PTCV/src/ptcv/protocol_search/clinicaltrials_service.py
- C:/Dev/PTCV/src/ptcv/sdtm/__init__.py
- C:/Dev/PTCV/src/ptcv/sdtm/models.py
- C:/Dev/PTCV/src/ptcv/soa_extractor/mapper.py
- C:/Dev/PTCV/src/ptcv/ui/app.py
- C:/Dev/PTCV/src/ptcv/ui/components/ich_regenerator.py
- C:/Dev/PTCV/src/ptcv/ui/components/schedule_of_visits.py
- C:/Dev/PTCV/tests/extraction/test_pdf_extractor.py
- C:/Dev/PTCV/tests/ich_parser/test_parser.py
- C:/Dev/PTCV/tests/soa_extractor/test_mapper.py
- C:/Dev/PTCV/tests/ui/test_schedule_of_visits.py

**Test Coverage:**
11 test file(s) found
Branch: `master` | Commits: 28700ea, fc72881, a7eb13d

---
## [2026-03-04T20:17:18Z] PTCV-63: PDF text extraction fails on landscape-oriented pages

**Status Change:** In Progress -> Done
**Actor:** Claude Code

**Summary:**
PDF text extraction fails on landscape-oriented pages

**Files Modified:**
- C:/Dev/PTCV/src/ptcv/extraction/extraction_service.py
- C:/Dev/PTCV/src/ptcv/extraction/format_detector.py
- C:/Dev/PTCV/src/ptcv/extraction/models.py
- C:/Dev/PTCV/src/ptcv/extraction/pdf_extractor.py
- C:/Dev/PTCV/src/ptcv/pipeline/orchestrator.py
- C:/Dev/PTCV/src/ptcv/protocol_search/__init__.py
- C:/Dev/PTCV/src/ptcv/protocol_search/clinicaltrials_service.py
- C:/Dev/PTCV/src/ptcv/sdtm/__init__.py
- C:/Dev/PTCV/src/ptcv/sdtm/models.py
- C:/Dev/PTCV/src/ptcv/soa_extractor/mapper.py
- C:/Dev/PTCV/src/ptcv/ui/app.py
- C:/Dev/PTCV/src/ptcv/ui/components/ich_regenerator.py
- C:/Dev/PTCV/src/ptcv/ui/components/schedule_of_visits.py
- C:/Dev/PTCV/tests/extraction/test_pdf_extractor.py
- C:/Dev/PTCV/tests/ich_parser/test_parser.py
- C:/Dev/PTCV/tests/soa_extractor/test_mapper.py
- C:/Dev/PTCV/tests/ui/test_schedule_of_visits.py

**Test Coverage:**
11 test file(s) found
Branch: `master` | Commits: 28700ea, fc72881, a7eb13d

---
## [2026-03-04T20:17:17Z] PTCV-62: Add Anthropic_API_Key secret to PTCV secrets manager

**Status Change:** In Progress -> Done
**Actor:** Claude Code

**Summary:**
Add Anthropic_API_Key secret to PTCV secrets manager

**Files Modified:**
- C:/Dev/PTCV/src/ptcv/extraction/extraction_service.py
- C:/Dev/PTCV/src/ptcv/extraction/format_detector.py
- C:/Dev/PTCV/src/ptcv/extraction/models.py
- C:/Dev/PTCV/src/ptcv/extraction/pdf_extractor.py
- C:/Dev/PTCV/src/ptcv/pipeline/orchestrator.py
- C:/Dev/PTCV/src/ptcv/protocol_search/__init__.py
- C:/Dev/PTCV/src/ptcv/protocol_search/clinicaltrials_service.py
- C:/Dev/PTCV/src/ptcv/sdtm/__init__.py
- C:/Dev/PTCV/src/ptcv/sdtm/models.py
- C:/Dev/PTCV/src/ptcv/soa_extractor/mapper.py
- C:/Dev/PTCV/src/ptcv/ui/app.py
- C:/Dev/PTCV/src/ptcv/ui/components/ich_regenerator.py
- C:/Dev/PTCV/src/ptcv/ui/components/schedule_of_visits.py
- C:/Dev/PTCV/tests/extraction/test_pdf_extractor.py
- C:/Dev/PTCV/tests/ich_parser/test_parser.py
- C:/Dev/PTCV/tests/soa_extractor/test_mapper.py
- C:/Dev/PTCV/tests/ui/test_schedule_of_visits.py

**Test Coverage:**
11 test file(s) found
Branch: `master` | Commits: 28700ea, fc72881, a7eb13d

---
## [2026-03-04T20:17:17Z] PTCV-58: Integrate CDISC Library REST API for controlled terminology validation

**Status Change:** In Progress -> Done
**Actor:** Claude Code

**Summary:**
Integrate CDISC Library REST API for controlled terminology validation

**Files Modified:**
- C:/Dev/PTCV/src/ptcv/extraction/extraction_service.py
- C:/Dev/PTCV/src/ptcv/extraction/format_detector.py
- C:/Dev/PTCV/src/ptcv/extraction/models.py
- C:/Dev/PTCV/src/ptcv/extraction/pdf_extractor.py
- C:/Dev/PTCV/src/ptcv/pipeline/orchestrator.py
- C:/Dev/PTCV/src/ptcv/protocol_search/__init__.py
- C:/Dev/PTCV/src/ptcv/protocol_search/clinicaltrials_service.py
- C:/Dev/PTCV/src/ptcv/sdtm/__init__.py
- C:/Dev/PTCV/src/ptcv/sdtm/models.py
- C:/Dev/PTCV/src/ptcv/soa_extractor/mapper.py
- C:/Dev/PTCV/src/ptcv/ui/app.py
- C:/Dev/PTCV/src/ptcv/ui/components/ich_regenerator.py
- C:/Dev/PTCV/src/ptcv/ui/components/schedule_of_visits.py
- C:/Dev/PTCV/tests/extraction/test_pdf_extractor.py
- C:/Dev/PTCV/tests/ich_parser/test_parser.py
- C:/Dev/PTCV/tests/soa_extractor/test_mapper.py
- C:/Dev/PTCV/tests/ui/test_schedule_of_visits.py

**Test Coverage:**
11 test file(s) found
Branch: `master` | Commits: 28700ea, fc72881, a7eb13d

---
## [2026-03-04T20:17:17Z] PTCV-56: Implement ICH M11 CeSHarP machine-readable protocol parser for direct SoA extraction

**Status Change:** In Progress -> Done
**Actor:** Claude Code

**Summary:**
Implement ICH M11 CeSHarP machine-readable protocol parser for direct SoA extraction

**Files Modified:**
- C:/Dev/PTCV/src/ptcv/extraction/extraction_service.py
- C:/Dev/PTCV/src/ptcv/extraction/format_detector.py
- C:/Dev/PTCV/src/ptcv/extraction/models.py
- C:/Dev/PTCV/src/ptcv/extraction/pdf_extractor.py
- C:/Dev/PTCV/src/ptcv/pipeline/orchestrator.py
- C:/Dev/PTCV/src/ptcv/protocol_search/__init__.py
- C:/Dev/PTCV/src/ptcv/protocol_search/clinicaltrials_service.py
- C:/Dev/PTCV/src/ptcv/sdtm/__init__.py
- C:/Dev/PTCV/src/ptcv/sdtm/models.py
- C:/Dev/PTCV/src/ptcv/soa_extractor/mapper.py
- C:/Dev/PTCV/src/ptcv/ui/app.py
- C:/Dev/PTCV/src/ptcv/ui/components/ich_regenerator.py
- C:/Dev/PTCV/src/ptcv/ui/components/schedule_of_visits.py
- C:/Dev/PTCV/tests/extraction/test_pdf_extractor.py
- C:/Dev/PTCV/tests/ich_parser/test_parser.py
- C:/Dev/PTCV/tests/soa_extractor/test_mapper.py
- C:/Dev/PTCV/tests/ui/test_schedule_of_visits.py

**Test Coverage:**
11 test file(s) found
Branch: `master` | Commits: 28700ea, fc72881, a7eb13d

---
## [2026-03-04T20:17:17Z] PTCV-55: Parameterize natural history data search from SoA-derived endpoints and visit windows

**Status Change:** In Progress -> Done
**Actor:** Claude Code

**Summary:**
Parameterize natural history data search from SoA-derived endpoints and visit windows

**Files Modified:**
- C:/Dev/PTCV/src/ptcv/extraction/extraction_service.py
- C:/Dev/PTCV/src/ptcv/extraction/format_detector.py
- C:/Dev/PTCV/src/ptcv/extraction/models.py
- C:/Dev/PTCV/src/ptcv/extraction/pdf_extractor.py
- C:/Dev/PTCV/src/ptcv/pipeline/orchestrator.py
- C:/Dev/PTCV/src/ptcv/protocol_search/__init__.py
- C:/Dev/PTCV/src/ptcv/protocol_search/clinicaltrials_service.py
- C:/Dev/PTCV/src/ptcv/sdtm/__init__.py
- C:/Dev/PTCV/src/ptcv/sdtm/models.py
- C:/Dev/PTCV/src/ptcv/soa_extractor/mapper.py
- C:/Dev/PTCV/src/ptcv/ui/app.py
- C:/Dev/PTCV/src/ptcv/ui/components/ich_regenerator.py
- C:/Dev/PTCV/src/ptcv/ui/components/schedule_of_visits.py
- C:/Dev/PTCV/tests/extraction/test_pdf_extractor.py
- C:/Dev/PTCV/tests/ich_parser/test_parser.py
- C:/Dev/PTCV/tests/soa_extractor/test_mapper.py
- C:/Dev/PTCV/tests/ui/test_schedule_of_visits.py

**Test Coverage:**
11 test file(s) found
Branch: `master` | Commits: 28700ea, fc72881, a7eb13d

---
## [2026-03-04T20:17:17Z] PTCV-54: Generate synthetic SDTM datasets from SoA-derived schemas

**Status Change:** In Progress -> Done
**Actor:** Claude Code

**Summary:**
Generate synthetic SDTM datasets from SoA-derived schemas

**Files Modified:**
- C:/Dev/PTCV/src/ptcv/extraction/extraction_service.py
- C:/Dev/PTCV/src/ptcv/extraction/format_detector.py
- C:/Dev/PTCV/src/ptcv/extraction/models.py
- C:/Dev/PTCV/src/ptcv/extraction/pdf_extractor.py
- C:/Dev/PTCV/src/ptcv/pipeline/orchestrator.py
- C:/Dev/PTCV/src/ptcv/protocol_search/__init__.py
- C:/Dev/PTCV/src/ptcv/protocol_search/clinicaltrials_service.py
- C:/Dev/PTCV/src/ptcv/sdtm/__init__.py
- C:/Dev/PTCV/src/ptcv/sdtm/models.py
- C:/Dev/PTCV/src/ptcv/soa_extractor/mapper.py
- C:/Dev/PTCV/src/ptcv/ui/app.py
- C:/Dev/PTCV/src/ptcv/ui/components/ich_regenerator.py
- C:/Dev/PTCV/src/ptcv/ui/components/schedule_of_visits.py
- C:/Dev/PTCV/tests/extraction/test_pdf_extractor.py
- C:/Dev/PTCV/tests/ich_parser/test_parser.py
- C:/Dev/PTCV/tests/soa_extractor/test_mapper.py
- C:/Dev/PTCV/tests/ui/test_schedule_of_visits.py

**Test Coverage:**
11 test file(s) found
Branch: `master` | Commits: 28700ea, fc72881, a7eb13d

---
## [2026-03-04T20:17:17Z] PTCV-53: Map extracted SoA tables to mock SDTM domains (TV, TA, TE, SE)

**Status Change:** In Progress -> Done
**Actor:** Claude Code

**Summary:**
Map extracted SoA tables to mock SDTM domains (TV, TA, TE, SE)

**Files Modified:**
- C:/Dev/PTCV/src/ptcv/extraction/extraction_service.py
- C:/Dev/PTCV/src/ptcv/extraction/format_detector.py
- C:/Dev/PTCV/src/ptcv/extraction/models.py
- C:/Dev/PTCV/src/ptcv/extraction/pdf_extractor.py
- C:/Dev/PTCV/src/ptcv/pipeline/orchestrator.py
- C:/Dev/PTCV/src/ptcv/protocol_search/__init__.py
- C:/Dev/PTCV/src/ptcv/protocol_search/clinicaltrials_service.py
- C:/Dev/PTCV/src/ptcv/sdtm/__init__.py
- C:/Dev/PTCV/src/ptcv/sdtm/models.py
- C:/Dev/PTCV/src/ptcv/soa_extractor/mapper.py
- C:/Dev/PTCV/src/ptcv/ui/app.py
- C:/Dev/PTCV/src/ptcv/ui/components/ich_regenerator.py
- C:/Dev/PTCV/src/ptcv/ui/components/schedule_of_visits.py
- C:/Dev/PTCV/tests/extraction/test_pdf_extractor.py
- C:/Dev/PTCV/tests/ich_parser/test_parser.py
- C:/Dev/PTCV/tests/soa_extractor/test_mapper.py
- C:/Dev/PTCV/tests/ui/test_schedule_of_visits.py

**Test Coverage:**
11 test file(s) found
Branch: `master` | Commits: 28700ea, fc72881, a7eb13d

---
## [2026-03-04T20:17:17Z] PTCV-52: Targeted extraction pipeline for ICH E6(R3) B4, B5, B10, B14 sections

**Status Change:** In Progress -> Done
**Actor:** Claude Code

**Summary:**
Targeted extraction pipeline for ICH E6(R3) B4, B5, B10, B14 sections

**Files Modified:**
- C:/Dev/PTCV/src/ptcv/extraction/extraction_service.py
- C:/Dev/PTCV/src/ptcv/extraction/format_detector.py
- C:/Dev/PTCV/src/ptcv/extraction/models.py
- C:/Dev/PTCV/src/ptcv/extraction/pdf_extractor.py
- C:/Dev/PTCV/src/ptcv/pipeline/orchestrator.py
- C:/Dev/PTCV/src/ptcv/protocol_search/__init__.py
- C:/Dev/PTCV/src/ptcv/protocol_search/clinicaltrials_service.py
- C:/Dev/PTCV/src/ptcv/sdtm/__init__.py
- C:/Dev/PTCV/src/ptcv/sdtm/models.py
- C:/Dev/PTCV/src/ptcv/soa_extractor/mapper.py
- C:/Dev/PTCV/src/ptcv/ui/app.py
- C:/Dev/PTCV/src/ptcv/ui/components/ich_regenerator.py
- C:/Dev/PTCV/src/ptcv/ui/components/schedule_of_visits.py
- C:/Dev/PTCV/tests/extraction/test_pdf_extractor.py
- C:/Dev/PTCV/tests/ich_parser/test_parser.py
- C:/Dev/PTCV/tests/soa_extractor/test_mapper.py
- C:/Dev/PTCV/tests/ui/test_schedule_of_visits.py

**Test Coverage:**
11 test file(s) found
Branch: `master` | Commits: 28700ea, fc72881, a7eb13d

---
## [2026-03-04T19:52:37Z] PTCV-48: [Fallback Tier] Expand RAGClassifier context window from 3K to 8K chars

**Status Change:** In Progress -> Done
**Actor:** Claude Code

**Summary:**
[Fallback Tier] Expand RAGClassifier context window from 3K to 8K chars

**Files Modified:**
- C:/Dev/PTCV/src/ptcv/ich_parser/classifier.py
- C:/Dev/PTCV/tests/ich_parser/test_classifier_context.py
- C:/Dev/PTCV/tests/ich_parser/conftest.py
- C:/Dev/PTCV/tests/ich_parser/test_classifier.py

**Test Coverage:**
1 test file(s) found
Branch: `master` | Commits: 28700ea, fc72881, a7eb13d

---
## [2026-03-04T19:09:10Z] PTCV-47: Implement dynamic few-shot prompting in LlmRetemplater classification prompt

**Status Change:** In Progress -> Done
**Actor:** Claude Code

**Summary:**
Implement dynamic few-shot prompting in LlmRetemplater classification prompt

**Files Modified:**
- C:/Dev/PTCV/src/ptcv/ich_parser/llm_retemplater.py
- C:/Dev/PTCV/tests/ich_parser/test_retemplater_few_shot.py

**Test Coverage:**
1 test file(s) found
Branch: `master` | Commits: 28700ea, fc72881, a7eb13d

---
## [2026-03-04T18:36:08Z] PTCV-57: Implement visit schedule feasibility validator with cross-visit window and ordering checks

**Status Change:** In Progress -> Done
**Actor:** Claude Code

**Summary:**
Implement visit schedule feasibility validator with cross-visit window and ordering checks

**Files Modified:**
- C:/Dev/PTCV/src/ptcv/sdtm/validation/schedule_validator.py
- C:/Dev/PTCV/src/ptcv/sdtm/validation/models.py
- C:/Dev/PTCV/src/ptcv/sdtm/validation/validation_service.py
- C:/Dev/PTCV/src/ptcv/sdtm/validation/__init__.py
- C:/Dev/PTCV/tests/sdtm/validation/test_schedule_validator.py

**Test Coverage:**
3 test file(s) found
Branch: `master` | Commits: 28700ea, fc72881, a7eb13d

---
## [2026-03-04T18:33:00Z] PTCV-65: Implement retemplating fidelity checker to detect hallucinations and content loss

**Status Change:** In Progress -> Done
**Actor:** Claude Code

**Summary:**
Implement retemplating fidelity checker to detect hallucinations and content loss

**Files Modified:**
- C:/Dev/PTCV/src/ptcv/ich_parser/fidelity_checker.py
- C:/Dev/PTCV/tests/ich_parser/test_fidelity_checker.py

**Test Coverage:**
1 test file(s) found
Branch: `master` | Commits: 28700ea, fc72881, a7eb13d

---
## [2026-03-04T18:18:35Z] PTCV-68: Create stage-specific compressed ICH prompt templates with relevance filtering

**Status Change:** In Progress -> Done
**Actor:** Claude Code

**Summary:**
Create stage-specific compressed ICH prompt templates with relevance filtering

**Files Modified:**
- C:/Dev/PTCV/data/templates/ich_e6r3_schema.yaml
- C:/Dev/PTCV/src/ptcv/ich_parser/schema_loader.py
- C:/Dev/PTCV/src/ptcv/ich_parser/llm_retemplater.py
- C:/Dev/PTCV/tests/ich_parser/test_schema_loader.py
- C:/Dev/PTCV/scripts/benchmark_ich_prompt_tokens.py
- C:/Dev/PTCV/tests/scripts/test_benchmark_ich_prompt.py

**Test Coverage:**
3 test file(s) found
Branch: `master` | Commits: 28700ea, fc72881, a7eb13d

---
## [2026-03-04T18:18:00Z] PTCV-64: Implement real ICH E6(R3) retemplater via Claude Opus 4.6 language transformation

**Status Change:** In Progress -> Done
**Actor:** Claude Code

**Summary:**
Implement real ICH E6(R3) retemplater via Claude Opus 4.6 language transformation

**Files Modified:**
- C:/Dev/PTCV/data/templates/ich_e6r3_schema.yaml
- C:/Dev/PTCV/src/ptcv/ich_parser/llm_retemplater.py
- C:/Dev/PTCV/src/ptcv/ich_parser/models.py
- C:/Dev/PTCV/src/ptcv/ich_parser/coverage_reviewer.py
- C:/Dev/PTCV/src/ptcv/ich_parser/classifier.py
- C:/Dev/PTCV/src/ptcv/ich_parser/schema_loader.py
- C:/Dev/PTCV/src/ptcv/ich_parser/__init__.py
- C:/Dev/PTCV/src/ptcv/pipeline/models.py
- C:/Dev/PTCV/src/ptcv/pipeline/orchestrator.py
- C:/Dev/PTCV/src/ptcv/ui/app.py
- C:/Dev/PTCV/src/ptcv/ui/components/ich_regenerator.py
- C:/Dev/PTCV/tests/ich_parser/test_format_detector.py
- C:/Dev/PTCV/tests/ich_parser/test_format_verdict.py
- C:/Dev/PTCV/tests/ich_parser/test_schema_loader.py
- C:/Dev/PTCV/src/ptcv/extraction/parquet_writer.py

**Test Coverage:**
10 test file(s) found
Branch: `master` | Commits: 28700ea, fc72881, a7eb13d

---
## [2026-03-04T14:51:26Z] PTCV-67: Externalize ICH E6(R3) section definitions to a single YAML schema file

**Status Change:** In Progress -> Done
**Actor:** Claude Code

**Summary:**
Externalize ICH E6(R3) section definitions to a single YAML schema file

**Files Modified:**
- C:/Dev/PTCV/data/templates/ich_e6r3_schema.yaml
- C:/Dev/PTCV/src/ptcv/ich_parser/schema_loader.py
- C:/Dev/PTCV/src/ptcv/ich_parser/classifier.py
- C:/Dev/PTCV/src/ptcv/ich_parser/llm_retemplater.py
- C:/Dev/PTCV/tests/ich_parser/test_schema_loader.py
- C:/Dev/PTCV/.gitignore

**Test Coverage:**
3 test file(s) found
Branch: `master` | Commits: 28700ea, fc72881, a7eb13d

---
## [2026-03-04T12:52:35Z] PTCV-60: Reorder pipeline: document-first SoA extraction with ICH retemplating

**Status Change:** In Progress -> Done
**Actor:** Claude Code

**Summary:**
Reorder pipeline: document-first SoA extraction with ICH retemplating

**Files Modified:**
- C:/Dev/PTCV/requirements.txt
- C:/Dev/PTCV/src/ptcv/ich_parser/llm_retemplater.py
- C:/Dev/PTCV/src/ptcv/ich_parser/coverage_reviewer.py
- C:/Dev/PTCV/src/ptcv/pipeline/orchestrator.py
- C:/Dev/PTCV/src/ptcv/pipeline/models.py
- C:/Dev/PTCV/src/ptcv/soa_extractor/extractor.py
- C:/Dev/PTCV/tests/ich_parser/test_llm_retemplater.py
- C:/Dev/PTCV/tests/ich_parser/test_coverage_reviewer.py
- C:/Dev/PTCV/tests/pipeline/test_orchestrator.py
- C:/Dev/PTCV/tests/pipeline/conftest.py

**Test Coverage:**
5 test file(s) found
Branch: `master` | Commits: 28700ea, fc72881, a7eb13d

---
## [2026-03-04T12:42:31Z] PTCV-59: Implement protocol amendment diff engine for versioned SoA tracking

**Status Change:** In Progress -> Done
**Actor:** Claude Code

**Summary:**
Implement protocol amendment diff engine for versioned SoA tracking

**Files Modified:**
- C:/Dev/PTCV/.gitignore
- C:/Dev/PTCV/jira_changelog.md
- C:/Dev/PTCV/requirements.txt
- C:/Dev/PTCV/src/ptcv/extraction/extraction_service.py
- C:/Dev/PTCV/src/ptcv/extraction/format_detector.py
- C:/Dev/PTCV/src/ptcv/extraction/models.py
- C:/Dev/PTCV/src/ptcv/extraction/parquet_writer.py
- C:/Dev/PTCV/src/ptcv/extraction/pdf_extractor.py
- C:/Dev/PTCV/src/ptcv/ich_parser/__init__.py
- C:/Dev/PTCV/src/ptcv/ich_parser/classifier.py
- C:/Dev/PTCV/src/ptcv/ich_parser/format_detector.py
- C:/Dev/PTCV/src/ptcv/pipeline/__init__.py
- C:/Dev/PTCV/src/ptcv/pipeline/models.py
- C:/Dev/PTCV/src/ptcv/pipeline/orchestrator.py
- C:/Dev/PTCV/src/ptcv/protocol_search/__init__.py
- C:/Dev/PTCV/src/ptcv/protocol_search/clinicaltrials_service.py
- C:/Dev/PTCV/src/ptcv/sdtm/__init__.py
- C:/Dev/PTCV/src/ptcv/sdtm/models.py
- C:/Dev/PTCV/src/ptcv/sdtm/validation/__init__.py
- C:/Dev/PTCV/src/ptcv/soa_extractor/extractor.py
- C:/Dev/PTCV/src/ptcv/ui/app.py
- C:/Dev/PTCV/tests/extraction/test_pdf_extractor.py
- C:/Dev/PTCV/tests/ich_parser/test_format_detector.py
- C:/Dev/PTCV/tests/ich_parser/test_format_verdict.py
- C:/Dev/PTCV/tests/pipeline/conftest.py
- C:/Dev/PTCV/tests/pipeline/test_models.py
- C:/Dev/PTCV/tests/pipeline/test_orchestrator.py

**Test Coverage:**
13 test file(s) found
Branch: `master` | Commits: 28700ea, fc72881, a7eb13d

---
## [2026-03-03T23:06:49Z] PTCV-61: Reconcile Streamlit UI with PTCV-60 document-first pipeline reorder

**Status Change:** In Progress -> Done
**Actor:** Claude Code

**Summary:**
Reconcile Streamlit UI with PTCV-60 document-first pipeline reorder

**Files Modified:**
- C:/Dev/PTCV/src/ptcv/ui/app.py

**Test Coverage:**
1 test file(s) found
Branch: `master` | Commits: 28700ea, fc72881, a7eb13d

---
## [2026-03-03T18:34:13Z] PTCV-51: SoA extraction pipeline ignores pre-extracted tables.parquet data

**Status Change:** In Progress -> Done
**Actor:** Claude Code

**Summary:**
SoA extraction pipeline ignores pre-extracted tables.parquet data

**Files Modified:**
- C:/Dev/PTCV/src/ptcv/soa_extractor/table_bridge.py
- C:/Dev/PTCV/src/ptcv/soa_extractor/table_discovery.py
- C:/Dev/PTCV/src/ptcv/soa_extractor/extractor.py
- C:/Dev/PTCV/src/ptcv/soa_extractor/__init__.py
- C:/Dev/PTCV/src/ptcv/soa_extractor/parser.py
- C:/Dev/PTCV/src/ptcv/pipeline/orchestrator.py
- C:/Dev/PTCV/src/ptcv/ui/app.py
- C:/Dev/PTCV/tests/soa_extractor/test_table_bridge.py
- C:/Dev/PTCV/tests/soa_extractor/test_table_discovery.py
- C:/Dev/PTCV/tests/annotations/test_models.py

**Test Coverage:**
6 test file(s) found
Branch: `master` | Commits: 7aadfce, f711be6, 20d488e

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
