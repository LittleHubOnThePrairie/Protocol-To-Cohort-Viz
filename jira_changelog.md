# Jira Changelog

This file tracks Jira issue status changes made by Claude Code.
Use "Post summary of all Jira changes to Confluence" to sync to Confluence.

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
