"""Shared fixtures for ICH parser tests."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Ensure src/ is on the path for non-installed package
sys.path.insert(0, str(Path(__file__).parents[2] / "src"))


# A minimal but realistic protocol excerpt covering several ICH sections
SAMPLE_PROTOCOL_TEXT = """
1. GENERAL INFORMATION

Protocol Title: A Phase 2 Randomised Double-Blind Placebo-Controlled Study of
Drug X in Patients with Advanced Oncology

Protocol Number: EUCT-2024-000001-01
Sponsor: Acme Pharma Ltd
Version: 1.0
Date: 2024-01-15
EudraCT Number: 2024-000001-11

2. BACKGROUND INFORMATION

Background and Rationale:
Drug X is a novel kinase inhibitor with preclinical evidence of efficacy in
KRAS-mutant solid tumours. Prior nonclinical studies demonstrated dose-dependent
tumour regression in mouse xenograft models. A Phase 1 study showed acceptable
tolerability at doses up to 200 mg BID.

Literature review confirms that current treatment options for KRAS-mutant cancers
remain limited, with an unmet medical need for new therapeutic approaches.

3. OBJECTIVES AND PURPOSE

Primary Objective:
To evaluate the efficacy of Drug X as measured by overall response rate (ORR)
in patients with KRAS-mutant advanced solid tumours.

Secondary Endpoints:
- Progression-free survival (PFS)
- Overall survival (OS)
- Duration of response (DoR)
- Safety and tolerability

Hypothesis: Drug X will demonstrate a clinically meaningful improvement in ORR
compared to placebo.

4. TRIAL DESIGN

Study Design:
This is a randomised, double-blind, placebo-controlled, parallel-group Phase 2
study. Patients will be randomised 2:1 (Drug X:placebo) using a central
randomisation system. Blinding will be maintained through matching placebo.

5. SELECTION OF SUBJECTS

Inclusion Criteria:
1. Adults aged >= 18 years
2. Histologically confirmed KRAS-mutant solid tumour
3. ECOG performance status 0-1
4. Adequate organ function

Exclusion Criteria:
1. Prior treatment with a KRAS inhibitor
2. Active CNS metastases
3. Contraindication to study drug

6. TREATMENT

Investigational Medicinal Product: Drug X 150 mg BID oral
Dosing Schedule: Continuous daily dosing in 28-day cycles
Concomitant Medications: Supportive care permitted; no other antineoplastics

7. SAFETY ASSESSMENTS

Adverse Events: All adverse events will be graded using CTCAE v5.0
Serious Adverse Events (SAE): Reported within 24 hours to sponsor
Vital Signs: Assessed at each clinic visit
Laboratory Tests: Haematology, biochemistry, urinalysis at screening and Day 1 each cycle
ECG: At screening and Day 1 Cycle 1

8. STATISTICAL METHODS

Statistical Analysis Plan: See separate SAP document
Sample Size: 90 patients (60 Drug X, 30 placebo) provides 80% power
Significance Level: One-sided alpha = 0.025
Analysis Population: Modified intention-to-treat (mITT) — all randomised patients
who received at least one dose
Missing Data: Patients lost to follow-up treated as non-responders

9. QUALITY ASSURANCE

GCP Compliance: This study will be conducted in accordance with ICH E6(R3) GCP
Audit: The sponsor may conduct audits of the study sites
Monitoring: Regular site monitoring visits per monitoring plan
Source Data Verification: 100% SDV for primary endpoint data
Data Management: Electronic data capture (EDC) system with audit trail
"""


@pytest.fixture
def sample_text() -> str:
    return SAMPLE_PROTOCOL_TEXT


@pytest.fixture
def tmp_gateway(tmp_path: Path):
    """FilesystemAdapter pointed at a temporary directory."""
    from ptcv.storage import FilesystemAdapter
    gw = FilesystemAdapter(root=tmp_path)
    gw.initialise()
    return gw


@pytest.fixture
def tmp_review_queue(tmp_path: Path):
    """ReviewQueue backed by a temporary SQLite file."""
    from ptcv.ich_parser import ReviewQueue
    rq = ReviewQueue(db_path=tmp_path / "review_queue.db")
    rq.initialise()
    return rq
