"""Shared fixtures for ICH parser tests."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Ensure src/ is on the path for non-installed package
sys.path.insert(0, str(Path(__file__).parents[2] / "src"))


# A realistic protocol excerpt covering several ICH sections.
# Each section must be >=500 chars to survive block merging (PTCV-48).
SAMPLE_PROTOCOL_TEXT = """
1. GENERAL INFORMATION

Protocol Title: A Phase 2 Randomised Double-Blind Placebo-Controlled Study of
Drug X in Patients with Advanced Oncology

Protocol Number: EUCT-2024-000001-01
Sponsor: Acme Pharma Ltd
Version: 1.0
Date: 2024-01-15
EudraCT Number: 2024-000001-11
Regulatory Authority: European Medicines Agency (EMA)
IND Reference: IND-2024-0042
Clinical Trial Application: CTA submitted 2024-01-10
Coordinating Investigator: Prof. J. Smith, University Hospital
Medical Monitor: Dr. A. Jones, Acme Pharma Ltd
Contract Research Organisation: Global CRO Partners Inc.

2. BACKGROUND INFORMATION

Background and Rationale:
Drug X is a novel kinase inhibitor with preclinical evidence of efficacy in
KRAS-mutant solid tumours. Prior nonclinical studies demonstrated dose-dependent
tumour regression in mouse xenograft models. A Phase 1 study showed acceptable
tolerability at doses up to 200 mg BID with manageable gastrointestinal toxicity.

Literature review confirms that current treatment options for KRAS-mutant cancers
remain limited, with an unmet medical need for new therapeutic approaches.
Recent publications (Smith et al. 2023, Jones et al. 2024) have demonstrated
the potential of allosteric kinase inhibitors in this disease setting, providing
the scientific rationale for this Phase 2 investigation.

Nonclinical pharmacology studies showed IC50 values of 12 nM for KRAS G12C and
28 nM for KRAS G12D mutations. In vivo efficacy was observed across multiple
xenograft models including patient-derived xenografts with confirmed KRAS
mutations. The therapeutic window between efficacy and toxicity was favourable.

3. OBJECTIVES AND PURPOSE

Primary Objective:
To evaluate the efficacy of Drug X as measured by overall response rate (ORR)
in patients with KRAS-mutant advanced solid tumours, assessed by blinded
independent central review (BICR) per RECIST v1.1 criteria.

Secondary Endpoints:
- Progression-free survival (PFS) by BICR per RECIST v1.1
- Overall survival (OS) at 12 and 24 months
- Duration of response (DoR) in confirmed responders
- Safety and tolerability as assessed by CTCAE v5.0
- Pharmacokinetic profile of Drug X at steady state
- Quality of life assessed by EORTC QLQ-C30

Hypothesis: Drug X will demonstrate a clinically meaningful improvement in ORR
compared to placebo, with an expected ORR of at least 30% versus 5% for placebo.

4. TRIAL DESIGN

Study Design:
This is a randomised, double-blind, placebo-controlled, parallel-group Phase 2
study. Patients will be randomised 2:1 (Drug X:placebo) using a central
interactive web response system (IWRS). Blinding will be maintained through
matching placebo capsules identical in appearance, taste, and packaging.
Stratification factors include KRAS mutation subtype (G12C vs other), ECOG
performance status (0 vs 1), and number of prior lines of therapy (1 vs >=2).
The study will be conducted at approximately 30 sites across 8 countries in
Europe and North America. An independent Data Monitoring Committee (DMC) will
review unblinded safety data at regular intervals per the DMC charter.

5. SELECTION OF SUBJECTS

Inclusion Criteria:
1. Adults aged >= 18 years at the time of informed consent
2. Histologically or cytologically confirmed KRAS-mutant solid tumour
3. ECOG performance status 0-1 at screening and baseline
4. Adequate organ function as defined by protocol-specified laboratory values
5. At least one measurable lesion per RECIST v1.1 by CT or MRI
6. Life expectancy of at least 12 weeks
7. Documented progression on or after last systemic therapy
8. Willing and able to provide written informed consent

Exclusion Criteria:
1. Prior treatment with a KRAS inhibitor or related pathway inhibitor
2. Active CNS metastases (treated and stable CNS disease allowed)
3. Known hypersensitivity or contraindication to study drug components
4. Clinically significant cardiovascular disease within 6 months
5. Active autoimmune disease requiring systemic immunosuppression
6. Concurrent participation in another interventional clinical study

6. TREATMENT

Investigational Medicinal Product: Drug X 150 mg BID oral administration
Dosing Schedule: Continuous daily dosing in 28-day cycles until disease
progression, unacceptable toxicity, withdrawal of consent, or investigator
decision. Drug X capsules should be taken with food, approximately 12 hours
apart. Dose modifications are permitted according to the protocol-defined
algorithm: first reduction to 100 mg BID, second reduction to 50 mg BID.
Patients requiring dose reduction below 50 mg BID will discontinue treatment.
Comparator: Matching placebo capsules administered on the same schedule.
Concomitant Medications: Standard supportive care is permitted including
antiemetics, analgesics, and bisphosphonates. No concurrent antineoplastic
agents, investigational drugs, or strong CYP3A4 inhibitors are allowed.

7. SAFETY ASSESSMENTS

Adverse Events: All adverse events will be graded using CTCAE v5.0 criteria
and recorded from informed consent through 30 days after last dose.
Serious Adverse Events (SAE): Reported within 24 hours to sponsor safety
department. SAEs include death, life-threatening events, hospitalisation,
persistent disability, congenital anomaly, and other medically significant
events per ICH E2A guidelines.
Vital Signs: Blood pressure, pulse rate, temperature, and respiratory rate
assessed at each clinic visit and prior to each dosing on Day 1 of each cycle.
Laboratory Tests: Haematology (CBC with differential), serum chemistry
(comprehensive metabolic panel), coagulation (PT/INR, aPTT), and urinalysis
at screening, Day 1 of each cycle, and at end of treatment.
ECG: 12-lead ECG at screening, Day 1 Cycle 1, Day 1 Cycle 2, and every
3 cycles thereafter. QTc monitoring with centralised reading.

8. STATISTICAL METHODS

Statistical Analysis Plan: Detailed in a separate SAP document, finalised
prior to database lock and unblinding.
Sample Size: 90 patients (60 Drug X, 30 placebo) provides 80% power to detect
an ORR difference of 25% (30% vs 5%) using a two-proportion z-test at a
one-sided significance level of 0.025.
Significance Level: One-sided alpha = 0.025 for the primary endpoint.
Analysis Population: Modified intention-to-treat (mITT) population includes
all randomised patients who received at least one dose of study medication.
Per-protocol population for sensitivity analysis.
Missing Data: Patients lost to follow-up treated as non-responders in
primary analysis. Sensitivity analyses using multiple imputation and
tipping-point analysis will be conducted. Kaplan-Meier methods for
time-to-event endpoints. Cox proportional hazards model for PFS and OS.

9. QUALITY ASSURANCE

GCP Compliance: This study will be conducted in full accordance with the
principles of ICH E6(R3) Good Clinical Practice, the Declaration of Helsinki,
and applicable local regulatory requirements in each participating country.
Audit: The sponsor reserves the right to conduct quality assurance audits of
study sites, the CRO, and central laboratories at any time during the study.
Regulatory authorities may also conduct inspections per local requirements.
Monitoring: Risk-based monitoring approach with regular on-site and remote
monitoring visits per the study-specific monitoring plan. Central statistical
monitoring for data anomaly detection will supplement on-site visits.
Source Data Verification: 100% SDV for primary endpoint data, informed consent,
SAEs, and eligibility criteria. Risk-proportionate SDV for other data points.
Data Management: Electronic data capture (EDC) system with full audit trail,
21 CFR Part 11 compliance, and automatic edit checks at point of entry.
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
