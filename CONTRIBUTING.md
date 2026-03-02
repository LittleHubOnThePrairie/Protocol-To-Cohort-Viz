# Contributing to PTCV (ProtocolToCohortViz)

PTCV is ML-enabled clinical trial software. Every contribution — from a human
developer or an AI agent — must satisfy requirements from:

- **NIST AI RMF 1.0** — AI risk management (adopted as project standard)
- **21 CFR Part 11** — Electronic records and signatures (mandatory, FDA)
- **ICH E6(R3)** — Good Clinical Practice (mandatory, FDA-adopted Sept 2025)
- **ALCOA+** — Data integrity principles (mandatory for GxP, PIC/S PI 041-1)

Compliance is not optional. An FDA inspector will evaluate the artifacts your
code produces — audit logs, validation reports, data lineage records, electronic
signatures. This document ensures those artifacts are complete and defensible.

For research background and regulatory crosswalk, see:
https://littlehubonprairie.atlassian.net/wiki/spaces/PTCV/pages/31031298

---

## Boundaries

### Always Do (Autonomous — no approval needed)

- **Audit trail on every data mutation.** Any function that creates, modifies,
  or deletes a regulated record MUST write to the append-only audit log with
  WHO (user ID), WHAT (operation + before/after values), WHEN (UTC ms timestamp),
  WHY (justification string). [21 CFR 11.10(e), ALCOA+ Attributable/Contemporaneous]

- **Hash at every pipeline stage boundary.** Compute SHA-256 of input and output
  at each stage transition. Store hashes alongside data. This is how an inspector
  verifies nothing was altered between stages. [ALCOA+ Original, 21 CFR 11.10(e)]

- **Preserve raw data.** Never overwrite source data in-place. Store originals
  immutably; create new versions for transformations. An inspector needs to see
  the original alongside every derivative. [ALCOA+ Original/Complete, E6(R3)]

- **RBAC on every data endpoint.** All API routes and data access functions MUST
  check permissions before returning data. Use unique user IDs — no shared
  accounts, no generic service logins for human-initiated actions.
  [21 CFR 11.10(d), E6(R3) Security, ALCOA+ Attributable]

- **Type hints + mypy.** All public functions MUST have complete type annotations.
  Run `mypy --strict` on modified files before committing.

- **Deterministic training.** Fix and record all random seeds. Pin environment
  via Docker or conda lockfile. Record GPU/CUDA versions. If a training run
  cannot be reproduced, it cannot be validated.
  [NIST AI RMF MEASURE, E6(R3) QbD, ALCOA+ Consistent]

- **Test docstrings with regulatory traceability.** Every test module MUST state:
  qualification phase (IQ/OQ/PQ), regulatory requirement validated, risk tier.

### Ask First (Confirmation required from appropriate stakeholder)

- **Model deployment.** All model deployments require multi-stakeholder approval:
  ML Lead (technical), QA (validation), Clinical (safety), Regulatory (clearance).
  Each approver provides an electronic signature (user ID + timestamp + meaning).
  [21 CFR 11.50–11.200, E6(R3) Sponsor Oversight]

- **Training data changes.** Adding, removing, or relabeling training data
  requires data steward review. Document: what changed, why, impact assessment,
  representativeness verification. [ALCOA+ Complete/Accurate, E6(R3) Data Governance]

- **Monitoring threshold changes.** Modifying drift detection parameters, alert
  thresholds, or performance floors requires quality review with documented
  justification. [E6(R3) RBQM, NIST AI RMF MANAGE]

- **Schema changes.** Database or API schema modifications require formal change
  control: before/after documentation, regression testing, reviewer sign-off.
  [21 CFR 11.10(e), ALCOA+ Consistent]

- **Risk tier reclassification.** Changing a module's risk classification
  (HIGH/MEDIUM/LOW) affects testing requirements. Requires regulatory review.
  [E6(R3) QbD/RBQM]

### Never Do (Prohibited — no exceptions)

- **Never delete data without an audit trail entry** that includes the records
  affected and justification. Silent data deletion is a regulatory finding.
  [21 CFR 11.10(e), ALCOA+ Complete]

- **Never provide a mechanism to disable audit trails.** No feature flag, admin
  toggle, environment variable, or code path should allow audit logging to be
  turned off in any environment. [21 CFR 11.10(e), E6(R3) Data Governance]

- **Never let AI make autonomous regulated decisions.** AI/ML outputs that affect
  patient safety, clinical decisions, or regulatory records are decision-support
  only. A qualified human MUST review and e-sign. [E6(R3) Human Oversight, NIST]

- **Never log PHI.** No patient names, medical record numbers, SSNs, dates of
  birth, or other HIPAA identifiers in application logs, error messages, debug
  output, test fixtures, or code comments. Use tokenized references only.
  [HIPAA, E6(R3) Security]

- **Never use shared credentials** for any system that touches regulated data.
  Every action must be attributable to an individual.
  [21 CFR 11.10(d), ALCOA+ Attributable]

- **Never commit binary model files without metadata.** Model artifacts MUST be
  accompanied by: training data hash, code commit, environment hash,
  hyperparameters, validation metrics. [ALCOA+ Legible, NIST AI RMF MEASURE]

---

## Contribution Workflow

### Step 1: Understand Before You Build

Before writing code, every contributor MUST:

1. **Read the Jira ticket** — parse GHERKIN acceptance criteria. If none exist,
   add them before writing any code.
2. **Classify the risk tier** — does this code touch patient data (HIGH), model
   predictions (HIGH), audit records (HIGH), data pipelines (MEDIUM), or
   administrative functions (LOW)?
3. **Identify applicable regulations** — use the table in the Risk Classification
   section to determine which compliance patterns are required.
4. **Check existing patterns** — search the codebase for how similar compliance
   requirements were implemented. Consistency matters to auditors.

### Step 2: Build with Compliance Embedded

While writing code, apply the compliance action required for what you're writing:

| What You're Writing | What You Must Do |
|---------------------|-----------------|
| Data mutation (CRUD) | Add audit trail entry (WHO/WHAT/WHEN/WHY) |
| Data transformation | Hash input + output; preserve raw; version transform |
| Data access | RBAC check before return; no PHI in logs |
| Model training | Record seed, env hash, data version, hyperparams, metrics |
| Model inference | Log model version, input hash, prediction, confidence |
| Clinical decision support | Require human review gate + e-signature |

### Step 3: Self-Verify Before Submission

Run this checklist — all items must pass before opening a PR:

- [ ] Audit trail coverage: every data mutation has an `AuditLogger` call
- [ ] Hash verification: SHA-256 at every pipeline stage boundary
- [ ] Raw data preserved: no in-place overwrites of source data
- [ ] RBAC enforced: all data endpoints check permissions before returning
- [ ] No PHI in logs, comments, test data, or error messages
- [ ] Deterministic config: seeds fixed, environment pinned (if training code)
- [ ] Tests match risk tier: HIGH = unit + integration + E2E; MEDIUM = unit + integration
- [ ] Test docstrings: IQ/OQ/PQ phase + regulatory requirement stated
- [ ] `mypy --strict` passes on all modified files
- [ ] Change control documented (if model/data/config changed)

### Step 4: Regulatory Code Review

Reviewers verify both functional and regulatory correctness. A PR cannot merge
if any regulatory review item fails — regardless of functional correctness.

Regulatory review checklist (reviewers verify):

| # | Check | Common Failures |
|---|-------|----------------|
| 1 | Audit trail: every CREATE/MODIFY/DELETE has WHO/WHAT/WHEN/WHY | Missing "why" field; no before/after values |
| 2 | Data integrity: hash at boundaries; raw data preserved | Overwriting source files; hashing after transformation |
| 3 | Access controls: RBAC check before data return | Endpoint returns data without permission check |
| 4 | PHI containment: no patient data in logs/errors/fixtures | Exception handler logging full request body |
| 5 | Reproducibility: seeds fixed; environment pinned | `random.random()` without seed; no lockfile |
| 6 | Test traceability: IQ/OQ/PQ phase in docstrings | Tests exist but no regulatory linkage |
| 7 | Human-in-the-loop: AI outputs for regulated decisions need sign-off | Model prediction used directly in clinical workflow |
| 8 | Change control: model/data/config changes documented | Hyperparameter change with no justification |

### Step 5: Post-Merge

- Update model registry if model artifacts changed
- Document change control if config/thresholds modified
- Update Jira ticket with compliance patterns applied

---

## Risk Classification

Every module has a risk tier that determines testing depth and required
compliance patterns. When uncertain, classify UP.

| Risk Tier | Criteria | Tests Required | Compliance Patterns |
|-----------|----------|----------------|-------------------|
| **HIGH** | Patient data, model predictions, clinical decisions, audit records | Unit (IQ) + Integration (OQ) + E2E (PQ) + Subgroup equity + Stress | All: audit trail, hash, RBAC, e-sig, model registry |
| **MEDIUM** | Data pipelines, feature engineering, monitoring, configuration | Unit (IQ) + Integration (OQ) | Audit trail, hash, RBAC |
| **LOW** | Administrative UI, reporting, non-clinical utilities | Unit (IQ) + Checklist | Basic code quality |

---

## Compliance Patterns

### Pattern 1: Audit Trail (21 CFR 11.10(e), ALCOA+)

```python
from ptcv.compliance.audit import AuditLogger, AuditAction

audit = AuditLogger(module="training_data")

audit.log(
    action=AuditAction.MODIFY,
    record_id=dataset.id,
    user_id=current_user.id,
    reason="Corrected mislabeled samples per QA review QA-2026-042",
    before={"label_count": 1200},
    after={"label_count": 1198},
)
```

Required fields: `who` (user ID — no shared accounts), `what` (action +
before/after), `when` (UTC ms timestamp — NTP-synchronized), `why`
(justification — mandatory, never optional).

Storage: append-only backend. No UPDATE or DELETE operations on audit records.
No mechanism to disable audit logging.

### Pattern 2: Data Integrity Guard (ALCOA+ all 9 principles)

```python
from ptcv.compliance.integrity import DataIntegrityGuard

guard = DataIntegrityGuard()

# Checkpoint at ingestion — captures hash of raw data
raw_hash = guard.checkpoint(raw_data, stage="ingestion")

# Transform — raw is preserved, hash chain maintained
processed = transform(raw_data)
guard.checkpoint(processed, stage="preprocessing", parent_hash=raw_hash)

# Verify at any time
assert guard.verify(raw_data, expected_hash=raw_hash)
```

### Pattern 3: Electronic Signature (21 CFR 11.50–11.200)

```python
from ptcv.compliance.signatures import ESignature, SignatureMeaning

# Model deployment requires multi-stakeholder approval
approval = ESignature.request(
    record_type="model_deployment",
    record_id=model.version_id,
    signers=[
        ("ml_lead@ptcv.org", SignatureMeaning.TECHNICAL_REVIEW),
        ("qa_lead@ptcv.org", SignatureMeaning.QUALITY_APPROVAL),
        ("clinical@ptcv.org", SignatureMeaning.CLINICAL_REVIEW),
        ("regulatory@ptcv.org", SignatureMeaning.REGULATORY_CLEARANCE),
    ],
)
# Each signature = user_id + timestamp + meaning, linked to the signed record
```

### Pattern 4: Model Registry (E6(R3) Change Control, NIST MANAGE)

```python
from ptcv.compliance.registry import ModelRegistry

registry = ModelRegistry()
registry.register(
    model_id="adverse_event_classifier_v3",
    training_data_hash="sha256:abc123...",
    code_commit="git:def456...",
    environment_hash="docker:ghi789...",
    validation_results={
        "iq": {"status": "PASSED", "report_id": "IQ-2026-003"},
        "oq": {"status": "PASSED", "report_id": "OQ-2026-003"},
        "pq": {"status": "PASSED", "report_id": "PQ-2026-003"},
    },
    risk_classification="HIGH",
    intended_use="Adverse event detection in clinical trial data",
)
```

Every model retrain = new registry entry + new multi-stakeholder approval.
Model versions are immutable once deployed.

---

## Testing Standards

### IQ/OQ/PQ Mapping

| Test Type | Qualification Phase | Purpose | When Required |
|-----------|-------------------|---------|---------------|
| Unit tests | IQ — Installation | Components configured and installed correctly | All changes |
| Integration tests | OQ — Operational | System operates under defined conditions | All integration points |
| E2E tests | PQ — Performance | Clinical performance thresholds met in realistic conditions | HIGH risk modules |
| Subgroup equity tests | PQ extension | No demographic performance degradation | All clinical ML models |
| Stress / robustness tests | PQ extension | Edge cases and adversarial inputs handled | All models |

### Required Test Docstring Format

Every test module MUST include a docstring with qualification phase, regulatory
requirement, and risk tier:

```python
"""IQ Test Suite: Training Data Ingestion Module

Qualification: Installation Qualification (IQ)
Regulatory: 21 CFR 11.10(a), ALCOA+ (Original, Complete)
Risk Tier: HIGH — training data directly affects model performance
Module: ptcv.data.ingestion
"""
```

### Minimum Test Requirements by Risk Tier

| Risk Tier | Unit (IQ) | Integration (OQ) | E2E (PQ) | Subgroup | Stress |
|-----------|-----------|------------------|----------|----------|--------|
| HIGH | Required | Required | Required | Required | Required |
| MEDIUM | Required | Required | Recommended | Recommended | Recommended |
| LOW | Required | Checklist | Optional | Optional | Optional |

---

## ML-Specific Standards

### Training Data Governance

Before any model training begins:

1. Document provenance for all data sources (institution, method, date, consent status)
2. Version with DVC or equivalent — hash + schema + row count per version
3. Run bias detection — demographic distribution analysis before training
4. Verify representativeness against intended patient population (FDA GMLP Principle 3)
5. Document train/validation/test split method and verify independence
6. Run data quality assessment: missing values, outliers, inter-rater reliability

### Model Development

1. Fix all random seeds; record in experiment metadata
2. Pin environment — Docker image or conda lockfile; GPU/CUDA versions recorded
3. Log every experiment: hyperparameters, metrics, data version, code commit
4. Create model traceability matrix: clinical risks → inputs → algorithms → outputs

### Validation Protocols

All clinical models require:

1. K-fold cross-validation (reduces variance in performance estimates)
2. Stratified holdout with demographic subgroup representation
3. Subgroup equity testing — performance parity across age, sex, race/ethnicity
4. Robustness testing — edge cases, adversarial inputs, noisy data
5. External validation (for HIGH risk / multi-site deployment: data from different institution)

### Production Monitoring

Configure before any model goes live:

1. Drift detection: PSI (>0.1 warning, >0.2 critical) and KS test
2. Rolling performance metrics: AUC-ROC, F1, sensitivity, specificity
3. Automated alerting with clinical review triggers
4. Per-subgroup error rate tracking (false positive/negative by demographic)

### Retraining Governance

Every model retrain is a new software release. Required steps:

1. Document the trigger (drift alert, performance degradation, incident, scheduled)
2. Validate new training data meets ALCOA+ requirements
3. Compare retrained model vs. original on held-out set and original test set
4. Verify no subgroup performance degradation
5. IQ/OQ/PQ re-qualification (full or partial depending on scope of change)
6. Multi-stakeholder e-signature approval
7. Canary deployment with monitoring before full rollout

---

## Git Workflow

### Branch Naming

| Type | Pattern | Example |
|------|---------|---------|
| Feature | `feature/PTCV-XXX-description` | `feature/PTCV-42-add-audit-logger` |
| Bugfix | `bugfix/PTCV-XXX-description` | `bugfix/PTCV-43-fix-drift-alert` |
| Hotfix | `hotfix/description` | `hotfix/critical-phi-exposure` |

### Commit Format

```
<type>(scope): <description>

[Body — include regulatory context if compliance-relevant]

Regulatory: <framework reference>   ← include when change is compliance-driven
PTCV-XXX                            ← Jira issue key
Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
```

---

## Regulatory Quick Reference

**NIST AI RMF 1.0** — Voluntary framework, adopted as PTCV standard. Four
functions: GOVERN (policies, accountability), MAP (context, risk identification),
MEASURE (testing, metrics, bias), MANAGE (deployment, monitoring, incidents).
Covers 7 trustworthy AI characteristics — the only framework here that explicitly
addresses fairness and transparency. [nist.gov/artificial-intelligence/ai-risk-management-framework]

**21 CFR Part 11** — FDA regulation, legally binding. The non-negotiable baseline
for electronic records and signatures. Covers audit trails (11.10(e)), access
controls (11.10(d)), system validation (11.10(a)), and electronic signatures
(Subpart C). Does not address AI/ML specifically — all ML artifacts are electronic
records subject to Part 11. [ecfr.gov/current/title-21/chapter-I/subchapter-A/part-11]

**ICH E6(R3)** — Good Clinical Practice guideline, adopted January 2025, published
by FDA September 2025. Most significant GCP update in 30 years. Introduces Quality
by Design (embed quality from requirements, not bolted on post-development),
Risk-Based Quality Management (validate proportionally to clinical risk), and an
explicit Data Governance section. AI/ML-specific guidance deferred to ICH E21
(under development). [fda.gov/regulatory-information/search-fda-guidance-documents/e6r3-good-clinical-practice-gcp]

**ALCOA+** — Nine data integrity principles: Attributable, Legible,
Contemporaneous, Original, Accurate, Complete, Consistent, Enduring, Available.
Codified in PIC/S PI 041-1. The standard by which an inspector evaluates whether
your data can be trusted. If your data fails ALCOA+, your model's clinical
validity is irrelevant regardless of its performance metrics.

---

## Maintaining This Document

This CONTRIBUTING.md is a controlled document. Changes require:

1. PR with proposed changes targeting `main`
2. Regulatory reviewer approval for any compliance section changes
3. Changelog entry (date, change description, regulatory basis)
4. Retroactive assessment — when a new rule is added, evaluate whether existing
   code complies and create remediation Jira tickets if not

For AI agents: changes to this file take effect on the next session. Write rules
as unambiguous directives (MUST/MUST NOT). Include code examples for every
compliance pattern — agents follow patterns more reliably than prose descriptions.

### Changelog

| Date | Change | Regulatory Basis |
|------|--------|-----------------|
| 2026-03-02 | Initial version — synthesized from PRD-1 research (PTCV-3 through PTCV-8) | NIST AI RMF 1.0, 21 CFR Part 11, ICH E6(R3), ALCOA+ |
