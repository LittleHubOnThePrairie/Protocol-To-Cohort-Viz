# Claude Code Configuration — PTCV

Project-specific instructions for Protocol-to-Cohort-Viz. Inherits global conventions from the [LittleHub CLAUDE.md](https://github.com/LittleHubOnThePrairie/ClaudeCodeAssets/blob/master/CLAUDE.md).

---

## Project Context

| Attribute | Value |
|-----------|-------|
| **Jira Project** | PTCV |
| **Confluence Space** | PTCV |
| **GitHub** | LittleHubOnThePrairie/Protocol-To-Cohort-Viz |
| **Domain** | ML-enabled clinical trial visualization |

---

## Regulatory Context

This project operates under regulatory compliance requirements:

- **NIST AI RMF 1.0** — AI Risk Management Framework (GOVERN, MAP, MEASURE, MANAGE)
- **21 CFR Part 11** — Electronic Records and Signatures
- **FDA GMLP** — Good Machine Learning Practice (10 guiding principles)
- **ICH E6(R3)** — Good Clinical Practice guidelines
- **ALCOA+** — Data integrity framework (Attributable, Legible, Contemporaneous, Original, Accurate + Complete, Consistent, Enduring, Available)

All code contributions must consider audit trail requirements and data integrity obligations.

---

## Code Style

Follow [CONTRIBUTING.md](CONTRIBUTING.md) for detailed conventions.

**Python:** Google Style — 4 spaces, 80 chars, `snake_case`, `PascalCase` classes, type hints required.
**TypeScript:** 2 spaces, `camelCase`, `PascalCase` types, explicit types, async/await.

---

## Git Workflow

**Branches:** `feature/PTCV-<id>-<desc>`, `bugfix/PTCV-<id>-<desc>`
**Commits:** `<type>(scope): <desc>` + `Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>`

---

## Secrets

Files: `.secrets` (never commit), `.secrets.example` (template)
Load: `source ./load-secrets.sh`
