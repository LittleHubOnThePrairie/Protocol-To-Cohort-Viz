# Contributing to Protocol-to-Cohort-Viz (PTCV)

## Code Style

### Python

- **Style guide:** Google Python Style Guide
- **Indentation:** 4 spaces (no tabs)
- **Line length:** 80 characters max
- **Naming:** `snake_case` for functions/variables, `PascalCase` for classes, `UPPER_SNAKE_CASE` for constants
- **Type hints:** Required on all public function signatures
- **Docstrings:** Google-style docstrings on all public functions and classes
- **Linting:** Run `mypy` on modified files before committing
- **Imports:** stdlib, third-party, local — separated by blank lines, alphabetized within groups

### TypeScript

- **Indentation:** 2 spaces
- **Naming:** `camelCase` for functions/variables, `PascalCase` for types/interfaces/classes
- **Types:** Explicit return types on exported functions; avoid `any`
- **Async:** Use `async`/`await` over raw Promises

### General

- Prefer editing existing files over creating new ones
- Keep solutions simple — avoid over-engineering
- No commented-out code in commits
- Tests required for new functionality

## Git Workflow

### Branch Naming

```
feature/PTCV-<id>-<short-description>
bugfix/PTCV-<id>-<short-description>
hotfix/<description>
```

### Commit Messages

Use [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <description>

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
```

**Types:** `feat`, `fix`, `docs`, `refactor`, `test`, `chore`, `ci`

### Pre-Commit Checklist

- [ ] Code passes `mypy` type checking
- [ ] No secrets or credentials in committed files
- [ ] Tests pass for modified code
- [ ] Commit message follows conventional format

## Regulatory Considerations

This project is subject to regulatory compliance (NIST AI RMF, 21 CFR Part 11, ALCOA+). When contributing:

- **Audit trails:** All data transformations must be traceable
- **Data integrity:** Follow ALCOA+ principles for ML artifacts
- **Validation:** Document IQ/OQ/PQ qualification for ML model changes
- **Electronic records:** Ensure compliance with 21 CFR Part 11 requirements for electronic signatures and access controls

## Issue Workflow

1. Pick up a Jira issue from the PTCV board
2. Verify GHERKIN acceptance criteria exist
3. Create a feature branch: `feature/PTCV-<id>-<desc>`
4. Implement and test
5. Create PR referencing the Jira issue key
6. Request review
