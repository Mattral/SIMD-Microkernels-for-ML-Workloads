# Contributing to GuardRail Studio

> *"Excellence is not negotiable. Latency is not negotiable. Tests are not negotiable."*

Thank you for considering a contribution to GuardRail Studio. This document codifies
the **non-negotiable** engineering practices that keep the platform shippable to
billions of requests per day.

**Companion Docs:**
- [Engineering Philosophy →](./PHILOSOPHY.md)
- [System Design →](./SYSTEM_DESIGN.md)
- [Security Posture →](./SECURITY.md)

---

## Table of Contents

1. [Code of Conduct](#1-code-of-conduct)
2. [Branching Strategy](#2-branching-strategy)
3. [Commit Conventions](#3-commit-conventions)
4. [Quality Gates](#4-quality-gates)
5. [Local Setup for Contributors](#5-local-setup-for-contributors)
6. [Testing Requirements](#6-testing-requirements)
7. [Pull-Request Workflow & Review Rubric](#7-pull-request-workflow--review-rubric)
8. [Documentation Requirements](#8-documentation-requirements)
9. [Releases & Versioning](#9-releases--versioning)
10. [Postmortems](#10-postmortems)

---

## 1. Code of Conduct

We follow the [Contributor Covenant v2.1](https://www.contributor-covenant.org/version/2/1/code_of_conduct/).
Bug reports, ideas, and disagreements are welcome — personal attacks are not.

---

## 2. Branching Strategy

We practise **trunk-based development with short-lived feature branches**. There is
no `develop` branch. `main` is always releasable.

```
                       merge after CI green
                       and 2 approving reviews
                            │
                            ▼
main  ───●──────●──────●─────●──────●──────●──────▶
          \      \      \              \    \
           \      \      \              \    \
    feat/  ●──────●       \              \    \
    fix/                   ●─────●         \    \
    chore/                        \         \    \
                                   ●─────●   \
    release/v1.2.0                          ●─────●  (cherry-picked hotfix only)
```

Rules:

- Branches **MUST** be ≤ 72 hours old when merged.
- Branch names: `feat/<slug>`, `fix/<slug>`, `chore/<slug>`, `docs/<slug>`, `perf/<slug>`.
- Hotfix branches: `hotfix/<release>-<slug>` and cherry-pick to the release branch
  *and* `main`.

---

## 3. Commit Conventions

We use **Conventional Commits** (https://www.conventionalcommits.org/). This is
enforced by the CI lane in
[`.github/workflows/ci_cd.yaml`](../.github/workflows/ci_cd.yaml).

```
<type>(<scope>): <imperative summary>            (≤ 72 chars)
<blank>
<body — what + why, not how>
<blank>
<footer — BREAKING CHANGE: ..., Refs: #1234>
```

| Type | Use |
| --- | --- |
| `feat` | new user-visible capability |
| `fix` | bug fix |
| `perf` | latency / memory improvement |
| `refactor` | no behavioural change |
| `test` | tests only |
| `docs` | docs only |
| `chore` | tooling / CI / deps |
| `build` | build system / Dockerfile |

Example:

```
perf(inference): batch tokenisation by 4 reducing p99 by 1.2ms

Previously each request tokenised separately, paying the Rust FFI overhead
per call. Aggregating into batches of 4 reduces FFI cost amortised over
the batch, observed on k6_chaos at 25k RPS.

Refs: #482
```

---

## 4. Quality Gates

Every PR runs the gates below. **Any red gate blocks merge.** See
[`.github/workflows/ci_cd.yaml`](../.github/workflows/ci_cd.yaml) for the
exact workflow definition.

| Gate | Tool | Threshold |
| --- | --- | --- |
| Lint | `ruff check` | zero violations |
| Format | `black --check` | zero diffs |
| Static types | `mypy --strict` | zero errors |
| Unit tests | `pytest` | 100% pass |
| Coverage | `pytest --cov` | **≥ 90%** lines |
| Model parity | `pytest tests/ml/test_model_parity.py` | max abs diff < 1e-5 |
| Container scan | Trivy | zero CRITICAL CVEs |
| Frontend lint | ESLint | zero warnings (`--max-warnings 0`) |
| Frontend build | `yarn build` | success |
| k6 smoke | `k6 run --vus 50 --duration 30s` | p99 < 15 ms |

---

## 5. Local Setup for Contributors

```bash
# 1. Clone & enter
$ git clone git@github.com:emergent-labs/guardrail-studio.git
$ cd guardrail-studio

# 2. Backend
$ python3.11 -m venv .venv && source .venv/bin/activate
$ pip install -r backend/requirements.txt
$ pip install -r backend/requirements-dev.txt   # ruff, black, mypy, pytest-cov

# 3. Frontend
$ cd frontend && yarn install && cd ..

# 4. Git hooks (pre-commit lint + format)
$ pre-commit install

# 5. Smoke test
$ pytest backend/tests -q
$ ruff check backend/src
$ black --check backend/src
$ mypy --strict backend/src
```

---

## 6. Testing Requirements

### 6.1 Layer Discipline

| Layer | Test Type | Tooling | Coverage Target |
| --- | --- | --- | --- |
| `src/services/*` | Unit + Integration | `pytest-asyncio` | 95% |
| `src/api/routes/*` | API contract | `httpx.AsyncClient` | 95% |
| `src/db/*` | Integration | testcontainers (Postgres) | 90% |
| `ml_pipelines/*` | Property + Parity | `pytest` + `numpy.testing` | 90% |
| `frontend/*` | Component + e2e | Vitest + Playwright | 80% |

### 6.2 Required tests for every PR that touches…

| Area | Mandatory test |
| --- | --- |
| Inference client | end-to-end mock Triton round-trip |
| Circuit breaker | state-machine transitions (closed → open → half_open → closed) |
| Threshold logic | property-based test with `hypothesis` |
| Drift detector | golden PSI fixtures |
| Model export | bit-parity ≤ 1e-5 against PyTorch |
| Postgres schema | partition pruning verified with `EXPLAIN` |
| Public API | OpenAPI snapshot diff |

### 6.3 Performance regression test

Every PR with the `perf` label must include a k6 run output in the PR description
comparing baseline vs candidate p50/p95/p99. CI auto-fails the PR if regression > 5%.

---

## 7. Pull-Request Workflow & Review Rubric

### 7.1 Workflow

```
   ┌──────────────────────┐
   │   open draft PR      │
   └──────────┬───────────┘
              ▼
   ┌──────────────────────┐
   │  CI lanes run        │
   │  • lint              │
   │  • types             │
   │  • tests + coverage  │
   │  • trivy             │
   │  • model parity      │
   │  • frontend          │
   └──────────┬───────────┘
              ▼
   ┌──────────────────────┐
   │  mark "ready for     │
   │  review" only when   │
   │  ALL gates are green │
   └──────────┬───────────┘
              ▼
   ┌──────────────────────┐
   │ 2 approving reviews  │
   │ (1 must be a CODEOWNER)│
   └──────────┬───────────┘
              ▼
   ┌──────────────────────┐
   │  squash-and-merge    │
   │  conventional title  │
   └──────────────────────┘
```

### 7.2 Reviewer Rubric

Each reviewer should explicitly answer these in the PR thread:

1. **Latency**: did this PR add ANY synchronous I/O, allocation, or lock to the hot path?
2. **Safety**: does this PR open a new threat surface? If yes, where is the mitigation?
3. **Tests**: are the new branches exercised? `coverage report` to prove it.
4. **Observability**: are new failure modes wired into a metric, trace, or log?
5. **Docs**: are the `docs/` files updated to reflect new behaviour?

---

## 8. Documentation Requirements

A change is incomplete until the documentation is updated.

| Change | Docs that MUST update |
| --- | --- |
| New endpoint | `docs/SYSTEM_DESIGN.md §10` + OpenAPI |
| New metric | `docs/USER_GUIDE_AND_UI.md §3` |
| New runbook event | `docs/SETUP_AND_OPERATIONS.md §11` |
| New threat surface | `docs/SECURITY.md` |
| Build/CI change | `docs/CONTRIBUTING.md` + this section |

Cross-link everything with **relative markdown** (`./X.md`, `../foo/bar.py`). Broken
links are caught by the `docs-lint` CI lane.

---

## 9. Releases & Versioning

We follow **SemVer 2.0.0** (`MAJOR.MINOR.PATCH`).

- **MAJOR**: breaking wire-format change, breaking model output shape.
- **MINOR**: new endpoints, new model versions, backward-compatible config.
- **PATCH**: bug fix, perf, doc only.

Release is automated:

```
$ git tag -a v1.2.0 -m "v1.2.0"
$ git push origin v1.2.0
```

→ CI builds release container images, signs them with Cosign, publishes to GHCR,
and opens a PR against `deploy/k8s/production-stack.yaml` updating the image tags
(GitOps).

---

## 10. Postmortems

Every Sev-1 / Sev-2 incident requires a **blameless postmortem** within 5 business days.

### 10.1 Template

```markdown
# Postmortem — <Incident Title> — <Date>

## Summary (≤ 200 words)
…

## Impact
- Users affected:
- Duration:
- Revenue/SLO impact:

## Timeline (UTC)
- HH:MM  Alert fired
- HH:MM  On-call paged
- HH:MM  Mitigation applied
- HH:MM  Resolved

## Root Cause
(use the "5 whys" technique)

## What went well
…

## What didn't go well
…

## Action items (each with an OWNER and a DUE date)
- [ ] OWNER  DUE  Description
- [ ] OWNER  DUE  Description
```

All postmortems live in `docs/postmortems/` and are linked from this file as they
are written.

---

> Continue to [`docs/PHILOSOPHY.md`](./PHILOSOPHY.md) for the *spirit* behind these
> rules.

