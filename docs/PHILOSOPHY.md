# GuardRail Studio — Engineering Philosophy

> *"What you accept, you teach. What you tolerate, you become."*

This document is the **constitution** of GuardRail Studio. Code may be rewritten, but
these principles are immutable. They were chosen because every shortcut around them
has cost a real production outage at companies that came before us.

**Companion Docs:**
- [Contributing →](./CONTRIBUTING.md)
- [System Design →](./SYSTEM_DESIGN.md)

---

## Table of Contents

1. [The Four Pillars](#1-the-four-pillars)
2. [Latency Is Not Negotiable](#2-latency-is-not-negotiable)
3. [Type Safety Is Self-Documentation](#3-type-safety-is-self-documentation)
4. [Asynchronous by Default](#4-asynchronous-by-default)
5. [Tests Are a First-Class Deliverable](#5-tests-are-a-first-class-deliverable)
6. [Observability Is a Feature](#6-observability-is-a-feature)
7. [Security Is a Posture, Not a Check](#7-security-is-a-posture-not-a-check)
8. [Operability over Cleverness](#8-operability-over-cleverness)
9. [Boring Technology, Sharp Edges](#9-boring-technology-sharp-edges)
10. [Decision Records](#10-decision-records)

---

## 1. The Four Pillars

Every design decision in this repo can be justified against one of four pillars:

```
┌───────────────────────────────────────────────────────────────────┐
│                                                                   │
│      LATENCY        TRUST         OBSERVABILITY     RECOVERY      │
│      ───────       ──────         ─────────────     ────────      │ 
│   sub-10ms p99    zero-trust      every span        auto rollback │
│   gRPC binary     mTLS + WAF      every metric      Flagger gate  │
│   ONNX/TensorRT   KMS + SM        every log         circuit brk   │
│                                                                   │
└───────────────────────────────────────────────────────────────────┘
```

If a PR cannot point to a pillar it is improving, the PR is closed.

---

## 2. Latency Is Not Negotiable

The hot path budget is **10 milliseconds at p99**. Every microsecond is counted.

- **No synchronous I/O on the hot path.** All DB writes are fire-and-forget background tasks.
- **No JSON on the wire to inference servers.** gRPC + protobuf only.
- **No Python loops inside Triton callbacks.** NumPy / Rust tokenisers only.
- **No allocation in the request critical section** that can be pre-allocated.
- **Profile before optimising, profile after optimising.** No "feel good" perf PRs.

If a feature cannot meet the budget, it lives off the hot path.

---

## 3. Type Safety Is Self-Documentation

We run `mypy --strict` in CI with the following knobs:

```
--disallow-untyped-defs
--disallow-any-generics
--no-implicit-reexport
--warn-unused-ignores
--warn-return-any
--warn-redundant-casts
```

This is intentional. Type annotations are **the cheapest form of documentation** and
**the most reliable form of contract**. A bug found by mypy in 0.4 seconds is one
that is not found by a customer at 4 AM.

Pydantic models are the only acceptable form of dict-shaped data at API boundaries.
See [`backend/src/schemas/firewall.py`](../backend/src/schemas/firewall.py).

---

## 4. Asynchronous by Default

```python
# Wrong — blocks the event loop
def write_log(row): db.insert(row)

# Right — non-blocking, fire-and-forget
async def write_log(row): await db.insert(row)

# Even better — does not delay the response
asyncio.create_task(write_log(row))
```

Rules of thumb:

- Every I/O call is `async`.
- Every blocking CPU operation lives in a `ThreadPoolExecutor` or a Dask worker.
- Background bookkeeping uses `asyncio.create_task(...)` with a structured logger
  in the exception handler — **never silently swallow a task exception**.

---

## 5. Tests Are a First-Class Deliverable

A feature is not "done" when it works. It is done when:

1. It is covered by tests at the right layer (see
   [Contributing §6](./CONTRIBUTING.md#6-testing-requirements)).
2. The tests run in CI under 5 minutes.
3. A new failure mode is exercised by at least one negative test.
4. A regression test is added the moment a bug is discovered, **before** the fix.

The bit-parity gate at [`tests/ml/test_model_parity.py`](../tests/ml/test_model_parity.py)
is the canonical example: a single line difference between PyTorch and ONNX of
**1e-5** is enough to block a release. This is intentional — at scale, every micro-
divergence becomes a real customer regression.

---

## 6. Observability Is a Feature

If a metric is not emitted, the feature does not exist. If a span is not added, the
feature is invisible at 4 AM. Hence:

- **Every** public endpoint emits a histogram and a counter.
- **Every** outbound call (gRPC, SQL, HTTP) is traced via OpenTelemetry.
- **Every** business decision (block/allow, breaker open/close) is logged with the
  request ID, and **the request ID is propagated end-to-end**.

The wiring lives in [`backend/src/core/observability.py`](../backend/src/core/observability.py).

---

## 7. Security Is a Posture, Not a Check

Security is not a CI step you add and forget. It is a posture:

- **Least privilege.** Every IAM role assumes only the resources it needs.
- **Encrypt everywhere.** TLS in transit (mTLS inside the mesh), KMS at rest.
- **Never log secrets, payloads, or tokens.** Use the
  [structured logger](../backend/src/core/logging.py) which scrubs sensitive keys.
- **Threat-model every new endpoint.** See [`docs/SECURITY.md`](./SECURITY.md).
- **Assume breach.** If an attacker has shell on a pod, what is the blast radius?

---

## 8. Operability over Cleverness

```
A clever solution that pages the on-call at 3 AM is
a worse solution than a boring one that does not.
```

Concretely:

- Prefer **explicit** state machines (`CircuitState.CLOSED` etc.) over implicit flags.
- Prefer **named** thresholds in config over magic numbers in code.
- Prefer **declarative** Kubernetes manifests over imperative `kubectl` runbooks.
- Prefer **idempotent** deploys (Helm + Flux/Argo) over imperative scripts.

Anyone on the team must be able to read any module in <15 minutes and explain its
contract. If that is not possible, the module is over-engineered.

---

## 9. Boring Technology, Sharp Edges

We deliberately choose **boring technology** for the platform spine — Postgres,
Kubernetes, gRPC, OpenTelemetry, Helm, Terraform. They are *boring* because they
have been running planet-scale workloads for a decade. We reserve **sharp edges**
(ONNX, TensorRT, LoRA, Dask) only where they earn their keep with measurable
latency or accuracy wins.

The chart looks like this:

```
                Latency Sensitivity
                       ▲
                       │     ┌─────────────────────────────┐
                       │     │  Triton + TensorRT + ONNX   │ ← sharp
                       │     │  HNSW Qdrant                │
                       │     │  gRPC binary protocol       │
                       │     └─────────────────────────────┘
                       │
                       │ ┌─────────────────────────────────┐
                       │ │  Postgres + Aurora              │ ← boring
                       │ │  Helm + Flagger                 │
                       │ │  OpenTelemetry + Grafana        │
                       │ └─────────────────────────────────┘
                       └─────────────────────────────────────▶
                                  Operational Risk
```

---

## 10. Decision Records

Every meaningful architectural decision should leave an **ADR** (Architecture
Decision Record) in `docs/adr/NNN-<slug>.md`. The template is intentionally
short — long ADRs do not get written.

```markdown
# ADR-NNN: <Title>

- Status: proposed | accepted | superseded by ADR-XXX
- Date: YYYY-MM-DD
- Authors: @handle1 @handle2

## Context
What problem are we solving?

## Decision
What did we choose?

## Alternatives Considered
Bullet list with one-sentence rejection rationale.

## Consequences
Positive / negative / neutral.
```

The first three ADRs in this repo are:

- ADR-001: Choose ONNX + Triton over native PyTorch serving.
- ADR-002: Range-partition `firewall_logs` by day.
- ADR-003: Adopt LoRA over full fine-tuning for continuous adaptation.

---

> *"This is the way."*  — see you in the PR queue.

