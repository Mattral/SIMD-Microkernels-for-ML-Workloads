# GuardRail Studio — System Design

> *"In a world racing toward AGI, latency is the new currency and trust is the new firewall."*

This document is the canonical architectural reference for **GuardRail Studio**, an
ultra-low-latency, high-throughput LLM Firewall & Observability Platform. It is intended
for principal engineers, security architects, and reliability engineers responsible for
operating the platform at planetary scale.

**Audience:** Staff/Principal SWE • SRE • Security Architecture • MLOps

**Companion Docs:**
- [Setup & Operations →](./SETUP_AND_OPERATIONS.md)
- [User Guide & UI →](./USER_GUIDE_AND_UI.md)
- [Security Threat Model →](./SECURITY.md)
- [Engineering Philosophy →](./PHILOSOPHY.md)

---

## 1. Executive Summary

GuardRail Studio sits in the synchronous request path between a calling application
(typically an LLM-powered SaaS) and an upstream foundation model. Every prompt is
classified, scored, and either allowed, redacted, or rejected within a **p99 budget of
10 ms**. The platform additionally streams telemetry to a distributed analytics plane
that performs continuous drift detection and triggers LoRA-based model
re-fine-tuning when distributional shift is observed.

### 1.1 Design Goals (Hard Constraints)

| Goal | Target | Mechanism |
| --- | --- | --- |
| Inference Latency | p99 ≤ 10 ms | ONNX + Triton + gRPC + binary protobuf |
| Throughput | ≥ 25,000 RPS / pod | Async FastAPI + Triton dynamic batching |
| Availability | 99.99% | Dual-layer circuit breakers + Istio outlier ejection |
| Drift Detection | < 5 min lag | Dask streaming PSI/KL divergence over Postgres partitions |
| Model Recovery | < 30 min | Airflow DAG → LoRA fine-tune → Flagger canary |
| Security | Zero plaintext PII at rest | Tokenisation + AWS KMS + WAF + IRSA |

### 1.2 Anti-Goals

- **Not** a chat orchestrator. We do not own the LLM call itself.
- **Not** a logging sink. We are an inline classifier with telemetry side-effects.
- **Not** a rules engine. Heuristics exist only as a circuit breaker fallback.

---

## 2. Macro Topology — The 30,000 ft View

```
                                         ┌─────────────────────────┐
                                         │   Calling Application   │
                                         │  (Customer LLM SaaS)    │
                                         └────────────┬────────────┘
                                                      │ HTTPS / mTLS
                                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          AWS Edge & Ingress Plane                            │
│  ┌────────────┐   ┌────────────┐   ┌────────────────────────────────────┐  │
│  │  Route53   │──▶│   AWS WAF  │──▶│  NLB ──▶ Istio IngressGateway      │  │
│  │  GeoDNS    │   │  Rate/Bot  │   │   (mTLS, JWT, header propagation)  │  │
│  └────────────┘   └────────────┘   └─────────────────┬──────────────────┘  │
└──────────────────────────────────────────────────────┼──────────────────────┘
                                                       │
                                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                       EKS Data Plane (multi-AZ)                              │
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────┐    │
│  │   FastAPI Backend (Deployment, HPA, PDB)                           │    │
│  │   ┌──────────────────────────────────────────────────────────┐    │    │
│  │   │  ASGI workers (uvloop) ── async/await pipeline           │    │    │
│  │   │   ├─ Strategy: GuardrailService                          │    │    │
│  │   │   ├─ Singleton: ProductionInferenceClient (gRPC pool)    │    │    │
│  │   │   ├─ Circuit Breaker (CLOSED → HALF_OPEN → OPEN)         │    │    │
│  │   │   ├─ Fallback Strategy: Regex Heuristic Classifier       │    │    │
│  │   │   └─ OpenTelemetry → OTLP/gRPC                           │    │    │
│  │   └──────────────────────────────────────────────────────────┘    │    │
│  └─────────────┬──────────────────┬─────────────────┬────────────────┘    │
│                │ gRPC (8001)      │ TCP (5432)      │ HTTP (6333)         │
│                ▼                  ▼                  ▼                     │
│  ┌──────────────────────┐ ┌────────────────┐ ┌──────────────────────┐    │
│  │ Triton Inference     │ │  PostgreSQL    │ │  Qdrant Vector DB    │    │
│  │  Server (StatefulSet)│ │  (RDS Aurora,  │ │  (StatefulSet, NVMe) │    │
│  │  ├─ ONNX Runtime     │ │   time-series  │ │   Adversarial        │    │
│  │  ├─ TensorRT engine  │ │   partitions)  │ │   pattern recall     │    │
│  │  ├─ Dyn. batching    │ │                │ │                      │    │
│  │  └─ Model repo (S3)  │ │                │ │                      │    │
│  └──────────────────────┘ └────────┬───────┘ └──────────────────────┘    │
└─────────────────────────────────────┼──────────────────────────────────────┘
                                      │ logical replication / WAL slot
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                       Analytics & Re-training Plane                          │
│                                                                              │
│  ┌─────────────┐   ┌──────────────────┐   ┌──────────────────────────────┐ │
│  │ Dask        │──▶│ Drift Detector   │──▶│  Apache Airflow              │ │
│  │ Distributed │   │ (PSI/KL/χ²)      │   │   drift_retrain_dag          │ │
│  │ Out-of-core │   │ per-day partition│   │   ├─ pull labels (S3)        │ │
│  └─────────────┘   └──────────────────┘   │   ├─ LoRA/PEFT fine-tune     │ │
│                                            │   ├─ ONNX export + parity   │ │
│                                            │   └─ Triton repo push (S3)  │ │
│                                            └──────────────┬──────────────┘ │
│                                                            │                │
│                                                            ▼                │
│                                            ┌──────────────────────────────┐ │
│                                            │  Flagger Canary Controller   │ │
│                                            │   Istio VirtualService split │ │
│                                            │   1% → 10% → 50% → 100%      │ │
│                                            │   p99/error-rate gating      │ │
│                                            └──────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
                  ┌───────────────────────────────────────┐
                  │  Observability Plane                   │
                  │  ├─ Prometheus (metrics, recording)    │
                  │  ├─ Grafana Tempo (OTLP traces)        │
                  │  ├─ Loki (structured logs)             │
                  │  ├─ Grafana (dashboards, alerts)       │
                  │  └─ Weights & Biases (runs, artifacts) │
                  └───────────────────────────────────────┘
```

---

## 3. Synchronous Inference Path (Hot Path)

Every authenticated request that arrives at `POST /api/firewall/check` traverses
**exactly one round-trip** through the hot path. This is the budget against which we
optimise everything.

### 3.1 Request Lifecycle — Sequence Diagram

```
Client          Istio Gateway     FastAPI Pod       Triton Pod      Qdrant
  │                  │                 │                │              │
  │── POST /check ─▶ │                 │                │              │
  │                  │── mTLS + JWT ──▶│                │              │
  │                  │                 │── tokenize ──▶ (HF Fast / Rust)
  │                  │                 │── gRPC infer ─▶│              │
  │                  │                 │                │── ONNX run ─▶│ (intra-pod)
  │                  │                 │                │◀── logits ───│
  │                  │                 │◀── logits ─────│              │
  │                  │                 │── KNN search ──┼────────────▶ │
  │                  │                 │◀── neighbours ─┼──────────────│
  │                  │                 │                │              │
  │                  │                 │ async log ──▶ Postgres (fire-and-forget)
  │                  │                 │ async span ─▶ OTLP collector
  │                  │◀── 200 + JSON ──│                │              │
  │◀── 200 + JSON ───│                 │                │              │
  │                                                                    │
  │  TOTAL BUDGET: 10 ms p99 (8 ms inference, 2 ms transport+aux)      │
```

### 3.2 Latency Budget Allocation

| Stage | Budget | Notes |
| --- | ---:| --- |
| Istio mTLS + JWT validate | 0.5 ms | LDS/RDS cached, JWKS pre-warmed |
| FastAPI request parse | 0.2 ms | `orjson`, no Pydantic v1 nesting |
| HF fast tokenisation | 0.8 ms | Rust-backed, batch-friendly |
| gRPC marshal (protobuf) | 0.3 ms | binary, no JSON |
| Triton dynamic batching wait | ≤ 2.0 ms | `max_queue_delay_microseconds = 2000` |
| ONNX Runtime exec | 3.5 ms | INT8 quantised, OpenVINO EP |
| gRPC unmarshal + softmax | 0.4 ms | NumPy, vectorised |
| Qdrant ANN (top-3) | 1.0 ms | HNSW M=16, ef=64 |
| Response serialise | 0.3 ms | `orjson` |
| **Total p99 target** | **≤ 10 ms** | hard SLO |

---

## 4. Dual-Layer Circuit Breaker

A single circuit breaker is insufficient. Triton can be healthy from Istio's perspective
(TCP up, gRPC handshake fine) yet returning degraded models. We therefore implement
**two independent breakers** at different layers of the stack.

### 4.1 Layer 1 — Application Breaker (FastAPI)

Lives inside [`backend/src/services/inference_client_triton.py`](../backend/src/services/inference_client_triton.py).
Implements the canonical three-state machine:

```
                ┌──────────────┐
   success ────▶│   CLOSED     │◀──── default
                │  (healthy)   │
                └──────┬───────┘
                       │ failures >= 5 in window
                       ▼
                ┌──────────────┐
                │     OPEN     │── 30s cooldown ──┐
                │  (fallback)  │                  │
                └──────┬───────┘                  │
                       │                          │
                       ▼                          │
                ┌──────────────┐                  │
                │  HALF_OPEN   │◀─────────────────┘
                │  (probing)   │
                └──────┬───────┘
            success    │    failure
                ┌──────┴──────┐
                ▼             ▼
            CLOSED          OPEN
```

When `OPEN`, requests are immediately rerouted to the **regex heuristic fallback**
(prompt-injection patterns, SSN/CC/passport regex). This keeps the firewall
functioning at degraded accuracy rather than failing open.

### 4.2 Layer 2 — Mesh Breaker (Istio DestinationRule)

Istio's outlier detection ejects unhealthy Triton pods from the load-balancing pool
*before* the application breaker even sees the failure. This protects against:

- gRPC stream stalls (TCP healthy, protocol stuck)
- Per-pod GPU memory exhaustion
- Slow-loris-style degraded latency

```yaml
# excerpt — deploy/k8s/istio_flagger/canary-triton.yaml
trafficPolicy:
  outlierDetection:
    consecutive5xxErrors: 3
    interval: 10s
    baseEjectionTime: 30s
    maxEjectionPercent: 50
```

### 4.3 Why Two Breakers?

| Failure Mode | Detected By | Fallback |
| --- | --- | --- |
| Triton pod crash | Istio (5xx) | Eject pod, route to healthy pod |
| Model corruption / wrong logits shape | App breaker | Heuristic regex |
| Network partition | Both | Mesh ejects, app falls back if all pods ejected |
| Slow GPU (thermal throttle) | App breaker (latency >10ms) | Heuristic regex |

---

## 5. Triton Inference Cluster Topology

```
                       ┌──────────────────────────────────────┐
                       │  Istio VirtualService                │
                       │   model-version weighting (canary)    │
                       └─────────┬────────────────────┬────────┘
                                 │ 90%                │ 10%
                                 ▼                    ▼
                  ┌─────────────────────┐  ┌─────────────────────┐
                  │ Triton Primary      │  │ Triton Canary       │
                  │ (StatefulSet, 4 pods)│  │ (StatefulSet, 1 pod)│
                  │  ├─ guardrail v3.2  │  │  ├─ guardrail v3.3  │
                  │  ├─ ONNX backend    │  │  ├─ ONNX backend    │
                  │  ├─ dyn. batch=64   │  │  ├─ dyn. batch=64   │
                  │  └─ NVMe model cache│  │  └─ NVMe model cache│
                  └─────────┬───────────┘  └─────────┬───────────┘
                            │                        │
                            └───────────┬────────────┘
                                        ▼
                            ┌───────────────────────┐
                            │  S3 Model Repository  │
                            │  (immutable artifacts)│
                            │  guardrail/           │
                            │   ├─ 3.2/             │
                            │   │   ├─ model.onnx   │
                            │   │   └─ config.pbtxt │
                            │   └─ 3.3/             │
                            │       ├─ model.onnx   │
                            │       └─ config.pbtxt │
                            └───────────────────────┘
```

Configuration lives in [`deploy/triton/model_repository/guardrail_model/config.pbtxt`](../deploy/triton/model_repository/guardrail_model/config.pbtxt).

Key choices:

- **`max_batch_size: 32`** — sweet spot for our p99 budget on g4dn.xlarge.
- **`dynamic_batching.max_queue_delay_microseconds: 2000`** — caps latency tax at 2 ms.
- **`instance_group [{ count: 2, kind: KIND_GPU }]`** — 2 model instances per pod for
  concurrent stream execution.
- **`optimization.execution_accelerators.gpu_execution_accelerator [{ name: "tensorrt" }]`** —
  TensorRT optimisation post model load.

---

## 6. Distributed Drift Engine

Drift detection cannot be in the hot path — it would blow the latency SLO. Instead, we
asynchronously stream firewall decisions to time-series partitioned Postgres and
process them in batches with **Dask Distributed**.

### 6.1 Data Flow

```
FastAPI ─async─▶ Postgres (firewall_logs)
                 │     (range partition by day:
                 │      firewall_logs_2026_02_15, ...)
                 ▼
            ┌─────────────────────────────────┐
            │  Dask Scheduler (k8s pod)       │
            │   ├─ Workers (HPA, 4–32 pods)   │
            │   └─ Reads partitions out-of-core│
            └────────────┬────────────────────┘
                         │
                         ▼
            ┌─────────────────────────────────┐
            │  drift_detector.py              │
            │   ├─ Compute PSI per feature    │
            │   ├─ KL divergence on logits    │
            │   ├─ χ² on threat-type counts   │
            │   └─ Emit DriftReport JSON      │
            └────────────┬────────────────────┘
                         │ if drift > threshold
                         ▼
            ┌─────────────────────────────────┐
            │  Airflow Sensor                 │
            │   triggers drift_retrain_dag    │
            └─────────────────────────────────┘
```

### 6.2 Why Postgres Range Partitioning?

The schema in [`backend/src/db/migrations/001_initial_schema.sql`](../backend/src/db/migrations/001_initial_schema.sql)
creates a **partitioned parent table** with **per-day child partitions**. This unlocks:

- **O(1) partition pruning** — Dask reads only the partitions it needs.
- **Cheap rollups** — drop partitions older than retention window with one DDL.
- **Hot/cold tiering** — recent partitions on NVMe, older partitions on S3 via
  `pg_partman` foreign data wrappers.

---

## 7. Active Learning & Continuous Fine-Tuning Loop

```
        ┌──────────────────────┐
        │ Drift event detected │
        └──────────┬───────────┘
                   ▼
        ┌─────────────────────────┐
        │ Airflow DAG starts      │
        │ drift_retrain_dag.py    │
        └──────────┬──────────────┘
                   │
        ┌──────────┴──────────────────┐
        ▼                              ▼
┌────────────────────┐      ┌────────────────────┐
│ Pull mis-labelled  │      │ Pull true labels   │
│ samples from       │      │ from human-in-loop │
│ Postgres partitions│      │ review queue (S3)  │
└──────────┬─────────┘      └─────────┬──────────┘
           │                          │
           └──────────────┬───────────┘
                          ▼
            ┌──────────────────────────┐
            │ continuous_finetuning.py │
            │  ├─ Load base PyTorch    │
            │  ├─ Apply LoRA adapter   │
            │  │   (r=8, α=16, q,v)    │
            │  ├─ Train 3 epochs       │
            │  └─ Save adapter only    │
            └──────────────┬───────────┘
                           ▼
            ┌──────────────────────────┐
            │ export_model.py           │
            │  ├─ Merge LoRA + base    │
            │  ├─ Export ONNX          │
            │  ├─ Validate parity      │
            │  │   (1e-5 tolerance)    │
            │  └─ Push to S3 repo      │
            └──────────────┬───────────┘
                           ▼
            ┌──────────────────────────┐
            │ Triton model_repository  │
            │ auto-discovery polling   │
            │ loads new version        │
            └──────────────┬───────────┘
                           ▼
            ┌──────────────────────────┐
            │ Flagger detects new ver. │
            │ progressive canary roll  │
            │ 1% → 10% → 50% → 100%    │
            │ Auto-rollback on SLI loss│
            └──────────────────────────┘
```

The relevant code lives in:
- [`ml_pipelines/continuous_finetuning.py`](../ml_pipelines/continuous_finetuning.py)
- [`ml_pipelines/export_model.py`](../ml_pipelines/export_model.py)
- [`deploy/airflow/dags/drift_retrain_dag.py`](../deploy/airflow/dags/drift_retrain_dag.py)
- [`deploy/k8s/istio_flagger/canary-triton.yaml`](../deploy/k8s/istio_flagger/canary-triton.yaml)

---

## 8. Design Patterns Catalogue

GuardRail Studio is built on a deliberate, opinionated catalogue of design patterns.
Every component below is implemented somewhere in the codebase and the file paths
are provided for direct inspection.

### 8.1 Singleton — Connection Pool Managers

The gRPC channel to Triton is *expensive* to construct (TLS handshake, HPACK table
warmup, JWKS fetch). We instantiate exactly one per process and reuse it.

```python
# backend/src/services/inference_client_triton.py (excerpt)
class ProductionInferenceClient:
    _instance: Optional['ProductionInferenceClient'] = None
    _lock = asyncio.Lock()

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
```

The same pattern is applied to:
- `db_manager` in [`backend/src/db/postgres.py`](../backend/src/db/postgres.py) — SQLAlchemy AsyncEngine pool.
- `qdrant_manager` in [`backend/src/db/qdrant.py`](../backend/src/db/qdrant.py) — Qdrant HTTP/2 client.

### 8.2 Strategy — Pluggable Classifier Policies

`GuardrailService` does not know whether it is being served by Triton or by regex
fallback. It owns a *Strategy reference* (`inference_client`) and an
**explicit policy decision** (threshold per `ThreatType`):

```python
# backend/src/services/guardrail_service.py
self.thresholds = {
    ThreatType.PROMPT_INJECTION: settings.prompt_injection_threshold,  # 0.85
    ThreatType.PII_DETECTION:    settings.pii_detection_threshold,     # 0.80
    ThreatType.TOXICITY:         settings.toxicity_threshold,          # 0.75
}
```

This allows hot-swapping detectors without touching the service layer.

### 8.3 Circuit Breaker — Three-State Machine

Covered in §4 above. Implementation:
[`backend/src/services/inference_client_triton.py`](../backend/src/services/inference_client_triton.py).

### 8.4 Repository — DB-Agnostic Data Access

All Postgres I/O is mediated through repositories:
[`backend/src/repositories/telemetry_repo.py`](../backend/src/repositories/telemetry_repo.py).
This isolates the service layer from any future move to e.g. ClickHouse.

### 8.5 Sidecar — Mesh, Logging, Telemetry

Istio sidecars handle mTLS, retry, and outlier detection. The application is unaware.
OpenTelemetry agent runs as DaemonSet to scrape OTLP from each pod.

### 8.6 Bulkhead — Worker Isolation

Dask workers and FastAPI workers run in **separate Deployments** with separate HPAs.
A drift-detection job cannot starve the hot path of CPU.

---

## 9. Data Models

The canonical schema (abbreviated; full DDL in
[`backend/src/db/migrations/001_initial_schema.sql`](../backend/src/db/migrations/001_initial_schema.sql)):

```sql
CREATE TABLE firewall_logs (
    id           BIGSERIAL,
    request_id   TEXT        NOT NULL,
    ts           TIMESTAMPTZ NOT NULL,
    latency_ms   NUMERIC(8,3) NOT NULL,
    passed       BOOLEAN     NOT NULL,
    threat_type  TEXT        NOT NULL,
    confidence   NUMERIC(5,4) NOT NULL,
    model_name   TEXT        NOT NULL,
    tokens       INTEGER     NOT NULL,
    PRIMARY KEY (id, ts)
) PARTITION BY RANGE (ts);
```

Wire schemas (Pydantic) live in
[`backend/src/schemas/firewall.py`](../backend/src/schemas/firewall.py).

---

## 10. Capacity & Scaling Model

| Component | Scaling Trigger | Min | Max | Notes |
| --- | --- | --- | --- | --- |
| FastAPI Deployment | CPU 60% / latency p95 > 6ms | 4 | 128 | HPA + KEDA |
| Triton StatefulSet | GPU util > 70% | 2 | 32 | Cluster autoscaler g4dn.xlarge |
| RDS Aurora | r/w split, read replicas | 1 primary + 2 replicas | + 8 replicas | Aurora autoscaling |
| Qdrant StatefulSet | RAM > 80% | 3 | 9 | Tied to NVMe local-ssd |
| Dask Workers | Pending tasks > 50 | 4 | 64 | KEDA on `dask_scheduler_pending` |
| Airflow Workers | Queue depth > 10 | 2 | 8 | Celery executor |

---

## 11. Failure Mode & Effects Analysis (FMEA, abridged)

| Failure | Detection | Mitigation | Blast Radius |
| --- | --- | --- | --- |
| Triton GPU OOM | App breaker latency probe | Eject pod, scale up, fallback heuristic | 1 pod |
| RDS primary failover | App-level retry | Aurora failover (~30s); write buffering in app | All pods, 30s degraded |
| Bad model rollout | Flagger SLI gate | Auto-rollback to prior version | 1–10% of traffic |
| Region failure | Route53 health check | DNS failover to secondary region | Region-wide |
| Drift detector lag | Airflow SLA miss alert | Page on-call; manual re-run | None (no hot-path impact) |

---

## 12. Open Questions / Roadmap

- [ ] Move ONNX → TensorRT FP8 for further latency win (need H100 nodes).
- [ ] Replace Qdrant with HNSW-on-FoundationDB for stronger consistency.
- [ ] Add WebAssembly heuristic engine to push fallback closer to client.

---

> See [`docs/PHILOSOPHY.md`](./PHILOSOPHY.md) for the engineering principles that
> govern *how* we make these design choices, not just *what* they are.
