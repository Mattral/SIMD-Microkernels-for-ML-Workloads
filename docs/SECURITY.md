# GuardRail Studio — Security Posture & Threat Model

> *"In the era of AGI, the firewall is not the perimeter — it is the prompt itself."*

This document is the **authoritative security reference** for GuardRail Studio. It
describes the threats the platform must defend against, the controls deployed at
every layer, and the operational procedures used to keep them sharp.

**Companion Docs:**
- [System Design →](./SYSTEM_DESIGN.md)
- [Setup & Operations →](./SETUP_AND_OPERATIONS.md)
- [Contributing (security review rubric) →](./CONTRIBUTING.md#72-reviewer-rubric)

---

## Table of Contents

1. [Threat Model — STRIDE](#1-threat-model--stride)
2. [Adversarial LLM Attack Surface](#2-adversarial-llm-attack-surface)
3. [Defence-in-Depth Topology](#3-defence-in-depth-topology)
4. [IAM Least-Privilege Matrix](#4-iam-least-privilege-matrix)
5. [TLS & End-to-End Encryption](#5-tls--end-to-end-encryption)
6. [AWS WAF Ruleset](#6-aws-waf-ruleset)
7. [Secrets Management](#7-secrets-management)
8. [PII Handling & Data Residency](#8-pii-handling--data-residency)
9. [Audit Logging & SIEM Pipeline](#9-audit-logging--siem-pipeline)
10. [Incident Response Plan](#10-incident-response-plan)
11. [Vulnerability Management](#11-vulnerability-management)
12. [Compliance & Attestations](#12-compliance--attestations)

---

## 1. Threat Model — STRIDE

| Category | Threat | Likelihood | Impact | Mitigation |
| --- | --- | --- | --- | --- |
| **S**poofing | Forged JWT / API key reuse | High | Critical | mTLS at mesh + JWT validation via JWKS pinning; per-tenant API keys rotated 90 d |
| **T**ampering | In-transit prompt mutation | Medium | High | TLS 1.3 + Istio mTLS + payload HMAC |
| **R**epudiation | Tenant denies a request | Medium | Medium | Immutable audit log → CloudWatch + W&B run lineage |
| **I**nformation Disclosure | PII leakage in logs | High | Critical | Logger scrubber + payload redaction + KMS at rest |
| **D**enial of Service | Token-flood / amplification | High | High | WAF rate-limit + Istio per-tenant quota + HPA |
| **E**levation of Privilege | Container escape | Low | Critical | distroless images, gVisor sandbox, IRSA scoped roles |

---

## 2. Adversarial LLM Attack Surface

GuardRail Studio is a *defensive* layer; its job is to detect and block LLM-specific
attacks. The taxonomy below is the canonical surface we cover.

### 2.1 Threat Vectors

| Vector | Description | Detection | Mitigation |
| --- | --- | --- | --- |
| **Prompt Injection** | Adversary embeds instructions like *"ignore previous instructions"*. | DistilRoBERTa classifier (label=1) + regex fallback in `inference_client_triton.py`. | Block at threshold ≥ 0.85; log + alert. |
| **Jailbreaking** | Multi-step role-play designed to evade safety prompts. | Same classifier with `prompt_injection` head; conversation-level entropy. | Block; surface to W&B for active-learning re-train. |
| **PII Leakage (outbound)** | Model echoes SSN/CC/passport-like patterns. | Classifier label=2 + regex (SSN/CC/passport). | Redact response; ledger event. |
| **Data Poisoning** | Adversary supplies training data to bias the model. | Drift detector PSI/KL on training distribution; manual approval gate on Airflow DAG. | LoRA-only updates limit blast radius; rollback < 5 min via Flagger. |
| **Model Theft / Exfil** | Bulk inference to clone decision boundary. | WAF rate-limit + per-tenant quota + entropy anomaly on response distribution. | Throttle, raise risk score, alert SOC. |
| **Denial-of-Wallet** | Bursting expensive inference to inflate cost. | Per-tenant budget meter + WAF. | Hard cutover to fallback heuristics; alert. |
| **Embedding Inversion** | Reconstruct training data from exposed embeddings. | Embeddings never returned outside the trust boundary. | API never exposes raw embeddings to tenants. |
| **Side-channel via timings** | Tenant infers model class from latency. | Response padding and jitter are considered to reduce timing leakage. | Target is to keep latency noise within a narrow band where practical. |

### 2.2 Threat Map vs Code

```
Threat                    Detected by                              Test
────────────────────────  ───────────────────────────────────────  ─────────────────────────
Prompt Injection          inference_client_triton.py::_infer_triton  tests/ml/test_model_parity.py
                          + _infer_fallback_heuristic                + tests/unit/test_classifier.py
PII Leakage               inference_client_triton.py + regex set     tests/unit/test_pii_regex.py
Data Poisoning            drift_detector.py + airflow approval gate  tests/integration/test_drift.py
Denial-of-Wallet          Istio per-tenant quota + WAF               tests/load_testing/k6_chaos_test.js
```

---

## 3. Defence-in-Depth Topology

```
       ┌────────────────────────────────────────────────────────────────┐
       │                          INTERNET                              │
       └─────────────────────────────┬──────────────────────────────────┘
                                     ▼
       ┌────────────────────────────────────────────────────────────────┐
       │  L1 — AWS Shield + Route53 GeoDNS                              │
       │       (volumetric DDoS, geographic shaping)                    │
       └─────────────────────────────┬──────────────────────────────────┘
                                     ▼
       ┌────────────────────────────────────────────────────────────────┐
       │  L2 — AWS WAF                                                  │
       │       OWASP top-10 + LLM-specific rule pack                    │
       │       per-tenant rate-limit + bot challenge                    │
       └─────────────────────────────┬──────────────────────────────────┘
                                     ▼
       ┌────────────────────────────────────────────────────────────────┐
       │  L3 — Istio IngressGateway (mTLS, JWT, header propagation)     │
       └─────────────────────────────┬──────────────────────────────────┘
                                     ▼
       ┌────────────────────────────────────────────────────────────────┐
       │  L4 — Application Layer                                        │
       │       FastAPI input validation (Pydantic strict)               │
       │       Per-route auth dependency                                │
       │       Rate limit middleware                                    │
       └─────────────────────────────┬──────────────────────────────────┘
                                     ▼
       ┌────────────────────────────────────────────────────────────────┐
       │  L5 — ML Layer                                                 │
       │       DistilRoBERTa classifier (Triton)                        │
       │       Heuristic fallback (regex)                               │
       │       Threshold gating                                         │
       │       Embedding-based historical match (Qdrant)                │
       └─────────────────────────────┬──────────────────────────────────┘
                                     ▼
       ┌────────────────────────────────────────────────────────────────┐
       │  L6 — Data Layer                                               │
       │       Postgres (TDE via KMS, IAM auth, row-level security)     │
       │       Qdrant (KMS at-rest, mTLS in-mesh)                       │
       └────────────────────────────────────────────────────────────────┘
```

---

## 4. IAM Least-Privilege Matrix

This architecture is designed so each Kubernetes workload uses an **IRSA** role with
minimal scope. The backend service account currently implements IRSA in
[`deploy/terraform/modules/eks/main.tf`](../deploy/terraform/modules/eks/main.tf).

The following matrix shows the recommended end-state role mapping for the full
platform architecture:

| Workload | Role | Allowed Actions | Resources |
| --- | --- | --- | --- |
| `guardrail-backend` | `gr-backend-role` | `secretsmanager:GetSecretValue` | `arn:aws:secretsmanager:*:*:secret:guardrail/backend/*` |
| | | `s3:GetObject` | `arn:aws:s3:::guardrail-models/*` (read) |
| | | `kms:Decrypt` | `arn:aws:kms:*:*:key/guardrail-data` |
| `triton-server` | `gr-triton-role` | `s3:GetObject` | `arn:aws:s3:::guardrail-models/*` |
| | | `s3:ListBucket` | `arn:aws:s3:::guardrail-models` |
| `airflow-worker` | `gr-airflow-role` | `s3:Get*`, `s3:Put*` | `arn:aws:s3:::guardrail-training/*` |
| | | `secretsmanager:GetSecretValue` | `arn:aws:secretsmanager:*:*:secret:guardrail/airflow/*` |
| `dask-worker` | `gr-dask-role` | `rds-db:connect` | RDS read-replica only |
| | | `s3:GetObject` | `arn:aws:s3:::guardrail-analytics/*` |

**No workload should have wildcard `s3:*` in a least-privilege deployment.**

---

## 5. TLS & End-to-End Encryption

```
                           Public TLS (1.3, AES-GCM)
                                   │
                                   ▼
                       ┌─────────────────────────┐
                       │  AWS ACM cert on NLB    │  cert auto-rotated 60 d
                       └───────────┬─────────────┘
                                   ▼
                       ┌─────────────────────────┐
                       │  Istio mTLS PERMISSIVE→STRICT │
                       └───────────┬─────────────┘
                                   ▼
                       ┌─────────────────────────┐
                       │  intra-pod loopback     │  unencrypted (loopback ns)
                       └───────────┬─────────────┘
                                   ▼
                       ┌─────────────────────────┐
                       │  PostgreSQL TLS (verify-ca) │  enforced by `sslmode=verify-ca`
                       └─────────────────────────┘
```

- TLS 1.3 only — TLS 1.2 disabled at the NLB.
- Cipher suites: AEAD only (no CBC, no RC4, no 3DES).
- mTLS inside the mesh enforced via `PeerAuthentication { mode: STRICT }`.
- KMS-backed at-rest encryption on every EBS volume, every RDS cluster, every S3
  bucket, every Secrets Manager secret.

---

## 6. AWS WAF Ruleset

Managed rule groups enabled at the WebACL:

- `AWSManagedRulesCommonRuleSet`
- `AWSManagedRulesKnownBadInputsRuleSet`
- `AWSManagedRulesAmazonIpReputationList`
- `AWSManagedRulesAnonymousIpList`
- `AWSManagedRulesBotControlRuleSet` (target mode)

Custom rules:

| Rule | Action | Notes |
| --- | --- | --- |
| `rate-limit-per-tenant` | Block 5 min | 1000 RPS / tenant header `X-Tenant-Id` |
| `payload-size-cap` | Block | Body > 32 KiB rejected |
| `prompt-injection-signature` | Count + risk-score | regex signature list, used as *signal* not block |
| `geo-block` | Block | OFAC list |
| `path-allowlist` | Block | only `^/api/` paths allowed |

---

## 7. Secrets Management

**Primary source of truth: AWS Secrets Manager.** The architecture favors Secrets
Manager as the principal secret store and minimizes static credentials in repository
or cluster configuration.

```
                     ┌─────────────────────────────┐
                     │  AWS Secrets Manager        │
                     │   guardrail/backend/db-url  │
                     │   guardrail/backend/jwks    │
                     │   guardrail/airflow/wandb   │
                     └──────────────┬──────────────┘
                                    │ IRSA + IAM
                                    ▼
                     ┌─────────────────────────────┐
                     │  Optional sync layer        │  e.g. External Secrets Operator
                     │  (refresh interval configurable)         │
                     └──────────────┬──────────────┘
                                    ▼
                     ┌─────────────────────────────┐
                     │  k8s Secret  (ephemeral)    │
                     │  mounted as env into pod    │
                     └─────────────────────────────┘
```

Rules and policy intent:

- **Avoid `.env` in git history.** The repository aims to keep static secrets out of version control.
- **No long-lived static creds.** Database access is intended to use IAM auth tokens where available.
- **Do not store secrets in ConfigMaps.** ConfigMaps are for configuration only.
- **Rotation cadence**: WAF tokens 30 d, DB IAM N/A, JWT signing keys 90 d, KMS CMK 365 d.

---

## 8. PII Handling & Data Residency

GuardRail Studio is designed to **minimise PII exposure**:

1. **The architecture intends to avoid persisting request bodies** unless the request
   is *blocked* for audit purposes. If persisted, blocked payloads should be hashed
   (SHA-256) and the raw payload retained only under strict expiration and encryption.
2. **Embeddings are never returned to tenants** to prevent inversion attacks.
3. **All training data is anonymised** at ingestion via the
   `continuous_finetuning.py::redact_pii()` helper.
4. **Data residency**: customer A's traffic and storage live in their selected
   region. Cross-region replication is opt-in and disabled by default.

---

## 9. Audit Logging & SIEM Pipeline

```
   FastAPI structlog ──┐
                       │
   Istio access logs ──┼──▶ FluentBit ──▶ Kinesis Firehose ──▶ S3 (raw)
                       │                                       │
   AWS WAF logs ───────┤                                       ▼
                       │                                  AWS Athena
   RDS audit logs ─────┘                                       │
                                                               ▼
                                                  Datadog / Splunk SIEM
                                                  + auto-generated CloudWatch
                                                  alarms on anomalous patterns
```

Retention:

- **Raw access / audit logs**: 365 days S3 + Glacier after 90 days.
- **Auth events**: 7 years (compliance).
- **Postgres slow-query log**: 30 days CloudWatch.

---

## 10. Incident Response Plan

**Severity tiers**:

| Sev | Examples | Response Time | Comms |
| --- | --- | --- | --- |
| Sev-1 | Active exfil / customer-impacting breach | 15 min | PagerDuty + exec page + customer advisory |
| Sev-2 | Critical CVE in dependency / WAF down | 1 h | PagerDuty + #security-incidents |
| Sev-3 | Suspicious activity, no confirmed compromise | 4 h | Slack thread + ticket |

Procedure:

1. **Contain** — disable WAF rule, eject pod, rotate credential.
2. **Eradicate** — push patched image, force secret rotation.
3. **Recover** — verify metrics, scale back up.
4. **Postmortem** — see
   [`docs/CONTRIBUTING.md#10-postmortems`](./CONTRIBUTING.md#10-postmortems).

The on-call runbook lives in
[`docs/SETUP_AND_OPERATIONS.md §11`](./SETUP_AND_OPERATIONS.md#11-day-2-operations--runbooks).

---

## 11. Vulnerability Management

- **Trivy** scans every container image in CI (see
  [`.github/workflows/ci_cd.yaml`](../.github/workflows/ci_cd.yaml)). CRITICAL CVEs
  break the build; HIGH CVEs require a `risk-accepted` label and a 14-day SLA.
- **Dependabot** raises PRs weekly for Python + Node deps; auto-merge enabled for
  patch versions with green CI.
- **Pip-audit** & **npm audit** run on a weekly schedule.
- **Cosign** signs every release image; deployment manifests pin by digest.

---

## 12. Compliance & Attestations

Targeted alignment (the platform itself is the *control*, not the auditor):

- **SOC 2 Type II** — controls live in `compliance/soc2/` (not in this repo).
- **GDPR / CCPA** — PII handling §8 is the operative control.
- **HIPAA** — opt-in BAA tier; KMS, audit log, encryption-at-rest already meet
  the technical requirements.
- **ISO 27001 Annex A** — IAM matrix in §4 maps to A.9.

Audit evidence is collected automatically by the GitOps pipeline and exported to
the `compliance-evidence` S3 bucket.

---

> *"Security is the work nobody applauds — until the day everyone needs it."*

