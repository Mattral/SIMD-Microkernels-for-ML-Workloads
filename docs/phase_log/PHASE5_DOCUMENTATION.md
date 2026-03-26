# Phase 5: Progressive Delivery, Automated Learning & Chaos Engineering

## Overview

Phase 5 implements the "Holy Grail" of MLOps with:
- **Progressive Delivery**: Istio + Flagger for automated canary deployments
- **Continuous Learning**: LoRA-based fine-tuning for rapid adaptation
- **Chaos Engineering**: k6 load tests validating circuit breaker and autoscaling

---

## Component 1: Istio + Flagger Progressive Delivery

### Architecture

```
Model Update Trigger
    ↓
Flagger Detects New Deployment
    ↓
Create Canary Pods (new model)
    ↓
Traffic Shifting Loop:
    5% → 10% → 15% → ... → 100%
    (Every 1 minute)
    ↓
    At Each Step:
    ├─ Check HTTP 500 rate < 1%
    ├─ Check p99 latency < 15ms
    └─ Check throughput > 10 req/s
    ↓
    ├─ [ALL PASS] → Continue to next step
    └─ [ANY FAIL] → ROLLBACK immediately
    ↓
[Success] Promote Canary → Primary
[Failure] Rollback to Previous Version
```

### Deployment Timeline

| Time | Primary | Canary | Status | Action |
|------|---------|--------|--------|--------|
| 0m   | 100%    | 0%     | Init   | Canary created |
| 1m   | 95%     | 5%     | Analyzing | Check metrics |
| 2m   | 90%     | 10%    | Analyzing | Check metrics |
| 3m   | 85%     | 15%    | Analyzing | Check metrics |
| ...  | ...     | ...    | ... | ... |
| 19m  | 5%      | 95%    | Analyzing | Final check |
| 20m  | 0%      | 100%   | Promoted | Swap primary/canary |

**Total Duration**: ~20 minutes for full promotion

### Installation

```bash
# 1. Install Istio
curl -L https://istio.io/downloadIstio | sh -
cd istio-*
export PATH=$PWD/bin:$PATH
istioctl install --set profile=default -y

# 2. Enable Istio for namespace
kubectl label namespace guardrail-studio istio-injection=enabled

# 3. Install Flagger
helm repo add flagger https://flagger.app
helm upgrade -i flagger flagger/flagger \
  --namespace istio-system \
  --set meshProvider=istio \
  --set metricsServer=http://prometheus.monitoring:9090

# 4. Deploy canary configuration
kubectl apply -f /app/deploy/k8s/istio_flagger/canary-triton.yaml
```

### Trigger Canary Deployment

```bash
# Method 1: Update image
kubectl -n guardrail-studio set image deployment/triton \
  triton=nvcr.io/nvidia/tritonserver:23.11-py3

# Method 2: Update annotation (model version)
kubectl -n guardrail-studio annotate deployment/triton \
  model-version="v2" --overwrite

# Method 3: Update ConfigMap with new model path
kubectl -n guardrail-studio edit configmap triton-config
```

### Monitor Canary

```bash
# Watch canary status
watch kubectl -n guardrail-studio get canary triton-canary

# View events
kubectl -n guardrail-studio describe canary triton-canary

# Check traffic distribution
kubectl -n guardrail-studio get virtualservice triton-vs -o yaml
```

### Prometheus Metrics

```promql
# HTTP Error Rate
sum(rate(istio_requests_total{
  destination_workload=~"triton-canary",
  response_code=~"5.*"
}[1m])) / sum(rate(istio_requests_total{
  destination_workload=~"triton-canary"
}[1m])) * 100

# p99 Latency
histogram_quantile(0.99, sum(rate(
  istio_request_duration_milliseconds_bucket{
    destination_workload=~"triton-canary"
  }[1m]
)) by (le))

# Request Throughput
sum(rate(istio_requests_total{
  destination_workload=~"triton-canary"
}[1m]))
```

### Automatic Rollback

Flagger rolls back if:
1. **HTTP 500 error rate** > 1%
2. **p99 latency** > 15ms
3. **Throughput** < 10 req/s
4. **Webhook validation** fails

**Rollback Actions**:
- Traffic immediately shifts back to primary (100%)
- Canary pods are scaled down
- Deployment marked as "Failed"
- Slack/webhook notifications sent

---

## Component 2: LoRA Continuous Fine-Tuning

### Architecture

**Parameter-Efficient Fine-Tuning (PEFT)**:
- Only trains **~0.5%** of model parameters (LoRA adapters)
- Training time: **~10 minutes** (vs 2+ hours full fine-tuning)
- Memory: **2GB** (vs 12GB full model)
- Accuracy: **95-98%** of full fine-tuning performance

### LoRA Configuration

```python
LoraConfig(
    task_type=TaskType.SEQ_CLS,
    r=8,              # Rank (number of trainable parameters)
    lora_alpha=16,    # Scaling factor
    lora_dropout=0.1,
    target_modules=["query", "value"]  # Attention modules only
)
```

**Parameter Breakdown**:
- Base RoBERTa: 125M parameters
- LoRA adapters: **~590K parameters** (0.47%)
- Reduction: **99.53%** fewer trainable parameters

### Workflow

```
1. Drift Detection (from Airflow DAG)
    ↓
2. Extract Drift Dataset (100-1000 samples)
    ↓
3. Load Base Model + Inject LoRA Adapters
    ↓
4. Fine-Tune Adapters (3 epochs, ~10 min)
    ↓
5. Merge LoRA Weights → Base Model
    ↓
6. Export to ONNX (via export_model.py)
    ↓
7. Upload to Triton Model Repository
    ↓
8. Trigger Flagger Canary Deployment
    ↓
9. Progressive Rollout (20 min)
    ↓
10. Promote if Successful / Rollback if Failed
```

### Usage

```bash
# Manual trigger
cd /app/ml_pipelines
python continuous_finetuning.py

# Automated trigger (from Airflow)
# Already integrated in drift_retrain_dag.py
```

### W&B Tracking

The pipeline logs to W&B:
- **Training loss** curve
- **Evaluation loss** curve
- **Parameter counts** (base vs LoRA)
- **Training duration**
- **ONNX export status**

### Expected Results

| Metric | Full Fine-Tuning | LoRA Fine-Tuning |
|--------|------------------|------------------|
| Training Time | 2-4 hours | 10-15 minutes |
| Memory Usage | 12GB | 2GB |
| Parameters Trained | 125M (100%) | 590K (0.47%) |
| Accuracy Retention | 100% | 95-98% |
| Inference Latency | Same | Same |

---

## Component 3: k6 Chaos Engineering

### Test Profile

**Thundering Herd Attack**:
```
Stage 1 (3 min):  Ramp 0 → 5000 VUs
Stage 2 (2 min):  Hold 5000 VUs
Stage 3 (1 min):  Ramp 5000 → 0 VUs
Total: 6 minutes
Expected RPS: 10,000-15,000 req/sec
```

### Success Criteria

| Metric | Threshold | Purpose |
|--------|-----------|---------|
| HTTP Success Rate | > 99% | Including fallback responses |
| p95 Latency | < 50ms | Global latency SLA |
| p99 Latency | < 100ms | Tail latency SLA |
| Fallback Activation | > 1% | Circuit breaker validation |
| Check Pass Rate | > 95% | Response validation |

### Installation

```bash
# Install k6 (Linux)
sudo apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-keys C5AD17C747E3415A3642D57D77C6C491D6AC1D69
echo "deb https://dl.k6.io/deb stable main" | sudo tee /etc/apt/sources.list.d/k6.list
sudo apt-get update
sudo apt-get install k6

# Install k6 (macOS)
brew install k6
```

### Execution

```bash
# Set backend URL
export API_URL="http://$(kubectl -n guardrail-studio get svc backend-external -o jsonpath='{.status.loadBalancer.ingress[0].hostname}')"

# Run chaos test
k6 run --vus 5000 --duration 6m /app/tests/load_testing/k6_chaos_test.js

# Run with custom stages
k6 run --stage 2m:1000 --stage 3m:5000 --stage 1m:0 \
  /app/tests/load_testing/k6_chaos_test.js

# Run and export results
k6 run --out json=results.json \
  /app/tests/load_testing/k6_chaos_test.js
```

### What It Tests

**1. Circuit Breaker Activation**:
- At ~2000 VUs, Triton will start experiencing queue buildup
- Circuit breaker opens after 5 consecutive failures
- Requests automatically fallback to regex heuristics
- Validation: `fallback_activation_rate > 1%`

**2. Horizontal Pod Autoscaling**:
- CPU/memory pressure triggers HPA
- Backend scales from 2 → 10 pods
- Triton scales from 1 → 5 pods (GPU)
- Validation: Monitor with `kubectl get hpa -w`

**3. Graceful Degradation**:
- Even with Triton down, system returns 200 OK
- Fallback heuristics maintain >85% accuracy
- No connection refused or 5xx errors
- Validation: `http_req_failed < 1%`

**4. Latency Under Load**:
- p95 latency stays < 50ms despite 5000 VUs
- Triton inference: <10ms
- Fallback heuristics: <1ms
- Validation: Prometheus histograms

### Expected Output

```
     ✓ status is 200
     ✓ status is not 5xx
     ✓ response has request_id
     ✓ response has classification
     ✓ response time < 100ms
     ✓ response is valid JSON

     checks.........................: 98.23% ✓ 589380    ✗ 10620
     data_received..................: 245 MB 41 MB/s
     data_sent......................: 198 MB 33 MB/s
     http_req_blocked...............: avg=1.2ms    p(95)=3.5ms
     http_req_duration..............: avg=28.4ms   p(95)=45.2ms p(99)=78.3ms
     http_reqs......................: 600000 100000/s
     success_rate...................: 99.12% ✓ 594720    ✗ 5280
     fallback_activation_rate.......: 12.45% ✓ 74700     ✗ 525300
     blocked_rate...................: 23.67% ✓ 142000    ✗ 458000
```

**Interpretation**:
- ✅ 99.12% success rate (threshold: >99%)
- ✅ p95 45.2ms < 50ms (threshold: <50ms)
- ✅ 12.45% fallback rate (circuit breaker active)
- ✅ 100,000 req/sec throughput maintained

---

## Integration: Complete Automated Loop

### End-to-End Flow

```
1. Production Traffic → Triton Inference
    ↓
2. Metrics logged to PostgreSQL (partitioned)
    ↓
3. Daily Airflow DAG (02:00 UTC)
    ├─ Extract last 24h logs
    ├─ Run Drift Analysis (Dask + PSI/Wasserstein)
    └─ Check PSI > 0.2 threshold
    ↓
4. [Drift Detected] → Trigger LoRA Fine-Tuning
    ├─ Extract drift dataset
    ├─ Train LoRA adapters (10 min)
    ├─ Merge weights
    └─ Export ONNX v2
    ↓
5. Upload to Triton Model Repository
    ↓
6. Update Deployment Annotation
    ↓
7. Flagger Detects New Version
    ├─ Create Canary Pods
    ├─ Traffic: 5% → 10% → ... → 100%
    ├─ Validate metrics at each step
    └─ Promote or Rollback
    ↓
8. [Success] New Model in Production
    [Failure] Rollback to Previous Model
    ↓
9. W&B Logs Complete Training + Deployment Pipeline
```

### Kubernetes Job for Automated Training

```yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: lora-finetuning
  namespace: guardrail-studio
spec:
  template:
    spec:
      containers:
        - name: trainer
          image: guardrail-studio/ml-trainer:latest
          command: ["python", "/app/ml_pipelines/continuous_finetuning.py"]
          envFrom:
            - secretRef:
                name: database-credentials
            - secretRef:
                name: wandb-credentials
          resources:
            requests:
              memory: "4Gi"
              cpu: "2000m"
            limits:
              memory: "8Gi"
              cpu: "4000m"
      restartPolicy: Never
  backoffLimit: 3
```

---

## Monitoring & Observability

### Flagger Metrics (Prometheus)

```promql
# Canary phase duration
flagger_canary_phase_duration_seconds{namespace="guardrail-studio"}

# Canary status (0=failed, 1=success)
flagger_canary_status{namespace="guardrail-studio"}

# Traffic weight
flagger_canary_weight{namespace="guardrail-studio"}
```

### Grafana Dashboards

**1. Progressive Delivery Dashboard**:
- Traffic weight over time (primary vs canary)
- HTTP error rate comparison
- Latency distribution (p50, p95, p99)
- Rollback events timeline

**2. Circuit Breaker Dashboard**:
- Circuit state (CLOSED/OPEN/HALF_OPEN)
- Fallback activation rate
- Failure rate over time
- Recovery timeline

**3. Autoscaling Dashboard**:
- Pod count over time (backend, Triton)
- CPU/memory utilization
- HPA scaling events
- Request throughput correlation

---

## Troubleshooting

### Canary Stuck "Progressing"

```bash
# Check Flagger logs
kubectl -n istio-system logs deployment/flagger --tail=100

# Verify Prometheus connection
kubectl -n guardrail-studio exec -it deployment/flagger -- \
  curl http://prometheus.monitoring:9090/api/v1/query?query=up

# Check metric templates
kubectl -n guardrail-studio get metrictemplate
kubectl -n guardrail-studio describe metrictemplate request-success-rate
```

### Canary Rollback Loop

**Symptoms**: Canary repeatedly fails and rolls back

**Diagnosis**:
```bash
# View canary events
kubectl -n guardrail-studio describe canary triton-canary

# Check metric violations
kubectl -n guardrail-studio logs deployment/triton-canary --tail=50

# Compare primary vs canary metrics
kubectl -n guardrail-studio exec prometheus -- \
  promtool query instant 'http_req_duration{workload=~"triton.*"}'
```

**Solutions**:
1. **Loosen thresholds** temporarily: Edit MetricTemplate
2. **Extend analysis interval**: Increase `interval` in Canary spec
3. **Add more stepWeights**: Slower traffic increase
4. **Check new model quality**: Validate ONNX parity

### k6 Test Failures

**Threshold Failed: http_req_duration p95 > 50ms**

**Diagnosis**:
```bash
# Check HPA scaling
kubectl get hpa -n guardrail-studio -w

# Check pod resources
kubectl top pods -n guardrail-studio

# Increase backend replicas
kubectl scale deployment backend -n guardrail-studio --replicas=10
```

---

## Best Practices

### Progressive Delivery

1. **Start Conservative**: 5% traffic increments, 1min intervals
2. **Monitor Continuously**: Watch Grafana during rollout
3. **Set Strict SLAs**: p99 < 15ms, error rate < 1%
4. **Enable Webhooks**: Slack/PagerDuty notifications
5. **Test Rollbacks**: Manually trigger failures in staging

### Continuous Learning

1. **Validate Drift**: Confirm PSI > 0.2 before retraining
2. **Hold-Out Validation**: Always evaluate on separate set
3. **Track Experiments**: W&B for every training run
4. **Merge Carefully**: Verify ONNX parity after merge
5. **A/B Test**: Use Flagger canary to validate new model

### Chaos Engineering

1. **Run Regularly**: Weekly chaos tests in staging
2. **Ramp Gradually**: Don't jump to 5000 VUs immediately
3. **Monitor Blast Radius**: Ensure other services unaffected
4. **Automate Recovery**: Verify HPA and circuit breaker work
5. **Document Results**: Track improvements over time

---

## References

- [Flagger Documentation](https://docs.flagger.app/)
- [Istio Traffic Management](https://istio.io/latest/docs/concepts/traffic-management/)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)
- [PEFT Library](https://github.com/huggingface/peft)
- [k6 Documentation](https://k6.io/docs/)
- [Progressive Delivery](https://redmonk.com/jgovernor/2018/08/06/towards-progressive-delivery/)
