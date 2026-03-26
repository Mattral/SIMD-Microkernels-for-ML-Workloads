# Flagger Installation and Setup Guide
# ====================================

## Prerequisites

1. Kubernetes cluster with Istio installed
2. Prometheus for metrics
3. Flagger controller

## Installation

### 1. Install Istio

```bash
# Download Istio
curl -L https://istio.io/downloadIstio | sh -
cd istio-*
export PATH=$PWD/bin:$PATH

# Install Istio with default profile
istioctl install --set profile=default -y

# Enable sidecar injection for guardrail-studio namespace
kubectl label namespace guardrail-studio istio-injection=enabled
```

### 2. Install Flagger

```bash
# Add Flagger Helm repository
helm repo add flagger https://flagger.app
helm repo update

# Install Flagger with Istio support
helm upgrade -i flagger flagger/flagger \
  --namespace istio-system \
  --set meshProvider=istio \
  --set metricsServer=http://prometheus.monitoring:9090 \
  --set slack.url=https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK \
  --set slack.channel=guardrail-alerts \
  --set slack.user=flagger

# Install Flagger Grafana dashboards
helm upgrade -i flagger-grafana flagger/grafana \
  --namespace istio-system \
  --set url=http://prometheus.monitoring:9090
```

### 3. Install Prometheus (if not already installed)

```bash
# Install Prometheus Operator
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm upgrade -i prometheus prometheus-community/kube-prometheus-stack \
  --namespace monitoring --create-namespace
```

## Deploy Canary Configuration

```bash
# Apply all Flagger resources
kubectl apply -f /app/deploy/k8s/istio_flagger/canary-triton.yaml

# Verify Flagger controller
kubectl -n istio-system logs deployment/flagger -f

# Check canary status
kubectl -n guardrail-studio get canary triton-canary
```

## Trigger Canary Deployment

To trigger a canary deployment, update the Triton deployment with a new model:

```bash
# Update Triton deployment with new model version
kubectl -n guardrail-studio set image deployment/triton \
  triton=nvcr.io/nvidia/tritonserver:23.11-py3

# Or update via annotation
kubectl -n guardrail-studio annotate deployment/triton \
  model-version="v2" --overwrite
```

## Monitor Canary Progress

```bash
# Watch canary progression
watch kubectl -n guardrail-studio get canary triton-canary

# View detailed canary events
kubectl -n guardrail-studio describe canary triton-canary

# Check traffic distribution
kubectl -n guardrail-studio get vs triton-vs -o yaml

# View Prometheus metrics
kubectl -n monitoring port-forward svc/prometheus-kube-prometheus-prometheus 9090:9090
# Open: http://localhost:9090
```

## Rollback Scenarios

Flagger automatically rolls back if:
1. HTTP 500 error rate > 1%
2. p99 latency > 15ms
3. Request throughput drops below 10 req/s
4. Any webhook returns failure

### Manual Rollback

```bash
# Rollback to previous version
kubectl -n guardrail-studio rollout undo deployment/triton

# Flagger will detect and abort canary
```

## Traffic Shifting Timeline

For a successful deployment:

```
Time  | Primary | Canary | Status
------|---------|--------|------------------
0m    | 100%    | 0%     | Initialized
1m    | 95%     | 5%     | Analyzing (Gate 1)
2m    | 90%     | 10%    | Analyzing (Gate 2)
3m    | 85%     | 15%    | Analyzing (Gate 3)
...   | ...     | ...    | ...
19m   | 5%      | 95%    | Analyzing (Gate 19)
20m   | 0%      | 100%   | Promoted!
```

Total deployment time: ~20 minutes (with all gates passing)

## Verification

```bash
# Test primary endpoint
curl http://triton-service-primary.guardrail-studio:8001/v2/health/ready

# Test canary endpoint
curl http://triton-service-canary.guardrail-studio:8001/v2/health/ready

# Check Istio proxy stats
istioctl proxy-status

# View traffic distribution
istioctl dashboard envoy deployment/triton
```

## Troubleshooting

### Canary stuck in "Progressing" state

```bash
# Check Flagger logs
kubectl -n istio-system logs deployment/flagger --tail=100

# Verify Prometheus metrics
kubectl -n guardrail-studio describe metrictemplate

# Check canary events
kubectl -n guardrail-studio get events --field-selector involvedObject.name=triton-canary
```

### Metrics not available

```bash
# Verify ServiceMonitor
kubectl -n guardrail-studio get servicemonitor

# Check Prometheus targets
kubectl -n monitoring port-forward svc/prometheus-kube-prometheus-prometheus 9090:9090
# Navigate to: http://localhost:9090/targets
```

## Cleanup

```bash
# Remove canary
kubectl -n guardrail-studio delete canary triton-canary

# Uninstall Flagger
helm -n istio-system delete flagger

# Uninstall Istio
istioctl uninstall --purge -y
```

## References

- [Flagger Documentation](https://docs.flagger.app/)
- [Istio Traffic Management](https://istio.io/latest/docs/concepts/traffic-management/)
- [Progressive Delivery Best Practices](https://flagger.app/blog/2019/12/progressive-delivery/)
