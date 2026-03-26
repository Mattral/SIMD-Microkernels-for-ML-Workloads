# Kubernetes Deployment Guide

## GPU Memory HPA Prerequisites

The `triton-hpa` resource requires the full DCGM → Prometheus → Adapter chain:

### Step 1: Deploy DCGM Exporter

```bash
helm repo add gpu-helm-charts https://nvidia.github.io/dcgm-exporter/helm-charts
helm install dcgm-exporter gpu-helm-charts/dcgm-exporter \
  --namespace monitoring \
  --set serviceMonitor.enabled=true
```

### Step 2: Deploy Prometheus Adapter

```bash
helm install prometheus-adapter prometheus-community/prometheus-adapter \
  --namespace monitoring \
  -f deploy/k8s/prometheus-adapter-values.yaml
```

### Step 3: Verify custom metrics API

```bash
kubectl get --raw "/apis/custom.metrics.k8s.io/v1beta1" | jq .
```

## Notes

- The backend HPA now scales primarily on `http_request_duration_p95_ms`, with CPU as a fallback.
- The Triton HPA uses the external DCGM metric `DCGM_FI_DEV_MEM_COPY_UTIL`.
- Without DCGM exporter and Prometheus adapter, the Triton HPA will not find metrics and will fall back to the CPU metric.

## Troubleshooting

- If the backend HPA reports metric not found, verify that the custom metric is exposed by the application and scraped by Prometheus.
- If the Triton HPA fails, verify the Prometheus Adapter installation and the presence of the DCGM metric in the metrics API.
