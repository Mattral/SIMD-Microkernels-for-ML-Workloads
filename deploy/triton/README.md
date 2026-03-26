# Triton Inference Server Deployment Guide

## Directory Structure

```
deploy/triton/
├── model_repository/
│   └── guardrail_model/
│       ├── config.pbtxt          # Triton model configuration
│       └── 1/
│           └── model.onnx        # ONNX model file (copy from ml_pipelines/artifacts/)
├── docker-compose.yml            # Docker deployment
└── README.md                     # This file
```

## Setup Instructions

### 1. Export Model

First, run the model export pipeline:

```bash
cd /app/ml_pipelines
python export_model.py
```

This generates `artifacts/guardrail_model.onnx`.

### 2. Copy Model to Triton Repository

```bash
cp /app/ml_pipelines/artifacts/guardrail_model.onnx \
   /app/deploy/triton/model_repository/guardrail_model/1/model.onnx
```

### 3. Start Triton Server

#### Option A: Docker (Recommended for Production)

```bash
docker run --rm \
  --name triton-server \
  -p 8000:8000 -p 8001:8001 -p 8002:8002 \
  -v /app/deploy/triton/model_repository:/models \
  nvcr.io/nvidia/tritonserver:23.10-py3 \
  tritonserver --model-repository=/models \
    --strict-model-config=true \
    --log-verbose=1
```

#### Option B: Local Binary

```bash
tritonserver \
  --model-repository=/app/deploy/triton/model_repository \
  --strict-model-config=true \
  --log-verbose=1
```

### 4. Verify Server Health

```bash
curl http://localhost:8000/v2/health/ready
```

Expected response:
```json
{"model_name":"guardrail_model","model_version":"1","ready":true}
```

## Configuration Tuning

### GPU Optimization (Production)

Edit `config.pbtxt`:

```protobuf
instance_group [
  {
    kind: KIND_GPU
    count: 2
    gpus: [ 0 ]
  }
]

optimization {
  execution_accelerators {
    gpu_execution_accelerator [
      {
        name: "tensorrt"
        parameters [
          {
            key: "precision_mode"
            value: "FP16"
          }
        ]
      }
    ]
  }
}
```

### Dynamic Batching Tuning

Adjust for latency vs throughput trade-off:

```protobuf
dynamic_batching {
  preferred_batch_size: [ 1, 2, 4, 8, 16, 32 ]
  max_queue_delay_microseconds: 500  # Lower = better latency
  preserve_ordering: false
}
```

## Performance Benchmarking

### Using perf_analyzer

```bash
perf_analyzer \
  -m guardrail_model \
  -u localhost:8001 \
  --grpc-protocol \
  --input-data test_data.json \
  --concurrency-range 1:8:2 \
  --measurement-interval 10000
```

### Expected Metrics

| Configuration | p99 Latency | Throughput |
|--------------|-------------|------------|
| CPU-only (4 instances) | 8-12ms | 400 req/s |
| GPU FP32 (1 instance) | 3-5ms | 800 req/s |
| GPU FP16 (2 instances) | 2-4ms | 1200+ req/s |

## Monitoring

### Prometheus Metrics

Triton exposes metrics at `http://localhost:8002/metrics`:

- `nv_inference_request_success`
- `nv_inference_queue_duration_us`
- `nv_inference_compute_infer_duration_us`
- `nv_gpu_utilization`

### Key Metrics to Monitor

1. **Queue Time**: Should be < 1ms
2. **Compute Time**: Should be < 8ms (p99)
3. **GPU Utilization**: Target 70-85%
4. **Batch Size Distribution**: Should align with preferred_batch_size

## Troubleshooting

### Model Failed to Load

```bash
# Check server logs
docker logs triton-server

# Validate config
tritonserver --model-repository=/models --strict-model-config=true --log-verbose=3
```

### High Latency

1. Reduce `max_queue_delay_microseconds`
2. Increase instance count
3. Enable GPU acceleration
4. Check for CPU throttling

### OOM Errors

1. Reduce `max_batch_size`
2. Decrease instance count
3. Lower `max_workspace_size_bytes` for TensorRT

## Production Checklist

- [ ] ONNX model validated with parity tests
- [ ] GPU acceleration enabled (TensorRT FP16)
- [ ] Dynamic batching configured
- [ ] Model warmup defined
- [ ] Prometheus monitoring integrated
- [ ] Health checks configured in load balancer
- [ ] Model versioning strategy defined
- [ ] Rollback plan documented

## References

- [Triton Documentation](https://github.com/triton-inference-server/server)
- [ONNX Runtime Optimization](https://onnxruntime.ai/docs/performance/)
- [TensorRT Best Practices](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/)
