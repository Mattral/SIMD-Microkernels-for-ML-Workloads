# Phase 3 & 4: Distributed Analytics + Cloud Infrastructure

## Overview

Phases 3 and 4 complete the GuardRail Studio production architecture with:
- Distributed drift detection using Dask
- Production PostgreSQL with time-series partitioning
- Apache Airflow orchestration for daily analytics
- Terraform IaC for AWS (EKS, RDS, VPC)
- Production Kubernetes manifests with autoscaling

---

## Phase 3: Distributed Analytics & Orchestration

### 1. Drift Detection Engine (`src/analytics/drift_detector.py`)

**Purpose**: Monitor production ML performance and detect algorithmic drift.

**Features**:
- Dask distributed processing for out-of-core operations
- Population Stability Index (PSI) computation
- Wasserstein Distance (Earth Mover's Distance)
- Automatic W&B alerting on drift detection
- Async database integration

**Statistical Metrics**:

**PSI (Population Stability Index)**:
```
PSI = Σ (actual% - expected%) × ln(actual% / expected%)
```
- PSI < 0.1: No drift
- 0.1 < PSI < 0.2: Moderate drift
- PSI > 0.2: Significant drift (retraining recommended)

**Wasserstein Distance**:
- Measures minimum cost to transform one distribution to another
- Lower values = more similar distributions
- Threshold: 0.15

**Usage**:
```bash
cd /app/backend
python src/analytics/drift_detector.py
```

**Expected Output**:
```
================================================================================
DRIFT ANALYSIS SUMMARY
================================================================================
Status: success
Drift Detected: True
PSI (Confidence): 0.2345
PSI (Latency): 0.1234
Wasserstein (Confidence): 0.1678
Requests Analyzed: 15234
================================================================================
```

---

### 2. Production PostgreSQL Schema

**File**: `src/db/migrations/001_initial_schema.sql`

**Key Features**:
- Time-series table partitioning (RANGE by timestamp)
- Strategic indexing (B-tree, GIN, partial indexes)
- Automated partition management
- 90-day retention policy

**Table Structure**:

```sql
firewall_requests (partitioned)
├── Partitions by month: firewall_requests_2024_01, _02, ...
├── Indexes:
│   ├── idx_firewall_timestamp (B-tree DESC)
│   ├── idx_firewall_blocked (partial, WHERE blocked = true)
│   ├── idx_firewall_input_text_trgm (GIN for full-text search)
│   └── Composite indexes for common query patterns
└── Performance tuning:
    ├── shared_buffers = 8GB
    ├── max_connections = 200
    └── parallel_workers = 4
```

**Deployment**:
```bash
# Connect to RDS instance
psql -h <rds-endpoint> -U guardrail_admin -d guardrail_studio

# Run migration
\i /app/backend/src/db/migrations/001_initial_schema.sql

# Verify partitions
SELECT tablename FROM pg_tables WHERE tablename LIKE 'firewall_requests%';
```

**Maintenance Functions**:
```sql
-- Create future partitions (run monthly)
SELECT create_future_firewall_partitions(3);

-- Drop old partitions (run weekly)
SELECT drop_old_firewall_partitions(90);
```

---

### 3. Airflow Orchestration DAG

**File**: `deploy/airflow/dags/drift_retrain_dag.py`

**Schedule**: Daily @ 02:00 UTC

**Pipeline Flow**:
```
START
  ↓
extract_firewall_logs (24h)
  ↓
run_drift_detection (Dask + PSI/Wasserstein)
  ↓
check_drift_threshold (Branch)
  ├─→ [Drift Detected] → trigger_model_retraining
  └─→ [No Drift] → update_qdrant_vectors
  ↓
END
```

**Task Descriptions**:

1. **extract_firewall_logs**: Query partitioned PostgreSQL table for last 24h
2. **run_drift_detection**: Compute drift metrics using distributed Dask
3. **check_drift_threshold**: Branch based on PSI > 0.2
4. **trigger_model_retraining**: Initiate retraining job (mocked in Phase 3)
5. **update_qdrant_vectors**: Refresh adversarial pattern index

**Airflow Setup**:
```bash
# Set Airflow home
export AIRFLOW_HOME=/app/deploy/airflow

# Initialize database
airflow db init

# Create admin user
airflow users create \
  --username admin \
  --password admin \
  --firstname Admin \
  --lastname User \
  --role Admin \
  --email admin@guardrailstudio.ai

# Copy DAG
cp /app/deploy/airflow/dags/drift_retrain_dag.py $AIRFLOW_HOME/dags/

# Start scheduler and webserver
airflow scheduler &
airflow webserver --port 8080
```

**Access UI**: http://localhost:8080

---

## Phase 4: Cloud-Native Infrastructure

### 1. Terraform Infrastructure as Code

**Directory**: `/app/deploy/terraform/`

**Architecture**:
```
AWS Account
├── VPC (10.0.0.0/16)
│   ├── Public Subnets (3 AZs)
│   ├── Private Subnets (3 AZs)
│   ├── Internet Gateway
│   └── NAT Gateways (1 or 3, configurable)
├── EKS Cluster (k8s 1.28)
│   ├── CPU Node Group (m5.xlarge, 2-10 nodes)
│   └── GPU Node Group (g4dn.xlarge, 1-5 nodes)
└── RDS PostgreSQL (15.4)
    ├── Multi-AZ (production)
    ├── db.r6g.xlarge
    └── 100-1000GB auto-scaling storage
```

**Terraform Modules**:
- `modules/networking/`: VPC, subnets, NAT, IGW
- `modules/eks/`: EKS cluster, node groups, IAM roles
- `modules/rds/`: PostgreSQL instance, security groups, parameter groups

**Deployment**:
```bash
cd /app/deploy/terraform

# Initialize Terraform
terraform init

# Review plan
terraform plan \
  -var="environment=production" \
  -var="aws_region=us-west-2"

# Apply infrastructure
terraform apply \
  -var="environment=production" \
  -var="aws_region=us-west-2" \
  -auto-approve

# Get outputs
terraform output eks_cluster_name
terraform output rds_endpoint
terraform output configure_kubectl
```

**Configure kubectl**:
```bash
aws eks update-kubeconfig \
  --region us-west-2 \
  --name guardrail-studio-production
```

**Cost Estimation** (Monthly, us-west-2):
- EKS Control Plane: $73
- CPU Nodes (3x m5.xlarge): ~$450
- GPU Nodes (2x g4dn.xlarge): ~$700
- RDS (db.r6g.xlarge Multi-AZ): ~$600
- Data Transfer & Storage: ~$100
- **Total**: ~$1,900/month

---

### 2. Kubernetes Production Manifests

**File**: `/app/deploy/k8s/production-stack.yaml`

**Components**:

**Backend Deployment**:
- Replicas: 3 (min 2, max 10 with HPA)
- Resources: 500m-2000m CPU, 512Mi-2Gi memory
- Health probes: liveness, readiness, startup
- Node selector: CPU nodes (role=cpu)
- Anti-affinity: spread across nodes

**Triton Deployment**:
- Replicas: 2 (min 1, max 5 with HPA)
- Resources: 2-4 CPU, 4-8Gi memory, 1 GPU
- GPU: nvidia.com/gpu=1 (via device plugin)
- Node selector: GPU nodes (role=gpu)
- Toleration: nvidia.com/gpu taint

**Services**:
- `backend-service`: ClusterIP (internal)
- `backend-external`: LoadBalancer (AWS NLB)
- `triton-service`: ClusterIP (internal gRPC)

**Horizontal Pod Autoscalers**:
- Backend: 70% CPU, 80% memory → 2-10 pods
- Triton: 70% CPU, 75% memory → 1-5 pods

**Deployment**:
```bash
# Create namespace
kubectl apply -f /app/deploy/k8s/production-stack.yaml

# Verify deployments
kubectl get deployments -n guardrail-studio

# Check pods
kubectl get pods -n guardrail-studio

# Get external endpoint
kubectl get svc backend-external -n guardrail-studio
```

**Update ConfigMap** (RDS endpoint):
```bash
# Edit database-credentials secret
kubectl edit secret database-credentials -n guardrail-studio

# Update POSTGRES_URL with actual RDS endpoint from Terraform
```

**Deploy Model to Triton**:
```bash
# Copy ONNX model to pod
kubectl cp /app/ml_pipelines/artifacts/guardrail_model.onnx \
  guardrail-studio/triton-0:/models/guardrail_model/1/model.onnx

# Verify model loaded
kubectl exec -n guardrail-studio triton-0 -- \
  curl localhost:8000/v2/models/guardrail_model
```

---

## Integration & Testing

### 1. End-to-End Deployment

**Prerequisites**:
- AWS CLI configured
- Terraform installed
- kubectl installed
- ONNX model exported

**Step-by-Step**:
```bash
# 1. Deploy infrastructure
cd /app/deploy/terraform
terraform apply -auto-approve

# 2. Configure kubectl
aws eks update-kubeconfig --name <cluster-name>

# 3. Deploy Kubernetes resources
kubectl apply -f /app/deploy/k8s/production-stack.yaml

# 4. Update secrets with real credentials
# Edit database-credentials, wandb-credentials

# 5. Deploy model to Triton PVC
# Upload model.onnx to persistent volume

# 6. Verify health
kubectl get pods -n guardrail-studio
kubectl logs -n guardrail-studio <backend-pod>
kubectl logs -n guardrail-studio <triton-pod>

# 7. Test external endpoint
BACKEND_URL=$(kubectl get svc backend-external -n guardrail-studio -o jsonpath='{.status.loadBalancer.ingress[0].hostname}')
curl http://$BACKEND_URL/api/health/
```

### 2. Drift Detection Test

```bash
# Run drift analysis (from Airflow or manually)
cd /app/backend
python src/analytics/drift_detector.py

# Expected: PSI and Wasserstein metrics logged
# If drift detected: W&B alert triggered
```

### 3. Autoscaling Test

```bash
# Generate load on backend
kubectl run -n guardrail-studio load-generator \
  --image=busybox \
  --restart=Never \
  -- /bin/sh -c "while true; do wget -q -O- http://backend-service:8001/api/health/; done"

# Watch HPA scale
kubectl get hpa -n guardrail-studio -w

# Expected: Backend pods scale from 2 → 10 based on CPU/memory
```

---

## Monitoring & Observability

### Prometheus Metrics

Both backend and Triton expose Prometheus metrics:
- Backend: `http://<pod>:8001/metrics`
- Triton: `http://<pod>:8002/metrics`

**Key Metrics**:
- `http_requests_total`: Total HTTP requests
- `http_request_duration_seconds`: Request latency histogram
- `nv_inference_request_success`: Triton inference success count
- `nv_inference_compute_infer_duration_us`: Triton compute duration

### Grafana Dashboards

Create dashboards for:
1. **Request Throughput**: req/sec, blocked rate, threat rate
2. **Latency Distribution**: p50, p95, p99 histograms
3. **Drift Metrics**: PSI, Wasserstein trends over time
4. **GPU Utilization**: nvidia_gpu_utilization, memory usage
5. **Database Performance**: Connection pool, query duration

### Alerts

**Critical**:
- Drift detected (PSI > 0.2)
- p99 latency > 15ms
- GPU memory > 90%
- Pod CrashLoopBackOff

**Warning**:
- Drift approaching threshold (PSI > 0.15)
- p99 latency > 12ms
- Database connection pool > 80%

---

## Cost Optimization

1. **Use Spot Instances** for GPU nodes (70% cost reduction)
2. **Enable Cluster Autoscaler** to scale down idle nodes
3. **RDS Read Replicas** for analytics queries (reduce primary load)
4. **S3 Lifecycle Policies** for old partition archives
5. **Reserved Instances** for steady-state workloads (40% savings)

---

## Security Hardening

1. **Network Policies**: Restrict pod-to-pod communication
2. **Pod Security Standards**: Enforce restricted profile
3. **Secrets Management**: Migrate to AWS Secrets Manager
4. **RBAC**: Principle of least privilege
5. **WAF**: AWS WAF on LoadBalancer
6. **Encryption**: At-rest (RDS, EBS) and in-transit (TLS)

---

## Disaster Recovery

**Backup Strategy**:
- RDS automated backups (7-day retention)
- Point-in-time recovery (PITR) enabled
- Cross-region snapshot replication

**RTO/RPO Targets**:
- Recovery Time Objective (RTO): < 1 hour
- Recovery Point Objective (RPO): < 15 minutes

**Runbook**:
```bash
# 1. Restore RDS from snapshot
aws rds restore-db-instance-from-db-snapshot \
  --db-instance-identifier guardrail-studio-restore \
  --db-snapshot-identifier <snapshot-id>

# 2. Update K8s secret with new endpoint
kubectl edit secret database-credentials -n guardrail-studio

# 3. Rolling restart
kubectl rollout restart deployment backend -n guardrail-studio
```

---

## References

- [Terraform AWS Provider](https://registry.terraform.io/providers/hashicorp/aws/latest/docs)
- [Amazon EKS Best Practices](https://aws.github.io/aws-eks-best-practices/)
- [PostgreSQL Partitioning](https://www.postgresql.org/docs/15/ddl-partitioning.html)
- [Apache Airflow Documentation](https://airflow.apache.org/docs/)
- [NVIDIA Device Plugin](https://github.com/NVIDIA/k8s-device-plugin)
