"""
Production Airflow DAG: Drift Detection & Model Retraining Pipeline
===================================================================

This DAG orchestrates the daily analytical operations for GuardRail Studio:
1. Extract firewall logs from partitioned PostgreSQL tables
2. Execute distributed drift analysis (PSI, Wasserstein distance)
3. Conditional retraining trigger based on drift detection
4. Update Qdrant threat vector database with fresh patterns

Schedule: Daily @ 02:00 UTC
Catchup: Disabled (only process current data)
Concurrency: 1 (sequential execution)
Retries: 3 with exponential backoff

Engineering Patterns:
- TaskFlow API (@task decorator) for Python callables
- XCom for inter-task data passing
- Jinja templating for dynamic parameters
- BranchPythonOperator for conditional execution
- Slack/PagerDuty alerts on failure

Author: Principal Data Platform Engineer
"""

from datetime import datetime, timedelta
from typing import Dict, Any, List
import os
import sys
import asyncio

from airflow import DAG
from airflow.decorators import task
from airflow.operators.python import BranchPythonOperator
from airflow.operators.bash import BashOperator
from airflow.operators.dummy import DummyOperator
from airflow.utils.dates import days_ago
from airflow.models import Variable
from airflow.utils.trigger_rule import TriggerRule

# Add project path for imports
sys.path.insert(0, '/app/backend')


# =============================================================================
# DAG Configuration
# =============================================================================

DAG_ID = "drift_detection_retraining_pipeline"

default_args = {
    'owner': 'mlops-team',
    'depends_on_past': False,
    'email': ['alerts@guardrailstudio.ai'],
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 3,
    'retry_delay': timedelta(minutes=5),
    'retry_exponential_backoff': True,
    'max_retry_delay': timedelta(minutes=30),
    'execution_timeout': timedelta(hours=2),
}

# DAG documentation
dag_doc_md = """
# Drift Detection & Retraining Pipeline

## Overview
This pipeline monitors production ML model performance and triggers
retraining when statistical drift is detected.

## Pipeline Stages

### 1. Extract Logs (extract_firewall_logs)
- Queries last 24h from partitioned `firewall_requests` table
- Extracts confidence scores, latencies, token counts
- Stores results in XCom for downstream tasks

### 2. Drift Analysis (run_drift_detection)
- Computes PSI and Wasserstein distance
- Compares against baseline validation distribution
- Logs metrics to W&B with automatic alerting
- Returns drift_detected boolean

### 3. Conditional Branch (check_drift_threshold)
- Routes to retraining if drift_detected == True
- Routes to monitoring if drift_detected == False

### 4A. Trigger Retraining (trigger_model_retraining)
- Initiates model fine-tuning job
- Updates model registry
- Triggers deployment pipeline

### 4B. Update Threat Index (update_qdrant_vectors)
- Extracts new adversarial patterns from production logs
- Computes embeddings via Triton inference
- Updates Qdrant nearest-neighbor index

## Monitoring
- W&B: https://wandb.ai/guardrail-studio/drift-validation
- Slack: #guardrail-alerts
- PagerDuty: Critical drift events

## SLA
- DAG execution: < 30 minutes (p95)
- Drift detection: < 10 minutes
- Vector update: < 5 minutes
"""


# =============================================================================
# DAG Definition
# =============================================================================

dag = DAG(
    dag_id=DAG_ID,
    default_args=default_args,
    description='Daily drift detection and model retraining orchestration',
    doc_md=dag_doc_md,
    schedule_interval='0 2 * * *',  # Daily at 02:00 UTC
    start_date=days_ago(1),
    catchup=False,
    max_active_runs=1,
    tags=['ml-ops', 'drift-detection', 'production'],
)


# =============================================================================
# Task Definitions
# =============================================================================

@task(task_id='extract_firewall_logs', dag=dag)
def extract_firewall_logs(hours: int = 24, **context) -> Dict[str, Any]:
    """Extract firewall logs from PostgreSQL for drift analysis.
    
    This task queries the partitioned firewall_requests table and
    extracts the necessary fields for statistical analysis.
    
    Args:
        hours: Number of hours of logs to extract
        
    Returns:
        Dictionary with extracted log statistics
    """
    import asyncio
    from sqlalchemy import select, func
    from src.db.production_manager import production_db
    from src.db.postgres import FirewallRequest
    from src.core.logging import setup_logging, get_logger
    
    setup_logging("INFO")
    logger = get_logger(__name__)
    
    async def _extract():
        # Initialize database
        await production_db.initialize()
        
        # Calculate time range
        from datetime import datetime, timedelta, timezone
        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(hours=hours)
        
        # Query logs
        async with production_db._session_maker() as session:
            # Count total requests
            count_stmt = select(func.count()).select_from(FirewallRequest).where(
                FirewallRequest.timestamp >= start_time,
                FirewallRequest.timestamp <= end_time
            )
            result = await session.execute(count_stmt)
            total_requests = result.scalar() or 0
            
            # Get aggregated statistics
            stats_stmt = select(
                func.count(FirewallRequest.id).label('count'),
                func.avg(FirewallRequest.confidence_score).label('avg_confidence'),
                func.avg(FirewallRequest.latency_ms).label('avg_latency'),
                func.sum(FirewallRequest.input_tokens).label('total_tokens'),
                func.sum(func.cast(FirewallRequest.blocked, func.Integer())).label('blocked_count'),
                func.sum(func.cast(FirewallRequest.threat_detected, func.Integer())).label('threat_count')\n            ).where(\n                FirewallRequest.timestamp >= start_time,\n                FirewallRequest.timestamp <= end_time\n            )\n            \n            result = await session.execute(stats_stmt)\n            row = result.first()\n        \n        await production_db.close()\n        \n        stats = {\n            'total_requests': row.count or 0,\n            'avg_confidence': float(row.avg_confidence or 0.0),\n            'avg_latency_ms': float(row.avg_latency or 0.0),\n            'total_tokens': int(row.total_tokens or 0),\n            'blocked_count': int(row.blocked_count or 0),\n            'threat_count': int(row.threat_count or 0),\n            'extraction_time': datetime.now(timezone.utc).isoformat(),\n            'time_range_hours': hours\n        }\n        \n        logger.info(f\"Extracted {stats['total_requests']} requests from last {hours}h\")\n        return stats
    \n    # Run async extraction\n    return asyncio.run(_extract())


@task(task_id='run_drift_detection', dag=dag)
def run_drift_detection(log_stats: Dict[str, Any], **context) -> Dict[str, Any]:
    """Execute distributed drift detection analysis.
    
    This task runs the drift detection engine which computes:
    - Population Stability Index (PSI)
    - Wasserstein Distance
    - Distribution comparison metrics
    
    Args:
        log_stats: Statistics from extract_firewall_logs task
        
    Returns:
        Drift analysis results including drift_detected boolean
    """
    import asyncio
    from src.analytics.drift_detector import DriftDetectionEngine
    from src.core.logging import setup_logging, get_logger
    
    setup_logging("INFO")
    logger = get_logger(__name__)
    
    logger.info(f"Starting drift detection on {log_stats['total_requests']} requests\")
    \n    # Initialize drift detection engine\n    engine = DriftDetectionEngine(\n        psi_threshold=0.2,\n        wasserstein_threshold=0.15,\n        enable_wandb=True,\n        n_workers=4\n    )\n    \n    # Run drift analysis\n    result = asyncio.run(engine.run_drift_analysis(hours=24))\n    \n    logger.info(f\"Drift detection complete: drift_detected={result.get('drift_detected', False)}\")\n    \n    return result


def check_drift_threshold(**context) -> str:\n    \"\"\"Branch task: Route based on drift detection result.
    \n    Returns:\n        Next task_id to execute\n    \"\"\"\n    ti = context['ti']\n    drift_result = ti.xcom_pull(task_ids='run_drift_detection')\n    \n    drift_detected = drift_result.get('drift_detected', False)\n    \n    if drift_detected:\n        print(\"⚠️ DRIFT DETECTED - Routing to retraining pipeline\")\n        return 'trigger_model_retraining'\n    else:\n        print(\"✓ No drift detected - Routing to monitoring\")\n        return 'update_qdrant_vectors'\n\n\n@task(task_id='trigger_model_retraining', dag=dag)\ndef trigger_model_retraining(drift_result: Dict[str, Any], **context):\n    \"\"\"Trigger model retraining job when drift is detected.
    \n    In production, this would:\n    1. Pull latest training data from warehouse\n    2. Fine-tune model on Kubernetes GPU cluster\n    3. Update model registry with new version\n    4. Trigger canary deployment\n    \n    For Phase 3, we mock this operation.\n    \"\"\"\n    from src.core.logging import get_logger\n    logger = get_logger(__name__)\n    \n    metrics = drift_result.get('metrics', {})\n    psi = metrics.get('psi_confidence', 0)\n    \n    logger.critical(\n        \"MODEL RETRAINING TRIGGERED\",\n        extra={\n            'psi_confidence': psi,\n            'trigger_reason': 'drift_threshold_exceeded',\n            'drift_metrics': metrics\n        }\n    )\n    \n    # Mock retraining job\n    print(\"=\"*80)\n    print(\"🔥 RETRAINING JOB TRIGGERED\")\n    print(f\"   PSI Confidence: {psi:.4f}\")\n    print(f\"   Wasserstein: {metrics.get('wasserstein_confidence', 0):.4f}\")\n    print(f\"   Requests Analyzed: {metrics.get('num_requests', 0)}\")\n    print(\"=\"*80)\n    \n    # In production: Submit Kubernetes Job or trigger ML pipeline\n    # kubectl apply -f k8s/jobs/model-training.yaml\n    \n    return {\n        'status': 'triggered',\n        'job_id': f\"retrain_{context['execution_date'].strftime('%Y%m%d_%H%M%S')}\",\n        'metrics': metrics\n    }


@task(task_id='update_qdrant_vectors', dag=dag)\ndef update_qdrant_vectors(**context):\n    \"\"\"Update Qdrant vector database with fresh adversarial patterns.
    \n    This task:\n    1. Extracts unique adversarial patterns from recent logs\n    2. Computes embeddings via Triton inference\n    3. Updates Qdrant nearest-neighbor index\n    4. Prunes old patterns based on LRU policy\n    \"\"\"\n    from src.core.logging import get_logger\n    from src.db.qdrant import qdrant_manager\n    \n    logger = get_logger(__name__)\n    \n    # Initialize Qdrant\n    qdrant_manager.initialize()\n    \n    # Mock vector update (in production: extract patterns, compute embeddings, upsert)\n    logger.info(\"Updating Qdrant adversarial pattern index\")\n    \n    # Simulate discovering new patterns\n    new_patterns_count = 15\n    total_vectors = 1500\n    \n    print(\"=\"*80)\n    print(\"✓ QDRANT VECTOR UPDATE COMPLETE\")\n    print(f\"   New Patterns: {new_patterns_count}\")\n    print(f\"   Total Vectors: {total_vectors}\")\n    print(\"   Index Status: Healthy\")\n    print(\"=\"*80)\n    \n    return {\n        'new_patterns': new_patterns_count,\n        'total_vectors': total_vectors,\n        'update_time': context['execution_date'].isoformat()\n    }


# =============================================================================
# DAG Task Flow
# =============================================================================

with dag:
    # Start marker
    start = DummyOperator(task_id='start')
    
    # Extract logs from PostgreSQL
    log_stats = extract_firewall_logs(hours=24)
    
    # Run drift detection analysis
    drift_result = run_drift_detection(log_stats=log_stats)
    
    # Branch based on drift detection
    drift_branch = BranchPythonOperator(
        task_id='check_drift_threshold',
        python_callable=check_drift_threshold,
        provide_context=True
    )
    
    # Path A: Drift detected → Trigger retraining
    retraining_job = trigger_model_retraining(drift_result=drift_result)
    
    # Path B: No drift → Update vectors only
    vector_update = update_qdrant_vectors()
    
    # Converge paths
    end = DummyOperator(
        task_id='end',
        trigger_rule=TriggerRule.NONE_FAILED_MIN_ONE_SUCCESS\n    )
    
    # Define flow
    start >> log_stats >> drift_result >> drift_branch
    drift_branch >> retraining_job >> end
    drift_branch >> vector_update >> end


# =============================================================================
# DAG Validation
# =============================================================================

if __name__ == \"__main__\":
    dag.test()
