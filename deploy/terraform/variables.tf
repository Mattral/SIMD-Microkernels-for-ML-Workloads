# Input Variables for GuardRail Studio Infrastructure

# =============================================================================
# General Configuration
# =============================================================================

variable "project_name" {
  description = "Project name used for resource naming"
  type        = string
  default     = "guardrail-studio"
}

variable "environment" {
  description = "Environment name (dev, staging, production)"
  type        = string
  default     = "production"
  
  validation {
    condition     = contains(["dev", "staging", "production"], var.environment)
    error_message = "Environment must be dev, staging, or production."
  }
}

variable "aws_region" {
  description = "AWS region for resource deployment"
  type        = string
  default     = "us-west-2"
}

# =============================================================================
# Networking Configuration
# =============================================================================

variable "vpc_cidr" {
  description = "CIDR block for VPC"
  type        = string
  default     = "10.0.0.0/16"
}

# =============================================================================
# EKS Configuration
# =============================================================================

variable "eks_cluster_version" {
  description = "Kubernetes version for EKS cluster"
  type        = string
  default     = "1.28"
}

variable "cpu_instance_types" {
  description = "EC2 instance types for CPU node group"
  type        = list(string)
  default     = ["m5.xlarge", "m5.2xlarge"]
}

variable "cpu_node_desired" {
  description = "Desired number of CPU nodes"
  type        = number
  default     = 3
}

variable "cpu_node_min" {
  description = "Minimum number of CPU nodes"
  type        = number
  default     = 2
}

variable "cpu_node_max" {
  description = "Maximum number of CPU nodes"
  type        = number
  default     = 10
}

variable "gpu_instance_types" {
  description = "EC2 instance types for GPU node group"
  type        = list(string)
  default     = ["g4dn.xlarge", "g5.xlarge"]
}

variable "gpu_node_desired" {
  description = "Desired number of GPU nodes"
  type        = number
  default     = 2
}

variable "gpu_node_min" {
  description = "Minimum number of GPU nodes"
  type        = number
  default     = 1
}

variable "gpu_node_max" {
  description = "Maximum number of GPU nodes"
  type        = number
  default     = 5
}

# =============================================================================
# RDS Configuration
# =============================================================================

variable "rds_postgres_version" {
  description = "PostgreSQL engine version"
  type        = string
  default     = "15.4"
}

variable "rds_instance_class" {
  description = "RDS instance class"
  type        = string
  default     = "db.r6g.xlarge"
}

variable "rds_allocated_storage" {
  description = "Initial allocated storage in GB"
  type        = number
  default     = 100
}

variable "rds_max_allocated_storage" {
  description = "Maximum storage autoscaling limit in GB"
  type        = number
  default     = 1000
}

variable "rds_backup_retention" {
  description = "Backup retention period in days"
  type        = number
  default     = 7
}

variable "database_name" {
  description = "Name of the PostgreSQL database"
  type        = string
  default     = "guardrail_studio"
}

variable "database_username" {
  description = "Master username for PostgreSQL"
  type        = string
  default     = "guardrail_admin"
  sensitive   = true
}
