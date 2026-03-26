# GuardRail Studio - Terraform Infrastructure as Code
# ==================================================
#
# This Terraform configuration provisions a production-grade AWS infrastructure
# for GuardRail Studio with:
# - Amazon EKS cluster with CPU and GPU node groups
# - RDS PostgreSQL with Multi-AZ deployment
# - VPC with public/private subnets
# - Security groups and IAM roles
# - Auto-scaling configurations
#
# Author: Principal Infrastructure Engineer

terraform {
  required_version = ">= 1.6.0"
  
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.23"
    }
  }
  
  # Backend configuration for remote state
  # Uncomment and configure for production
  # backend "s3" {
  #   bucket         = "guardrail-studio-terraform-state"
  #   key            = "production/terraform.tfstate"
  #   region         = "us-west-2"
  #   encrypt        = true
  #   dynamodb_table = "terraform-state-lock"
  # }
}

provider "aws" {
  region = var.aws_region
  
  default_tags {
    tags = {
      Project     = "GuardRail Studio"
      Environment = var.environment
      ManagedBy   = "Terraform"
      Team        = "MLOps"
    }
  }
}

# Data sources
data "aws_availability_zones" "available" {
  state = "available"
}

data "aws_caller_identity" "current" {}

# =============================================================================
# Networking Module
# =============================================================================

module "networking" {
  source = "./modules/networking"
  
  project_name         = var.project_name
  environment          = var.environment
  vpc_cidr             = var.vpc_cidr
  availability_zones   = data.aws_availability_zones.available.names
  enable_nat_gateway   = true
  single_nat_gateway   = var.environment != "production"
}

# =============================================================================
# EKS Cluster Module
# =============================================================================

module "eks" {
  source = "./modules/eks"
  
  project_name       = var.project_name
  environment        = var.environment
  cluster_version    = var.eks_cluster_version
  
  vpc_id             = module.networking.vpc_id
  private_subnet_ids = module.networking.private_subnet_ids
  
  # CPU Node Group Configuration
  cpu_instance_types = var.cpu_instance_types
  cpu_desired_size   = var.cpu_node_desired
  cpu_min_size       = var.cpu_node_min
  cpu_max_size       = var.cpu_node_max
  
  # GPU Node Group Configuration
  gpu_instance_types = var.gpu_instance_types
  gpu_desired_size   = var.gpu_node_desired
  gpu_min_size       = var.gpu_node_min
  gpu_max_size       = var.gpu_node_max
  
  enable_cluster_autoscaler = true
}

# =============================================================================
# RDS PostgreSQL Module
# =============================================================================

module "rds" {
  source = "./modules/rds"
  
  project_name        = var.project_name
  environment         = var.environment
  
  vpc_id              = module.networking.vpc_id
  private_subnet_ids  = module.networking.private_subnet_ids
  
  engine_version      = var.rds_postgres_version
  instance_class      = var.rds_instance_class
  allocated_storage   = var.rds_allocated_storage
  max_allocated_storage = var.rds_max_allocated_storage
  
  database_name       = var.database_name
  master_username     = var.database_username
  
  multi_az            = var.environment == "production"
  backup_retention_period = var.rds_backup_retention
  
  # Performance Insights
  performance_insights_enabled = true
  
  # Allow access from EKS cluster
  allowed_security_group_ids = [module.eks.cluster_security_group_id]
}

# =============================================================================
# Outputs
# =============================================================================

output "vpc_id" {
  description = "VPC ID"
  value       = module.networking.vpc_id
}

output "eks_cluster_name" {
  description = "EKS cluster name"
  value       = module.eks.cluster_name
}

output "eks_cluster_endpoint" {
  description = "EKS cluster endpoint"
  value       = module.eks.cluster_endpoint
  sensitive   = true
}

output "rds_endpoint" {
  description = "RDS endpoint"
  value       = module.rds.endpoint
  sensitive   = true
}

output "rds_database_name" {
  description = "RDS database name"
  value       = module.rds.database_name
}

output "configure_kubectl" {
  description = "Command to configure kubectl"
  value       = "aws eks update-kubeconfig --region ${var.aws_region} --name ${module.eks.cluster_name}"
}
