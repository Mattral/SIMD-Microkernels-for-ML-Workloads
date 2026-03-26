# RDS PostgreSQL Module

variable "project_name" {}
variable "environment" {}
variable "vpc_id" {}
variable "private_subnet_ids" {}
variable "engine_version" {}
variable "instance_class" {}
variable "allocated_storage" {}
variable "max_allocated_storage" {}
variable "database_name" {}
variable "master_username" {}
variable "multi_az" {}
variable "backup_retention_period" {}
variable "performance_insights_enabled" {}
variable "allowed_security_group_ids" {}

locals {
  db_identifier = "${var.project_name}-${var.environment}"
}

# DB Subnet Group
resource "aws_db_subnet_group" "main" {
  name       = "${local.db_identifier}-subnet-group"
  subnet_ids = var.private_subnet_ids
}

# Security Group
resource "aws_security_group" "rds" {
  name        = "${local.db_identifier}-rds-sg"
  description = "Security group for RDS PostgreSQL"
  vpc_id      = var.vpc_id
  
  ingress {
    description     = "PostgreSQL from EKS"
    from_port       = 5432
    to_port         = 5432
    protocol        = "tcp"
    security_groups = var.allowed_security_group_ids
  }
  
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

# Random password
resource "random_password" "master" {
  length  = 32
  special = true
}

# RDS Instance
resource "aws_db_instance" "main" {
  identifier     = local.db_identifier
  engine         = "postgres"
  engine_version = var.engine_version
  instance_class = var.instance_class
  
  allocated_storage     = var.allocated_storage
  max_allocated_storage = var.max_allocated_storage
  storage_type          = "gp3"
  storage_encrypted     = true
  
  db_name  = var.database_name
  username = var.master_username
  password = random_password.master.result
  
  multi_az               = var.multi_az
  db_subnet_group_name   = aws_db_subnet_group.main.name
  vpc_security_group_ids = [aws_security_group.rds.id]
  
  backup_retention_period = var.backup_retention_period
  backup_window          = "03:00-04:00"
  maintenance_window     = "sun:04:00-sun:05:00"
  
  performance_insights_enabled = var.performance_insights_enabled
  
  enabled_cloudwatch_logs_exports = ["postgresql", "upgrade"]
  
  skip_final_snapshot = var.environment != "production"
  
  # Production-tuned parameters
  parameter_group_name = aws_db_parameter_group.main.name
}

# Parameter Group
resource "aws_db_parameter_group" "main" {
  name   = "${local.db_identifier}-params"
  family = "postgres15"
  
  parameter {
    name  = "shared_buffers"
    value = "8388608"  # 8GB
  }
  
  parameter {
    name  = "max_connections"
    value = "200"
  }
  
  parameter {
    name  = "work_mem"
    value = "65536"  # 64MB
  }
  
  parameter {
    name  = "maintenance_work_mem"
    value = "2097152"  # 2GB
  }
  
  parameter {
    name  = "effective_cache_size"
    value = "25165824"  # 24GB
  }
}

output "endpoint" {
  value = aws_db_instance.main.endpoint
}

output "database_name" {
  value = aws_db_instance.main.db_name
}

output "master_password" {
  value     = random_password.master.result
  sensitive = true
}
