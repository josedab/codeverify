# CodeVerify Terraform Configuration
# AWS Infrastructure

terraform {
  required_version = ">= 1.0"
  
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.0"
    }
  }
  
  backend "s3" {
    bucket         = "codeverify-terraform-state"
    key            = "production/terraform.tfstate"
    region         = "us-east-1"
    encrypt        = true
    dynamodb_table = "codeverify-terraform-locks"
  }
}

provider "aws" {
  region = var.aws_region
  
  default_tags {
    tags = {
      Project     = "CodeVerify"
      Environment = var.environment
      ManagedBy   = "Terraform"
    }
  }
}

# Variables
variable "aws_region" {
  description = "AWS region"
  type        = string
  default     = "us-east-1"
}

variable "environment" {
  description = "Environment name"
  type        = string
  default     = "production"
}

variable "domain_name" {
  description = "Main domain name"
  type        = string
  default     = "codeverify.io"
}

# VPC
module "vpc" {
  source = "terraform-aws-modules/vpc/aws"
  
  name = "codeverify-vpc"
  cidr = "10.0.0.0/16"
  
  azs             = ["${var.aws_region}a", "${var.aws_region}b", "${var.aws_region}c"]
  private_subnets = ["10.0.1.0/24", "10.0.2.0/24", "10.0.3.0/24"]
  public_subnets  = ["10.0.101.0/24", "10.0.102.0/24", "10.0.103.0/24"]
  
  enable_nat_gateway   = true
  single_nat_gateway   = false
  enable_dns_hostnames = true
  enable_dns_support   = true
}

# EKS Cluster
module "eks" {
  source = "terraform-aws-modules/eks/aws"
  
  cluster_name    = "codeverify-${var.environment}"
  cluster_version = "1.28"
  
  vpc_id     = module.vpc.vpc_id
  subnet_ids = module.vpc.private_subnets
  
  cluster_endpoint_public_access = true
  
  eks_managed_node_groups = {
    main = {
      name           = "codeverify-nodes"
      instance_types = ["t3.medium"]
      
      min_size     = 3
      max_size     = 10
      desired_size = 5
      
      labels = {
        Environment = var.environment
      }
    }
    
    workers = {
      name           = "codeverify-workers"
      instance_types = ["t3.large"]
      
      min_size     = 2
      max_size     = 20
      desired_size = 5
      
      labels = {
        Environment = var.environment
        NodeType    = "worker"
      }
      
      taints = [
        {
          key    = "workload"
          value  = "analysis"
          effect = "NO_SCHEDULE"
        }
      ]
    }
  }
}

# RDS PostgreSQL
resource "aws_db_subnet_group" "codeverify" {
  name       = "codeverify-db-subnet"
  subnet_ids = module.vpc.private_subnets
}

resource "aws_security_group" "rds" {
  name   = "codeverify-rds-sg"
  vpc_id = module.vpc.vpc_id
  
  ingress {
    from_port       = 5432
    to_port         = 5432
    protocol        = "tcp"
    security_groups = [module.eks.cluster_security_group_id]
  }
}

resource "aws_db_instance" "codeverify" {
  identifier = "codeverify-${var.environment}"
  
  engine         = "postgres"
  engine_version = "15"
  instance_class = "db.t3.medium"
  
  allocated_storage     = 100
  max_allocated_storage = 500
  storage_type          = "gp3"
  storage_encrypted     = true
  
  db_name  = "codeverify"
  username = "codeverify"
  password = random_password.db_password.result
  
  db_subnet_group_name   = aws_db_subnet_group.codeverify.name
  vpc_security_group_ids = [aws_security_group.rds.id]
  
  backup_retention_period = 7
  backup_window          = "03:00-04:00"
  maintenance_window     = "sun:04:00-sun:05:00"
  
  multi_az               = true
  deletion_protection    = true
  skip_final_snapshot    = false
  final_snapshot_identifier = "codeverify-final-snapshot"
}

resource "random_password" "db_password" {
  length  = 32
  special = false
}

# ElastiCache Redis
resource "aws_elasticache_subnet_group" "codeverify" {
  name       = "codeverify-redis-subnet"
  subnet_ids = module.vpc.private_subnets
}

resource "aws_security_group" "redis" {
  name   = "codeverify-redis-sg"
  vpc_id = module.vpc.vpc_id
  
  ingress {
    from_port       = 6379
    to_port         = 6379
    protocol        = "tcp"
    security_groups = [module.eks.cluster_security_group_id]
  }
}

resource "aws_elasticache_replication_group" "codeverify" {
  replication_group_id = "codeverify-${var.environment}"
  description          = "CodeVerify Redis cluster"
  
  node_type            = "cache.t3.medium"
  num_cache_clusters   = 2
  
  engine               = "redis"
  engine_version       = "7.0"
  port                 = 6379
  
  subnet_group_name    = aws_elasticache_subnet_group.codeverify.name
  security_group_ids   = [aws_security_group.redis.id]
  
  automatic_failover_enabled = true
  multi_az_enabled          = true
  
  at_rest_encryption_enabled = true
  transit_encryption_enabled = true
}

# S3 Bucket for artifacts
resource "aws_s3_bucket" "artifacts" {
  bucket = "codeverify-artifacts-${var.environment}"
}

resource "aws_s3_bucket_versioning" "artifacts" {
  bucket = aws_s3_bucket.artifacts.id
  
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "artifacts" {
  bucket = aws_s3_bucket.artifacts.id
  
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "aws:kms"
    }
  }
}

# CloudWatch Log Group
resource "aws_cloudwatch_log_group" "codeverify" {
  name              = "/codeverify/${var.environment}"
  retention_in_days = 30
}

# Outputs
output "eks_cluster_endpoint" {
  description = "EKS cluster endpoint"
  value       = module.eks.cluster_endpoint
}

output "rds_endpoint" {
  description = "RDS endpoint"
  value       = aws_db_instance.codeverify.endpoint
}

output "redis_endpoint" {
  description = "Redis endpoint"
  value       = aws_elasticache_replication_group.codeverify.primary_endpoint_address
}

output "db_password" {
  description = "Database password"
  value       = random_password.db_password.result
  sensitive   = true
}
