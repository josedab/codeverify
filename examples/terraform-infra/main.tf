# Example Terraform configuration with intentional security issues.
# CodeVerify's multi-modal verification catches these.

provider "aws" {
  region = "us-east-1"
}

# BUG: S3 bucket is public
# CodeVerify finds: resource is publicly accessible
resource "aws_s3_bucket" "data" {
  bucket = "my-company-data"

  tags = {
    Environment = "production"
  }
}

resource "aws_s3_bucket_public_access_block" "data" {
  bucket = aws_s3_bucket.data.id

  block_public_acls       = false  # Should be true
  block_public_policy     = false  # Should be true
  ignore_public_acls      = false  # Should be true
  restrict_public_buckets = false  # Should be true
}

# BUG: Security group allows traffic from anywhere
# CodeVerify finds: open CIDR 0.0.0.0/0
resource "aws_security_group" "web" {
  name        = "web-sg"
  description = "Web server security group"

  ingress {
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]  # Too broad
  }

  ingress {
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]  # SSH from anywhere — critical!
  }
}

# BUG: RDS instance not encrypted
# CodeVerify finds: encryption is disabled
resource "aws_db_instance" "main" {
  engine         = "postgres"
  instance_class = "db.t3.medium"
  allocated_storage = 50

  storage_encrypted = false  # Should be true
  publicly_accessible = true  # Should be false

  username = "admin"
  password = "SuperSecret123!"  # Hardcoded password!
}
