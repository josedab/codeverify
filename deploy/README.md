# Deployment Guide

This directory contains deployment configurations for CodeVerify.

## Overview

- `kubernetes/` - Kubernetes manifests for container orchestration
- `terraform/` - Infrastructure as Code for AWS

## Kubernetes Deployment

### Prerequisites

- kubectl configured
- Kubernetes cluster (EKS, GKE, or similar)
- Helm 3.x

### Deploy

```bash
# Apply all manifests
kubectl apply -f kubernetes/codeverify.yaml

# Verify deployment
kubectl get pods -n codeverify
kubectl get services -n codeverify
```

### Secrets

Before deploying, update the secrets in `codeverify.yaml`:

```bash
# Create secrets from .env file
kubectl create secret generic codeverify-secrets \
  --from-env-file=.env \
  -n codeverify
```

## Terraform (AWS)

### Prerequisites

- Terraform >= 1.0
- AWS CLI configured
- S3 bucket for state (create manually first)

### Initialize

```bash
cd terraform

# Initialize Terraform
terraform init

# Plan changes
terraform plan -out=tfplan

# Apply
terraform apply tfplan
```

### Infrastructure Created

- VPC with public/private subnets
- EKS cluster with managed node groups
- RDS PostgreSQL (Multi-AZ)
- ElastiCache Redis (cluster mode)
- S3 bucket for artifacts
- CloudWatch log groups

## Production Checklist

### Security
- [ ] Update all secrets in Kubernetes
- [ ] Enable encryption at rest
- [ ] Configure network policies
- [ ] Set up WAF rules
- [ ] Enable audit logging

### Monitoring
- [ ] Deploy Prometheus/Grafana
- [ ] Configure alerting rules
- [ ] Set up Sentry DSN
- [ ] Enable CloudWatch alarms

### Backup
- [ ] Verify RDS automated backups
- [ ] Test restore procedure
- [ ] Document disaster recovery

### DNS
- [ ] Configure Route 53 records
- [ ] Set up SSL certificates
- [ ] Verify ingress configuration

## Scaling

### Horizontal Pod Autoscaler

The HPA is configured to scale:
- API: 3-10 replicas based on CPU (70%)
- Workers: 3-20 replicas based on CPU (70%)

### Manual Scaling

```bash
# Scale workers for high load
kubectl scale deployment codeverify-worker --replicas=10 -n codeverify
```

## Troubleshooting

### Check pod logs
```bash
kubectl logs -f deployment/codeverify-api -n codeverify
```

### Check events
```bash
kubectl get events -n codeverify --sort-by='.lastTimestamp'
```

### Debug pod
```bash
kubectl exec -it deployment/codeverify-api -n codeverify -- /bin/sh
```
