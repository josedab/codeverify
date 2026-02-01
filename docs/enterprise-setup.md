# Enterprise Setup Guide

This guide covers deploying CodeVerify for enterprise environments with high availability, security, and compliance requirements.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     Enterprise CodeVerify Architecture                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────┐     ┌──────────────────────────────────────────────────┐  │
│  │   WAF/CDN    │────▶│              Load Balancer (ALB/NLB)             │  │
│  │ (CloudFlare) │     └────────────────────┬─────────────────────────────┘  │
│  └──────────────┘                          │                                 │
│                                            ▼                                 │
│                          ┌─────────────────────────────────┐                │
│                          │        Kubernetes Cluster        │                │
│                          │           (EKS/GKE)             │                │
│  ┌───────────────────────┼─────────────────────────────────┼─────────────┐  │
│  │                       │                                 │             │  │
│  │  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐              │  │
│  │  │  API Pods   │     │ Worker Pods │     │ GitHub App  │              │  │
│  │  │  (3-10)     │     │   (3-20)    │     │  Pods (2-5) │              │  │
│  │  └──────┬──────┘     └──────┬──────┘     └──────┬──────┘              │  │
│  │         │                   │                   │                     │  │
│  │         └───────────────────┼───────────────────┘                     │  │
│  │                             │                                         │  │
│  │                             ▼                                         │  │
│  │              ┌──────────────────────────────┐                        │  │
│  │              │         Service Mesh          │                        │  │
│  │              │       (Istio/Linkerd)        │                        │  │
│  │              └──────────────────────────────┘                        │  │
│  │                                                                       │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                          Data Layer                                   │  │
│  │  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐            │  │
│  │  │  PostgreSQL  │    │    Redis     │    │     S3       │            │  │
│  │  │  (RDS HA)    │    │ (ElastiCache)│    │  (Artifacts) │            │  │
│  │  │  Multi-AZ    │    │   Cluster    │    │              │            │  │
│  │  └──────────────┘    └──────────────┘    └──────────────┘            │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Prerequisites

### Infrastructure Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| Kubernetes | 1.28+ | 1.29+ |
| PostgreSQL | 14, db.r5.large | 15, db.r5.xlarge Multi-AZ |
| Redis | 7.0, cache.r6g.large | 7.2, cache.r6g.xlarge cluster |
| Worker Nodes | 3x m5.xlarge | 5x m5.2xlarge |
| Storage | 100GB gp3 | 500GB gp3 |

### Network Requirements

- Private subnets for workloads
- NAT Gateway for outbound traffic
- VPC endpoints for AWS services
- TLS 1.3 for all communications

## Deployment Options

### Option 1: AWS with EKS (Recommended)

```bash
cd deploy/terraform

# Initialize
terraform init

# Create infrastructure
terraform plan -var-file=enterprise.tfvars -out=tfplan
terraform apply tfplan
```

**enterprise.tfvars:**
```hcl
environment = "production"
region      = "us-east-1"

# EKS Configuration
eks_cluster_version = "1.29"
eks_node_groups = {
  general = {
    instance_types = ["m5.2xlarge"]
    min_size       = 3
    max_size       = 10
    desired_size   = 5
  }
  workers = {
    instance_types = ["c5.2xlarge"]
    min_size       = 3
    max_size       = 20
    desired_size   = 5
  }
}

# RDS Configuration
rds_instance_class    = "db.r5.xlarge"
rds_multi_az          = true
rds_storage_encrypted = true
rds_backup_retention  = 30

# ElastiCache Configuration
redis_node_type       = "cache.r6g.xlarge"
redis_num_cache_nodes = 3
redis_cluster_mode    = true

# Security
enable_waf           = true
enable_shield        = true
enable_guardduty     = true
```

### Option 2: GCP with GKE

```bash
cd deploy/terraform-gcp

terraform init
terraform plan -var-file=enterprise.tfvars -out=tfplan
terraform apply tfplan
```

### Option 3: Self-Hosted Kubernetes

For on-premises or custom Kubernetes:

```bash
# Apply Kubernetes manifests
kubectl apply -f deploy/kubernetes/namespace.yaml
kubectl apply -f deploy/kubernetes/secrets.yaml
kubectl apply -f deploy/kubernetes/configmaps.yaml
kubectl apply -f deploy/kubernetes/services.yaml
kubectl apply -f deploy/kubernetes/deployments.yaml
kubectl apply -f deploy/kubernetes/hpa.yaml
kubectl apply -f deploy/kubernetes/pdb.yaml
```

## High Availability Configuration

### Database HA

**PostgreSQL with RDS Multi-AZ:**
```yaml
# terraform
rds_multi_az = true
rds_backup_retention_period = 30
rds_deletion_protection = true
rds_performance_insights_enabled = true
```

**Connection Pooling with PgBouncer:**
```yaml
# kubernetes/pgbouncer.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: pgbouncer
spec:
  replicas: 2
  template:
    spec:
      containers:
        - name: pgbouncer
          image: pgbouncer/pgbouncer:1.21.0
          env:
            - name: DATABASES_HOST
              value: "codeverify-db.xxxxx.rds.amazonaws.com"
            - name: POOL_MODE
              value: "transaction"
            - name: MAX_CLIENT_CONN
              value: "1000"
            - name: DEFAULT_POOL_SIZE
              value: "50"
```

### Redis HA

**ElastiCache Cluster Mode:**
```yaml
redis_cluster_mode = true
redis_num_node_groups = 3
redis_replicas_per_node_group = 2
```

### Application HA

**Pod Disruption Budgets:**
```yaml
# kubernetes/pdb.yaml
apiVersion: policy/v1
kind: PodDisruptionBudget
metadata:
  name: api-pdb
spec:
  minAvailable: 2
  selector:
    matchLabels:
      app: codeverify-api

---
apiVersion: policy/v1
kind: PodDisruptionBudget
metadata:
  name: worker-pdb
spec:
  minAvailable: 3
  selector:
    matchLabels:
      app: codeverify-worker
```

**Horizontal Pod Autoscaler:**
```yaml
# kubernetes/hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: api-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: codeverify-api
  minReplicas: 3
  maxReplicas: 10
  metrics:
    - type: Resource
      resource:
        name: cpu
        target:
          type: Utilization
          averageUtilization: 70
    - type: Resource
      resource:
        name: memory
        target:
          type: Utilization
          averageUtilization: 80
```

## Security Configuration

### Network Security

**Network Policies:**
```yaml
# kubernetes/network-policies.yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: api-network-policy
spec:
  podSelector:
    matchLabels:
      app: codeverify-api
  policyTypes:
    - Ingress
    - Egress
  ingress:
    - from:
        - namespaceSelector:
            matchLabels:
              name: ingress-nginx
      ports:
        - port: 8000
  egress:
    - to:
        - namespaceSelector:
            matchLabels:
              name: codeverify
    - to:
        - ipBlock:
            cidr: 10.0.0.0/8  # Internal services
```

### Secrets Management

**Using AWS Secrets Manager:**
```yaml
# kubernetes/external-secrets.yaml
apiVersion: external-secrets.io/v1beta1
kind: ExternalSecret
metadata:
  name: codeverify-secrets
spec:
  refreshInterval: 1h
  secretStoreRef:
    kind: ClusterSecretStore
    name: aws-secrets-manager
  target:
    name: codeverify-secrets
  data:
    - secretKey: database-url
      remoteRef:
        key: codeverify/production
        property: DATABASE_URL
    - secretKey: jwt-secret
      remoteRef:
        key: codeverify/production
        property: JWT_SECRET
```

**Using HashiCorp Vault:**
```yaml
apiVersion: secrets.hashicorp.com/v1beta1
kind: VaultStaticSecret
metadata:
  name: codeverify-secrets
spec:
  vaultAuthRef: vault-auth
  mount: secret
  path: codeverify/production
  destination:
    name: codeverify-secrets
    create: true
```

### Encryption

**Data at Rest:**
- RDS: AES-256 encryption enabled
- S3: SSE-KMS encryption
- EBS: Encrypted volumes

**Data in Transit:**
- TLS 1.3 for all external traffic
- mTLS within service mesh
- Certificate rotation via cert-manager

```yaml
# kubernetes/certificate.yaml
apiVersion: cert-manager.io/v1
kind: Certificate
metadata:
  name: codeverify-tls
spec:
  secretName: codeverify-tls-secret
  duration: 2160h # 90 days
  renewBefore: 360h # 15 days
  issuerRef:
    name: letsencrypt-prod
    kind: ClusterIssuer
  dnsNames:
    - api.codeverify.yourcompany.com
    - dashboard.codeverify.yourcompany.com
```

## SSO/SAML Configuration

### Configure SAML IdP

1. **In your IdP (Okta, Azure AD, etc.):**
   - Create new SAML application
   - Set ACS URL: `https://api.codeverify.yourcompany.com/sso/saml/acs`
   - Set Entity ID: `https://api.codeverify.yourcompany.com`
   - Configure attribute mappings

2. **In CodeVerify:**
   ```bash
   curl -X POST https://api.codeverify.yourcompany.com/sso/config/ORG_ID \
     -H "Authorization: Bearer ADMIN_TOKEN" \
     -H "Content-Type: application/json" \
     -d '{
       "idp_entity_id": "https://your-idp.example.com/saml",
       "idp_sso_url": "https://your-idp.example.com/sso",
       "idp_certificate": "-----BEGIN CERTIFICATE-----\n...",
       "enforce_sso": true,
       "domains": ["yourcompany.com"]
     }'
   ```

### SCIM Provisioning

Enable automatic user provisioning:

```yaml
# Environment variables
SCIM_ENABLED=true
SCIM_TOKEN=your-scim-token
```

## Compliance

### Audit Logging

All actions are logged to the `audit_logs` table and can be exported:

```bash
# Export audit logs
curl "https://api.codeverify.yourcompany.com/export/audit-logs?start_date=2026-01-01&end_date=2026-01-31" \
  -H "Authorization: Bearer ADMIN_TOKEN" \
  -o audit-logs.csv
```

**Log Retention:**
```yaml
# Environment variable
AUDIT_LOG_RETENTION_DAYS=365
```

### SOC 2 Compliance

CodeVerify supports SOC 2 Type II compliance with:

- [ ] Access controls (RBAC)
- [ ] Encryption at rest and in transit
- [ ] Audit logging
- [ ] Vulnerability scanning
- [ ] Incident response procedures

**Generate Compliance Report:**
```bash
curl "https://api.codeverify.yourcompany.com/compliance/report?framework=soc2" \
  -H "Authorization: Bearer ADMIN_TOKEN" \
  -o soc2-report.pdf
```

### GDPR

Data processing controls:

```yaml
# Environment variables
GDPR_MODE=true
DATA_RETENTION_DAYS=90
ANONYMIZE_ON_DELETE=true
```

## Backup & Recovery

### Database Backup

**Automated RDS Backups:**
```hcl
# terraform
rds_backup_retention_period = 30
rds_backup_window = "03:00-04:00"
rds_maintenance_window = "Mon:04:00-Mon:05:00"
```

**Manual Snapshot:**
```bash
aws rds create-db-snapshot \
  --db-instance-identifier codeverify-prod \
  --db-snapshot-identifier codeverify-manual-$(date +%Y%m%d)
```

### Disaster Recovery

**Cross-Region Replication:**
```hcl
# terraform
rds_cross_region_replica = true
rds_replica_region = "us-west-2"
```

**Recovery Procedure:**
1. Promote read replica to primary
2. Update DNS to point to new region
3. Scale up workers in new region
4. Verify webhook connectivity

**RTO/RPO Targets:**
| Scenario | RTO | RPO |
|----------|-----|-----|
| Single AZ failure | 5 min | 0 |
| Region failure | 30 min | 5 min |
| Database corruption | 1 hour | 1 hour |

## Performance Tuning

### Database Optimization

```sql
-- Recommended PostgreSQL settings
ALTER SYSTEM SET shared_buffers = '4GB';
ALTER SYSTEM SET effective_cache_size = '12GB';
ALTER SYSTEM SET maintenance_work_mem = '1GB';
ALTER SYSTEM SET checkpoint_completion_target = 0.9;
ALTER SYSTEM SET wal_buffers = '64MB';
ALTER SYSTEM SET default_statistics_target = 100;
ALTER SYSTEM SET random_page_cost = 1.1;
ALTER SYSTEM SET effective_io_concurrency = 200;
ALTER SYSTEM SET work_mem = '50MB';
ALTER SYSTEM SET min_wal_size = '1GB';
ALTER SYSTEM SET max_wal_size = '4GB';
```

### Worker Optimization

```yaml
# Environment variables
CELERY_WORKER_CONCURRENCY=8
CELERY_WORKER_PREFETCH_MULTIPLIER=1
VERIFICATION_TIMEOUT=60
MAX_PARALLEL_ANALYSES=4
```

### API Optimization

```yaml
# Environment variables
UVICORN_WORKERS=4
DB_POOL_SIZE=20
DB_MAX_OVERFLOW=10
REDIS_MAX_CONNECTIONS=50
```

## Monitoring Setup

See [Monitoring Guide](monitoring.md) for detailed setup.

Quick setup:

```bash
# Install Prometheus stack
helm install prometheus prometheus-community/kube-prometheus-stack \
  -f deploy/monitoring/prometheus-values.yaml

# Install Grafana dashboards
kubectl apply -f deploy/monitoring/dashboards/
```

## Support

Enterprise customers have access to:

- **Priority Support**: 4-hour response SLA
- **Dedicated Slack Channel**: Direct access to engineering team
- **Quarterly Business Reviews**: Performance and roadmap discussions
- **Custom Integrations**: Engineering assistance for custom needs

Contact: enterprise@codeverify.io
