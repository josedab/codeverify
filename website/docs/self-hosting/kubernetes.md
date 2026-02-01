---
sidebar_position: 3
---

# Kubernetes Deployment

Deploy CodeVerify on Kubernetes for production workloads.

## Prerequisites

- Kubernetes 1.24+
- kubectl configured
- Helm 3.0+
- Ingress controller (nginx-ingress or similar)
- Storage class with dynamic provisioning

## Quick Start with Helm

### 1. Add Repository

```bash
helm repo add codeverify https://charts.codeverify.dev
helm repo update
```

### 2. Install

```bash
helm install codeverify codeverify/codeverify \
  --namespace codeverify \
  --create-namespace \
  --set global.domain=codeverify.yourcompany.com \
  --set api.secretKey=$(openssl rand -hex 32) \
  --set postgresql.auth.password=$(openssl rand -hex 16)
```

### 3. Wait for Ready

```bash
kubectl -n codeverify get pods -w
```

### 4. Get Ingress IP

```bash
kubectl -n codeverify get ingress
```

## Helm Values

### Minimal Production Values

```yaml
# values.yaml
global:
  domain: codeverify.yourcompany.com

api:
  replicas: 2
  secretKey: "your-secret-key"
  resources:
    requests:
      cpu: 500m
      memory: 1Gi
    limits:
      cpu: 2
      memory: 2Gi

worker:
  replicas: 4
  resources:
    requests:
      cpu: 1
      memory: 2Gi
    limits:
      cpu: 4
      memory: 8Gi

dashboard:
  replicas: 2
  resources:
    requests:
      cpu: 250m
      memory: 512Mi

postgresql:
  auth:
    password: "secure-password"
  persistence:
    size: 100Gi
    storageClass: standard

redis:
  auth:
    password: "secure-redis-password"
  persistence:
    size: 10Gi

ingress:
  enabled: true
  className: nginx
  tls:
    enabled: true
    secretName: codeverify-tls
```

### Install with Values File

```bash
helm install codeverify codeverify/codeverify \
  --namespace codeverify \
  --create-namespace \
  -f values.yaml
```

## Architecture

```yaml
# Kubernetes resources created
apiVersion: apps/v1
kind: Deployment  # api (2 replicas)
---
apiVersion: apps/v1
kind: Deployment  # dashboard (2 replicas)
---
apiVersion: apps/v1
kind: Deployment  # worker (4 replicas)
---
apiVersion: apps/v1
kind: StatefulSet  # postgresql
---
apiVersion: apps/v1
kind: StatefulSet  # redis
---
apiVersion: v1
kind: Service  # api, dashboard, postgresql, redis
---
apiVersion: networking.k8s.io/v1
kind: Ingress  # external access
```

## Scaling

### Horizontal Pod Autoscaler

```yaml
# values.yaml
api:
  autoscaling:
    enabled: true
    minReplicas: 2
    maxReplicas: 10
    targetCPUUtilizationPercentage: 70

worker:
  autoscaling:
    enabled: true
    minReplicas: 4
    maxReplicas: 20
    targetCPUUtilizationPercentage: 80
```

### Manual Scaling

```bash
kubectl -n codeverify scale deployment codeverify-worker --replicas=8
```

## External Database

Use an external PostgreSQL (RDS, Cloud SQL, etc.):

```yaml
# values.yaml
postgresql:
  enabled: false

externalPostgresql:
  host: your-rds-instance.region.rds.amazonaws.com
  port: 5432
  database: codeverify
  username: codeverify
  existingSecret: codeverify-db-credentials
  existingSecretPasswordKey: password
```

Create the secret:

```bash
kubectl -n codeverify create secret generic codeverify-db-credentials \
  --from-literal=password=your-database-password
```

## External Redis

Use an external Redis (ElastiCache, etc.):

```yaml
# values.yaml
redis:
  enabled: false

externalRedis:
  host: your-redis-cluster.region.cache.amazonaws.com
  port: 6379
  existingSecret: codeverify-redis-credentials
```

## TLS Configuration

### With cert-manager

```yaml
# values.yaml
ingress:
  enabled: true
  className: nginx
  annotations:
    cert-manager.io/cluster-issuer: letsencrypt-prod
  tls:
    enabled: true
    secretName: codeverify-tls
```

### With Existing Certificate

```bash
kubectl -n codeverify create secret tls codeverify-tls \
  --cert=fullchain.pem \
  --key=privkey.pem
```

## Secrets Management

### Using External Secrets Operator

```yaml
apiVersion: external-secrets.io/v1beta1
kind: ExternalSecret
metadata:
  name: codeverify-secrets
spec:
  secretStoreRef:
    kind: ClusterSecretStore
    name: aws-secrets-manager
  target:
    name: codeverify-secrets
  data:
    - secretKey: api-secret-key
      remoteRef:
        key: codeverify/api
        property: secret_key
    - secretKey: openai-api-key
      remoteRef:
        key: codeverify/openai
        property: api_key
```

### Using Sealed Secrets

```bash
kubeseal --format yaml < secret.yaml > sealed-secret.yaml
kubectl apply -f sealed-secret.yaml
```

## Monitoring

### Prometheus + Grafana

```yaml
# values.yaml
metrics:
  enabled: true
  serviceMonitor:
    enabled: true
    namespace: monitoring

dashboard:
  grafanaDashboards:
    enabled: true
```

### Metrics Endpoint

```bash
curl http://codeverify-api:8000/metrics
```

## Logging

### With Loki

```yaml
# values.yaml
logging:
  format: json

podAnnotations:
  prometheus.io/scrape: "true"
```

### Log Aggregation

```bash
kubectl -n codeverify logs -l app=codeverify-worker --tail=100 -f
```

## Network Policies

Restrict traffic between components:

```yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: codeverify-api
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
        - podSelector:
            matchLabels:
              app: codeverify-dashboard
      ports:
        - port: 8000
  egress:
    - to:
        - podSelector:
            matchLabels:
              app: codeverify-postgresql
        - podSelector:
            matchLabels:
              app: codeverify-redis
```

## Backup

### Database Backup

```bash
kubectl -n codeverify exec -it codeverify-postgresql-0 -- \
  pg_dump -U postgres codeverify > backup.sql
```

### Automated Backups with CronJob

```yaml
apiVersion: batch/v1
kind: CronJob
metadata:
  name: codeverify-backup
spec:
  schedule: "0 2 * * *"
  jobTemplate:
    spec:
      template:
        spec:
          containers:
            - name: backup
              image: postgres:15
              command:
                - /bin/sh
                - -c
                - pg_dump -h $DB_HOST -U $DB_USER $DB_NAME | gzip > /backups/backup-$(date +%Y%m%d).sql.gz
```

## Upgrading

```bash
helm repo update
helm upgrade codeverify codeverify/codeverify \
  --namespace codeverify \
  -f values.yaml
```

## Rollback

```bash
helm rollback codeverify --namespace codeverify
```

## Troubleshooting

### Pod Not Starting

```bash
kubectl -n codeverify describe pod codeverify-api-xxx
kubectl -n codeverify logs codeverify-api-xxx
```

### Database Connection Issues

```bash
kubectl -n codeverify exec -it codeverify-api-xxx -- \
  python -c "from app.db import engine; print(engine.connect())"
```

### Worker Queue Issues

```bash
kubectl -n codeverify exec -it codeverify-worker-xxx -- \
  celery -A app.worker inspect active
```

## Next Steps

- [Configuration](/docs/self-hosting/configuration) — Full configuration reference
- [Monitoring](/docs/resources/troubleshooting#monitoring) — Monitoring setup

