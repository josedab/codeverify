# Database Schema Documentation

This document describes the CodeVerify database schema.

## Overview

CodeVerify uses PostgreSQL as its primary database. The schema is managed via Alembic migrations located in `apps/api/alembic/versions/`.

## Entity Relationship Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         CodeVerify Database Schema                           │
└─────────────────────────────────────────────────────────────────────────────┘

┌──────────────┐       ┌──────────────┐       ┌──────────────┐
│organizations │───┬──▶│ repositories │───┬──▶│   analyses   │
└──────────────┘   │   └──────────────┘   │   └──────┬───────┘
       │           │          │           │          │
       │           │          │           │          ▼
       ▼           │          │           │   ┌──────────────┐
┌──────────────┐   │          │           │   │   findings   │
│    users     │◀──┤          │           │   └──────────────┘
└──────────────┘   │          │           │
       │           │          │           │   ┌──────────────┐
       ▼           │          │           └──▶│analysis_stages│
┌──────────────┐   │          │               └──────────────┘
│org_memberships│◀─┘          │
└──────────────┘              │          ┌──────────────────┐
                              ├─────────▶│  custom_rules    │
┌──────────────┐              │          └──────────────────┘
│ installations│◀─────────────┤
└──────────────┘              │          ┌──────────────────┐
                              └─────────▶│  codebase_scans  │
┌──────────────┐                         └──────────────────┘
│   api_keys   │◀─────────────┐
└──────────────┘              │
                              │
┌──────────────┐              │          ┌──────────────────┐
│  audit_logs  │◀─────────────┼─────────▶│    webhooks      │
└──────────────┘              │          └────────┬─────────┘
                              │                   │
┌──────────────┐              │                   ▼
│notification_ │◀─────────────┤          ┌──────────────────┐
│   configs    │              │          │webhook_deliveries│
└──────────────┘              │          └──────────────────┘
                              │
┌──────────────┐              │
│scan_schedules│◀─────────────┘
└──────────────┘
```

## Core Tables

### organizations

Stores GitHub organizations that have installed CodeVerify.

| Column | Type | Nullable | Default | Description |
|--------|------|----------|---------|-------------|
| `id` | UUID | No | uuid_generate_v4() | Primary key |
| `github_id` | BIGINT | No | - | GitHub organization ID |
| `name` | VARCHAR(255) | No | - | Display name |
| `login` | VARCHAR(255) | No | - | GitHub login/slug |
| `avatar_url` | TEXT | Yes | - | Avatar URL |
| `settings` | JSON | No | {} | Organization settings |
| `created_at` | TIMESTAMPTZ | No | now() | Created timestamp |
| `updated_at` | TIMESTAMPTZ | No | now() | Updated timestamp |

**Indexes:**
- `organizations_pkey` on `id`
- `organizations_github_id_key` UNIQUE on `github_id`

---

### users

Stores authenticated users.

| Column | Type | Nullable | Default | Description |
|--------|------|----------|---------|-------------|
| `id` | UUID | No | uuid_generate_v4() | Primary key |
| `github_id` | BIGINT | No | - | GitHub user ID |
| `username` | VARCHAR(255) | No | - | GitHub username |
| `email` | VARCHAR(255) | Yes | - | Email address |
| `avatar_url` | TEXT | Yes | - | Avatar URL |
| `access_token_encrypted` | TEXT | Yes | - | Encrypted GitHub token |
| `created_at` | TIMESTAMPTZ | No | now() | Created timestamp |
| `updated_at` | TIMESTAMPTZ | No | now() | Updated timestamp |

**Indexes:**
- `users_pkey` on `id`
- `users_github_id_key` UNIQUE on `github_id`

---

### org_memberships

Maps users to organizations with roles.

| Column | Type | Nullable | Default | Description |
|--------|------|----------|---------|-------------|
| `id` | UUID | No | uuid_generate_v4() | Primary key |
| `org_id` | UUID | No | - | FK to organizations |
| `user_id` | UUID | No | - | FK to users |
| `role` | VARCHAR(50) | No | 'member' | Role (admin, member) |
| `created_at` | TIMESTAMPTZ | No | now() | Created timestamp |

**Indexes:**
- `org_memberships_pkey` on `id`
- `org_memberships_org_id_user_id_key` UNIQUE on `(org_id, user_id)`

**Foreign Keys:**
- `org_id` → `organizations(id)` ON DELETE CASCADE
- `user_id` → `users(id)` ON DELETE CASCADE

---

### repositories

Stores repositories enabled for CodeVerify analysis.

| Column | Type | Nullable | Default | Description |
|--------|------|----------|---------|-------------|
| `id` | UUID | No | uuid_generate_v4() | Primary key |
| `github_id` | BIGINT | No | - | GitHub repository ID |
| `org_id` | UUID | Yes | - | FK to organizations |
| `name` | VARCHAR(255) | No | - | Repository name |
| `full_name` | VARCHAR(512) | No | - | Full name (org/repo) |
| `private` | BOOLEAN | No | false | Is private repo |
| `default_branch` | VARCHAR(255) | No | 'main' | Default branch |
| `settings` | JSON | No | {} | Repository-specific settings |
| `enabled` | BOOLEAN | No | true | Analysis enabled |
| `created_at` | TIMESTAMPTZ | No | now() | Created timestamp |
| `updated_at` | TIMESTAMPTZ | No | now() | Updated timestamp |

**Indexes:**
- `repositories_pkey` on `id`
- `repositories_github_id_key` UNIQUE on `github_id`

**Foreign Keys:**
- `org_id` → `organizations(id)` ON DELETE CASCADE

---

### installations

Tracks GitHub App installations.

| Column | Type | Nullable | Default | Description |
|--------|------|----------|---------|-------------|
| `id` | UUID | No | uuid_generate_v4() | Primary key |
| `github_installation_id` | BIGINT | No | - | GitHub installation ID |
| `org_id` | UUID | Yes | - | FK to organizations |
| `account_type` | VARCHAR(50) | No | - | 'Organization' or 'User' |
| `permissions` | JSON | No | {} | Granted permissions |
| `events` | JSON | No | [] | Subscribed events |
| `suspended_at` | TIMESTAMPTZ | Yes | - | Suspension timestamp |
| `created_at` | TIMESTAMPTZ | No | now() | Created timestamp |
| `updated_at` | TIMESTAMPTZ | No | now() | Updated timestamp |

**Indexes:**
- `installations_pkey` on `id`
- `installations_github_installation_id_key` UNIQUE on `github_installation_id`

**Foreign Keys:**
- `org_id` → `organizations(id)` ON DELETE CASCADE

---

### analyses

Stores PR analysis records.

| Column | Type | Nullable | Default | Description |
|--------|------|----------|---------|-------------|
| `id` | UUID | No | uuid_generate_v4() | Primary key |
| `repo_id` | UUID | No | - | FK to repositories |
| `pr_number` | INTEGER | No | - | Pull request number |
| `pr_title` | TEXT | Yes | - | Pull request title |
| `head_sha` | VARCHAR(40) | No | - | Head commit SHA |
| `base_sha` | VARCHAR(40) | Yes | - | Base commit SHA |
| `status` | VARCHAR(50) | No | 'pending' | Status: pending, running, completed, failed |
| `triggered_by` | UUID | Yes | - | FK to users (who triggered) |
| `started_at` | TIMESTAMPTZ | Yes | - | Analysis start time |
| `completed_at` | TIMESTAMPTZ | Yes | - | Analysis completion time |
| `error_message` | TEXT | Yes | - | Error message if failed |
| `metadata` | JSON | No | {} | Additional metadata |
| `created_at` | TIMESTAMPTZ | No | now() | Created timestamp |
| `updated_at` | TIMESTAMPTZ | No | now() | Updated timestamp |

**Indexes:**
- `analyses_pkey` on `id`
- `idx_analyses_repo_pr` on `(repo_id, pr_number)`
- `idx_analyses_status` on `status`
- `idx_analyses_created` on `created_at`

**Foreign Keys:**
- `repo_id` → `repositories(id)` ON DELETE CASCADE
- `triggered_by` → `users(id)`

---

### findings

Stores individual findings from analyses.

| Column | Type | Nullable | Default | Description |
|--------|------|----------|---------|-------------|
| `id` | UUID | No | uuid_generate_v4() | Primary key |
| `analysis_id` | UUID | No | - | FK to analyses |
| `category` | VARCHAR(100) | No | - | Category: verification, security, semantic |
| `severity` | VARCHAR(50) | No | - | Severity: critical, high, medium, low |
| `title` | TEXT | No | - | Finding title |
| `description` | TEXT | Yes | - | Detailed description |
| `file_path` | TEXT | No | - | File path |
| `line_start` | INTEGER | Yes | - | Starting line number |
| `line_end` | INTEGER | Yes | - | Ending line number |
| `code_snippet` | TEXT | Yes | - | Relevant code snippet |
| `fix_suggestion` | TEXT | Yes | - | Suggested fix description |
| `fix_diff` | TEXT | Yes | - | Diff for suggested fix |
| `confidence` | FLOAT | Yes | - | Confidence score (0-1) |
| `verification_type` | VARCHAR(50) | Yes | - | Type: formal, ai, pattern |
| `verification_proof` | TEXT | Yes | - | Z3 proof or explanation |
| `metadata` | JSON | No | {} | Additional metadata (CWE, OWASP, etc.) |
| `dismissed` | BOOLEAN | No | false | Is dismissed |
| `dismissed_by` | UUID | Yes | - | FK to users (who dismissed) |
| `dismissed_reason` | TEXT | Yes | - | Dismissal reason |
| `created_at` | TIMESTAMPTZ | No | now() | Created timestamp |

**Indexes:**
- `findings_pkey` on `id`
- `idx_findings_analysis` on `analysis_id`
- `idx_findings_category` on `category`
- `idx_findings_severity` on `severity`

**Foreign Keys:**
- `analysis_id` → `analyses(id)` ON DELETE CASCADE
- `dismissed_by` → `users(id)`

---

### analysis_stages

Tracks individual stages of an analysis.

| Column | Type | Nullable | Default | Description |
|--------|------|----------|---------|-------------|
| `id` | UUID | No | uuid_generate_v4() | Primary key |
| `analysis_id` | UUID | No | - | FK to analyses |
| `stage_name` | VARCHAR(100) | No | - | Stage name: parse, semantic, formal, security |
| `status` | VARCHAR(50) | No | 'pending' | Status |
| `started_at` | TIMESTAMPTZ | Yes | - | Stage start time |
| `completed_at` | TIMESTAMPTZ | Yes | - | Stage completion time |
| `duration_ms` | INTEGER | Yes | - | Duration in milliseconds |
| `result` | JSON | Yes | - | Stage results |
| `error_message` | TEXT | Yes | - | Error message if failed |
| `created_at` | TIMESTAMPTZ | No | now() | Created timestamp |

**Indexes:**
- `analysis_stages_pkey` on `id`
- `idx_stages_analysis` on `analysis_id`

**Foreign Keys:**
- `analysis_id` → `analyses(id)` ON DELETE CASCADE

## Configuration Tables

### custom_rules

Stores custom rule definitions.

| Column | Type | Nullable | Default | Description |
|--------|------|----------|---------|-------------|
| `id` | UUID | No | uuid_generate_v4() | Primary key |
| `org_id` | UUID | Yes | - | FK to organizations (org-wide rule) |
| `repo_id` | UUID | Yes | - | FK to repositories (repo-specific) |
| `name` | VARCHAR(255) | No | - | Rule name |
| `description` | TEXT | Yes | - | Rule description |
| `rule_type` | VARCHAR(100) | No | - | Type: pattern, ast, semantic |
| `rule_config` | JSON | No | - | Rule configuration |
| `severity` | VARCHAR(50) | No | 'warning' | Default severity |
| `enabled` | BOOLEAN | No | true | Is enabled |
| `created_by` | UUID | Yes | - | FK to users |
| `created_at` | TIMESTAMPTZ | No | now() | Created timestamp |
| `updated_at` | TIMESTAMPTZ | No | now() | Updated timestamp |

**Foreign Keys:**
- `org_id` → `organizations(id)` ON DELETE CASCADE
- `repo_id` → `repositories(id)` ON DELETE CASCADE
- `created_by` → `users(id)`

---

### notification_configs

Stores notification channel configurations.

| Column | Type | Nullable | Default | Description |
|--------|------|----------|---------|-------------|
| `id` | UUID | No | uuid_generate_v4() | Primary key |
| `org_id` | UUID | No | - | FK to organizations |
| `channel_type` | VARCHAR(50) | No | - | Type: slack, teams, email |
| `webhook_url` | TEXT | Yes | - | Webhook URL |
| `config` | JSON | No | {} | Channel-specific config |
| `events` | JSON | No | [] | Events to notify on |
| `enabled` | BOOLEAN | No | true | Is enabled |
| `created_by` | UUID | Yes | - | FK to users |
| `created_at` | TIMESTAMPTZ | No | now() | Created timestamp |
| `updated_at` | TIMESTAMPTZ | No | now() | Updated timestamp |

**Indexes:**
- `ix_notification_configs_org_id` on `org_id`

**Foreign Keys:**
- `org_id` → `organizations(id)` ON DELETE CASCADE
- `created_by` → `users(id)`

## Webhook Tables

### webhooks

Stores webhook subscriptions.

| Column | Type | Nullable | Default | Description |
|--------|------|----------|---------|-------------|
| `id` | UUID | No | - | Primary key |
| `org_id` | UUID | No | - | FK to organizations |
| `url` | TEXT | No | - | Webhook endpoint URL |
| `secret_hash` | VARCHAR(255) | Yes | - | Hashed signing secret |
| `events` | JSON | No | [] | Events to deliver |
| `active` | BOOLEAN | No | true | Is active |
| `last_triggered_at` | TIMESTAMPTZ | Yes | - | Last delivery time |
| `failure_count` | INTEGER | No | 0 | Consecutive failures |
| `created_by` | UUID | Yes | - | FK to users |
| `created_at` | TIMESTAMPTZ | No | now() | Created timestamp |
| `updated_at` | TIMESTAMPTZ | No | now() | Updated timestamp |

**Indexes:**
- `ix_webhooks_org_id` on `org_id`

**Foreign Keys:**
- `org_id` → `organizations(id)` ON DELETE CASCADE
- `created_by` → `users(id)`

---

### webhook_deliveries

Stores webhook delivery attempts.

| Column | Type | Nullable | Default | Description |
|--------|------|----------|---------|-------------|
| `id` | UUID | No | - | Primary key |
| `webhook_id` | UUID | No | - | FK to webhooks |
| `event` | VARCHAR(100) | No | - | Event type |
| `payload` | JSON | No | - | Delivered payload |
| `response_status` | INTEGER | Yes | - | HTTP response status |
| `response_body` | TEXT | Yes | - | Response body |
| `success` | BOOLEAN | No | false | Was successful |
| `delivered_at` | TIMESTAMPTZ | No | now() | Delivery timestamp |

**Indexes:**
- `ix_webhook_deliveries_webhook_id` on `webhook_id`
- `ix_webhook_deliveries_delivered_at` on `delivered_at`

**Foreign Keys:**
- `webhook_id` → `webhooks(id)` ON DELETE CASCADE

## Scanning Tables

### codebase_scans

Stores full codebase scan records.

| Column | Type | Nullable | Default | Description |
|--------|------|----------|---------|-------------|
| `id` | UUID | No | - | Primary key |
| `repo_id` | UUID | No | - | FK to repositories |
| `branch` | VARCHAR(255) | No | 'main' | Branch to scan |
| `status` | VARCHAR(50) | No | 'pending' | Scan status |
| `triggered_by` | VARCHAR(50) | No | 'manual' | Trigger: manual, schedule, api |
| `config` | JSON | No | {} | Scan configuration |
| `started_at` | TIMESTAMPTZ | Yes | - | Scan start time |
| `completed_at` | TIMESTAMPTZ | Yes | - | Scan completion time |
| `files_scanned` | INTEGER | No | 0 | Number of files scanned |
| `findings_count` | INTEGER | No | 0 | Number of findings |
| `results` | JSON | Yes | - | Scan results summary |
| `error_message` | TEXT | Yes | - | Error message if failed |
| `created_at` | TIMESTAMPTZ | No | now() | Created timestamp |

**Indexes:**
- `ix_codebase_scans_repo_id` on `repo_id`
- `ix_codebase_scans_status` on `status`

**Foreign Keys:**
- `repo_id` → `repositories(id)` ON DELETE CASCADE

---

### scan_schedules

Stores scheduled scan configurations.

| Column | Type | Nullable | Default | Description |
|--------|------|----------|---------|-------------|
| `id` | UUID | No | - | Primary key |
| `repo_id` | UUID | No | - | FK to repositories |
| `branch` | VARCHAR(255) | No | 'main' | Branch to scan |
| `cron_expression` | VARCHAR(100) | No | - | Cron schedule |
| `config` | JSON | No | {} | Scan configuration |
| `enabled` | BOOLEAN | No | true | Is enabled |
| `last_run_at` | TIMESTAMPTZ | Yes | - | Last execution time |
| `next_run_at` | TIMESTAMPTZ | Yes | - | Next scheduled time |
| `created_by` | UUID | Yes | - | FK to users |
| `created_at` | TIMESTAMPTZ | No | now() | Created timestamp |
| `updated_at` | TIMESTAMPTZ | No | now() | Updated timestamp |

**Indexes:**
- `ix_scan_schedules_repo_id` on `repo_id`
- `ix_scan_schedules_next_run_at` on `next_run_at`

**Foreign Keys:**
- `repo_id` → `repositories(id)` ON DELETE CASCADE
- `created_by` → `users(id)`

## Security Tables

### api_keys

Stores API key metadata.

| Column | Type | Nullable | Default | Description |
|--------|------|----------|---------|-------------|
| `id` | UUID | No | uuid_generate_v4() | Primary key |
| `org_id` | UUID | No | - | FK to organizations |
| `name` | VARCHAR(255) | No | - | Key name/description |
| `key_hash` | VARCHAR(255) | No | - | Hashed API key |
| `key_prefix` | VARCHAR(10) | No | - | Key prefix for identification |
| `scopes` | JSON | No | ["read"] | Allowed scopes |
| `expires_at` | TIMESTAMPTZ | Yes | - | Expiration time |
| `last_used_at` | TIMESTAMPTZ | Yes | - | Last usage time |
| `created_by` | UUID | Yes | - | FK to users |
| `created_at` | TIMESTAMPTZ | No | now() | Created timestamp |

**Foreign Keys:**
- `org_id` → `organizations(id)` ON DELETE CASCADE
- `created_by` → `users(id)`

---

### audit_logs

Stores audit trail for compliance.

| Column | Type | Nullable | Default | Description |
|--------|------|----------|---------|-------------|
| `id` | UUID | No | uuid_generate_v4() | Primary key |
| `org_id` | UUID | Yes | - | FK to organizations |
| `user_id` | UUID | Yes | - | FK to users |
| `action` | VARCHAR(255) | No | - | Action performed |
| `resource_type` | VARCHAR(100) | Yes | - | Resource type affected |
| `resource_id` | UUID | Yes | - | Resource ID affected |
| `details` | JSON | No | {} | Additional details |
| `ip_address` | VARCHAR(45) | Yes | - | Client IP address |
| `user_agent` | TEXT | Yes | - | Client user agent |
| `created_at` | TIMESTAMPTZ | No | now() | Event timestamp |

**Indexes:**
- `idx_audit_org` on `(org_id, created_at)`

**Foreign Keys:**
- `org_id` → `organizations(id)` ON DELETE CASCADE
- `user_id` → `users(id)`

## Cache Tables

### trust_score_cache

Caches computed trust scores.

| Column | Type | Nullable | Default | Description |
|--------|------|----------|---------|-------------|
| `id` | UUID | No | - | Primary key |
| `repo_id` | UUID | No | - | FK to repositories |
| `file_path` | TEXT | No | - | File path |
| `commit_sha` | VARCHAR(40) | No | - | Commit SHA |
| `score` | FLOAT | No | - | Trust score (0-100) |
| `risk_level` | VARCHAR(50) | No | - | Risk: low, medium, high, critical |
| `ai_probability` | FLOAT | No | 0.0 | AI-generated probability |
| `factors` | JSON | No | {} | Score factors |
| `computed_at` | TIMESTAMPTZ | No | now() | Computation timestamp |

**Indexes:**
- `ix_trust_score_cache_repo_id` on `repo_id`
- `ix_trust_score_cache_file_commit` UNIQUE on `(repo_id, file_path, commit_sha)`

**Foreign Keys:**
- `repo_id` → `repositories(id)` ON DELETE CASCADE

---

### diff_summary_cache

Caches AI-generated PR summaries.

| Column | Type | Nullable | Default | Description |
|--------|------|----------|---------|-------------|
| `id` | UUID | No | - | Primary key |
| `repo_id` | UUID | No | - | FK to repositories |
| `pr_number` | INTEGER | No | - | Pull request number |
| `head_sha` | VARCHAR(40) | No | - | Head commit SHA |
| `title` | TEXT | No | - | Generated title |
| `description` | TEXT | No | - | Generated description |
| `changes` | JSON | No | [] | Change summary |
| `risk_assessment` | JSON | No | {} | Risk assessment |
| `computed_at` | TIMESTAMPTZ | No | now() | Computation timestamp |

**Indexes:**
- `ix_diff_summary_cache_repo_id` on `repo_id`
- `ix_diff_summary_cache_pr` UNIQUE on `(repo_id, pr_number, head_sha)`

**Foreign Keys:**
- `repo_id` → `repositories(id)` ON DELETE CASCADE

## Attestation Tables

### verification_attestations

Stores cryptographic verification attestations.

| Column | Type | Nullable | Default | Description |
|--------|------|----------|---------|-------------|
| `id` | UUID | No | - | Primary key |
| `org_id` | UUID | Yes | - | FK to organizations |
| `repo_id` | UUID | Yes | - | FK to repositories |
| `analysis_id` | UUID | Yes | - | FK to analyses |
| `attestation_type` | VARCHAR(50) | No | - | Type: pr, commit, release |
| `repo_full_name` | VARCHAR(512) | No | - | Full repository name |
| `ref` | VARCHAR(255) | Yes | - | Git ref (branch/tag) |
| `commit_sha` | VARCHAR(40) | Yes | - | Commit SHA |
| `file_paths` | JSON | No | [] | Files covered |
| `scope` | JSON | No | {} | Verification scope |
| `evidence` | JSON | No | {} | Verification evidence |
| `tier` | VARCHAR(50) | Yes | - | Certification tier |
| `passed` | BOOLEAN | No | false | Did verification pass |
| `signature` | TEXT | Yes | - | Cryptographic signature |
| `attestation_hash` | VARCHAR(64) | No | - | Attestation hash |
| `parent_attestation_id` | UUID | Yes | - | Parent attestation (for chains) |
| `metadata` | JSON | No | {} | Additional metadata |
| `expires_at` | TIMESTAMPTZ | Yes | - | Expiration time |
| `created_at` | TIMESTAMPTZ | No | now() | Created timestamp |

**Indexes:**
- `ix_verification_attestations_org_id` on `org_id`
- `ix_verification_attestations_repo_id` on `repo_id`
- `ix_verification_attestations_repo_full_name` on `repo_full_name`
- `ix_verification_attestations_tier` on `tier`
- `ix_verification_attestations_created_at` on `created_at`

**Foreign Keys:**
- `org_id` → `organizations(id)` ON DELETE CASCADE
- `repo_id` → `repositories(id)` ON DELETE CASCADE
- `analysis_id` → `analyses(id)` ON DELETE SET NULL
- `parent_attestation_id` → `verification_attestations(id)`

---

### verification_badges

Stores badge tokens for embedding.

| Column | Type | Nullable | Default | Description |
|--------|------|----------|---------|-------------|
| `id` | UUID | No | - | Primary key |
| `attestation_id` | UUID | No | - | FK to attestations |
| `repo_id` | UUID | Yes | - | FK to repositories |
| `repo_full_name` | VARCHAR(512) | No | - | Full repository name |
| `token` | VARCHAR(32) | No | - | Unique badge token |
| `tier` | VARCHAR(50) | Yes | - | Certification tier |
| `passed` | BOOLEAN | No | false | Current status |
| `style` | VARCHAR(50) | No | 'flat' | Badge style |
| `config` | JSON | No | {} | Badge configuration |
| `view_count` | INTEGER | No | 0 | Number of views |
| `last_viewed_at` | TIMESTAMPTZ | Yes | - | Last view time |
| `expires_at` | TIMESTAMPTZ | Yes | - | Expiration time |
| `created_at` | TIMESTAMPTZ | No | now() | Created timestamp |

**Indexes:**
- `ix_verification_badges_token` UNIQUE on `token`
- `ix_verification_badges_repo_full_name` on `repo_full_name`

**Foreign Keys:**
- `attestation_id` → `verification_attestations(id)` ON DELETE CASCADE
- `repo_id` → `repositories(id)` ON DELETE CASCADE

## Migrations

### Running Migrations

```bash
# Apply all pending migrations
cd apps/api && alembic upgrade head

# Rollback one migration
alembic downgrade -1

# Create new migration
alembic revision --autogenerate -m "Add new table"

# Show migration history
alembic history
```

### Migration Files

| Migration | Description |
|-----------|-------------|
| `001_initial` | Core tables (orgs, users, repos, analyses, findings) |
| `002_next_gen_features` | Webhooks, scans, notifications, caches |
| `003_verification_badges` | Attestations and badges |

## Maintenance

### Recommended Indexes

Already created in migrations. For additional query patterns, consider:

```sql
-- For dashboard queries
CREATE INDEX idx_analyses_org_created 
ON analyses(repo_id, created_at DESC);

-- For finding searches
CREATE INDEX idx_findings_text_search 
ON findings USING gin(to_tsvector('english', title || ' ' || description));
```

### Partitioning (Optional)

For high-volume deployments, consider partitioning:

```sql
-- Partition findings by analysis date
CREATE TABLE findings (
    -- ... columns ...
) PARTITION BY RANGE (created_at);

CREATE TABLE findings_2026_01 PARTITION OF findings
FOR VALUES FROM ('2026-01-01') TO ('2026-02-01');
```

### Backup

```bash
# Full backup
pg_dump -Fc codeverify > backup.dump

# Restore
pg_restore -d codeverify backup.dump
```
