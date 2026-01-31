"""Add webhook, scan, and notification models.

Revision ID: 002
Revises: 001_initial
Create Date: 2024-01-29

"""
from typing import Sequence, Union

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql
from alembic import op


# revision identifiers, used by Alembic.
revision: str = "002_next_gen_features"
down_revision: Union[str, None] = "001_initial"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Create webhooks table
    op.create_table(
        "webhooks",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("org_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("url", sa.Text(), nullable=False),
        sa.Column("secret_hash", sa.String(255), nullable=True),
        sa.Column("events", postgresql.JSON(), nullable=False, default=[]),
        sa.Column("active", sa.Boolean(), nullable=False, default=True),
        sa.Column("last_triggered_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("failure_count", sa.Integer(), nullable=False, default=0),
        sa.Column("created_by", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(["org_id"], ["organizations.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["created_by"], ["users.id"]),
    )
    op.create_index("ix_webhooks_org_id", "webhooks", ["org_id"])

    # Create webhook_deliveries table
    op.create_table(
        "webhook_deliveries",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("webhook_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("event", sa.String(100), nullable=False),
        sa.Column("payload", postgresql.JSON(), nullable=False),
        sa.Column("response_status", sa.Integer(), nullable=True),
        sa.Column("response_body", sa.Text(), nullable=True),
        sa.Column("success", sa.Boolean(), nullable=False, default=False),
        sa.Column("delivered_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(["webhook_id"], ["webhooks.id"], ondelete="CASCADE"),
    )
    op.create_index("ix_webhook_deliveries_webhook_id", "webhook_deliveries", ["webhook_id"])
    op.create_index("ix_webhook_deliveries_delivered_at", "webhook_deliveries", ["delivered_at"])

    # Create codebase_scans table
    op.create_table(
        "codebase_scans",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("repo_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("branch", sa.String(255), nullable=False, default="main"),
        sa.Column("status", sa.String(50), nullable=False, default="pending"),
        sa.Column("triggered_by", sa.String(50), nullable=False, default="manual"),
        sa.Column("config", postgresql.JSON(), nullable=False, default={}),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("files_scanned", sa.Integer(), nullable=False, default=0),
        sa.Column("findings_count", sa.Integer(), nullable=False, default=0),
        sa.Column("results", postgresql.JSON(), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(["repo_id"], ["repositories.id"], ondelete="CASCADE"),
    )
    op.create_index("ix_codebase_scans_repo_id", "codebase_scans", ["repo_id"])
    op.create_index("ix_codebase_scans_status", "codebase_scans", ["status"])

    # Create scan_schedules table
    op.create_table(
        "scan_schedules",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("repo_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("branch", sa.String(255), nullable=False, default="main"),
        sa.Column("cron_expression", sa.String(100), nullable=False),
        sa.Column("config", postgresql.JSON(), nullable=False, default={}),
        sa.Column("enabled", sa.Boolean(), nullable=False, default=True),
        sa.Column("last_run_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("next_run_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_by", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(["repo_id"], ["repositories.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["created_by"], ["users.id"]),
    )
    op.create_index("ix_scan_schedules_repo_id", "scan_schedules", ["repo_id"])
    op.create_index("ix_scan_schedules_next_run_at", "scan_schedules", ["next_run_at"])

    # Create notification_configs table
    op.create_table(
        "notification_configs",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("org_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("channel_type", sa.String(50), nullable=False),
        sa.Column("webhook_url", sa.Text(), nullable=True),
        sa.Column("config", postgresql.JSON(), nullable=False, default={}),
        sa.Column("events", postgresql.JSON(), nullable=False, default=[]),
        sa.Column("enabled", sa.Boolean(), nullable=False, default=True),
        sa.Column("created_by", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(["org_id"], ["organizations.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["created_by"], ["users.id"]),
    )
    op.create_index("ix_notification_configs_org_id", "notification_configs", ["org_id"])

    # Create trust_score_cache table
    op.create_table(
        "trust_score_cache",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("repo_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("file_path", sa.Text(), nullable=False),
        sa.Column("commit_sha", sa.String(40), nullable=False),
        sa.Column("score", sa.Float(), nullable=False),
        sa.Column("risk_level", sa.String(50), nullable=False),
        sa.Column("ai_probability", sa.Float(), nullable=False, default=0.0),
        sa.Column("factors", postgresql.JSON(), nullable=False, default={}),
        sa.Column("computed_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(["repo_id"], ["repositories.id"], ondelete="CASCADE"),
    )
    op.create_index("ix_trust_score_cache_repo_id", "trust_score_cache", ["repo_id"])
    op.create_index(
        "ix_trust_score_cache_file_commit",
        "trust_score_cache",
        ["repo_id", "file_path", "commit_sha"],
        unique=True,
    )

    # Create diff_summary_cache table
    op.create_table(
        "diff_summary_cache",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("repo_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("pr_number", sa.Integer(), nullable=False),
        sa.Column("head_sha", sa.String(40), nullable=False),
        sa.Column("title", sa.Text(), nullable=False),
        sa.Column("description", sa.Text(), nullable=False),
        sa.Column("changes", postgresql.JSON(), nullable=False, default=[]),
        sa.Column("risk_assessment", postgresql.JSON(), nullable=False, default={}),
        sa.Column("computed_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(["repo_id"], ["repositories.id"], ondelete="CASCADE"),
    )
    op.create_index("ix_diff_summary_cache_repo_id", "diff_summary_cache", ["repo_id"])
    op.create_index(
        "ix_diff_summary_cache_pr",
        "diff_summary_cache",
        ["repo_id", "pr_number", "head_sha"],
        unique=True,
    )


def downgrade() -> None:
    op.drop_table("diff_summary_cache")
    op.drop_table("trust_score_cache")
    op.drop_table("notification_configs")
    op.drop_table("scan_schedules")
    op.drop_table("codebase_scans")
    op.drop_table("webhook_deliveries")
    op.drop_table("webhooks")
