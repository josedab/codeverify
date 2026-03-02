"""Add multi-repository governance tables.

Revision ID: 005
Revises: 004_saas_multi_tenancy
Create Date: 2024-02-28

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "005_governance"
down_revision: str | None = "004_saas_multi_tenancy"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # Policy sets
    op.create_table(
        "policy_sets",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("org_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("version", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("rules", postgresql.JSON(), nullable=False, server_default="[]"),
        sa.Column("is_default", sa.Boolean(), server_default="false"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), onupdate=sa.func.now()),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_policy_sets_org_id", "policy_sets", ["org_id"])

    # Repo groups
    op.create_table(
        "repo_groups",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("org_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("policy_set_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("policy_sets.id", ondelete="SET NULL"), nullable=True),
        sa.Column("repo_patterns", postgresql.JSON(), server_default="[]"),
        sa.Column("repos", postgresql.JSON(), server_default="[]"),
        sa.Column("inherit_from", postgresql.UUID(as_uuid=True), sa.ForeignKey("repo_groups.id"), nullable=True),
        sa.Column("mode", sa.String(50), server_default="enforce"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), onupdate=sa.func.now()),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_repo_groups_org_id", "repo_groups", ["org_id"])
    op.create_index("ix_repo_groups_policy_set_id", "repo_groups", ["policy_set_id"])

    # Policy violations
    op.create_table(
        "policy_violations",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("org_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False),
        sa.Column("repo_full_name", sa.String(512), nullable=False),
        sa.Column("pr_number", sa.Integer(), nullable=False),
        sa.Column("policy_set_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("policy_sets.id"), nullable=False),
        sa.Column("rule_id", sa.String(255), nullable=False),
        sa.Column("rule_name", sa.String(255), nullable=False),
        sa.Column("severity", sa.String(50), nullable=False),
        sa.Column("action_taken", sa.String(50), nullable=False),
        sa.Column("finding_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("details", postgresql.JSON(), server_default="{}"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_policy_violations_org_id", "policy_violations", ["org_id"])
    op.create_index("ix_policy_violations_repo", "policy_violations", ["repo_full_name"])

    # Policy exceptions
    op.create_table(
        "policy_exceptions",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("org_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False),
        sa.Column("violation_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("policy_violations.id"), nullable=False),
        sa.Column("repo_full_name", sa.String(512), nullable=False),
        sa.Column("pr_number", sa.Integer(), nullable=False),
        sa.Column("rule_id", sa.String(255), nullable=False),
        sa.Column("reason", sa.Text(), nullable=False),
        sa.Column("requested_by", sa.String(255), nullable=False),
        sa.Column("approved_by", sa.String(255), nullable=True),
        sa.Column("status", sa.String(50), server_default="pending"),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_policy_exceptions_org_id", "policy_exceptions", ["org_id"])
    op.create_index("ix_policy_exceptions_status", "policy_exceptions", ["status"])

    # Governance audit log
    op.create_table(
        "governance_audit_log",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("org_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False),
        sa.Column("action", sa.String(255), nullable=False),
        sa.Column("actor", sa.String(255), nullable=False),
        sa.Column("resource_type", sa.String(100), nullable=False),
        sa.Column("resource_id", sa.String(255), nullable=False),
        sa.Column("details", postgresql.JSON(), server_default="{}"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_governance_audit_org_id", "governance_audit_log", ["org_id"])
    op.create_index("ix_governance_audit_action", "governance_audit_log", ["action"])
    op.create_index("ix_governance_audit_created_at", "governance_audit_log", ["created_at"])


def downgrade() -> None:
    op.drop_table("governance_audit_log")
    op.drop_table("policy_exceptions")
    op.drop_table("policy_violations")
    op.drop_table("repo_groups")
    op.drop_table("policy_sets")
