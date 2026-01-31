"""Add verification badges and attestation tables.

Revision ID: 003
Revises: 002_next_gen_features
Create Date: 2024-01-30

"""
from typing import Sequence, Union

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql
from alembic import op


# revision identifiers, used by Alembic.
revision: str = "003_verification_badges"
down_revision: Union[str, None] = "002_next_gen_features"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Create verification_attestations table
    op.create_table(
        "verification_attestations",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("org_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("repo_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("analysis_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("attestation_type", sa.String(50), nullable=False),
        sa.Column("repo_full_name", sa.String(512), nullable=False),
        sa.Column("ref", sa.String(255), nullable=True),
        sa.Column("commit_sha", sa.String(40), nullable=True),
        sa.Column("file_paths", postgresql.JSON(), nullable=False, default=[]),
        sa.Column("scope", postgresql.JSON(), nullable=False, default={}),
        sa.Column("evidence", postgresql.JSON(), nullable=False, default={}),
        sa.Column("tier", sa.String(50), nullable=True),
        sa.Column("passed", sa.Boolean(), nullable=False, default=False),
        sa.Column("signature", sa.Text(), nullable=True),
        sa.Column("attestation_hash", sa.String(64), nullable=False),
        sa.Column("parent_attestation_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("metadata", postgresql.JSON(), nullable=False, default={}),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(["org_id"], ["organizations.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["repo_id"], ["repositories.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["analysis_id"], ["analyses.id"], ondelete="SET NULL"),
        sa.ForeignKeyConstraint(["parent_attestation_id"], ["verification_attestations.id"]),
    )
    op.create_index("ix_verification_attestations_org_id", "verification_attestations", ["org_id"])
    op.create_index("ix_verification_attestations_repo_id", "verification_attestations", ["repo_id"])
    op.create_index("ix_verification_attestations_repo_full_name", "verification_attestations", ["repo_full_name"])
    op.create_index("ix_verification_attestations_tier", "verification_attestations", ["tier"])
    op.create_index("ix_verification_attestations_created_at", "verification_attestations", ["created_at"])

    # Create verification_badges table
    op.create_table(
        "verification_badges",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("attestation_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("repo_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("repo_full_name", sa.String(512), nullable=False),
        sa.Column("token", sa.String(32), nullable=False, unique=True),
        sa.Column("tier", sa.String(50), nullable=True),
        sa.Column("passed", sa.Boolean(), nullable=False, default=False),
        sa.Column("style", sa.String(50), nullable=False, default="flat"),
        sa.Column("config", postgresql.JSON(), nullable=False, default={}),
        sa.Column("view_count", sa.Integer(), nullable=False, default=0),
        sa.Column("last_viewed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(["attestation_id"], ["verification_attestations.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["repo_id"], ["repositories.id"], ondelete="CASCADE"),
    )
    op.create_index("ix_verification_badges_token", "verification_badges", ["token"], unique=True)
    op.create_index("ix_verification_badges_repo_full_name", "verification_badges", ["repo_full_name"])

    # Create certification_history table
    op.create_table(
        "certification_history",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("repo_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("attestation_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("previous_tier", sa.String(50), nullable=True),
        sa.Column("new_tier", sa.String(50), nullable=True),
        sa.Column("reason", sa.String(255), nullable=False),
        sa.Column("evidence_snapshot", postgresql.JSON(), nullable=False, default={}),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(["repo_id"], ["repositories.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["attestation_id"], ["verification_attestations.id"], ondelete="CASCADE"),
    )
    op.create_index("ix_certification_history_repo_id", "certification_history", ["repo_id"])
    op.create_index("ix_certification_history_created_at", "certification_history", ["created_at"])


def downgrade() -> None:
    op.drop_table("certification_history")
    op.drop_table("verification_badges")
    op.drop_table("verification_attestations")
