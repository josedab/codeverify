"""Add SaaS multi-tenancy tables with row-level security.

Revision ID: 004
Revises: 003_verification_badges
Create Date: 2024-02-28

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "004_saas_multi_tenancy"
down_revision: str | None = "003_verification_badges"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # Subscriptions
    op.create_table(
        "subscriptions",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column(
            "org_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("organizations.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("tier", sa.String(50), nullable=False, server_default="free"),
        sa.Column("status", sa.String(50), nullable=False, server_default="active"),
        sa.Column("stripe_customer_id", sa.String(255), nullable=True),
        sa.Column("stripe_subscription_id", sa.String(255), nullable=True),
        sa.Column("stripe_price_id", sa.String(255), nullable=True),
        sa.Column("current_period_start", sa.DateTime(timezone=True), nullable=False),
        sa.Column("current_period_end", sa.DateTime(timezone=True), nullable=False),
        sa.Column("cancel_at_period_end", sa.Boolean(), server_default="false"),
        sa.Column("trial_end", sa.DateTime(timezone=True), nullable=True),
        sa.Column("metadata", postgresql.JSON(), server_default="{}"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            onupdate=sa.func.now(),
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_subscriptions_org_id", "subscriptions", ["org_id"], unique=True)
    op.create_index("ix_subscriptions_stripe_customer_id", "subscriptions", ["stripe_customer_id"])
    op.create_index(
        "ix_subscriptions_stripe_subscription_id", "subscriptions", ["stripe_subscription_id"]
    )

    # Usage meters
    op.create_table(
        "usage_meters",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column(
            "org_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("organizations.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("event_type", sa.String(100), nullable=False),
        sa.Column("quantity", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("unit_price_cents", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("idempotency_key", sa.String(255), nullable=True),
        sa.Column("metadata", postgresql.JSON(), server_default="{}"),
        sa.Column("reported_to_stripe", sa.Boolean(), server_default="false"),
        sa.Column("recorded_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_usage_meters_org_id", "usage_meters", ["org_id"])
    op.create_index("ix_usage_meters_event_type", "usage_meters", ["event_type"])
    op.create_index("ix_usage_meters_recorded_at", "usage_meters", ["recorded_at"])
    op.create_index(
        "ix_usage_meters_idempotency_key", "usage_meters", ["idempotency_key"], unique=True
    )

    # Stripe webhook events (idempotent processing)
    op.create_table(
        "stripe_events",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("stripe_event_id", sa.String(255), nullable=False, unique=True),
        sa.Column("event_type", sa.String(255), nullable=False),
        sa.Column("data", postgresql.JSON(), nullable=False),
        sa.Column("processed", sa.Boolean(), server_default="false"),
        sa.Column("processed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_stripe_events_stripe_event_id", "stripe_events", ["stripe_event_id"], unique=True
    )

    # Enable Row-Level Security on tenant-scoped tables
    tables_with_rls = [
        "organizations",
        "repositories",
        "analyses",
        "findings",
        "subscriptions",
        "usage_meters",
        "api_keys",
        "audit_logs",
    ]
    for table in tables_with_rls:
        op.execute(f"ALTER TABLE {table} ENABLE ROW LEVEL SECURITY;")
        op.execute(f"ALTER TABLE {table} FORCE ROW LEVEL SECURITY;")

        # Policy: app role can only see rows belonging to the current tenant
        col = "org_id" if table != "organizations" else "id"
        op.execute(
            f"CREATE POLICY tenant_isolation_{table} ON {table} "
            f"USING ({col} = current_setting('app.current_org_id')::uuid);"
        )

    # Superuser bypass policy for admin operations
    for table in tables_with_rls:
        op.execute(
            f"CREATE POLICY admin_bypass_{table} ON {table} "
            f"FOR ALL TO codeverify_admin USING (true);"
        )


def downgrade() -> None:
    tables_with_rls = [
        "organizations",
        "repositories",
        "analyses",
        "findings",
        "subscriptions",
        "usage_meters",
        "api_keys",
        "audit_logs",
    ]
    for table in tables_with_rls:
        op.execute(f"DROP POLICY IF EXISTS tenant_isolation_{table} ON {table};")
        op.execute(f"DROP POLICY IF EXISTS admin_bypass_{table} ON {table};")
        op.execute(f"ALTER TABLE {table} DISABLE ROW LEVEL SECURITY;")

    op.drop_table("stripe_events")
    op.drop_table("usage_meters")
    op.drop_table("subscriptions")
