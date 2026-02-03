"""Audit logs router for compliance and security tracking."""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Literal
from uuid import UUID

from fastapi import APIRouter, Depends, Query, Response
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy import func, select, and_
from sqlalchemy.ext.asyncio import AsyncSession
import csv
import io
import json

from codeverify_api.auth.dependencies import get_current_user
from codeverify_api.db.database import get_db
from codeverify_api.db.models import AuditLog, User, Organization

router = APIRouter(prefix="/audit-logs", tags=["audit-logs"])


class AuditLogResponse(BaseModel):
    """Audit log response model."""
    id: str
    org_id: str | None
    user_id: str | None
    username: str | None
    action: str
    resource_type: str | None
    resource_id: str | None
    details: dict[str, Any]
    ip_address: str | None
    user_agent: str | None
    created_at: str

    class Config:
        from_attributes = True


class AuditLogListResponse(BaseModel):
    """Paginated audit log list response."""
    items: list[AuditLogResponse]
    total: int
    page: int
    page_size: int
    total_pages: int


class AuditLogStats(BaseModel):
    """Audit log statistics."""
    total_events: int
    events_today: int
    events_this_week: int
    by_action: dict[str, int]
    by_resource_type: dict[str, int]
    top_users: list[dict[str, Any]]


@router.get("", response_model=AuditLogListResponse)
async def list_audit_logs(
    organization_id: UUID | None = Query(None, description="Filter by organization"),
    user_id: UUID | None = Query(None, description="Filter by user"),
    action: str | None = Query(None, description="Filter by action type"),
    resource_type: str | None = Query(None, description="Filter by resource type"),
    start_date: datetime | None = Query(None, description="Filter by start date"),
    end_date: datetime | None = Query(None, description="Filter by end date"),
    search: str | None = Query(None, description="Search in action or details"),
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(50, ge=1, le=100, description="Items per page"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> AuditLogListResponse:
    """List audit logs with filtering and pagination."""
    # Build filter conditions
    conditions = []
    
    if organization_id:
        conditions.append(AuditLog.org_id == organization_id)
    if user_id:
        conditions.append(AuditLog.user_id == user_id)
    if action:
        conditions.append(AuditLog.action == action)
    if resource_type:
        conditions.append(AuditLog.resource_type == resource_type)
    if start_date:
        conditions.append(AuditLog.created_at >= start_date)
    if end_date:
        conditions.append(AuditLog.created_at <= end_date)
    if search:
        conditions.append(
            AuditLog.action.ilike(f"%{search}%") |
            AuditLog.details.cast(str).ilike(f"%{search}%")
        )
    
    # Count total
    count_query = select(func.count(AuditLog.id))
    if conditions:
        count_query = count_query.where(and_(*conditions))
    total_result = await db.execute(count_query)
    total = total_result.scalar() or 0
    
    # Fetch logs with user join
    query = (
        select(AuditLog, User.username)
        .outerjoin(User, AuditLog.user_id == User.id)
        .order_by(AuditLog.created_at.desc())
        .offset((page - 1) * page_size)
        .limit(page_size)
    )
    if conditions:
        query = query.where(and_(*conditions))
    
    result = await db.execute(query)
    rows = result.all()
    
    items = [
        AuditLogResponse(
            id=str(log.id),
            org_id=str(log.org_id) if log.org_id else None,
            user_id=str(log.user_id) if log.user_id else None,
            username=username,
            action=log.action,
            resource_type=log.resource_type,
            resource_id=str(log.resource_id) if log.resource_id else None,
            details=log.details or {},
            ip_address=log.ip_address,
            user_agent=log.user_agent,
            created_at=log.created_at.isoformat(),
        )
        for log, username in rows
    ]
    
    return AuditLogListResponse(
        items=items,
        total=total,
        page=page,
        page_size=page_size,
        total_pages=(total + page_size - 1) // page_size,
    )


@router.get("/stats", response_model=AuditLogStats)
async def get_audit_log_stats(
    organization_id: UUID | None = Query(None, description="Filter by organization"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> AuditLogStats:
    """Get audit log statistics."""
    base_filter = []
    if organization_id:
        base_filter.append(AuditLog.org_id == organization_id)
    
    # Total events
    total_query = select(func.count(AuditLog.id))
    if base_filter:
        total_query = total_query.where(and_(*base_filter))
    total_result = await db.execute(total_query)
    total_events = total_result.scalar() or 0
    
    # Events today
    today_start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
    today_query = select(func.count(AuditLog.id)).where(
        AuditLog.created_at >= today_start,
        *base_filter
    )
    today_result = await db.execute(today_query)
    events_today = today_result.scalar() or 0
    
    # Events this week
    week_start = today_start - timedelta(days=today_start.weekday())
    week_query = select(func.count(AuditLog.id)).where(
        AuditLog.created_at >= week_start,
        *base_filter
    )
    week_result = await db.execute(week_query)
    events_this_week = week_result.scalar() or 0
    
    # By action
    action_query = (
        select(AuditLog.action, func.count(AuditLog.id))
        .group_by(AuditLog.action)
    )
    if base_filter:
        action_query = action_query.where(and_(*base_filter))
    action_result = await db.execute(action_query)
    by_action = {row[0]: row[1] for row in action_result.all()}
    
    # By resource type
    resource_query = (
        select(AuditLog.resource_type, func.count(AuditLog.id))
        .where(AuditLog.resource_type.isnot(None))
        .group_by(AuditLog.resource_type)
    )
    if base_filter:
        resource_query = resource_query.where(and_(*base_filter))
    resource_result = await db.execute(resource_query)
    by_resource_type = {row[0]: row[1] for row in resource_result.all()}
    
    # Top users
    user_query = (
        select(User.username, func.count(AuditLog.id).label("count"))
        .join(User, AuditLog.user_id == User.id)
        .group_by(User.username)
        .order_by(func.count(AuditLog.id).desc())
        .limit(10)
    )
    if base_filter:
        user_query = user_query.where(and_(*base_filter))
    user_result = await db.execute(user_query)
    top_users = [
        {"username": row[0], "event_count": row[1]}
        for row in user_result.all()
    ]
    
    return AuditLogStats(
        total_events=total_events,
        events_today=events_today,
        events_this_week=events_this_week,
        by_action=by_action,
        by_resource_type=by_resource_type,
        top_users=top_users,
    )


@router.get("/actions")
async def get_audit_actions(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> list[str]:
    """Get list of unique audit actions."""
    query = select(AuditLog.action).distinct().order_by(AuditLog.action)
    result = await db.execute(query)
    return [row[0] for row in result.all()]


@router.get("/resource-types")
async def get_resource_types(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> list[str]:
    """Get list of unique resource types."""
    query = (
        select(AuditLog.resource_type)
        .where(AuditLog.resource_type.isnot(None))
        .distinct()
        .order_by(AuditLog.resource_type)
    )
    result = await db.execute(query)
    return [row[0] for row in result.all()]


@router.get("/export")
async def export_audit_logs(
    organization_id: UUID | None = Query(None, description="Filter by organization"),
    start_date: datetime | None = Query(None, description="Filter by start date"),
    end_date: datetime | None = Query(None, description="Filter by end date"),
    format: Literal["csv", "json"] = Query("csv", description="Export format"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> StreamingResponse:
    """Export audit logs for compliance."""
    conditions = []
    if organization_id:
        conditions.append(AuditLog.org_id == organization_id)
    if start_date:
        conditions.append(AuditLog.created_at >= start_date)
    if end_date:
        conditions.append(AuditLog.created_at <= end_date)
    
    query = (
        select(AuditLog, User.username)
        .outerjoin(User, AuditLog.user_id == User.id)
        .order_by(AuditLog.created_at.desc())
    )
    if conditions:
        query = query.where(and_(*conditions))
    
    # Limit export to 10000 records
    query = query.limit(10000)
    
    result = await db.execute(query)
    rows = result.all()
    
    if format == "csv":
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow([
            "ID", "Timestamp", "User", "Action", "Resource Type",
            "Resource ID", "IP Address", "User Agent", "Details"
        ])
        for log, username in rows:
            writer.writerow([
                str(log.id),
                log.created_at.isoformat(),
                username or "system",
                log.action,
                log.resource_type or "",
                str(log.resource_id) if log.resource_id else "",
                log.ip_address or "",
                log.user_agent or "",
                json.dumps(log.details) if log.details else "",
            ])
        
        output.seek(0)
        return StreamingResponse(
            iter([output.getvalue()]),
            media_type="text/csv",
            headers={
                "Content-Disposition": f"attachment; filename=audit-logs-{datetime.utcnow().strftime('%Y%m%d')}.csv"
            }
        )
    else:
        data = [
            {
                "id": str(log.id),
                "timestamp": log.created_at.isoformat(),
                "user": username or "system",
                "action": log.action,
                "resource_type": log.resource_type,
                "resource_id": str(log.resource_id) if log.resource_id else None,
                "ip_address": log.ip_address,
                "user_agent": log.user_agent,
                "details": log.details,
            }
            for log, username in rows
        ]
        
        return StreamingResponse(
            iter([json.dumps(data, indent=2)]),
            media_type="application/json",
            headers={
                "Content-Disposition": f"attachment; filename=audit-logs-{datetime.utcnow().strftime('%Y%m%d')}.json"
            }
        )


@router.get("/{log_id}", response_model=AuditLogResponse)
async def get_audit_log(
    log_id: UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> AuditLogResponse:
    """Get a specific audit log entry."""
    query = (
        select(AuditLog, User.username)
        .outerjoin(User, AuditLog.user_id == User.id)
        .where(AuditLog.id == log_id)
    )
    result = await db.execute(query)
    row = result.first()
    
    if not row:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="Audit log not found")
    
    log, username = row
    return AuditLogResponse(
        id=str(log.id),
        org_id=str(log.org_id) if log.org_id else None,
        user_id=str(log.user_id) if log.user_id else None,
        username=username,
        action=log.action,
        resource_type=log.resource_type,
        resource_id=str(log.resource_id) if log.resource_id else None,
        details=log.details or {},
        ip_address=log.ip_address,
        user_agent=log.user_agent,
        created_at=log.created_at.isoformat(),
    )
