"""Widget API endpoints.

Serves widget data (badges, trust scores, finding summaries) and
embed code for the embeddable verification widget.
"""

from typing import Any

from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel, Field

router = APIRouter()


class WidgetCreateRequest(BaseModel):
    """Request to create a widget."""

    widget_type: str = Field(default="badge", description="badge, trust_score, finding_summary, full_dashboard")
    repo: str = Field(..., description="Repository (owner/name)")
    theme: str = Field(default="light", description="light, dark, auto")
    width: int = Field(default=200, ge=50, le=800)
    height: int = Field(default=40, ge=20, le=400)


class WidgetResponse(BaseModel):
    """Widget configuration response."""

    id: str
    widget_type: str
    repo: str
    theme: str
    token: str
    embed_iframe: str
    embed_react: str
    embed_markdown: str


class BadgeDataRequest(BaseModel):
    """Request to set badge data."""

    status: str = Field(..., description="passing, warning, failing, pending")
    message: str = Field(default="", description="Badge message text")


class TrustScoreDataRequest(BaseModel):
    """Request to set trust score data."""

    score: float = Field(..., ge=0.0, le=100.0)
    risk_level: str = Field(default="medium")


class FindingDataRequest(BaseModel):
    """Request to set finding summary data."""

    critical: int = Field(default=0, ge=0)
    high: int = Field(default=0, ge=0)
    medium: int = Field(default=0, ge=0)
    low: int = Field(default=0, ge=0)


@router.post("", status_code=status.HTTP_201_CREATED)
async def create_widget(request: WidgetCreateRequest) -> WidgetResponse:
    """Create a new embeddable widget."""
    from codeverify_core.embed_widget import (
        EmbedFormat,
        EmbeddableWidgetService,
        WidgetTheme,
        WidgetType,
    )

    svc = EmbeddableWidgetService()

    type_map = {
        "badge": WidgetType.BADGE,
        "trust_score": WidgetType.TRUST_SCORE,
        "finding_summary": WidgetType.FINDING_SUMMARY,
        "full_dashboard": WidgetType.FULL_DASHBOARD,
    }
    theme_map = {
        "light": WidgetTheme.LIGHT,
        "dark": WidgetTheme.DARK,
        "auto": WidgetTheme.AUTO,
    }

    config = svc.create_widget(
        widget_type=type_map.get(request.widget_type, WidgetType.BADGE),
        repo=request.repo,
        theme=theme_map.get(request.theme, WidgetTheme.LIGHT),
        width=request.width,
        height=request.height,
    )

    return WidgetResponse(
        id=config.id,
        widget_type=config.widget_type.value,
        repo=config.repo,
        theme=config.theme.value,
        token=config.token,
        embed_iframe=svc.get_embed_code(config.id, EmbedFormat.IFRAME),
        embed_react=svc.get_embed_code(config.id, EmbedFormat.REACT),
        embed_markdown=svc.get_embed_code(config.id, EmbedFormat.MARKDOWN),
    )


@router.get("/{widget_id}/badge.svg")
async def get_badge_svg(
    widget_id: str,
    token: str = Query(..., description="Widget access token"),
) -> Any:
    """Get badge SVG for embedding."""
    from fastapi.responses import Response

    from codeverify_core.embed_widget import EmbeddableWidgetService

    svc = EmbeddableWidgetService()
    widget = svc.get_widget(widget_id)
    if not widget or widget.token != token:
        raise HTTPException(status_code=404, detail="Widget not found")

    svg = svc.render_badge(widget_id)
    return Response(content=svg, media_type="image/svg+xml")


@router.put("/{widget_id}/badge")
async def update_badge_data(widget_id: str, request: BadgeDataRequest) -> dict[str, str]:
    """Update badge status data."""
    from codeverify_core.embed_widget import BadgeStatus, EmbeddableWidgetService

    svc = EmbeddableWidgetService()
    status_map = {
        "passing": BadgeStatus.PASSING,
        "warning": BadgeStatus.WARNING,
        "failing": BadgeStatus.FAILING,
        "pending": BadgeStatus.PENDING,
    }
    badge_status = status_map.get(request.status, BadgeStatus.UNKNOWN)
    svc.set_badge_data(widget_id, badge_status, request.message)
    return {"status": "updated"}


@router.put("/{widget_id}/trust-score")
async def update_trust_score_data(widget_id: str, request: TrustScoreDataRequest) -> dict[str, str]:
    """Update trust score widget data."""
    from codeverify_core.embed_widget import EmbeddableWidgetService

    svc = EmbeddableWidgetService()
    svc.set_trust_score_data(widget_id, request.score, request.risk_level)
    return {"status": "updated"}


@router.put("/{widget_id}/findings")
async def update_finding_data(widget_id: str, request: FindingDataRequest) -> dict[str, str]:
    """Update finding summary widget data."""
    from codeverify_core.embed_widget import EmbeddableWidgetService

    svc = EmbeddableWidgetService()
    svc.set_finding_data(widget_id, request.critical, request.high, request.medium, request.low)
    return {"status": "updated"}


@router.get("/{widget_id}/data")
async def get_widget_data(
    widget_id: str,
    token: str = Query(..., description="Widget access token"),
) -> Any:
    """Get widget data as JSON."""
    from fastapi.responses import JSONResponse

    from codeverify_core.embed_widget import EmbeddableWidgetService

    svc = EmbeddableWidgetService()
    widget = svc.get_widget(widget_id)
    if not widget or widget.token != token:
        raise HTTPException(status_code=404, detail="Widget not found")

    json_str = svc.get_widget_data_json(widget_id)
    return JSONResponse(content={"widget_id": widget_id, "data": json_str})
