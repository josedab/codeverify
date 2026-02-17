"""Embeddable Verification Widget.

Lightweight, embeddable JavaScript-compatible widget specification
for displaying verification status, trust scores, and finding
summaries on any platform.

Features:
- Widget configuration and theme management
- Verification status badge generation (SVG, HTML)
- Trust score display with visual indicator
- Finding summary with severity breakdown
- Embed code generation (iframe, React, Web Component)
- Widget data API for dynamic updates
"""

from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

import structlog

logger = structlog.get_logger()


class WidgetType(str, Enum):
    BADGE = "badge"
    TRUST_SCORE = "trust_score"
    FINDING_SUMMARY = "finding_summary"
    FULL_DASHBOARD = "full_dashboard"


class EmbedFormat(str, Enum):
    IFRAME = "iframe"
    REACT = "react"
    WEB_COMPONENT = "web_component"
    MARKDOWN = "markdown"
    SVG = "svg"


class WidgetTheme(str, Enum):
    LIGHT = "light"
    DARK = "dark"
    AUTO = "auto"


class BadgeStatus(str, Enum):
    PASSING = "passing"
    WARNING = "warning"
    FAILING = "failing"
    PENDING = "pending"
    UNKNOWN = "unknown"


@dataclass
class WidgetConfig:
    """Configuration for a widget instance."""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    widget_type: WidgetType = WidgetType.BADGE
    theme: WidgetTheme = WidgetTheme.LIGHT
    repo: str = ""
    branch: str = "main"
    width: int = 200
    height: int = 40
    show_details: bool = False
    auto_refresh_seconds: int = 300
    custom_css: str = ""
    token: str = field(default_factory=lambda: hashlib.sha256(
        uuid.uuid4().bytes).hexdigest()[:16])


@dataclass
class BadgeData:
    """Data for rendering a status badge."""
    label: str = "CodeVerify"
    status: BadgeStatus = BadgeStatus.UNKNOWN
    message: str = ""
    color: str = "#666"
    url: str = ""

    @property
    def status_color(self) -> str:
        colors = {
            BadgeStatus.PASSING: "#4c1",
            BadgeStatus.WARNING: "#dfb317",
            BadgeStatus.FAILING: "#e05d44",
            BadgeStatus.PENDING: "#9f9f9f",
            BadgeStatus.UNKNOWN: "#666",
        }
        return colors.get(self.status, self.color)


@dataclass
class TrustScoreData:
    """Data for rendering a trust score widget."""
    score: float = 0.0
    risk_level: str = "unknown"
    trend: str = "stable"
    last_scan: str = ""
    details_url: str = ""


@dataclass
class FindingSummaryData:
    """Data for rendering a finding summary widget."""
    total: int = 0
    critical: int = 0
    high: int = 0
    medium: int = 0
    low: int = 0
    last_scan: str = ""
    trend: str = "stable"


@dataclass
class WidgetRenderResult:
    """Result of rendering a widget."""
    widget_id: str = ""
    format: EmbedFormat = EmbedFormat.IFRAME
    html: str = ""
    embed_code: str = ""
    data_url: str = ""


class BadgeRenderer:
    """Renders SVG and HTML status badges."""

    def render_svg(self, data: BadgeData) -> str:
        label_width = len(data.label) * 7 + 10
        msg = data.message or data.status.value
        msg_width = len(msg) * 7 + 10
        total_width = label_width + msg_width

        return (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{total_width}" height="20">'
            f'<rect width="{label_width}" height="20" fill="#555"/>'
            f'<rect x="{label_width}" width="{msg_width}" height="20" fill="{data.status_color}"/>'
            f'<text x="{label_width // 2}" y="14" fill="#fff" text-anchor="middle" '
            f'font-family="sans-serif" font-size="11">{data.label}</text>'
            f'<text x="{label_width + msg_width // 2}" y="14" fill="#fff" text-anchor="middle" '
            f'font-family="sans-serif" font-size="11">{msg}</text>'
            f'</svg>'
        )

    def render_html(self, data: BadgeData) -> str:
        return (
            f'<span style="display:inline-flex;font-family:sans-serif;font-size:12px;">'
            f'<span style="background:#555;color:#fff;padding:2px 6px;border-radius:3px 0 0 3px;">'
            f'{data.label}</span>'
            f'<span style="background:{data.status_color};color:#fff;padding:2px 6px;'
            f'border-radius:0 3px 3px 0;">{data.message or data.status.value}</span></span>'
        )


class EmbedCodeGenerator:
    """Generates embed code for various formats."""

    BASE_URL = "https://codeverify.dev/widget"

    def generate(self, config: WidgetConfig, fmt: EmbedFormat) -> str:
        widget_url = f"{self.BASE_URL}/{config.id}?token={config.token}&theme={config.theme.value}"

        if fmt == EmbedFormat.IFRAME:
            return (
                f'<iframe src="{widget_url}" '
                f'width="{config.width}" height="{config.height}" '
                f'frameborder="0" style="border:none;"></iframe>'
            )

        if fmt == EmbedFormat.REACT:
            return (
                f'import {{ CodeVerifyWidget }} from "@codeverify/widget";\n\n'
                f'<CodeVerifyWidget\n'
                f'  widgetId="{config.id}"\n'
                f'  token="{config.token}"\n'
                f'  theme="{config.theme.value}"\n'
                f'  type="{config.widget_type.value}"\n'
                f'/>'
            )

        if fmt == EmbedFormat.WEB_COMPONENT:
            return (
                f'<script src="https://cdn.codeverify.dev/widget.js"></script>\n'
                f'<codeverify-widget\n'
                f'  widget-id="{config.id}"\n'
                f'  token="{config.token}"\n'
                f'  theme="{config.theme.value}"\n'
                f'  type="{config.widget_type.value}">\n'
                f'</codeverify-widget>'
            )

        if fmt == EmbedFormat.MARKDOWN:
            badge_url = f"{self.BASE_URL}/{config.id}/badge.svg?token={config.token}"
            return f'[![CodeVerify]({badge_url})](https://codeverify.dev)'

        if fmt == EmbedFormat.SVG:
            return f"{self.BASE_URL}/{config.id}/badge.svg?token={config.token}"

        return ""


class EmbeddableWidgetService:
    """Main service for the embeddable verification widget."""

    def __init__(self) -> None:
        self._configs: dict[str, WidgetConfig] = {}
        self._badge_renderer = BadgeRenderer()
        self._embed_gen = EmbedCodeGenerator()
        self._widget_data: dict[str, dict[str, Any]] = {}

    def create_widget(
        self,
        widget_type: WidgetType = WidgetType.BADGE,
        repo: str = "",
        theme: WidgetTheme = WidgetTheme.LIGHT,
        **kwargs: Any,
    ) -> WidgetConfig:
        """Create a new widget configuration."""
        config = WidgetConfig(
            widget_type=widget_type, repo=repo, theme=theme,
            **{k: v for k, v in kwargs.items() if k in WidgetConfig.__dataclass_fields__},
        )
        self._configs[config.id] = config
        return config

    def set_badge_data(
        self, widget_id: str, status: BadgeStatus, message: str = ""
    ) -> BadgeData:
        """Set badge data for a widget."""
        data = BadgeData(status=status, message=message)
        self._widget_data[widget_id] = {"badge": data}
        return data

    def set_trust_score_data(
        self, widget_id: str, score: float, risk_level: str = "medium"
    ) -> TrustScoreData:
        data = TrustScoreData(score=score, risk_level=risk_level)
        self._widget_data[widget_id] = {"trust_score": data}
        return data

    def set_finding_data(
        self, widget_id: str, critical: int = 0, high: int = 0,
        medium: int = 0, low: int = 0,
    ) -> FindingSummaryData:
        data = FindingSummaryData(
            total=critical + high + medium + low,
            critical=critical, high=high, medium=medium, low=low,
        )
        self._widget_data[widget_id] = {"findings": data}
        return data

    def render_badge(self, widget_id: str) -> str:
        """Render a badge as SVG."""
        data = self._widget_data.get(widget_id, {}).get("badge")
        if not data:
            data = BadgeData(status=BadgeStatus.UNKNOWN, message="no data")
        return self._badge_renderer.render_svg(data)

    def render_badge_html(self, widget_id: str) -> str:
        """Render a badge as HTML."""
        data = self._widget_data.get(widget_id, {}).get("badge")
        if not data:
            data = BadgeData(status=BadgeStatus.UNKNOWN, message="no data")
        return self._badge_renderer.render_html(data)

    def get_embed_code(
        self, widget_id: str, fmt: EmbedFormat = EmbedFormat.IFRAME
    ) -> str:
        """Get embed code for a widget."""
        config = self._configs.get(widget_id)
        if not config:
            return ""
        return self._embed_gen.generate(config, fmt)

    def get_widget_data_json(self, widget_id: str) -> str:
        """Get widget data as JSON for API responses."""
        data = self._widget_data.get(widget_id, {})
        serializable: dict[str, Any] = {}
        for key, val in data.items():
            if hasattr(val, '__dataclass_fields__'):
                serializable[key] = {k: getattr(val, k) for k in val.__dataclass_fields__}
            else:
                serializable[key] = val
        return json.dumps(serializable, indent=2, default=str)

    def get_widget(self, widget_id: str) -> WidgetConfig | None:
        return self._configs.get(widget_id)

    def list_widgets(self, repo: str = "") -> list[WidgetConfig]:
        widgets = list(self._configs.values())
        if repo:
            widgets = [w for w in widgets if w.repo == repo]
        return widgets


# ─── Singleton Access ──────────────────────────────────────────────────


_widget_instance: EmbeddableWidgetService | None = None


def get_widget_service() -> EmbeddableWidgetService:
    global _widget_instance
    if _widget_instance is None:
        _widget_instance = EmbeddableWidgetService()
    return _widget_instance


def reset_widget_service() -> None:
    global _widget_instance
    _widget_instance = None
