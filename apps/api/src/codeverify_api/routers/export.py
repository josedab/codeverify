"""Export router for compliance reports (CSV/PDF)."""
from __future__ import annotations

import csv
import io
from datetime import datetime, timedelta
from typing import Any
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from codeverify_api.auth.dependencies import get_current_user
from codeverify_api.db.database import get_db
from codeverify_api.db.models import Analysis, User

router = APIRouter(prefix="/export", tags=["export"])


class ExportFilters(BaseModel):
    """Filters for export."""
    
    organization_id: UUID | None = None
    repository_id: UUID | None = None
    start_date: datetime | None = None
    end_date: datetime | None = None
    severities: list[str] | None = Field(default=None, description="Filter by severities")
    categories: list[str] | None = Field(default=None, description="Filter by categories")
    status: str | None = Field(default=None, description="Filter by analysis status")


class ExportResponse(BaseModel):
    """Export job response."""
    
    export_id: str
    status: str
    format: str
    download_url: str | None


def _generate_csv_content(analyses: list[dict[str, Any]]) -> str:
    """Generate CSV content from analyses."""
    output = io.StringIO()
    
    # Define columns
    fieldnames = [
        "analysis_id",
        "repository",
        "pr_number",
        "commit_sha",
        "status",
        "total_findings",
        "critical_count",
        "high_count",
        "medium_count",
        "low_count",
        "pass_fail",
        "started_at",
        "completed_at",
        "duration_seconds",
    ]
    
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()
    
    for analysis in analyses:
        summary = analysis.get("summary", {}) or {}
        started = analysis.get("started_at")
        completed = analysis.get("completed_at")
        
        duration = None
        if started and completed:
            try:
                if isinstance(started, str):
                    started = datetime.fromisoformat(started.replace("Z", "+00:00"))
                if isinstance(completed, str):
                    completed = datetime.fromisoformat(completed.replace("Z", "+00:00"))
                duration = (completed - started).total_seconds()
            except Exception:
                pass
        
        row = {
            "analysis_id": str(analysis.get("id", "")),
            "repository": analysis.get("repo_full_name", ""),
            "pr_number": analysis.get("pr_number", ""),
            "commit_sha": analysis.get("head_sha", "")[:8] if analysis.get("head_sha") else "",
            "status": analysis.get("status", ""),
            "total_findings": summary.get("total_issues", 0),
            "critical_count": summary.get("critical", 0),
            "high_count": summary.get("high", 0),
            "medium_count": summary.get("medium", 0),
            "low_count": summary.get("low", 0),
            "pass_fail": "PASS" if summary.get("pass", True) else "FAIL",
            "started_at": str(analysis.get("started_at", "")),
            "completed_at": str(analysis.get("completed_at", "")),
            "duration_seconds": duration,
        }
        writer.writerow(row)
    
    return output.getvalue()


def _generate_findings_csv(findings: list[dict[str, Any]]) -> str:
    """Generate CSV content from findings."""
    output = io.StringIO()
    
    fieldnames = [
        "finding_id",
        "analysis_id",
        "category",
        "severity",
        "title",
        "description",
        "file_path",
        "line_start",
        "line_end",
        "confidence",
        "verification_type",
        "fix_suggestion",
    ]
    
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()
    
    for finding in findings:
        row = {
            "finding_id": finding.get("id", ""),
            "analysis_id": finding.get("analysis_id", ""),
            "category": finding.get("category", ""),
            "severity": finding.get("severity", ""),
            "title": finding.get("title", ""),
            "description": (finding.get("description", "") or "")[:500],
            "file_path": finding.get("file_path", ""),
            "line_start": finding.get("line_start", ""),
            "line_end": finding.get("line_end", ""),
            "confidence": finding.get("confidence", ""),
            "verification_type": finding.get("verification_type", ""),
            "fix_suggestion": (finding.get("fix_suggestion", "") or "")[:200],
        }
        writer.writerow(row)
    
    return output.getvalue()


def _generate_pdf_content(
    analyses: list[dict[str, Any]],
    findings: list[dict[str, Any]],
    organization_name: str,
    date_range: str,
) -> bytes:
    """Generate PDF report content.
    
    Note: Uses simple text-based PDF generation.
    For production, consider using reportlab or weasyprint.
    """
    # Simple PDF structure (minimal implementation)
    # In production, use a proper PDF library
    
    lines = []
    lines.append("CodeVerify Compliance Report")
    lines.append("=" * 50)
    lines.append("")
    lines.append(f"Organization: {organization_name}")
    lines.append(f"Date Range: {date_range}")
    lines.append(f"Generated: {datetime.utcnow().isoformat()}")
    lines.append("")
    lines.append("=" * 50)
    lines.append("")
    
    # Summary statistics
    total_analyses = len(analyses)
    passed = sum(1 for a in analyses if (a.get("summary") or {}).get("pass", True))
    failed = total_analyses - passed
    
    total_findings = len(findings)
    critical = sum(1 for f in findings if f.get("severity") == "critical")
    high = sum(1 for f in findings if f.get("severity") == "high")
    medium = sum(1 for f in findings if f.get("severity") == "medium")
    low = sum(1 for f in findings if f.get("severity") == "low")
    
    lines.append("EXECUTIVE SUMMARY")
    lines.append("-" * 30)
    lines.append(f"Total Analyses: {total_analyses}")
    lines.append(f"  - Passed: {passed}")
    lines.append(f"  - Failed: {failed}")
    lines.append(f"  - Pass Rate: {passed/total_analyses*100:.1f}%" if total_analyses else "N/A")
    lines.append("")
    lines.append(f"Total Findings: {total_findings}")
    lines.append(f"  - Critical: {critical}")
    lines.append(f"  - High: {high}")
    lines.append(f"  - Medium: {medium}")
    lines.append(f"  - Low: {low}")
    lines.append("")
    lines.append("=" * 50)
    lines.append("")
    
    # Analyses detail
    lines.append("ANALYSES DETAIL")
    lines.append("-" * 30)
    
    for analysis in analyses[:50]:
        summary = analysis.get("summary") or {}
        lines.append(f"  {analysis.get('repo_full_name', 'Unknown')} PR #{analysis.get('pr_number', '?')}")
        lines.append(f"    Status: {'PASS' if summary.get('pass', True) else 'FAIL'}")
        lines.append(f"    Findings: {summary.get('total_issues', 0)}")
        lines.append("")
    
    if len(analyses) > 50:
        lines.append(f"  ... and {len(analyses) - 50} more analyses")
    
    lines.append("")
    lines.append("=" * 50)
    lines.append("")
    
    # Critical findings detail
    critical_findings = [f for f in findings if f.get("severity") in ("critical", "high")]
    
    if critical_findings:
        lines.append("CRITICAL & HIGH SEVERITY FINDINGS")
        lines.append("-" * 30)
        
        for finding in critical_findings[:20]:
            lines.append(f"  [{finding.get('severity', '').upper()}] {finding.get('title', 'Unknown')}")
            lines.append(f"    File: {finding.get('file_path', '')}:{finding.get('line_start', '')}")
            lines.append(f"    Category: {finding.get('category', '')}")
            if finding.get("description"):
                desc = finding["description"][:100]
                lines.append(f"    Description: {desc}...")
            lines.append("")
        
        if len(critical_findings) > 20:
            lines.append(f"  ... and {len(critical_findings) - 20} more critical/high findings")
    
    lines.append("")
    lines.append("=" * 50)
    lines.append("END OF REPORT")
    
    # Convert to bytes (simple text-based "PDF")
    # For production, use reportlab
    content = "\n".join(lines)
    return content.encode("utf-8")


# Mock data storage (in production, would query database)
_mock_analyses: list[dict[str, Any]] = [
    {
        "id": "analysis-001",
        "repo_full_name": "org/repo",
        "pr_number": 42,
        "head_sha": "abc123def456",
        "status": "completed",
        "summary": {"total_issues": 3, "critical": 0, "high": 1, "medium": 2, "low": 0, "pass": True},
        "started_at": datetime.utcnow() - timedelta(hours=2),
        "completed_at": datetime.utcnow() - timedelta(hours=1, minutes=55),
    },
]

_mock_findings: list[dict[str, Any]] = [
    {
        "id": "finding-001",
        "analysis_id": "analysis-001",
        "category": "security",
        "severity": "high",
        "title": "SQL Injection Vulnerability",
        "description": "User input is directly interpolated into SQL query",
        "file_path": "src/database.py",
        "line_start": 42,
        "line_end": 45,
        "confidence": 0.95,
        "verification_type": "ai",
        "fix_suggestion": "Use parameterized queries",
    },
]


@router.get("/analyses/csv")
async def export_analyses_csv(
    organization_id: UUID | None = None,
    repository_id: UUID | None = None,
    start_date: datetime | None = None,
    end_date: datetime | None = None,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> StreamingResponse:
    """Export analyses to CSV format.
    
    Generates a CSV file with analysis summaries for compliance reporting.
    """
    # In production, would filter by organization_id, repository_id, dates
    # For now, use mock data
    analyses = _mock_analyses
    
    csv_content = _generate_csv_content(analyses)
    
    # Generate filename with date
    filename = f"codeverify_analyses_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.csv"
    
    return StreamingResponse(
        iter([csv_content]),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


@router.get("/findings/csv")
async def export_findings_csv(
    organization_id: UUID | None = None,
    repository_id: UUID | None = None,
    severities: str | None = Query(None, description="Comma-separated severities"),
    categories: str | None = Query(None, description="Comma-separated categories"),
    start_date: datetime | None = None,
    end_date: datetime | None = None,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> StreamingResponse:
    """Export findings to CSV format.
    
    Generates a CSV file with all findings for compliance reporting.
    """
    findings = _mock_findings
    
    # Apply filters
    if severities:
        sev_list = [s.strip().lower() for s in severities.split(",")]
        findings = [f for f in findings if f.get("severity", "").lower() in sev_list]
    
    if categories:
        cat_list = [c.strip().lower() for c in categories.split(",")]
        findings = [f for f in findings if f.get("category", "").lower() in cat_list]
    
    csv_content = _generate_findings_csv(findings)
    
    filename = f"codeverify_findings_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.csv"
    
    return StreamingResponse(
        iter([csv_content]),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


@router.get("/report/pdf")
async def export_compliance_pdf(
    organization_id: UUID | None = None,
    start_date: datetime | None = None,
    end_date: datetime | None = None,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> StreamingResponse:
    """Generate PDF compliance report.
    
    Comprehensive report suitable for compliance audits and management review.
    Uses ReportLab for professional PDF generation.
    """
    from codeverify_api.services.pdf_generator import generate_pdf_report
    
    # In production, would query actual data
    analyses = _mock_analyses
    findings = _mock_findings
    
    # Format date range
    if start_date and end_date:
        date_range = f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}"
    else:
        date_range = "All time"
    
    organization_name = "Organization"  # Would get from database
    
    # Generate PDF using reportlab
    pdf_content = generate_pdf_report(
        analyses=analyses,
        findings=findings,
        organization_name=organization_name,
        date_range=date_range,
    )
    
    filename = f"codeverify_compliance_report_{datetime.utcnow().strftime('%Y%m%d')}.pdf"
    
    return StreamingResponse(
        iter([pdf_content]),
        media_type="application/pdf",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


@router.get("/summary")
async def get_export_summary(
    organization_id: UUID | None = None,
    start_date: datetime | None = None,
    end_date: datetime | None = None,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> dict[str, Any]:
    """Get summary of data available for export.
    
    Use this to preview what will be included in exports.
    """
    analyses = _mock_analyses
    findings = _mock_findings
    
    total_analyses = len(analyses)
    total_findings = len(findings)
    
    severity_breakdown = {}
    for finding in findings:
        sev = finding.get("severity", "unknown")
        severity_breakdown[sev] = severity_breakdown.get(sev, 0) + 1
    
    category_breakdown = {}
    for finding in findings:
        cat = finding.get("category", "unknown")
        category_breakdown[cat] = category_breakdown.get(cat, 0) + 1
    
    return {
        "total_analyses": total_analyses,
        "total_findings": total_findings,
        "severity_breakdown": severity_breakdown,
        "category_breakdown": category_breakdown,
        "date_range": {
            "earliest": min((a.get("started_at") for a in analyses), default=None),
            "latest": max((a.get("completed_at") for a in analyses), default=None),
        },
        "export_formats": ["csv", "pdf"],
        "export_endpoints": {
            "analyses_csv": "/api/v1/export/analyses/csv",
            "findings_csv": "/api/v1/export/findings/csv",
            "compliance_pdf": "/api/v1/export/report/pdf",
        },
    }
