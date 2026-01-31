"""PDF Report Generator using ReportLab.

Generates professional compliance reports in PDF format.
"""

from __future__ import annotations

import io
from datetime import datetime
from typing import Any

try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
        PageBreak, Image, HRFlowable
    )
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False


class PDFReportGenerator:
    """Generate professional PDF compliance reports."""
    
    def __init__(
        self,
        organization_name: str = "Organization",
        logo_path: str | None = None,
    ):
        self.organization_name = organization_name
        self.logo_path = logo_path
        
        if REPORTLAB_AVAILABLE:
            self.styles = getSampleStyleSheet()
            self._setup_custom_styles()
    
    def _setup_custom_styles(self) -> None:
        """Set up custom paragraph styles."""
        self.styles.add(ParagraphStyle(
            name='Title',
            parent=self.styles['Heading1'],
            fontSize=24,
            spaceAfter=30,
            alignment=TA_CENTER,
            textColor=colors.HexColor('#1a365d'),
        ))
        
        self.styles.add(ParagraphStyle(
            name='Subtitle',
            parent=self.styles['Normal'],
            fontSize=12,
            spaceAfter=20,
            alignment=TA_CENTER,
            textColor=colors.HexColor('#4a5568'),
        ))
        
        self.styles.add(ParagraphStyle(
            name='SectionHeader',
            parent=self.styles['Heading2'],
            fontSize=14,
            spaceBefore=20,
            spaceAfter=10,
            textColor=colors.HexColor('#2d3748'),
            borderPadding=5,
        ))
        
        self.styles.add(ParagraphStyle(
            name='FindingTitle',
            parent=self.styles['Normal'],
            fontSize=11,
            spaceBefore=10,
            spaceAfter=5,
            textColor=colors.HexColor('#1a202c'),
            fontName='Helvetica-Bold',
        ))
        
        self.styles.add(ParagraphStyle(
            name='FindingBody',
            parent=self.styles['Normal'],
            fontSize=10,
            spaceAfter=10,
            textColor=colors.HexColor('#4a5568'),
            leftIndent=20,
        ))
    
    def generate(
        self,
        analyses: list[dict[str, Any]],
        findings: list[dict[str, Any]],
        date_range: str = "All time",
    ) -> bytes:
        """Generate PDF report.
        
        Args:
            analyses: List of analysis results
            findings: List of findings
            date_range: Date range string for report header
            
        Returns:
            PDF content as bytes
        """
        if not REPORTLAB_AVAILABLE:
            return self._generate_fallback(analyses, findings, date_range)
        
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(
            buffer,
            pagesize=letter,
            rightMargin=0.75*inch,
            leftMargin=0.75*inch,
            topMargin=0.75*inch,
            bottomMargin=0.75*inch,
        )
        
        # Build document content
        story = []
        
        # Title page
        story.extend(self._build_title_page(date_range))
        story.append(PageBreak())
        
        # Executive summary
        story.extend(self._build_executive_summary(analyses, findings))
        story.append(PageBreak())
        
        # Analyses summary
        story.extend(self._build_analyses_section(analyses))
        
        # Findings detail
        if findings:
            story.append(PageBreak())
            story.extend(self._build_findings_section(findings))
        
        # Build PDF
        doc.build(story, onFirstPage=self._add_header_footer, onLaterPages=self._add_header_footer)
        
        buffer.seek(0)
        return buffer.getvalue()
    
    def _build_title_page(self, date_range: str) -> list[Any]:
        """Build title page elements."""
        elements = []
        
        elements.append(Spacer(1, 2*inch))
        
        # Logo placeholder
        if self.logo_path:
            try:
                elements.append(Image(self.logo_path, width=2*inch, height=1*inch))
                elements.append(Spacer(1, 0.5*inch))
            except Exception:
                pass
        
        elements.append(Paragraph("CodeVerify", self.styles['Title']))
        elements.append(Paragraph("Compliance Report", self.styles['Title']))
        elements.append(Spacer(1, 0.5*inch))
        
        elements.append(Paragraph(
            f"<b>Organization:</b> {self.organization_name}",
            self.styles['Subtitle']
        ))
        elements.append(Paragraph(
            f"<b>Report Period:</b> {date_range}",
            self.styles['Subtitle']
        ))
        elements.append(Paragraph(
            f"<b>Generated:</b> {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}",
            self.styles['Subtitle']
        ))
        
        elements.append(Spacer(1, 1*inch))
        
        # Disclaimer
        disclaimer = """
        This report contains confidential security analysis data. 
        Distribution should be limited to authorized personnel only.
        """
        elements.append(Paragraph(disclaimer, self.styles['Normal']))
        
        return elements
    
    def _build_executive_summary(
        self,
        analyses: list[dict[str, Any]],
        findings: list[dict[str, Any]],
    ) -> list[Any]:
        """Build executive summary section."""
        elements = []
        
        elements.append(Paragraph("Executive Summary", self.styles['SectionHeader']))
        elements.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#e2e8f0')))
        elements.append(Spacer(1, 0.25*inch))
        
        # Calculate statistics
        total_analyses = len(analyses)
        passed = sum(1 for a in analyses if (a.get("summary") or {}).get("pass", True))
        failed = total_analyses - passed
        pass_rate = (passed / total_analyses * 100) if total_analyses > 0 else 0
        
        total_findings = len(findings)
        critical = sum(1 for f in findings if f.get("severity") == "critical")
        high = sum(1 for f in findings if f.get("severity") == "high")
        medium = sum(1 for f in findings if f.get("severity") == "medium")
        low = sum(1 for f in findings if f.get("severity") == "low")
        
        # Summary table
        summary_data = [
            ['Metric', 'Value'],
            ['Total Analyses', str(total_analyses)],
            ['Passed', str(passed)],
            ['Failed', str(failed)],
            ['Pass Rate', f'{pass_rate:.1f}%'],
            ['', ''],
            ['Total Findings', str(total_findings)],
            ['Critical', str(critical)],
            ['High', str(high)],
            ['Medium', str(medium)],
            ['Low', str(low)],
        ]
        
        table = Table(summary_data, colWidths=[3*inch, 2*inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2d3748')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 11),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#f7fafc')),
            ('TEXTCOLOR', (0, 1), (-1, -1), colors.HexColor('#2d3748')),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 10),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#e2e8f0')),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('TOPPADDING', (0, 1), (-1, -1), 8),
            ('BOTTOMPADDING', (0, 1), (-1, -1), 8),
            # Highlight critical/high rows
            ('BACKGROUND', (0, 8), (-1, 8), colors.HexColor('#fed7d7') if critical > 0 else colors.HexColor('#f7fafc')),
            ('BACKGROUND', (0, 9), (-1, 9), colors.HexColor('#feebc8') if high > 0 else colors.HexColor('#f7fafc')),
        ]))
        
        elements.append(table)
        elements.append(Spacer(1, 0.5*inch))
        
        # Overall status
        if critical > 0 or high > 0:
            status_text = "⚠️ <b>ATTENTION REQUIRED:</b> Critical or high severity issues found."
            status_style = ParagraphStyle(
                'StatusWarning',
                parent=self.styles['Normal'],
                fontSize=12,
                textColor=colors.HexColor('#c53030'),
                backColor=colors.HexColor('#fed7d7'),
                borderPadding=10,
            )
        else:
            status_text = "✓ <b>COMPLIANT:</b> No critical or high severity issues."
            status_style = ParagraphStyle(
                'StatusOK',
                parent=self.styles['Normal'],
                fontSize=12,
                textColor=colors.HexColor('#276749'),
                backColor=colors.HexColor('#c6f6d5'),
                borderPadding=10,
            )
        
        elements.append(Paragraph(status_text, status_style))
        
        return elements
    
    def _build_analyses_section(self, analyses: list[dict[str, Any]]) -> list[Any]:
        """Build analyses summary section."""
        elements = []
        
        elements.append(Paragraph("Analysis Results", self.styles['SectionHeader']))
        elements.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#e2e8f0')))
        elements.append(Spacer(1, 0.25*inch))
        
        if not analyses:
            elements.append(Paragraph("No analyses in this period.", self.styles['Normal']))
            return elements
        
        # Analyses table
        table_data = [['Repository', 'PR #', 'Status', 'Issues', 'Date']]
        
        for analysis in analyses[:50]:  # Limit to 50 rows
            summary = analysis.get("summary") or {}
            status = "✓ Pass" if summary.get("pass", True) else "✗ Fail"
            issues = str(summary.get("total_issues", 0))
            
            completed = analysis.get("completed_at")
            if completed:
                if isinstance(completed, str):
                    date_str = completed[:10]
                else:
                    date_str = completed.strftime("%Y-%m-%d")
            else:
                date_str = "-"
            
            table_data.append([
                analysis.get("repo_full_name", "Unknown")[:30],
                str(analysis.get("pr_number", "-")),
                status,
                issues,
                date_str,
            ])
        
        table = Table(table_data, colWidths=[2.5*inch, 0.75*inch, 1*inch, 0.75*inch, 1*inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2d3748')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('ALIGN', (1, 0), (1, -1), 'CENTER'),
            ('ALIGN', (3, 0), (3, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('FONTSIZE', (0, 1), (-1, -1), 9),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 10),
            ('TOPPADDING', (0, 1), (-1, -1), 6),
            ('BOTTOMPADDING', (0, 1), (-1, -1), 6),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#e2e8f0')),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f7fafc')]),
        ]))
        
        elements.append(table)
        
        if len(analyses) > 50:
            elements.append(Spacer(1, 0.25*inch))
            elements.append(Paragraph(
                f"<i>Showing 50 of {len(analyses)} analyses</i>",
                self.styles['Normal']
            ))
        
        return elements
    
    def _build_findings_section(self, findings: list[dict[str, Any]]) -> list[Any]:
        """Build findings detail section."""
        elements = []
        
        elements.append(Paragraph("Findings Detail", self.styles['SectionHeader']))
        elements.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#e2e8f0')))
        elements.append(Spacer(1, 0.25*inch))
        
        # Group findings by severity
        severity_order = ["critical", "high", "medium", "low"]
        severity_colors = {
            "critical": colors.HexColor('#c53030'),
            "high": colors.HexColor('#dd6b20'),
            "medium": colors.HexColor('#d69e2e'),
            "low": colors.HexColor('#3182ce'),
        }
        
        for severity in severity_order:
            sev_findings = [f for f in findings if f.get("severity") == severity]
            if not sev_findings:
                continue
            
            # Severity header
            elements.append(Paragraph(
                f"{severity.upper()} ({len(sev_findings)})",
                ParagraphStyle(
                    f'{severity}Header',
                    parent=self.styles['Heading3'],
                    textColor=severity_colors.get(severity, colors.black),
                    spaceBefore=15,
                    spaceAfter=10,
                )
            ))
            
            for finding in sev_findings[:20]:  # Limit per severity
                # Finding title
                title = finding.get("title", "Unknown Issue")
                file_path = finding.get("file_path", "")
                line = finding.get("line_start", "")
                location = f"{file_path}:{line}" if file_path else ""
                
                elements.append(Paragraph(
                    f"<b>{title}</b>",
                    self.styles['FindingTitle']
                ))
                
                if location:
                    elements.append(Paragraph(
                        f"<i>Location: {location}</i>",
                        self.styles['FindingBody']
                    ))
                
                description = finding.get("description", "")
                if description:
                    elements.append(Paragraph(
                        description[:500],
                        self.styles['FindingBody']
                    ))
                
                confidence = finding.get("confidence", 0)
                elements.append(Paragraph(
                    f"Confidence: {int(confidence * 100)}% | Category: {finding.get('category', 'Unknown')}",
                    self.styles['FindingBody']
                ))
                
                elements.append(Spacer(1, 0.1*inch))
            
            if len(sev_findings) > 20:
                elements.append(Paragraph(
                    f"<i>...and {len(sev_findings) - 20} more {severity} findings</i>",
                    self.styles['Normal']
                ))
        
        return elements
    
    def _add_header_footer(self, canvas, doc) -> None:
        """Add header and footer to each page."""
        canvas.saveState()
        
        # Header
        canvas.setFont('Helvetica', 9)
        canvas.setFillColor(colors.HexColor('#718096'))
        canvas.drawString(0.75*inch, letter[1] - 0.5*inch, f"CodeVerify Report - {self.organization_name}")
        canvas.drawRightString(letter[0] - 0.75*inch, letter[1] - 0.5*inch, datetime.utcnow().strftime("%Y-%m-%d"))
        
        # Footer
        canvas.drawString(0.75*inch, 0.5*inch, "Confidential")
        canvas.drawRightString(letter[0] - 0.75*inch, 0.5*inch, f"Page {doc.page}")
        
        canvas.restoreState()
    
    def _generate_fallback(
        self,
        analyses: list[dict[str, Any]],
        findings: list[dict[str, Any]],
        date_range: str,
    ) -> bytes:
        """Generate simple text-based report when reportlab is not available."""
        lines = []
        lines.append("=" * 60)
        lines.append("CodeVerify Compliance Report")
        lines.append("=" * 60)
        lines.append("")
        lines.append(f"Organization: {self.organization_name}")
        lines.append(f"Date Range: {date_range}")
        lines.append(f"Generated: {datetime.utcnow().isoformat()}")
        lines.append("")
        lines.append("-" * 60)
        lines.append("EXECUTIVE SUMMARY")
        lines.append("-" * 60)
        
        total_analyses = len(analyses)
        passed = sum(1 for a in analyses if (a.get("summary") or {}).get("pass", True))
        
        lines.append(f"Total Analyses: {total_analyses}")
        lines.append(f"Passed: {passed}")
        lines.append(f"Failed: {total_analyses - passed}")
        lines.append("")
        
        total_findings = len(findings)
        critical = sum(1 for f in findings if f.get("severity") == "critical")
        high = sum(1 for f in findings if f.get("severity") == "high")
        
        lines.append(f"Total Findings: {total_findings}")
        lines.append(f"Critical: {critical}")
        lines.append(f"High: {high}")
        lines.append("")
        
        if critical > 0 or high > 0:
            lines.append("⚠️ ATTENTION REQUIRED: Critical or high severity issues found.")
        else:
            lines.append("✓ COMPLIANT: No critical or high severity issues.")
        
        lines.append("")
        lines.append("=" * 60)
        lines.append("END OF REPORT")
        
        return "\n".join(lines).encode("utf-8")


def generate_pdf_report(
    analyses: list[dict[str, Any]],
    findings: list[dict[str, Any]],
    organization_name: str = "Organization",
    date_range: str = "All time",
) -> bytes:
    """Convenience function to generate PDF report.
    
    Args:
        analyses: List of analysis results
        findings: List of findings
        organization_name: Name of the organization
        date_range: Date range for the report
        
    Returns:
        PDF content as bytes
    """
    generator = PDFReportGenerator(organization_name=organization_name)
    return generator.generate(analyses, findings, date_range)
