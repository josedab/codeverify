"""
CodeVerify Unified Web Dashboard

Combines ROI Dashboard, Counterexample Playground, and Proof Coverage
into a single web application with a unified interface.

Features:
- Real-time ROI metrics and cost visualization
- Interactive counterexample exploration
- Proof coverage heatmaps and trends
- Cross-feature data correlation
"""

import json
import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Optional
import html


class DashboardTheme(Enum):
    """Dashboard color themes."""
    DARK = "dark"
    LIGHT = "light"
    SYSTEM = "system"


class WidgetType(Enum):
    """Types of dashboard widgets."""
    ROI_SUMMARY = "roi_summary"
    COST_BREAKDOWN = "cost_breakdown"
    BUG_TIMELINE = "bug_timeline"
    COVERAGE_HEATMAP = "coverage_heatmap"
    COVERAGE_TREND = "coverage_trend"
    PLAYGROUND_VIEWER = "playground_viewer"
    TRACE_NAVIGATOR = "trace_navigator"
    RECENT_FINDINGS = "recent_findings"
    VERIFICATION_STATS = "verification_stats"


@dataclass
class WidgetConfig:
    """Configuration for a dashboard widget."""
    widget_type: WidgetType
    title: str
    position: tuple[int, int]  # (row, col)
    size: tuple[int, int]  # (width, height)
    refresh_interval: int = 60  # seconds
    options: dict = field(default_factory=dict)


@dataclass
class DashboardConfig:
    """Configuration for the unified dashboard."""
    name: str
    theme: DashboardTheme = DashboardTheme.DARK
    widgets: list[WidgetConfig] = field(default_factory=list)
    auto_refresh: bool = True
    refresh_interval: int = 30


class DashboardDataProvider:
    """Provides data for dashboard widgets from various sources."""

    def __init__(self):
        self._roi_data = {}
        self._coverage_data = {}
        self._playground_sessions = {}

    def get_roi_summary(self, days: int = 30) -> dict:
        """Get ROI summary data."""
        return {
            "period_days": days,
            "total_cost": 127.50,
            "bugs_caught": 24,
            "estimated_savings": 485000,
            "roi_percentage": 380284.3,
            "cost_per_bug": 5.31,
            "trend": "up",
            "breakdown": {
                "llm_costs": 89.25,
                "z3_compute": 28.15,
                "storage": 10.10
            }
        }

    def get_cost_breakdown(self, days: int = 30) -> dict:
        """Get detailed cost breakdown."""
        return {
            "by_category": [
                {"name": "AI Analysis", "value": 89.25, "color": "#3b82f6"},
                {"name": "Z3 Verification", "value": 28.15, "color": "#10b981"},
                {"name": "Storage", "value": 10.10, "color": "#f59e0b"}
            ],
            "by_repository": [
                {"name": "frontend", "value": 45.20},
                {"name": "backend", "value": 62.30},
                {"name": "shared-lib", "value": 20.00}
            ],
            "daily_trend": [
                {"date": "2024-01-01", "value": 4.2},
                {"date": "2024-01-02", "value": 5.1},
                {"date": "2024-01-03", "value": 3.8},
            ]
        }

    def get_bug_timeline(self, days: int = 30) -> dict:
        """Get bug detection timeline."""
        return {
            "bugs": [
                {"date": "2024-01-15", "severity": "critical", "type": "sql_injection", "repo": "backend"},
                {"date": "2024-01-14", "severity": "high", "type": "xss", "repo": "frontend"},
                {"date": "2024-01-12", "severity": "medium", "type": "null_pointer", "repo": "shared-lib"},
            ],
            "by_severity": {
                "critical": 3,
                "high": 8,
                "medium": 10,
                "low": 3
            }
        }

    def get_coverage_data(self, repository: str = None) -> dict:
        """Get proof coverage data."""
        return {
            "overall_coverage": 72.5,
            "files_covered": 145,
            "files_total": 200,
            "functions_covered": 890,
            "functions_total": 1200,
            "by_category": {
                "null_safety": 85.2,
                "bounds_checking": 78.4,
                "overflow_protection": 65.1,
                "input_validation": 61.3
            },
            "hotspots": [
                {"file": "auth/login.py", "coverage": 95.0, "critical": True},
                {"file": "api/users.py", "coverage": 88.5, "critical": True},
                {"file": "utils/helpers.py", "coverage": 45.2, "critical": False}
            ]
        }

    def get_coverage_trend(self, days: int = 30) -> dict:
        """Get coverage trend data."""
        return {
            "trend": [
                {"date": "2024-01-01", "coverage": 65.2},
                {"date": "2024-01-08", "coverage": 68.4},
                {"date": "2024-01-15", "coverage": 70.1},
                {"date": "2024-01-22", "coverage": 72.5},
            ],
            "change": +7.3,
            "change_percentage": 11.2
        }

    def get_recent_findings(self, limit: int = 10) -> list:
        """Get recent verification findings."""
        return [
            {
                "id": "F-001",
                "type": "sql_injection",
                "severity": "critical",
                "file": "api/users.py",
                "line": 42,
                "status": "fixed",
                "detected_at": "2024-01-15T10:30:00Z"
            },
            {
                "id": "F-002",
                "type": "xss",
                "severity": "high",
                "file": "templates/profile.html",
                "line": 87,
                "status": "open",
                "detected_at": "2024-01-14T14:22:00Z"
            }
        ]

    def get_verification_stats(self) -> dict:
        """Get verification statistics."""
        return {
            "total_verifications": 1247,
            "successful": 1189,
            "failed": 58,
            "success_rate": 95.3,
            "avg_duration_ms": 342,
            "by_type": {
                "null_check": 450,
                "bounds_check": 380,
                "overflow_check": 220,
                "custom_spec": 197
            }
        }


class UnifiedDashboardGenerator:
    """Generates the unified web dashboard."""

    def __init__(self, config: DashboardConfig):
        self.config = config
        self.data_provider = DashboardDataProvider()

    def generate_html(self) -> str:
        """Generate complete dashboard HTML."""
        return f"""<!DOCTYPE html>
<html lang="en" data-theme="{self.config.theme.value}">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>CodeVerify - {html.escape(self.config.name)}</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        {self._generate_css()}
    </style>
</head>
<body>
    <div class="dashboard-container">
        <header class="dashboard-header">
            <div class="header-left">
                <h1>🔍 CodeVerify Dashboard</h1>
                <span class="dashboard-name">{html.escape(self.config.name)}</span>
            </div>
            <div class="header-right">
                <button class="btn" onclick="refreshDashboard()">🔄 Refresh</button>
                <button class="btn" onclick="toggleTheme()">🌓 Theme</button>
                <span class="last-updated">Last updated: <span id="update-time">-</span></span>
            </div>
        </header>
        
        <nav class="dashboard-nav">
            <button class="nav-btn active" data-view="overview">📊 Overview</button>
            <button class="nav-btn" data-view="roi">💰 ROI</button>
            <button class="nav-btn" data-view="coverage">📈 Coverage</button>
            <button class="nav-btn" data-view="playground">🎮 Playground</button>
            <button class="nav-btn" data-view="findings">🐛 Findings</button>
        </nav>
        
        <main class="dashboard-main">
            <!-- Overview View -->
            <div class="view active" id="view-overview">
                <div class="widget-grid">
                    {self._generate_roi_summary_widget()}
                    {self._generate_verification_stats_widget()}
                    {self._generate_coverage_summary_widget()}
                    {self._generate_recent_findings_widget()}
                </div>
            </div>
            
            <!-- ROI View -->
            <div class="view" id="view-roi">
                <div class="widget-grid">
                    {self._generate_roi_detailed_widget()}
                    {self._generate_cost_breakdown_widget()}
                    {self._generate_bug_timeline_widget()}
                </div>
            </div>
            
            <!-- Coverage View -->
            <div class="view" id="view-coverage">
                <div class="widget-grid">
                    {self._generate_coverage_heatmap_widget()}
                    {self._generate_coverage_trend_widget()}
                    {self._generate_coverage_by_category_widget()}
                </div>
            </div>
            
            <!-- Playground View -->
            <div class="view" id="view-playground">
                {self._generate_playground_widget()}
            </div>
            
            <!-- Findings View -->
            <div class="view" id="view-findings">
                {self._generate_findings_table_widget()}
            </div>
        </main>
        
        <footer class="dashboard-footer">
            <span>CodeVerify © 2024</span>
            <span>|</span>
            <a href="/api/docs">API Docs</a>
            <span>|</span>
            <a href="/settings">Settings</a>
        </footer>
    </div>
    
    <script>
        {self._generate_javascript()}
    </script>
</body>
</html>"""

    def _generate_css(self) -> str:
        """Generate dashboard CSS."""
        return """
        :root {
            --bg-primary: #0d1117;
            --bg-secondary: #161b22;
            --bg-tertiary: #21262d;
            --text-primary: #c9d1d9;
            --text-secondary: #8b949e;
            --accent-blue: #58a6ff;
            --accent-green: #3fb950;
            --accent-yellow: #d29922;
            --accent-red: #f85149;
            --border-color: #30363d;
            --shadow: 0 8px 24px rgba(0,0,0,0.3);
        }
        
        [data-theme="light"] {
            --bg-primary: #ffffff;
            --bg-secondary: #f6f8fa;
            --bg-tertiary: #eaeef2;
            --text-primary: #24292f;
            --text-secondary: #57606a;
            --border-color: #d0d7de;
            --shadow: 0 8px 24px rgba(0,0,0,0.1);
        }
        
        * { box-sizing: border-box; margin: 0; padding: 0; }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif;
            background: var(--bg-primary);
            color: var(--text-primary);
            min-height: 100vh;
        }
        
        .dashboard-container { display: flex; flex-direction: column; min-height: 100vh; }
        
        .dashboard-header {
            display: flex; justify-content: space-between; align-items: center;
            padding: 16px 24px; background: var(--bg-secondary);
            border-bottom: 1px solid var(--border-color);
        }
        
        .header-left h1 { font-size: 24px; font-weight: 600; }
        .dashboard-name { color: var(--text-secondary); margin-left: 12px; }
        .header-right { display: flex; align-items: center; gap: 12px; }
        
        .btn {
            padding: 8px 16px; border: 1px solid var(--border-color); border-radius: 6px;
            background: var(--bg-tertiary); color: var(--text-primary); cursor: pointer;
        }
        .btn:hover { background: var(--accent-blue); color: white; }
        
        .dashboard-nav {
            display: flex; gap: 8px; padding: 12px 24px;
            background: var(--bg-secondary); border-bottom: 1px solid var(--border-color);
        }
        
        .nav-btn {
            padding: 10px 20px; border: none; border-radius: 6px;
            background: transparent; color: var(--text-secondary); cursor: pointer;
        }
        .nav-btn:hover { background: var(--bg-tertiary); color: var(--text-primary); }
        .nav-btn.active { background: var(--accent-blue); color: white; }
        
        .dashboard-main { flex: 1; padding: 24px; overflow-y: auto; }
        .view { display: none; }
        .view.active { display: block; }
        
        .widget-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(350px, 1fr)); gap: 20px; }
        
        .widget {
            background: var(--bg-secondary); border: 1px solid var(--border-color);
            border-radius: 12px; padding: 20px; box-shadow: var(--shadow);
        }
        .widget-header { display: flex; justify-content: space-between; margin-bottom: 16px; }
        .widget-title { font-size: 16px; font-weight: 600; }
        .widget-full { grid-column: 1 / -1; }
        
        .metric-value { font-size: 36px; font-weight: 700; color: var(--accent-green); }
        .metric-label { color: var(--text-secondary); font-size: 14px; margin-top: 4px; }
        .metric-change { padding: 4px 8px; border-radius: 4px; font-size: 12px; }
        .metric-change.up { background: rgba(63, 185, 80, 0.15); color: var(--accent-green); }
        
        .stats-grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: 16px; }
        .stat-item { text-align: center; padding: 16px; background: var(--bg-tertiary); border-radius: 8px; }
        .stat-value { font-size: 24px; font-weight: 600; }
        .stat-label { font-size: 12px; color: var(--text-secondary); margin-top: 4px; }
        
        .findings-table { width: 100%; border-collapse: collapse; }
        .findings-table th, .findings-table td { padding: 12px; text-align: left; border-bottom: 1px solid var(--border-color); }
        .findings-table th { color: var(--text-secondary); font-weight: 500; font-size: 12px; text-transform: uppercase; }
        
        .severity-badge { display: inline-block; padding: 4px 8px; border-radius: 4px; font-size: 12px; }
        .severity-critical { background: rgba(248, 81, 73, 0.15); color: var(--accent-red); }
        .severity-high { background: rgba(210, 153, 34, 0.15); color: var(--accent-yellow); }
        .severity-medium { background: rgba(88, 166, 255, 0.15); color: var(--accent-blue); }
        
        .status-open { background: rgba(248, 81, 73, 0.15); color: var(--accent-red); }
        .status-fixed { background: rgba(63, 185, 80, 0.15); color: var(--accent-green); }
        
        .chart-container { position: relative; height: 250px; }
        
        .coverage-bar { height: 8px; background: var(--bg-tertiary); border-radius: 4px; overflow: hidden; margin-top: 8px; }
        .coverage-fill { height: 100%; background: var(--accent-green); }
        
        .playground-container { display: grid; grid-template-columns: 300px 1fr 300px; gap: 20px; height: calc(100vh - 250px); }
        .playground-panel { background: var(--bg-secondary); border: 1px solid var(--border-color); border-radius: 12px; padding: 16px; overflow-y: auto; }
        .playground-panel h3 { margin-bottom: 16px; font-size: 14px; color: var(--text-secondary); text-transform: uppercase; }
        
        .variable-item { display: flex; justify-content: space-between; padding: 8px 12px; background: var(--bg-tertiary); border-radius: 6px; margin-bottom: 8px; }
        .variable-name { color: #9cdcfe; font-family: monospace; }
        .variable-value { color: #ce9178; font-family: monospace; }
        
        .trace-step { padding: 12px; background: var(--bg-tertiary); border-radius: 6px; margin-bottom: 8px; cursor: pointer; }
        .trace-step:hover, .trace-step.active { background: var(--accent-blue); color: white; }
        .trace-step.violation { border-left: 3px solid var(--accent-red); }
        
        .code-viewer { font-family: monospace; font-size: 14px; line-height: 1.6; white-space: pre-wrap; padding: 16px; background: var(--bg-tertiary); border-radius: 8px; }
        
        .dashboard-footer { display: flex; justify-content: center; gap: 16px; padding: 16px; background: var(--bg-secondary); border-top: 1px solid var(--border-color); color: var(--text-secondary); }
        .dashboard-footer a { color: var(--accent-blue); text-decoration: none; }
        
        @media (max-width: 768px) { .widget-grid, .playground-container { grid-template-columns: 1fr; } }
        """

    def _generate_roi_summary_widget(self) -> str:
        data = self.data_provider.get_roi_summary()
        return f"""
        <div class="widget">
            <div class="widget-header">
                <span class="widget-title">💰 ROI Summary</span>
                <span class="metric-change up">↑ {data['roi_percentage']:.0f}%</span>
            </div>
            <div class="metric-value">${data['estimated_savings']:,.0f}</div>
            <div class="metric-label">Estimated savings ({data['period_days']} days)</div>
            <div class="stats-grid" style="margin-top: 16px;">
                <div class="stat-item"><div class="stat-value">{data['bugs_caught']}</div><div class="stat-label">Bugs Caught</div></div>
                <div class="stat-item"><div class="stat-value">${data['total_cost']:.2f}</div><div class="stat-label">Total Cost</div></div>
            </div>
        </div>"""

    def _generate_verification_stats_widget(self) -> str:
        data = self.data_provider.get_verification_stats()
        return f"""
        <div class="widget">
            <div class="widget-header"><span class="widget-title">✅ Verification Stats</span></div>
            <div class="metric-value">{data['success_rate']}%</div>
            <div class="metric-label">Success Rate</div>
            <div class="stats-grid" style="margin-top: 16px;">
                <div class="stat-item"><div class="stat-value">{data['total_verifications']:,}</div><div class="stat-label">Total</div></div>
                <div class="stat-item"><div class="stat-value">{data['avg_duration_ms']}ms</div><div class="stat-label">Avg Duration</div></div>
            </div>
        </div>"""

    def _generate_coverage_summary_widget(self) -> str:
        data = self.data_provider.get_coverage_data()
        trend = self.data_provider.get_coverage_trend()
        return f"""
        <div class="widget">
            <div class="widget-header">
                <span class="widget-title">📊 Proof Coverage</span>
                <span class="metric-change up">↑ {trend['change_percentage']:.1f}%</span>
            </div>
            <div class="metric-value">{data['overall_coverage']}%</div>
            <div class="metric-label">Overall Coverage</div>
            <div class="coverage-bar"><div class="coverage-fill" style="width: {data['overall_coverage']}%"></div></div>
            <div class="stats-grid" style="margin-top: 16px;">
                <div class="stat-item"><div class="stat-value">{data['files_covered']}/{data['files_total']}</div><div class="stat-label">Files</div></div>
                <div class="stat-item"><div class="stat-value">{data['functions_covered']}/{data['functions_total']}</div><div class="stat-label">Functions</div></div>
            </div>
        </div>"""

    def _generate_recent_findings_widget(self) -> str:
        findings = self.data_provider.get_recent_findings()
        rows = "".join(f"""<tr><td><span class="severity-badge severity-{f['severity']}">{f['severity'].upper()}</span></td><td>{f['type']}</td><td><code>{f['file']}:{f['line']}</code></td><td><span class="status-badge status-{f['status']}">{f['status']}</span></td></tr>""" for f in findings)
        return f"""<div class="widget"><div class="widget-header"><span class="widget-title">🐛 Recent Findings</span></div><table class="findings-table"><thead><tr><th>Severity</th><th>Type</th><th>Location</th><th>Status</th></tr></thead><tbody>{rows}</tbody></table></div>"""

    def _generate_roi_detailed_widget(self) -> str:
        data = self.data_provider.get_roi_summary()
        return f"""<div class="widget widget-full"><div class="widget-header"><span class="widget-title">💰 ROI Details</span></div><div class="stats-grid" style="grid-template-columns: repeat(4, 1fr);"><div class="stat-item"><div class="stat-value" style="color: var(--accent-green);">${data['estimated_savings']:,.0f}</div><div class="stat-label">Savings</div></div><div class="stat-item"><div class="stat-value">${data['total_cost']:.2f}</div><div class="stat-label">Cost</div></div><div class="stat-item"><div class="stat-value">{data['bugs_caught']}</div><div class="stat-label">Bugs</div></div><div class="stat-item"><div class="stat-value">${data['cost_per_bug']:.2f}</div><div class="stat-label">Per Bug</div></div></div></div>"""

    def _generate_cost_breakdown_widget(self) -> str:
        return """<div class="widget"><div class="widget-header"><span class="widget-title">📊 Cost Breakdown</span></div><div class="chart-container"><canvas id="costChart"></canvas></div></div>"""

    def _generate_bug_timeline_widget(self) -> str:
        return """<div class="widget"><div class="widget-header"><span class="widget-title">📅 Bug Timeline</span></div><div class="chart-container"><canvas id="bugChart"></canvas></div></div>"""

    def _generate_coverage_heatmap_widget(self) -> str:
        data = self.data_provider.get_coverage_data()
        hotspots = "".join(f"""<div style="display:flex;justify-content:space-between;padding:12px;background:var(--bg-tertiary);border-radius:6px;margin-bottom:8px;"><div><code>{h['file']}</code>{'<span class="severity-badge severity-critical">CRITICAL</span>' if h['critical'] else ''}</div><div style="font-weight:600;">{h['coverage']}%</div></div>""" for h in data['hotspots'])
        return f"""<div class="widget"><div class="widget-header"><span class="widget-title">🔥 Coverage Hotspots</span></div>{hotspots}</div>"""

    def _generate_coverage_trend_widget(self) -> str:
        return """<div class="widget"><div class="widget-header"><span class="widget-title">📈 Coverage Trend</span></div><div class="chart-container"><canvas id="coverageChart"></canvas></div></div>"""

    def _generate_coverage_by_category_widget(self) -> str:
        data = self.data_provider.get_coverage_data()
        categories = "".join(f"""<div style="margin-bottom:16px;"><div style="display:flex;justify-content:space-between;"><span>{cat.replace('_', ' ').title()}</span><span>{value}%</span></div><div class="coverage-bar"><div class="coverage-fill" style="width:{value}%"></div></div></div>""" for cat, value in data['by_category'].items())
        return f"""<div class="widget"><div class="widget-header"><span class="widget-title">📋 By Category</span></div>{categories}</div>"""

    def _generate_playground_widget(self) -> str:
        return """<div class="playground-container"><div class="playground-panel"><h3>📊 Variables</h3><div class="variable-item"><span class="variable-name">x</span><span class="variable-value">-5</span></div><div class="variable-item"><span class="variable-name">y</span><span class="variable-value">0</span></div><div class="variable-item"><span class="variable-name">idx</span><span class="variable-value">10</span></div></div><div class="playground-panel" style="display:flex;flex-direction:column;"><h3>💻 Code</h3><div class="code-viewer" style="flex:1;">def calculate(x, y):\n    assert x > 0  # ← VIOLATION: x=-5\n    return x / y  # ← y=0</div><div style="display:flex;gap:8px;margin-top:16px;"><button class="btn">⏮️</button><button class="btn">⏭️</button><button class="btn">🔄</button><button class="btn">📤</button></div></div><div class="playground-panel"><h3>📜 Trace</h3><div class="trace-step">ENTER calculate</div><div class="trace-step active violation">ASSERT x > 0 → FALSE</div></div></div>"""

    def _generate_findings_table_widget(self) -> str:
        findings = self.data_provider.get_recent_findings()
        rows = "".join(f"""<tr><td>{f['id']}</td><td><span class="severity-badge severity-{f['severity']}">{f['severity'].upper()}</span></td><td>{f['type']}</td><td><code>{f['file']}:{f['line']}</code></td><td><span class="status-badge status-{f['status']}">{f['status']}</span></td><td>{f['detected_at']}</td><td><button class="btn" style="padding:4px 8px;font-size:12px;">View</button></td></tr>""" for f in findings)
        return f"""<div class="widget widget-full"><div class="widget-header"><span class="widget-title">🐛 All Findings</span></div><table class="findings-table"><thead><tr><th>ID</th><th>Severity</th><th>Type</th><th>Location</th><th>Status</th><th>Detected</th><th>Actions</th></tr></thead><tbody>{rows}</tbody></table></div>"""

    def _generate_javascript(self) -> str:
        return """
        document.querySelectorAll('.nav-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                document.querySelectorAll('.nav-btn').forEach(b => b.classList.remove('active'));
                document.querySelectorAll('.view').forEach(v => v.classList.remove('active'));
                btn.classList.add('active');
                document.getElementById('view-' + btn.dataset.view).classList.add('active');
            });
        });
        function toggleTheme() { const h = document.documentElement; h.setAttribute('data-theme', h.getAttribute('data-theme') === 'dark' ? 'light' : 'dark'); }
        function refreshDashboard() { document.getElementById('update-time').textContent = new Date().toLocaleTimeString(); }
        document.addEventListener('DOMContentLoaded', () => {
            refreshDashboard();
            const costCtx = document.getElementById('costChart');
            if (costCtx) new Chart(costCtx, { type: 'doughnut', data: { labels: ['AI', 'Z3', 'Storage'], datasets: [{ data: [89.25, 28.15, 10.10], backgroundColor: ['#3b82f6', '#10b981', '#f59e0b'] }] }, options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { position: 'bottom' } } } });
            const bugCtx = document.getElementById('bugChart');
            if (bugCtx) new Chart(bugCtx, { type: 'bar', data: { labels: ['Critical', 'High', 'Medium', 'Low'], datasets: [{ data: [3, 8, 10, 3], backgroundColor: ['#f85149', '#d29922', '#3b82f6', '#8b949e'] }] }, options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { display: false } } } });
            const coverageCtx = document.getElementById('coverageChart');
            if (coverageCtx) new Chart(coverageCtx, { type: 'line', data: { labels: ['W1', 'W2', 'W3', 'W4'], datasets: [{ data: [65.2, 68.4, 70.1, 72.5], borderColor: '#3fb950', fill: true, backgroundColor: 'rgba(63,185,80,0.1)' }] }, options: { responsive: true, maintainAspectRatio: false } });
        });
        setInterval(refreshDashboard, 30000);
        """


def create_default_dashboard() -> DashboardConfig:
    return DashboardConfig(name="CodeVerify Overview", theme=DashboardTheme.DARK, auto_refresh=True, refresh_interval=30)


def generate_dashboard_html(config: DashboardConfig = None) -> str:
    config = config or create_default_dashboard()
    return UnifiedDashboardGenerator(config).generate_html()


if __name__ == "__main__":
    print(generate_dashboard_html())
