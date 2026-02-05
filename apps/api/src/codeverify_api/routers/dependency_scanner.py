"""
Dependency Vulnerability Scanner API Router

Provides REST API endpoints for dependency vulnerability scanning:
- Scan dependencies for vulnerabilities
- Get vulnerable packages
- Get upgrade suggestions
- View dependency graphs
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

# Import Dependency Scanner
try:
    from codeverify_agents.dependency_scanner import (
        DependencyVulnerabilityScanner,
        PackageEcosystem,
        VulnerabilitySeverity,
    )
    DEPENDENCY_SCANNER_AVAILABLE = True
except ImportError:
    DEPENDENCY_SCANNER_AVAILABLE = False
    DependencyVulnerabilityScanner = None
    PackageEcosystem = None
    VulnerabilitySeverity = None


router = APIRouter(prefix="/api/v1/dependencies", tags=["dependency-scanner"])

# Singleton scanner instance
_scanner: Optional[DependencyVulnerabilityScanner] = None


def get_scanner() -> DependencyVulnerabilityScanner:
    """Get or create the scanner singleton."""
    global _scanner
    if _scanner is None and DEPENDENCY_SCANNER_AVAILABLE:
        _scanner = DependencyVulnerabilityScanner()
    return _scanner


# =============================================================================
# Request/Response Models
# =============================================================================


class ScanRequest(BaseModel):
    """Request to scan dependencies."""
    project_name: str = Field(..., description="Name of the project")
    files: Dict[str, str] = Field(..., description="Map of file path to content")


class ScanResponse(BaseModel):
    """Response with scan summary."""
    scan_id: str
    project_name: str
    ecosystem: str
    total_packages: int
    vulnerable_count: int
    critical_count: int
    high_count: int


# =============================================================================
# API Endpoints
# =============================================================================


@router.post(
    "/scan",
    response_model=ScanResponse,
    summary="Scan Dependencies",
    description="Scan project dependencies for vulnerabilities"
)
async def scan_dependencies(request: ScanRequest) -> ScanResponse:
    """
    Scan project dependencies for vulnerabilities.

    Provide dependency file contents (package.json, requirements.txt, etc.)
    to scan for known vulnerabilities.
    """
    if not DEPENDENCY_SCANNER_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="Dependency Vulnerability Scanner is not available"
        )

    scanner = get_scanner()
    if not scanner:
        raise HTTPException(
            status_code=503,
            detail="Failed to initialize Dependency Vulnerability Scanner"
        )

    result = await scanner.scan(
        project_name=request.project_name,
        files=request.files,
    )

    return ScanResponse(
        scan_id=result.id,
        project_name=result.project_name,
        ecosystem=result.ecosystem.value,
        total_packages=result.total_packages,
        vulnerable_count=len(result.vulnerable_packages),
        critical_count=result.critical_count,
        high_count=result.high_count,
    )


@router.get(
    "/scan/{scan_id}",
    summary="Get Scan Result",
    description="Get full scan result details"
)
async def get_scan_result(scan_id: str) -> Dict[str, Any]:
    """Get full scan result details."""
    if not DEPENDENCY_SCANNER_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="Dependency Vulnerability Scanner is not available"
        )

    scanner = get_scanner()
    if not scanner:
        raise HTTPException(
            status_code=503,
            detail="Failed to initialize Dependency Vulnerability Scanner"
        )

    result = scanner.get_result(scan_id)
    if not result:
        raise HTTPException(
            status_code=404,
            detail=f"Scan result not found: {scan_id}"
        )

    return result.to_dict()


@router.get(
    "/scan/{scan_id}/vulnerabilities",
    summary="Get Vulnerabilities",
    description="Get vulnerable packages from a scan"
)
async def get_vulnerabilities(
    scan_id: str,
    min_severity: Optional[str] = None,
) -> Dict[str, Any]:
    """Get vulnerable packages from a scan result."""
    if not DEPENDENCY_SCANNER_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="Dependency Vulnerability Scanner is not available"
        )

    scanner = get_scanner()
    if not scanner:
        raise HTTPException(
            status_code=503,
            detail="Failed to initialize Dependency Vulnerability Scanner"
        )

    result = scanner.get_result(scan_id)
    if not result:
        raise HTTPException(
            status_code=404,
            detail=f"Scan result not found: {scan_id}"
        )

    packages = scanner.get_vulnerable_packages(scan_id, min_severity)

    return {
        "vulnerable_packages": [p.to_dict() for p in packages],
        "count": len(packages),
        "scan_id": scan_id,
    }


@router.get(
    "/scan/{scan_id}/upgrades",
    summary="Get Upgrade Paths",
    description="Get suggested upgrade paths"
)
async def get_upgrade_paths(scan_id: str) -> Dict[str, Any]:
    """Get suggested upgrade paths from a scan result."""
    if not DEPENDENCY_SCANNER_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="Dependency Vulnerability Scanner is not available"
        )

    scanner = get_scanner()
    if not scanner:
        raise HTTPException(
            status_code=503,
            detail="Failed to initialize Dependency Vulnerability Scanner"
        )

    result = scanner.get_result(scan_id)
    if not result:
        raise HTTPException(
            status_code=404,
            detail=f"Scan result not found: {scan_id}"
        )

    return {
        "upgrade_paths": [u.to_dict() for u in result.upgrade_paths],
        "count": len(result.upgrade_paths),
        "scan_id": scan_id,
    }


@router.get(
    "/scan/{scan_id}/graph",
    summary="Get Dependency Graph",
    description="Get the dependency graph from a scan"
)
async def get_dependency_graph(scan_id: str) -> Dict[str, Any]:
    """Get the dependency graph from a scan result."""
    if not DEPENDENCY_SCANNER_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="Dependency Vulnerability Scanner is not available"
        )

    scanner = get_scanner()
    if not scanner:
        raise HTTPException(
            status_code=503,
            detail="Failed to initialize Dependency Vulnerability Scanner"
        )

    result = scanner.get_result(scan_id)
    if not result:
        raise HTTPException(
            status_code=404,
            detail=f"Scan result not found: {scan_id}"
        )

    return {
        "graph": {k: v.to_dict() for k, v in result.dependency_graph.items()},
        "node_count": len(result.dependency_graph),
        "scan_id": scan_id,
    }


@router.get(
    "/ecosystems",
    summary="Get Ecosystems",
    description="Get supported package ecosystems"
)
async def get_ecosystems() -> Dict[str, Any]:
    """Get supported package ecosystems."""
    if not DEPENDENCY_SCANNER_AVAILABLE:
        return {
            "available": False,
            "ecosystems": [],
        }

    ecosystems = [
        {
            "value": "npm",
            "name": "NPM",
            "description": "Node.js packages",
            "files": ["package.json", "package-lock.json"],
        },
        {
            "value": "pypi",
            "name": "PyPI",
            "description": "Python packages",
            "files": ["requirements.txt", "pyproject.toml", "setup.py"],
        },
        {
            "value": "maven",
            "name": "Maven",
            "description": "Java packages",
            "files": ["pom.xml"],
        },
        {
            "value": "nuget",
            "name": "NuGet",
            "description": ".NET packages",
            "files": ["*.csproj", "packages.config"],
        },
        {
            "value": "rubygems",
            "name": "RubyGems",
            "description": "Ruby packages",
            "files": ["Gemfile", "Gemfile.lock"],
        },
        {
            "value": "go",
            "name": "Go Modules",
            "description": "Go packages",
            "files": ["go.mod", "go.sum"],
        },
        {
            "value": "cargo",
            "name": "Cargo",
            "description": "Rust packages",
            "files": ["Cargo.toml", "Cargo.lock"],
        },
    ]

    return {
        "available": True,
        "ecosystems": ecosystems,
    }


@router.get(
    "/severity-levels",
    summary="Get Severity Levels",
    description="Get vulnerability severity levels"
)
async def get_severity_levels() -> Dict[str, Any]:
    """Get vulnerability severity levels."""
    if not DEPENDENCY_SCANNER_AVAILABLE:
        return {
            "available": False,
            "levels": [],
        }

    levels = [
        {"value": "critical", "name": "Critical", "cvss_range": "9.0 - 10.0", "color": "red"},
        {"value": "high", "name": "High", "cvss_range": "7.0 - 8.9", "color": "orange"},
        {"value": "medium", "name": "Medium", "cvss_range": "4.0 - 6.9", "color": "yellow"},
        {"value": "low", "name": "Low", "cvss_range": "0.1 - 3.9", "color": "blue"},
        {"value": "informational", "name": "Informational", "cvss_range": "0.0", "color": "gray"},
    ]

    return {
        "available": True,
        "levels": levels,
    }


@router.get(
    "/stats",
    summary="Get Statistics",
    description="Get scanner statistics"
)
async def get_stats() -> Dict[str, Any]:
    """Get scanner statistics."""
    if not DEPENDENCY_SCANNER_AVAILABLE:
        return {
            "available": False,
            "message": "Dependency Vulnerability Scanner is not available",
        }

    scanner = get_scanner()
    if not scanner:
        return {
            "available": False,
            "message": "Failed to initialize scanner",
        }

    stats = scanner.get_statistics()
    stats["available"] = True

    return stats
