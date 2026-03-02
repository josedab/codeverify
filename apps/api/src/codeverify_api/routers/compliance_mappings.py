"""Compliance framework control mapping database.

Maps CodeVerify verification checks to SOC 2, ISO 27001, HIPAA, PCI-DSS,
NIST CSF, and OWASP/CWE controls. Includes evidence artifacts and gap analysis.
"""

from __future__ import annotations

import hashlib
import uuid
from datetime import datetime
from typing import Any

from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel, Field

router = APIRouter()


# ---------------------------------------------------------------------------
# Control mapping database
# ---------------------------------------------------------------------------

CONTROL_MAPPINGS: dict[str, list[dict[str, Any]]] = {
    "null_safety": [
        {"framework": "owasp", "control": "A03:2021", "name": "Injection", "relationship": "prevents"},
        {"framework": "cwe", "control": "CWE-476", "name": "NULL Pointer Dereference", "relationship": "detects"},
        {"framework": "soc2", "control": "CC7.1", "name": "System Operations Monitoring", "relationship": "supports"},
        {"framework": "nist", "control": "SI-10", "name": "Information Input Validation", "relationship": "supports"},
        {"framework": "iso27001", "control": "A.14.2.1", "name": "Secure Development Policy", "relationship": "supports"},
    ],
    "bounds_check": [
        {"framework": "cwe", "control": "CWE-119", "name": "Buffer Overflow", "relationship": "prevents"},
        {"framework": "cwe", "control": "CWE-125", "name": "Out-of-bounds Read", "relationship": "prevents"},
        {"framework": "owasp", "control": "A03:2021", "name": "Injection", "relationship": "prevents"},
        {"framework": "nist", "control": "SI-10", "name": "Information Input Validation", "relationship": "supports"},
        {"framework": "pci_dss", "control": "6.5.2", "name": "Buffer Overflows", "relationship": "prevents"},
    ],
    "division_by_zero": [
        {"framework": "cwe", "control": "CWE-369", "name": "Divide By Zero", "relationship": "prevents"},
        {"framework": "soc2", "control": "CC7.2", "name": "Monitoring of Systems", "relationship": "supports"},
        {"framework": "nist", "control": "SI-10", "name": "Information Input Validation", "relationship": "supports"},
    ],
    "sql_injection": [
        {"framework": "owasp", "control": "A03:2021", "name": "Injection", "relationship": "prevents"},
        {"framework": "cwe", "control": "CWE-89", "name": "SQL Injection", "relationship": "detects"},
        {"framework": "pci_dss", "control": "6.5.1", "name": "Injection Flaws", "relationship": "prevents"},
        {"framework": "hipaa", "control": "164.312(a)(1)", "name": "Access Control", "relationship": "supports"},
        {"framework": "nist", "control": "SI-10", "name": "Information Input Validation", "relationship": "prevents"},
        {"framework": "iso27001", "control": "A.14.2.5", "name": "Secure System Engineering", "relationship": "supports"},
    ],
    "type_safety": [
        {"framework": "cwe", "control": "CWE-704", "name": "Incorrect Type Conversion", "relationship": "prevents"},
        {"framework": "soc2", "control": "CC8.1", "name": "Change Management", "relationship": "supports"},
        {"framework": "nist", "control": "SA-11", "name": "Developer Testing", "relationship": "supports"},
    ],
    "resource_leak": [
        {"framework": "cwe", "control": "CWE-404", "name": "Improper Resource Shutdown", "relationship": "detects"},
        {"framework": "cwe", "control": "CWE-772", "name": "Missing Release of Resource", "relationship": "detects"},
        {"framework": "soc2", "control": "CC6.1", "name": "Logical Access Security", "relationship": "supports"},
        {"framework": "nist", "control": "SC-4", "name": "Information in Shared System Resources", "relationship": "supports"},
    ],
    "authentication": [
        {"framework": "owasp", "control": "A07:2021", "name": "Identification and Authentication Failures", "relationship": "prevents"},
        {"framework": "cwe", "control": "CWE-287", "name": "Improper Authentication", "relationship": "detects"},
        {"framework": "pci_dss", "control": "8.1", "name": "User Identification", "relationship": "supports"},
        {"framework": "hipaa", "control": "164.312(d)", "name": "Person or Entity Authentication", "relationship": "supports"},
        {"framework": "nist", "control": "IA-2", "name": "Identification and Authentication", "relationship": "supports"},
        {"framework": "iso27001", "control": "A.9.2.1", "name": "User Registration and De-registration", "relationship": "supports"},
        {"framework": "soc2", "control": "CC6.1", "name": "Logical Access Security", "relationship": "supports"},
    ],
    "encryption": [
        {"framework": "owasp", "control": "A02:2021", "name": "Cryptographic Failures", "relationship": "prevents"},
        {"framework": "cwe", "control": "CWE-327", "name": "Use of a Broken or Risky Cryptographic Algorithm", "relationship": "detects"},
        {"framework": "pci_dss", "control": "3.4", "name": "Render PAN Unreadable", "relationship": "supports"},
        {"framework": "hipaa", "control": "164.312(a)(2)(iv)", "name": "Encryption and Decryption", "relationship": "supports"},
        {"framework": "nist", "control": "SC-13", "name": "Cryptographic Protection", "relationship": "supports"},
        {"framework": "iso27001", "control": "A.10.1.1", "name": "Policy on the Use of Cryptographic Controls", "relationship": "supports"},
    ],
    "hardcoded_secret": [
        {"framework": "owasp", "control": "A02:2021", "name": "Cryptographic Failures", "relationship": "prevents"},
        {"framework": "cwe", "control": "CWE-798", "name": "Use of Hard-coded Credentials", "relationship": "detects"},
        {"framework": "pci_dss", "control": "6.5.3", "name": "Insecure Cryptographic Storage", "relationship": "prevents"},
        {"framework": "soc2", "control": "CC6.7", "name": "Logical Access Restrictions", "relationship": "supports"},
        {"framework": "nist", "control": "IA-5", "name": "Authenticator Management", "relationship": "supports"},
    ],
}


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class ControlMapping(BaseModel):
    check_type: str
    framework: str
    control_id: str
    control_name: str
    relationship: str


class CoverageMatrix(BaseModel):
    framework: str
    total_controls: int
    covered_controls: int
    coverage_percentage: float
    covered: list[dict[str, Any]]
    gaps: list[dict[str, Any]]


class ComplianceReportData(BaseModel):
    id: str
    organization: str
    frameworks: list[str]
    generated_at: str
    coverage_matrices: list[CoverageMatrix]
    evidence_artifacts: list[dict[str, Any]]
    gap_analysis: list[dict[str, Any]]
    overall_score: float
    audit_hash: str


class AuditTrailEntry(BaseModel):
    id: str
    timestamp: str
    action: str
    resource_type: str
    resource_id: str
    actor: str
    details: dict[str, Any]
    integrity_hash: str
    previous_hash: str


# ---------------------------------------------------------------------------
# In-memory audit trail
# ---------------------------------------------------------------------------

_audit_trail: list[dict[str, Any]] = []
_compliance_reports: dict[str, dict[str, Any]] = {}


def _append_audit(action: str, resource_type: str, resource_id: str, actor: str, details: dict[str, Any]) -> dict[str, Any]:
    """Append an immutable audit entry with cryptographic chain."""
    previous_hash = _audit_trail[-1]["integrity_hash"] if _audit_trail else "0" * 64
    entry = {
        "id": str(uuid.uuid4()),
        "timestamp": datetime.utcnow().isoformat(),
        "action": action,
        "resource_type": resource_type,
        "resource_id": resource_id,
        "actor": actor,
        "details": details,
        "previous_hash": previous_hash,
    }
    payload = f"{entry['id']}:{entry['timestamp']}:{entry['action']}:{previous_hash}"
    entry["integrity_hash"] = hashlib.sha256(payload.encode()).hexdigest()
    _audit_trail.append(entry)
    return entry


# ---------------------------------------------------------------------------
# Framework control definitions
# ---------------------------------------------------------------------------

FRAMEWORK_CONTROLS: dict[str, list[dict[str, str]]] = {
    "soc2": [
        {"id": "CC6.1", "name": "Logical Access Security", "description": "Logical access restrictions"},
        {"id": "CC6.7", "name": "Restriction of Privileged Access", "description": "Privileged access is restricted"},
        {"id": "CC7.1", "name": "System Monitoring", "description": "Detection and monitoring"},
        {"id": "CC7.2", "name": "Anomaly Detection", "description": "Anomalies are monitored"},
        {"id": "CC8.1", "name": "Change Management", "description": "Changes are authorized and tested"},
    ],
    "iso27001": [
        {"id": "A.9.2.1", "name": "User Registration", "description": "Formal registration/de-registration"},
        {"id": "A.10.1.1", "name": "Cryptographic Controls", "description": "Policy on use of crypto"},
        {"id": "A.14.2.1", "name": "Secure Development Policy", "description": "Secure development rules"},
        {"id": "A.14.2.5", "name": "Secure System Engineering", "description": "Principles for system engineering"},
    ],
    "hipaa": [
        {"id": "164.312(a)(1)", "name": "Access Control", "description": "Access control mechanisms"},
        {"id": "164.312(a)(2)(iv)", "name": "Encryption", "description": "Encryption and decryption"},
        {"id": "164.312(d)", "name": "Authentication", "description": "Entity authentication"},
        {"id": "164.312(e)(1)", "name": "Transmission Security", "description": "Transmission integrity"},
    ],
    "pci_dss": [
        {"id": "3.4", "name": "Render PAN Unreadable", "description": "Protect stored cardholder data"},
        {"id": "6.5.1", "name": "Injection Flaws", "description": "Protect against injection"},
        {"id": "6.5.2", "name": "Buffer Overflows", "description": "Protect against buffer overflow"},
        {"id": "6.5.3", "name": "Insecure Crypto Storage", "description": "Protect cryptographic storage"},
        {"id": "8.1", "name": "User Identification", "description": "Identify all users"},
    ],
    "nist": [
        {"id": "IA-2", "name": "Identification and Authentication", "description": "Unique user identification"},
        {"id": "IA-5", "name": "Authenticator Management", "description": "Manage authenticators"},
        {"id": "SA-11", "name": "Developer Testing", "description": "Developer testing and evaluation"},
        {"id": "SC-4", "name": "Information in Shared Resources", "description": "Prevent unauthorized transfer"},
        {"id": "SC-13", "name": "Cryptographic Protection", "description": "Employ FIPS crypto"},
        {"id": "SI-10", "name": "Information Input Validation", "description": "Validate information inputs"},
    ],
}


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.get("/mappings")
async def get_control_mappings(
    check_type: str | None = Query(default=None),
    framework: str | None = Query(default=None),
) -> dict[str, Any]:
    """Get the control mapping database, optionally filtered."""
    results: list[ControlMapping] = []
    for ct, mappings in CONTROL_MAPPINGS.items():
        if check_type and ct != check_type:
            continue
        for m in mappings:
            if framework and m["framework"] != framework:
                continue
            results.append(ControlMapping(
                check_type=ct,
                framework=m["framework"],
                control_id=m["control"],
                control_name=m["name"],
                relationship=m["relationship"],
            ))
    return {"mappings": [r.model_dump() for r in results], "total": len(results)}


@router.post("/coverage-report", response_model=ComplianceReportData)
async def generate_coverage_report(
    organization: str = Query(description="Organization name"),
    frameworks: list[str] = Query(description="Frameworks to assess"),
    checks_passed: list[str] = Query(default=[], description="Check types that passed"),
) -> ComplianceReportData:
    """Generate a compliance coverage report with gap analysis."""
    report_id = str(uuid.uuid4())
    coverage_matrices: list[CoverageMatrix] = []
    all_gaps: list[dict[str, Any]] = []
    evidence: list[dict[str, Any]] = []

    for fw in frameworks:
        fw_controls = FRAMEWORK_CONTROLS.get(fw, [])
        covered = []
        gaps = []

        for control in fw_controls:
            # Check if any passing check maps to this control
            is_covered = False
            for check in checks_passed:
                mappings = CONTROL_MAPPINGS.get(check, [])
                if any(m["framework"] == fw and m["control"] == control["id"] for m in mappings):
                    is_covered = True
                    covered.append({**control, "verified_by": check})
                    evidence.append({
                        "control_id": control["id"],
                        "framework": fw,
                        "check_type": check,
                        "status": "verified",
                        "timestamp": datetime.utcnow().isoformat(),
                    })
                    break

            if not is_covered:
                gaps.append({
                    **control,
                    "recommendation": f"Add verification check covering {control['name']}",
                })
                all_gaps.append({"framework": fw, **control})

        total = len(fw_controls)
        covered_count = len(covered)
        coverage_matrices.append(CoverageMatrix(
            framework=fw,
            total_controls=total,
            covered_controls=covered_count,
            coverage_percentage=round(covered_count / max(total, 1) * 100, 1),
            covered=covered,
            gaps=gaps,
        ))

    scores = [cm.coverage_percentage for cm in coverage_matrices]
    overall_score = round(sum(scores) / max(len(scores), 1), 1)

    # Compute audit hash
    content = f"{report_id}:{organization}:{datetime.utcnow().isoformat()}"
    audit_hash = hashlib.sha256(content.encode()).hexdigest()

    report = ComplianceReportData(
        id=report_id,
        organization=organization,
        frameworks=frameworks,
        generated_at=datetime.utcnow().isoformat(),
        coverage_matrices=coverage_matrices,
        evidence_artifacts=evidence,
        gap_analysis=all_gaps,
        overall_score=overall_score,
        audit_hash=audit_hash,
    )

    _compliance_reports[report_id] = report.model_dump()
    _append_audit("compliance_report.generated", "compliance_report", report_id, organization, {
        "frameworks": frameworks,
        "score": overall_score,
    })

    return report


@router.get("/reports/{report_id}")
async def get_compliance_report(report_id: str) -> dict[str, Any]:
    """Retrieve a previously generated compliance report."""
    report = _compliance_reports.get(report_id)
    if not report:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Report not found")
    return report


@router.get("/audit-trail")
async def get_compliance_audit_trail(
    limit: int = Query(default=50, le=500),
    offset: int = Query(default=0, ge=0),
) -> dict[str, Any]:
    """Get the immutable compliance audit trail with integrity verification."""
    entries = _audit_trail[offset:offset + limit]

    # Verify chain integrity
    chain_valid = True
    for i, entry in enumerate(entries):
        if i > 0:
            expected_prev = entries[i - 1]["integrity_hash"]
            if entry["previous_hash"] != expected_prev:
                chain_valid = False
                break

    return {
        "entries": entries,
        "total": len(_audit_trail),
        "chain_integrity_valid": chain_valid,
        "offset": offset,
        "limit": limit,
    }


@router.get("/frameworks-detail")
async def get_frameworks_detail() -> dict[str, Any]:
    """Get all supported frameworks with their controls and CodeVerify mappings."""
    result: dict[str, Any] = {}
    for fw, controls in FRAMEWORK_CONTROLS.items():
        enriched = []
        for control in controls:
            mapped_checks = []
            for check_type, mappings in CONTROL_MAPPINGS.items():
                for m in mappings:
                    if m["framework"] == fw and m["control"] == control["id"]:
                        mapped_checks.append({"check_type": check_type, "relationship": m["relationship"]})
            enriched.append({**control, "codeverify_checks": mapped_checks})
        result[fw] = {"controls": enriched, "total": len(controls)}
    return result
