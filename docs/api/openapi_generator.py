"""
CodeVerify API Documentation Generator

Generates OpenAPI 3.1 specifications for all CodeVerify features.
Includes endpoints for:
- AI Fingerprinting
- Vulnerability Reachability
- SBOM Generation
- Agentic Auto-Fix
- Runtime Probes
- Codebase Intelligence
- ROI Dashboard
- Intent Traceability
- Counterexample Playground
- Universal Git Support
"""

import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Optional


class HTTPMethod(Enum):
    """HTTP methods."""
    GET = "get"
    POST = "post"
    PUT = "put"
    PATCH = "patch"
    DELETE = "delete"


@dataclass
class Parameter:
    """API parameter definition."""
    name: str
    location: str  # query, path, header
    description: str
    required: bool = False
    schema: dict = field(default_factory=lambda: {"type": "string"})


@dataclass
class RequestBody:
    """API request body definition."""
    description: str
    content_type: str = "application/json"
    schema: dict = field(default_factory=dict)
    required: bool = True


@dataclass
class Response:
    """API response definition."""
    status_code: int
    description: str
    content_type: str = "application/json"
    schema: dict = field(default_factory=dict)


@dataclass
class Endpoint:
    """API endpoint definition."""
    path: str
    method: HTTPMethod
    summary: str
    description: str
    operation_id: str
    tags: list[str] = field(default_factory=list)
    parameters: list[Parameter] = field(default_factory=list)
    request_body: Optional[RequestBody] = None
    responses: list[Response] = field(default_factory=list)
    security: list[dict] = field(default_factory=list)


class OpenAPIGenerator:
    """Generates OpenAPI 3.1 specification."""

    def __init__(self, title: str, version: str, description: str):
        self.spec = {
            "openapi": "3.1.0",
            "info": {
                "title": title,
                "version": version,
                "description": description,
                "contact": {
                    "name": "CodeVerify Team",
                    "url": "https://codeverify.dev",
                    "email": "support@codeverify.dev"
                },
                "license": {
                    "name": "MIT",
                    "url": "https://opensource.org/licenses/MIT"
                }
            },
            "servers": [
                {"url": "https://api.codeverify.dev/v1", "description": "Production"},
                {"url": "https://staging-api.codeverify.dev/v1", "description": "Staging"},
                {"url": "http://localhost:8000/v1", "description": "Local development"}
            ],
            "tags": [],
            "paths": {},
            "components": {
                "schemas": {},
                "securitySchemes": {
                    "bearerAuth": {
                        "type": "http",
                        "scheme": "bearer",
                        "bearerFormat": "JWT"
                    },
                    "apiKey": {
                        "type": "apiKey",
                        "in": "header",
                        "name": "X-API-Key"
                    }
                }
            },
            "security": [{"bearerAuth": []}, {"apiKey": []}]
        }
        self.endpoints: list[Endpoint] = []

    def add_tag(self, name: str, description: str):
        """Add API tag for grouping endpoints."""
        self.spec["tags"].append({"name": name, "description": description})

    def add_schema(self, name: str, schema: dict):
        """Add reusable schema component."""
        self.spec["components"]["schemas"][name] = schema

    def add_endpoint(self, endpoint: Endpoint):
        """Add endpoint to spec."""
        self.endpoints.append(endpoint)
        
        if endpoint.path not in self.spec["paths"]:
            self.spec["paths"][endpoint.path] = {}
        
        operation = {
            "summary": endpoint.summary,
            "description": endpoint.description,
            "operationId": endpoint.operation_id,
            "tags": endpoint.tags,
            "responses": {}
        }
        
        if endpoint.parameters:
            operation["parameters"] = [
                {
                    "name": p.name,
                    "in": p.location,
                    "description": p.description,
                    "required": p.required,
                    "schema": p.schema
                }
                for p in endpoint.parameters
            ]
        
        if endpoint.request_body:
            operation["requestBody"] = {
                "description": endpoint.request_body.description,
                "required": endpoint.request_body.required,
                "content": {
                    endpoint.request_body.content_type: {
                        "schema": endpoint.request_body.schema
                    }
                }
            }
        
        for resp in endpoint.responses:
            operation["responses"][str(resp.status_code)] = {
                "description": resp.description,
                "content": {
                    resp.content_type: {
                        "schema": resp.schema
                    }
                }
            }
        
        if endpoint.security:
            operation["security"] = endpoint.security
        
        self.spec["paths"][endpoint.path][endpoint.method.value] = operation

    def generate(self) -> dict:
        """Generate complete OpenAPI spec."""
        return self.spec

    def to_json(self, indent: int = 2) -> str:
        """Export as JSON string."""
        return json.dumps(self.spec, indent=indent)

    def to_yaml(self) -> str:
        """Export as YAML string."""
        import yaml
        return yaml.dump(self.spec, default_flow_style=False, sort_keys=False)


def create_codeverify_api_spec() -> OpenAPIGenerator:
    """Create complete CodeVerify API specification."""
    
    api = OpenAPIGenerator(
        title="CodeVerify API",
        version="1.0.0",
        description="""
# CodeVerify API

AI-powered code verification platform with formal verification capabilities.

## Features

- **AI Fingerprinting**: Detect AI-generated code
- **Vulnerability Analysis**: Reachability proofs for CVEs
- **SBOM Generation**: CycloneDX/SPDX with verification attestations
- **Auto-Fix**: AI-powered verified code fixes
- **Runtime Probes**: Runtime verification from Z3 specs
- **ROI Dashboard**: Cost and value tracking
- **Intent Traceability**: Ticket-to-code alignment
- **Counterexample Playground**: Interactive Z3 debugger

## Authentication

All endpoints require authentication via Bearer token or API key.
"""
    )

    # === Add Tags ===
    api.add_tag("fingerprinting", "AI Code Fingerprinting - Detect AI-generated code")
    api.add_tag("reachability", "Vulnerability Reachability Analysis")
    api.add_tag("sbom", "SBOM and SLSA Provenance")
    api.add_tag("autofix", "Agentic Auto-Fix")
    api.add_tag("probes", "Runtime Verification Probes")
    api.add_tag("intelligence", "Codebase Intelligence Engine")
    api.add_tag("roi", "ROI Dashboard")
    api.add_tag("traceability", "Intent-to-Code Traceability")
    api.add_tag("playground", "Counterexample Playground")
    api.add_tag("git", "Universal Git Support")
    api.add_tag("webhooks", "Webhook Processing")

    # === Add Schemas ===
    
    # Common schemas
    api.add_schema("Error", {
        "type": "object",
        "properties": {
            "error": {"type": "string"},
            "code": {"type": "string"},
            "details": {"type": "object"}
        },
        "required": ["error"]
    })

    api.add_schema("Pagination", {
        "type": "object",
        "properties": {
            "page": {"type": "integer", "minimum": 1},
            "per_page": {"type": "integer", "minimum": 1, "maximum": 100},
            "total": {"type": "integer"},
            "total_pages": {"type": "integer"}
        }
    })

    # Fingerprinting schemas
    api.add_schema("FingerprintRequest", {
        "type": "object",
        "properties": {
            "code": {"type": "string", "description": "Source code to analyze"},
            "filename": {"type": "string", "description": "Filename for context"},
            "language": {"type": "string", "enum": ["python", "javascript", "typescript", "java", "go"]},
            "include_signals": {"type": "boolean", "default": True}
        },
        "required": ["code"]
    })

    api.add_schema("FingerprintResult", {
        "type": "object",
        "properties": {
            "id": {"type": "string", "format": "uuid"},
            "origin": {"type": "string", "enum": ["human", "ai_generated", "ai_assisted", "unknown"]},
            "confidence_score": {"type": "number", "minimum": 0, "maximum": 1},
            "detected_tools": {
                "type": "array",
                "items": {"type": "string"}
            },
            "signals": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "score": {"type": "number"},
                        "description": {"type": "string"}
                    }
                }
            },
            "analyzed_at": {"type": "string", "format": "date-time"}
        }
    })

    # Reachability schemas
    api.add_schema("VulnerabilityInput", {
        "type": "object",
        "properties": {
            "cve_id": {"type": "string", "pattern": "^CVE-\\d{4}-\\d+$"},
            "function_name": {"type": "string"},
            "vulnerability_type": {
                "type": "string",
                "enum": ["sql_injection", "xss", "buffer_overflow", "path_traversal", "command_injection"]
            },
            "cvss_score": {"type": "number", "minimum": 0, "maximum": 10}
        },
        "required": ["cve_id", "function_name"]
    })

    api.add_schema("ReachabilityResult", {
        "type": "object",
        "properties": {
            "vulnerability_id": {"type": "string"},
            "status": {"type": "string", "enum": ["reachable", "unreachable", "unknown", "timeout"]},
            "proof_type": {"type": "string"},
            "entry_points": {"type": "array", "items": {"type": "string"}},
            "path": {"type": "array", "items": {"type": "string"}},
            "confidence": {"type": "number"},
            "z3_model": {"type": "string"},
            "analyzed_at": {"type": "string", "format": "date-time"}
        }
    })

    # SBOM schemas
    api.add_schema("SBOMRequest", {
        "type": "object",
        "properties": {
            "project_name": {"type": "string"},
            "version": {"type": "string"},
            "format": {"type": "string", "enum": ["cyclonedx_json", "cyclonedx_xml", "spdx_json"]},
            "include_verification": {"type": "boolean", "default": True},
            "slsa_level": {"type": "integer", "minimum": 1, "maximum": 4}
        },
        "required": ["project_name", "version"]
    })

    api.add_schema("SBOM", {
        "type": "object",
        "properties": {
            "id": {"type": "string"},
            "format": {"type": "string"},
            "project_name": {"type": "string"},
            "version": {"type": "string"},
            "components": {
                "type": "array",
                "items": {"$ref": "#/components/schemas/Component"}
            },
            "verification_attestation": {"type": "object"},
            "slsa_provenance": {"type": "object"},
            "generated_at": {"type": "string", "format": "date-time"}
        }
    })

    api.add_schema("Component", {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "version": {"type": "string"},
            "type": {"type": "string"},
            "purl": {"type": "string"},
            "licenses": {"type": "array", "items": {"type": "string"}},
            "hashes": {"type": "object"}
        }
    })

    # Auto-fix schemas
    api.add_schema("AutoFixRequest", {
        "type": "object",
        "properties": {
            "code": {"type": "string"},
            "finding": {
                "type": "object",
                "properties": {
                    "type": {"type": "string"},
                    "line": {"type": "integer"},
                    "message": {"type": "string"}
                }
            },
            "verify_fix": {"type": "boolean", "default": True},
            "generate_tests": {"type": "boolean", "default": True}
        },
        "required": ["code", "finding"]
    })

    api.add_schema("AutoFixResult", {
        "type": "object",
        "properties": {
            "id": {"type": "string"},
            "status": {"type": "string", "enum": ["success", "partial", "failed"]},
            "fixed_code": {"type": "string"},
            "verification_proof": {"type": "object"},
            "generated_tests": {"type": "array", "items": {"type": "string"}},
            "confidence": {"type": "number"}
        }
    })

    # Runtime Probes schemas
    api.add_schema("RuntimeSpec", {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "condition": {"type": "string"},
            "message": {"type": "string"},
            "spec_type": {"type": "string", "enum": ["precondition", "postcondition", "invariant"]},
            "mode": {"type": "string", "enum": ["enforce", "log", "sample", "off"]}
        },
        "required": ["name", "condition"]
    })

    api.add_schema("ProbeCode", {
        "type": "object",
        "properties": {
            "language": {"type": "string"},
            "code": {"type": "string"},
            "decorator": {"type": "string"},
            "middleware": {"type": "string"}
        }
    })

    # ROI schemas
    api.add_schema("ROIMetrics", {
        "type": "object",
        "properties": {
            "period_start": {"type": "string", "format": "date"},
            "period_end": {"type": "string", "format": "date"},
            "total_cost": {"type": "number"},
            "bugs_caught": {"type": "integer"},
            "estimated_savings": {"type": "number"},
            "roi_percentage": {"type": "number"},
            "cost_per_bug": {"type": "number"},
            "breakdown": {"type": "object"}
        }
    })

    # Playground schemas
    api.add_schema("PlaygroundSession", {
        "type": "object",
        "properties": {
            "session_id": {"type": "string"},
            "counterexample_id": {"type": "string"},
            "variables": {"type": "object"},
            "current_step": {"type": "integer"},
            "total_steps": {"type": "integer"},
            "share_link": {"type": "string"}
        }
    })

    # Webhook schemas
    api.add_schema("WebhookPayload", {
        "type": "object",
        "properties": {
            "provider": {"type": "string"},
            "event_type": {"type": "string"},
            "repository": {"type": "object"},
            "pull_request": {"type": "object"},
            "sender": {"type": "string"}
        }
    })

    # === Add Endpoints ===

    # Fingerprinting endpoints
    api.add_endpoint(Endpoint(
        path="/fingerprint",
        method=HTTPMethod.POST,
        summary="Analyze code for AI generation",
        description="Analyzes source code to determine if it was AI-generated, human-written, or AI-assisted.",
        operation_id="analyzeFingerprint",
        tags=["fingerprinting"],
        request_body=RequestBody(
            description="Code to analyze",
            schema={"$ref": "#/components/schemas/FingerprintRequest"}
        ),
        responses=[
            Response(200, "Fingerprint analysis result", schema={"$ref": "#/components/schemas/FingerprintResult"}),
            Response(400, "Invalid request", schema={"$ref": "#/components/schemas/Error"}),
            Response(401, "Unauthorized", schema={"$ref": "#/components/schemas/Error"})
        ]
    ))

    api.add_endpoint(Endpoint(
        path="/fingerprint/{id}",
        method=HTTPMethod.GET,
        summary="Get fingerprint result",
        description="Retrieve a previous fingerprint analysis by ID.",
        operation_id="getFingerprint",
        tags=["fingerprinting"],
        parameters=[
            Parameter("id", "path", "Fingerprint analysis ID", required=True, schema={"type": "string", "format": "uuid"})
        ],
        responses=[
            Response(200, "Fingerprint result", schema={"$ref": "#/components/schemas/FingerprintResult"}),
            Response(404, "Not found", schema={"$ref": "#/components/schemas/Error"})
        ]
    ))

    # Reachability endpoints
    api.add_endpoint(Endpoint(
        path="/reachability/analyze",
        method=HTTPMethod.POST,
        summary="Analyze vulnerability reachability",
        description="Uses Z3 to prove or disprove whether a vulnerable code path is reachable.",
        operation_id="analyzeReachability",
        tags=["reachability"],
        request_body=RequestBody(
            description="Vulnerability and code context",
            schema={
                "type": "object",
                "properties": {
                    "vulnerability": {"$ref": "#/components/schemas/VulnerabilityInput"},
                    "code": {"type": "string"},
                    "entry_points": {"type": "array", "items": {"type": "string"}}
                },
                "required": ["vulnerability", "code"]
            }
        ),
        responses=[
            Response(200, "Reachability analysis result", schema={"$ref": "#/components/schemas/ReachabilityResult"}),
            Response(400, "Invalid request", schema={"$ref": "#/components/schemas/Error"})
        ]
    ))

    # SBOM endpoints
    api.add_endpoint(Endpoint(
        path="/sbom/generate",
        method=HTTPMethod.POST,
        summary="Generate verified SBOM",
        description="Generates a CISA-compliant SBOM with embedded verification attestations.",
        operation_id="generateSBOM",
        tags=["sbom"],
        request_body=RequestBody(
            description="SBOM generation parameters",
            schema={"$ref": "#/components/schemas/SBOMRequest"}
        ),
        responses=[
            Response(201, "Generated SBOM", schema={"$ref": "#/components/schemas/SBOM"}),
            Response(400, "Invalid request", schema={"$ref": "#/components/schemas/Error"})
        ]
    ))

    api.add_endpoint(Endpoint(
        path="/sbom/{id}",
        method=HTTPMethod.GET,
        summary="Get SBOM by ID",
        description="Retrieve a previously generated SBOM.",
        operation_id="getSBOM",
        tags=["sbom"],
        parameters=[
            Parameter("id", "path", "SBOM ID", required=True),
            Parameter("format", "query", "Output format", schema={"type": "string", "enum": ["json", "xml"]})
        ],
        responses=[
            Response(200, "SBOM data", schema={"$ref": "#/components/schemas/SBOM"}),
            Response(404, "Not found", schema={"$ref": "#/components/schemas/Error"})
        ]
    ))

    # Auto-fix endpoints
    api.add_endpoint(Endpoint(
        path="/autofix",
        method=HTTPMethod.POST,
        summary="Generate verified code fix",
        description="AI agent generates a verified fix for the identified issue.",
        operation_id="generateAutoFix",
        tags=["autofix"],
        request_body=RequestBody(
            description="Code and finding to fix",
            schema={"$ref": "#/components/schemas/AutoFixRequest"}
        ),
        responses=[
            Response(200, "Fix result", schema={"$ref": "#/components/schemas/AutoFixResult"}),
            Response(400, "Invalid request", schema={"$ref": "#/components/schemas/Error"}),
            Response(422, "Unable to generate fix", schema={"$ref": "#/components/schemas/Error"})
        ]
    ))

    api.add_endpoint(Endpoint(
        path="/autofix/{id}/pr",
        method=HTTPMethod.POST,
        summary="Create PR from fix",
        description="Creates a pull request with the verified fix.",
        operation_id="createFixPR",
        tags=["autofix"],
        parameters=[
            Parameter("id", "path", "Fix ID", required=True)
        ],
        request_body=RequestBody(
            description="PR details",
            schema={
                "type": "object",
                "properties": {
                    "repository": {"type": "string"},
                    "base_branch": {"type": "string"},
                    "title": {"type": "string"}
                }
            }
        ),
        responses=[
            Response(201, "PR created", schema={"type": "object", "properties": {"pr_url": {"type": "string"}}}),
            Response(404, "Fix not found", schema={"$ref": "#/components/schemas/Error"})
        ]
    ))

    # Runtime Probes endpoints
    api.add_endpoint(Endpoint(
        path="/probes/specs",
        method=HTTPMethod.POST,
        summary="Register runtime specification",
        description="Register a new runtime verification specification.",
        operation_id="registerSpec",
        tags=["probes"],
        request_body=RequestBody(
            description="Runtime specification",
            schema={"$ref": "#/components/schemas/RuntimeSpec"}
        ),
        responses=[
            Response(201, "Spec registered", schema={"type": "object", "properties": {"spec_id": {"type": "string"}}}),
            Response(400, "Invalid spec", schema={"$ref": "#/components/schemas/Error"})
        ]
    ))

    api.add_endpoint(Endpoint(
        path="/probes/generate",
        method=HTTPMethod.POST,
        summary="Generate probe code",
        description="Generate runtime probe code from Z3 specification.",
        operation_id="generateProbe",
        tags=["probes"],
        request_body=RequestBody(
            description="Spec and target language",
            schema={
                "type": "object",
                "properties": {
                    "spec": {"$ref": "#/components/schemas/RuntimeSpec"},
                    "language": {"type": "string", "enum": ["python", "typescript", "javascript"]}
                }
            }
        ),
        responses=[
            Response(200, "Generated probe code", schema={"$ref": "#/components/schemas/ProbeCode"})
        ]
    ))

    api.add_endpoint(Endpoint(
        path="/probes/violations",
        method=HTTPMethod.GET,
        summary="List spec violations",
        description="List runtime specification violations.",
        operation_id="listViolations",
        tags=["probes"],
        parameters=[
            Parameter("spec_name", "query", "Filter by spec name"),
            Parameter("since", "query", "Filter by time", schema={"type": "string", "format": "date-time"}),
            Parameter("page", "query", "Page number", schema={"type": "integer", "default": 1}),
            Parameter("per_page", "query", "Items per page", schema={"type": "integer", "default": 20})
        ],
        responses=[
            Response(200, "Violations list", schema={
                "type": "object",
                "properties": {
                    "violations": {"type": "array"},
                    "pagination": {"$ref": "#/components/schemas/Pagination"}
                }
            })
        ]
    ))

    # ROI Dashboard endpoints
    api.add_endpoint(Endpoint(
        path="/roi/metrics",
        method=HTTPMethod.GET,
        summary="Get ROI metrics",
        description="Get ROI metrics for a time period.",
        operation_id="getROIMetrics",
        tags=["roi"],
        parameters=[
            Parameter("start_date", "query", "Period start", schema={"type": "string", "format": "date"}),
            Parameter("end_date", "query", "Period end", schema={"type": "string", "format": "date"}),
            Parameter("repository", "query", "Filter by repository")
        ],
        responses=[
            Response(200, "ROI metrics", schema={"$ref": "#/components/schemas/ROIMetrics"})
        ]
    ))

    api.add_endpoint(Endpoint(
        path="/roi/bugs",
        method=HTTPMethod.POST,
        summary="Record caught bug",
        description="Record a bug caught by verification for ROI tracking.",
        operation_id="recordBug",
        tags=["roi"],
        request_body=RequestBody(
            description="Bug details",
            schema={
                "type": "object",
                "properties": {
                    "severity": {"type": "string", "enum": ["critical", "high", "medium", "low"]},
                    "category": {"type": "string"},
                    "repository": {"type": "string"},
                    "finding_id": {"type": "string"}
                },
                "required": ["severity", "category"]
            }
        ),
        responses=[
            Response(201, "Bug recorded", schema={"type": "object", "properties": {"id": {"type": "string"}, "estimated_value": {"type": "number"}}})
        ]
    ))

    # Codebase Intelligence endpoints
    api.add_endpoint(Endpoint(
        path="/intelligence/patterns",
        method=HTTPMethod.GET,
        summary="Get detected patterns",
        description="Get code patterns detected in the codebase.",
        operation_id="getPatterns",
        tags=["intelligence"],
        parameters=[
            Parameter("repository", "query", "Repository to analyze"),
            Parameter("pattern_type", "query", "Filter by pattern type")
        ],
        responses=[
            Response(200, "Detected patterns", schema={"type": "object"})
        ]
    ))

    api.add_endpoint(Endpoint(
        path="/intelligence/dependencies",
        method=HTTPMethod.GET,
        summary="Get dependency graph",
        description="Get the dependency graph for a codebase.",
        operation_id="getDependencies",
        tags=["intelligence"],
        parameters=[
            Parameter("repository", "query", "Repository", required=True)
        ],
        responses=[
            Response(200, "Dependency graph", schema={"type": "object"})
        ]
    ))

    # Intent Traceability endpoints
    api.add_endpoint(Endpoint(
        path="/traceability/check",
        method=HTTPMethod.POST,
        summary="Check intent alignment",
        description="Check if code changes align with ticket intent.",
        operation_id="checkAlignment",
        tags=["traceability"],
        request_body=RequestBody(
            description="Ticket and code changes",
            schema={
                "type": "object",
                "properties": {
                    "ticket": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string"},
                            "title": {"type": "string"},
                            "description": {"type": "string"},
                            "provider": {"type": "string", "enum": ["jira", "linear", "github", "gitlab"]}
                        }
                    },
                    "diff": {"type": "string"},
                    "pr_number": {"type": "integer"}
                },
                "required": ["ticket", "diff"]
            }
        ),
        responses=[
            Response(200, "Alignment result", schema={
                "type": "object",
                "properties": {
                    "score": {"type": "number", "minimum": 0, "maximum": 1},
                    "is_aligned": {"type": "boolean"},
                    "scope_creep_detected": {"type": "boolean"},
                    "unmentioned_changes": {"type": "array", "items": {"type": "string"}}
                }
            })
        ]
    ))

    # Counterexample Playground endpoints
    api.add_endpoint(Endpoint(
        path="/playground/sessions",
        method=HTTPMethod.POST,
        summary="Create playground session",
        description="Create a new interactive counterexample playground session.",
        operation_id="createPlaygroundSession",
        tags=["playground"],
        request_body=RequestBody(
            description="Z3 output and optional source code",
            schema={
                "type": "object",
                "properties": {
                    "z3_output": {"type": "string"},
                    "source_code": {"type": "string"},
                    "function_name": {"type": "string"}
                },
                "required": ["z3_output"]
            }
        ),
        responses=[
            Response(201, "Session created", schema={"$ref": "#/components/schemas/PlaygroundSession"})
        ]
    ))

    api.add_endpoint(Endpoint(
        path="/playground/sessions/{session_id}",
        method=HTTPMethod.GET,
        summary="Get playground session",
        description="Get details of a playground session.",
        operation_id="getPlaygroundSession",
        tags=["playground"],
        parameters=[
            Parameter("session_id", "path", "Session ID", required=True)
        ],
        responses=[
            Response(200, "Session details", schema={"$ref": "#/components/schemas/PlaygroundSession"}),
            Response(404, "Session not found", schema={"$ref": "#/components/schemas/Error"})
        ]
    ))

    api.add_endpoint(Endpoint(
        path="/playground/sessions/{session_id}/navigate",
        method=HTTPMethod.POST,
        summary="Navigate trace",
        description="Navigate through execution trace.",
        operation_id="navigateTrace",
        tags=["playground"],
        parameters=[
            Parameter("session_id", "path", "Session ID", required=True)
        ],
        request_body=RequestBody(
            description="Navigation action",
            schema={
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": ["forward", "backward", "reset", "goto"]},
                    "step": {"type": "integer"}
                },
                "required": ["action"]
            }
        ),
        responses=[
            Response(200, "Navigation result", schema={"type": "object", "properties": {"current_step": {"type": "integer"}}})
        ]
    ))

    api.add_endpoint(Endpoint(
        path="/playground/sessions/{session_id}/export",
        method=HTTPMethod.GET,
        summary="Export session",
        description="Export session as HTML or Mermaid diagram.",
        operation_id="exportSession",
        tags=["playground"],
        parameters=[
            Parameter("session_id", "path", "Session ID", required=True),
            Parameter("format", "query", "Export format", required=True, schema={"type": "string", "enum": ["html", "mermaid"]})
        ],
        responses=[
            Response(200, "Exported content", schema={"type": "object", "properties": {"content": {"type": "string"}, "format": {"type": "string"}}})
        ]
    ))

    # Universal Git / Webhook endpoints
    api.add_endpoint(Endpoint(
        path="/webhooks/{provider}",
        method=HTTPMethod.POST,
        summary="Process webhook",
        description="Process incoming webhook from Git provider.",
        operation_id="processWebhook",
        tags=["webhooks", "git"],
        parameters=[
            Parameter("provider", "path", "Git provider", required=True, schema={"type": "string", "enum": ["github", "gitlab", "bitbucket", "gitea", "gerrit", "azure"]})
        ],
        responses=[
            Response(200, "Webhook processed", schema={"$ref": "#/components/schemas/WebhookPayload"}),
            Response(400, "Invalid payload", schema={"$ref": "#/components/schemas/Error"}),
            Response(401, "Invalid signature", schema={"$ref": "#/components/schemas/Error"})
        ],
        security=[]  # Webhooks use signature verification, not API keys
    ))

    api.add_endpoint(Endpoint(
        path="/git/verify",
        method=HTTPMethod.POST,
        summary="Verify local changes",
        description="Verify local Git changes via CLI integration.",
        operation_id="verifyLocalChanges",
        tags=["git"],
        request_body=RequestBody(
            description="Verification request",
            schema={
                "type": "object",
                "properties": {
                    "base": {"type": "string", "default": "HEAD~1"},
                    "target": {"type": "string", "default": "HEAD"},
                    "staged_only": {"type": "boolean", "default": False}
                }
            }
        ),
        responses=[
            Response(200, "Verification result", schema={"type": "object"})
        ]
    ))

    return api


def generate_api_docs():
    """Generate API documentation files."""
    api = create_codeverify_api_spec()
    
    return {
        "openapi.json": api.to_json(),
        "spec": api.generate()
    }


# Main execution
if __name__ == "__main__":
    docs = generate_api_docs()
    print(docs["openapi.json"])
