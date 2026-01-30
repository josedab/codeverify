"""Security Threat Modeling Agent - Generates threat models from architecture."""

import json
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import structlog

from codeverify_agents.base import AgentConfig, AgentResult, BaseAgent

logger = structlog.get_logger()


class STRIDECategory(str, Enum):
    """STRIDE threat categories."""
    SPOOFING = "spoofing"
    TAMPERING = "tampering"
    REPUDIATION = "repudiation"
    INFORMATION_DISCLOSURE = "information_disclosure"
    DENIAL_OF_SERVICE = "denial_of_service"
    ELEVATION_OF_PRIVILEGE = "elevation_of_privilege"


class OWASPCategory(str, Enum):
    """OWASP Top 10 2021 categories."""
    BROKEN_ACCESS_CONTROL = "A01:2021"
    CRYPTOGRAPHIC_FAILURES = "A02:2021"
    INJECTION = "A03:2021"
    INSECURE_DESIGN = "A04:2021"
    SECURITY_MISCONFIGURATION = "A05:2021"
    VULNERABLE_COMPONENTS = "A06:2021"
    AUTH_FAILURES = "A07:2021"
    SOFTWARE_DATA_INTEGRITY = "A08:2021"
    LOGGING_FAILURES = "A09:2021"
    SSRF = "A10:2021"


@dataclass
class AttackSurface:
    """Represents an attack surface in the system."""
    name: str
    surface_type: str  # api_endpoint, database, file_system, network, user_input
    entry_points: list[str] = field(default_factory=list)
    data_flows: list[str] = field(default_factory=list)
    trust_level: str = "untrusted"  # untrusted, semi-trusted, trusted


@dataclass
class Threat:
    """Represents an identified threat."""
    id: str
    title: str
    description: str
    stride_category: STRIDECategory
    owasp_category: OWASPCategory | None
    attack_surface: str
    likelihood: str  # low, medium, high, critical
    impact: str  # low, medium, high, critical
    risk_score: float  # 0-10
    affected_components: list[str] = field(default_factory=list)
    mitigations: list[str] = field(default_factory=list)
    code_locations: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class ThreatModel:
    """Complete threat model for a system."""
    system_name: str
    description: str
    attack_surfaces: list[AttackSurface] = field(default_factory=list)
    threats: list[Threat] = field(default_factory=list)
    trust_boundaries: list[dict[str, Any]] = field(default_factory=list)
    data_flows: list[dict[str, Any]] = field(default_factory=list)
    security_assumptions: list[str] = field(default_factory=list)
    overall_risk_score: float = 0.0
    recommendations: list[str] = field(default_factory=list)


THREAT_MODEL_SYSTEM_PROMPT = """You are an expert security architect specializing in threat modeling. Your task is to analyze code and architecture to generate comprehensive threat models using STRIDE methodology and mapping to OWASP Top 10.

## Your Analysis Process:

1. **Identify Attack Surfaces**:
   - API endpoints (REST, GraphQL, WebSocket)
   - Database connections and queries
   - File system operations
   - Network communications
   - User input handling points
   - Authentication/authorization boundaries

2. **Map Trust Boundaries**:
   - External users → Application
   - Application → Database
   - Application → External services
   - Admin interfaces → Core systems

3. **Apply STRIDE Analysis**:
   - **S**poofing: Can attackers impersonate users/systems?
   - **T**ampering: Can data be modified without authorization?
   - **R**epudiation: Can actions be denied without proof?
   - **I**nformation Disclosure: Can sensitive data be exposed?
   - **D**enial of Service: Can the system be made unavailable?
   - **E**levation of Privilege: Can attackers gain higher permissions?

4. **Map to OWASP Top 10 2021**:
   - A01: Broken Access Control
   - A02: Cryptographic Failures
   - A03: Injection
   - A04: Insecure Design
   - A05: Security Misconfiguration
   - A06: Vulnerable and Outdated Components
   - A07: Identification and Authentication Failures
   - A08: Software and Data Integrity Failures
   - A09: Security Logging and Monitoring Failures
   - A10: Server-Side Request Forgery

5. **Calculate Risk Scores**:
   - Likelihood × Impact = Risk Score (1-10 scale)
   - Consider exploitability, discoverability, and business impact

Respond in JSON format with the complete threat model."""


class ThreatModelingAgent(BaseAgent):
    """
    Agent for generating security threat models from code and architecture.
    
    This agent analyzes code to identify attack surfaces, map threats to STRIDE
    and OWASP categories, and provide actionable security recommendations.
    """

    def __init__(self, config: AgentConfig | None = None) -> None:
        """Initialize threat modeling agent."""
        super().__init__(config)
        if config is None:
            self.config.provider = "anthropic"
            self.config.anthropic_model = "claude-3-sonnet-20240229"

    async def analyze(self, code: str, context: dict[str, Any]) -> AgentResult:
        """
        Analyze code to generate a threat model.

        Args:
            code: The code to analyze (can be multiple files concatenated)
            context: Additional context including:
                - system_name: Name of the system being analyzed
                - architecture_description: High-level architecture description
                - file_paths: List of file paths being analyzed
                - language: Programming language(s)
                - framework: Web framework (e.g., "fastapi", "express")
                - deployment_context: Where/how the system is deployed

        Returns:
            AgentResult with threat model
        """
        start_time = time.time()
        
        system_name = context.get("system_name", "Unknown System")
        
        try:
            user_prompt = self._build_analysis_prompt(code, context)
            
            response = await self._call_llm(
                system_prompt=THREAT_MODEL_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                json_mode=True,
            )
            
            try:
                threat_model_data = json.loads(response["content"])
            except json.JSONDecodeError:
                threat_model_data = {"raw_response": response["content"]}
            
            # Post-process and validate the threat model
            threat_model = self._parse_threat_model(threat_model_data, system_name)
            
            elapsed_ms = (time.time() - start_time) * 1000
            
            logger.info(
                "Threat model generated",
                system=system_name,
                attack_surfaces=len(threat_model.attack_surfaces),
                threats=len(threat_model.threats),
                risk_score=threat_model.overall_risk_score,
                latency_ms=elapsed_ms,
            )
            
            return AgentResult(
                success=True,
                data=self._threat_model_to_dict(threat_model),
                tokens_used=response.get("tokens", 0),
                latency_ms=elapsed_ms,
            )
            
        except Exception as e:
            logger.error("Threat modeling failed", error=str(e), system=system_name)
            return AgentResult(
                success=False,
                error=str(e),
                latency_ms=(time.time() - start_time) * 1000,
            )

    def _build_analysis_prompt(self, code: str, context: dict[str, Any]) -> str:
        """Build the threat modeling analysis prompt."""
        system_name = context.get("system_name", "Unknown System")
        architecture = context.get("architecture_description", "")
        language = context.get("language", "python")
        framework = context.get("framework", "")
        deployment = context.get("deployment_context", "")
        file_paths = context.get("file_paths", [])
        
        parts = [
            f"# Threat Model Analysis for: {system_name}",
            "",
        ]
        
        if architecture:
            parts.extend([
                "## Architecture Description",
                architecture,
                "",
            ])
        
        if deployment:
            parts.extend([
                "## Deployment Context",
                deployment,
                "",
            ])
        
        parts.extend([
            f"## Code Analysis (Language: {language}, Framework: {framework})",
            "",
        ])
        
        if file_paths:
            parts.append(f"Files: {', '.join(file_paths)}")
            parts.append("")
        
        parts.extend([
            "```" + language,
            code[:50000],  # Limit code size
            "```",
            "",
            "Generate a comprehensive threat model including:",
            "1. All attack surfaces with entry points",
            "2. STRIDE threats for each attack surface",
            "3. OWASP Top 10 mappings",
            "4. Risk scores (likelihood × impact)",
            "5. Specific mitigations for each threat",
            "6. Code locations where threats manifest",
        ])
        
        return "\n".join(parts)

    def _parse_threat_model(
        self, data: dict[str, Any], system_name: str
    ) -> ThreatModel:
        """Parse LLM response into structured ThreatModel."""
        attack_surfaces = []
        for surface_data in data.get("attack_surfaces", []):
            attack_surfaces.append(AttackSurface(
                name=surface_data.get("name", "Unknown"),
                surface_type=surface_data.get("type", "unknown"),
                entry_points=surface_data.get("entry_points", []),
                data_flows=surface_data.get("data_flows", []),
                trust_level=surface_data.get("trust_level", "untrusted"),
            ))
        
        threats = []
        for i, threat_data in enumerate(data.get("threats", [])):
            stride_cat = threat_data.get("stride_category", "tampering")
            owasp_cat = threat_data.get("owasp_category")
            
            try:
                stride_enum = STRIDECategory(stride_cat.lower().replace(" ", "_"))
            except ValueError:
                stride_enum = STRIDECategory.TAMPERING
            
            owasp_enum = None
            if owasp_cat:
                try:
                    owasp_enum = OWASPCategory(owasp_cat)
                except ValueError:
                    pass
            
            threats.append(Threat(
                id=threat_data.get("id", f"THREAT-{i+1}"),
                title=threat_data.get("title", "Unknown Threat"),
                description=threat_data.get("description", ""),
                stride_category=stride_enum,
                owasp_category=owasp_enum,
                attack_surface=threat_data.get("attack_surface", ""),
                likelihood=threat_data.get("likelihood", "medium"),
                impact=threat_data.get("impact", "medium"),
                risk_score=float(threat_data.get("risk_score", 5.0)),
                affected_components=threat_data.get("affected_components", []),
                mitigations=threat_data.get("mitigations", []),
                code_locations=threat_data.get("code_locations", []),
            ))
        
        # Calculate overall risk score
        overall_risk = 0.0
        if threats:
            overall_risk = sum(t.risk_score for t in threats) / len(threats)
        
        return ThreatModel(
            system_name=system_name,
            description=data.get("description", ""),
            attack_surfaces=attack_surfaces,
            threats=threats,
            trust_boundaries=data.get("trust_boundaries", []),
            data_flows=data.get("data_flows", []),
            security_assumptions=data.get("security_assumptions", []),
            overall_risk_score=overall_risk,
            recommendations=data.get("recommendations", []),
        )

    def _threat_model_to_dict(self, model: ThreatModel) -> dict[str, Any]:
        """Convert ThreatModel to dictionary for serialization."""
        return {
            "system_name": model.system_name,
            "description": model.description,
            "attack_surfaces": [
                {
                    "name": s.name,
                    "type": s.surface_type,
                    "entry_points": s.entry_points,
                    "data_flows": s.data_flows,
                    "trust_level": s.trust_level,
                }
                for s in model.attack_surfaces
            ],
            "threats": [
                {
                    "id": t.id,
                    "title": t.title,
                    "description": t.description,
                    "stride_category": t.stride_category.value,
                    "owasp_category": t.owasp_category.value if t.owasp_category else None,
                    "attack_surface": t.attack_surface,
                    "likelihood": t.likelihood,
                    "impact": t.impact,
                    "risk_score": t.risk_score,
                    "affected_components": t.affected_components,
                    "mitigations": t.mitigations,
                    "code_locations": t.code_locations,
                }
                for t in model.threats
            ],
            "trust_boundaries": model.trust_boundaries,
            "data_flows": model.data_flows,
            "security_assumptions": model.security_assumptions,
            "overall_risk_score": model.overall_risk_score,
            "recommendations": model.recommendations,
            "threat_summary": {
                "total_threats": len(model.threats),
                "by_stride": self._count_by_stride(model.threats),
                "by_risk_level": self._count_by_risk(model.threats),
            },
        }

    def _count_by_stride(self, threats: list[Threat]) -> dict[str, int]:
        """Count threats by STRIDE category."""
        counts: dict[str, int] = {cat.value: 0 for cat in STRIDECategory}
        for threat in threats:
            counts[threat.stride_category.value] += 1
        return counts

    def _count_by_risk(self, threats: list[Threat]) -> dict[str, int]:
        """Count threats by risk level."""
        counts = {"critical": 0, "high": 0, "medium": 0, "low": 0}
        for threat in threats:
            if threat.risk_score >= 8:
                counts["critical"] += 1
            elif threat.risk_score >= 6:
                counts["high"] += 1
            elif threat.risk_score >= 4:
                counts["medium"] += 1
            else:
                counts["low"] += 1
        return counts

    async def analyze_architecture(
        self,
        architecture_description: str,
        components: list[dict[str, Any]],
        data_flows: list[dict[str, Any]],
    ) -> AgentResult:
        """
        Generate threat model from architecture description (without code).
        
        Args:
            architecture_description: High-level system description
            components: List of system components with their properties
            data_flows: List of data flows between components
            
        Returns:
            AgentResult with threat model
        """
        context = {
            "system_name": "Architecture Analysis",
            "architecture_description": architecture_description,
        }
        
        # Build a pseudo-code representation of the architecture
        arch_code = self._build_architecture_representation(components, data_flows)
        
        return await self.analyze(arch_code, context)

    def _build_architecture_representation(
        self,
        components: list[dict[str, Any]],
        data_flows: list[dict[str, Any]],
    ) -> str:
        """Build a text representation of the architecture for analysis."""
        lines = [
            "# System Architecture",
            "",
            "## Components",
        ]
        
        for comp in components:
            lines.append(f"- {comp.get('name', 'Unknown')}: {comp.get('type', 'service')}")
            if "interfaces" in comp:
                for iface in comp["interfaces"]:
                    lines.append(f"  - Interface: {iface}")
        
        lines.extend([
            "",
            "## Data Flows",
        ])
        
        for flow in data_flows:
            src = flow.get("source", "?")
            dst = flow.get("destination", "?")
            data = flow.get("data_type", "data")
            lines.append(f"- {src} -> {dst}: {data}")
        
        return "\n".join(lines)

    async def generate_data_flow_diagram(
        self, threat_model: ThreatModel
    ) -> dict[str, Any]:
        """
        Generate a data flow diagram (DFD) from the threat model.
        
        Returns a structured representation suitable for visualization.
        """
        nodes = []
        edges = []
        trust_zones = []
        
        # Add attack surfaces as nodes
        for surface in threat_model.attack_surfaces:
            node_type = "external" if surface.trust_level == "untrusted" else "internal"
            nodes.append({
                "id": surface.name,
                "label": surface.name,
                "type": node_type,
                "surface_type": surface.surface_type,
            })
        
        # Add data flows as edges
        for flow in threat_model.data_flows:
            edges.append({
                "source": flow.get("source", ""),
                "target": flow.get("destination", ""),
                "label": flow.get("data_type", ""),
                "is_sensitive": flow.get("is_sensitive", False),
            })
        
        # Add trust boundaries as zones
        for boundary in threat_model.trust_boundaries:
            trust_zones.append({
                "name": boundary.get("name", ""),
                "components": boundary.get("components", []),
                "trust_level": boundary.get("trust_level", ""),
            })
        
        return {
            "nodes": nodes,
            "edges": edges,
            "trust_zones": trust_zones,
            "threats_overlay": [
                {
                    "threat_id": t.id,
                    "attack_surface": t.attack_surface,
                    "risk_score": t.risk_score,
                }
                for t in threat_model.threats
            ],
        }
