"""Verification Memory Graph - Persistent knowledge graph of verified code patterns.

This module provides:
- Proof artifact serialization and storage
- Knowledge graph construction for verified patterns
- Cross-project learning capabilities
- Proof reuse and suggestion system
- Privacy-preserving proof aggregation
"""

import hashlib
import json
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, TypeVar
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

import structlog

logger = structlog.get_logger()


class ProofStatus(str, Enum):
    """Status of a verification proof."""
    
    VERIFIED = "verified"  # Proof is valid
    INVALID = "invalid"  # Proof was invalidated
    STALE = "stale"  # Code changed, proof needs revalidation
    PENDING = "pending"  # Not yet verified


class ProofType(str, Enum):
    """Type of formal proof."""
    
    SMT = "smt"  # Z3 SMT solver proof
    SYMBOLIC = "symbolic"  # Symbolic execution
    INVARIANT = "invariant"  # Loop/class invariant
    CONTRACT = "contract"  # Pre/post condition
    TYPE = "type"  # Type system proof
    PATTERN = "pattern"  # Pattern-based verification


class ConstraintKind(str, Enum):
    """Kind of Z3 constraint."""
    
    NULL_CHECK = "null_check"
    BOUNDS_CHECK = "bounds_check"
    OVERFLOW_CHECK = "overflow_check"
    DIV_ZERO_CHECK = "div_zero_check"
    TYPE_CHECK = "type_check"
    INVARIANT = "invariant"
    PRECONDITION = "precondition"
    POSTCONDITION = "postcondition"
    ASSERTION = "assertion"


# =============================================================================
# Proof Artifact Models
# =============================================================================

class SerializedConstraint(BaseModel):
    """A serialized Z3 constraint for storage."""
    
    id: UUID = Field(default_factory=uuid4)
    kind: ConstraintKind
    smt_lib: str  # SMT-LIB2 format
    variables: list[str] = Field(default_factory=list)
    description: str = ""
    source_location: dict[str, Any] = Field(default_factory=dict)


class ProofArtifact(BaseModel):
    """A complete proof artifact that can be stored and reused."""
    
    id: UUID = Field(default_factory=uuid4)
    proof_type: ProofType
    status: ProofStatus = ProofStatus.PENDING
    
    # Content identification
    code_hash: str  # Hash of the verified code
    pattern_hash: str  # Hash of the abstract pattern (for reuse matching)
    language: str
    
    # Proof content
    constraints: list[SerializedConstraint] = Field(default_factory=list)
    model: dict[str, Any] | None = None  # Z3 model if satisfiable
    counterexample: dict[str, Any] | None = None
    
    # Metadata
    function_name: str | None = None
    class_name: str | None = None
    file_path: str | None = None
    line_start: int | None = None
    line_end: int | None = None
    
    # Timing
    verification_time_ms: int = 0
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_used_at: datetime = Field(default_factory=datetime.utcnow)
    use_count: int = 0
    
    # Provenance
    organization_id: str | None = None
    repository_id: str | None = None
    
    def mark_used(self) -> None:
        """Mark this proof as used."""
        self.last_used_at = datetime.utcnow()
        self.use_count += 1
    
    def invalidate(self) -> None:
        """Invalidate this proof."""
        self.status = ProofStatus.INVALID
    
    def to_storage_dict(self) -> dict[str, Any]:
        """Convert to dictionary for storage."""
        data = self.model_dump(mode="json")
        data["id"] = str(self.id)
        data["constraints"] = [
            {**c.model_dump(mode="json"), "id": str(c.id)}
            for c in self.constraints
        ]
        return data
    
    @classmethod
    def from_storage_dict(cls, data: dict[str, Any]) -> "ProofArtifact":
        """Create from storage dictionary."""
        data["id"] = UUID(data["id"])
        data["constraints"] = [
            SerializedConstraint(**{**c, "id": UUID(c["id"])})
            for c in data.get("constraints", [])
        ]
        return cls(**data)


class PatternFingerprint(BaseModel):
    """Abstract fingerprint of a code pattern for matching."""
    
    id: UUID = Field(default_factory=uuid4)
    
    # Pattern characteristics
    node_types: list[str]  # AST node types involved
    control_flow_signature: str  # Simplified CFG signature
    data_flow_signature: str  # Data dependency signature
    variable_count: int
    loop_depth: int
    branch_count: int
    
    # For matching
    similarity_vector: list[float] = Field(default_factory=list)
    
    def compute_similarity(self, other: "PatternFingerprint") -> float:
        """Compute similarity score with another fingerprint."""
        if not self.similarity_vector or not other.similarity_vector:
            return 0.0
        
        # Cosine similarity
        dot_product = sum(a * b for a, b in zip(self.similarity_vector, other.similarity_vector))
        norm_a = sum(a * a for a in self.similarity_vector) ** 0.5
        norm_b = sum(b * b for b in other.similarity_vector) ** 0.5
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
        
        return dot_product / (norm_a * norm_b)


# =============================================================================
# Proof Storage Backend
# =============================================================================

class ProofStorageBackend:
    """Abstract backend for proof storage."""
    
    async def store(self, proof: ProofArtifact) -> None:
        """Store a proof artifact."""
        raise NotImplementedError
    
    async def get(self, proof_id: UUID) -> ProofArtifact | None:
        """Get a proof by ID."""
        raise NotImplementedError
    
    async def get_by_code_hash(self, code_hash: str) -> ProofArtifact | None:
        """Get a proof by code hash."""
        raise NotImplementedError
    
    async def find_similar(
        self,
        pattern_hash: str,
        limit: int = 10,
    ) -> list[ProofArtifact]:
        """Find proofs with similar patterns."""
        raise NotImplementedError
    
    async def invalidate(self, proof_id: UUID) -> None:
        """Invalidate a proof."""
        raise NotImplementedError
    
    async def delete(self, proof_id: UUID) -> None:
        """Delete a proof."""
        raise NotImplementedError


class InMemoryProofStorage(ProofStorageBackend):
    """In-memory proof storage for development/testing."""
    
    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self._proofs: dict[UUID, ProofArtifact] = {}
        self._by_code_hash: dict[str, UUID] = {}
        self._by_pattern_hash: dict[str, list[UUID]] = defaultdict(list)
    
    async def store(self, proof: ProofArtifact) -> None:
        # Evict old proofs if at capacity
        if len(self._proofs) >= self.max_size:
            self._evict_oldest()
        
        self._proofs[proof.id] = proof
        self._by_code_hash[proof.code_hash] = proof.id
        self._by_pattern_hash[proof.pattern_hash].append(proof.id)
    
    async def get(self, proof_id: UUID) -> ProofArtifact | None:
        return self._proofs.get(proof_id)
    
    async def get_by_code_hash(self, code_hash: str) -> ProofArtifact | None:
        proof_id = self._by_code_hash.get(code_hash)
        if proof_id:
            return self._proofs.get(proof_id)
        return None
    
    async def find_similar(
        self,
        pattern_hash: str,
        limit: int = 10,
    ) -> list[ProofArtifact]:
        proof_ids = self._by_pattern_hash.get(pattern_hash, [])[:limit]
        return [self._proofs[pid] for pid in proof_ids if pid in self._proofs]
    
    async def invalidate(self, proof_id: UUID) -> None:
        if proof_id in self._proofs:
            self._proofs[proof_id].invalidate()
    
    async def delete(self, proof_id: UUID) -> None:
        if proof_id in self._proofs:
            proof = self._proofs.pop(proof_id)
            self._by_code_hash.pop(proof.code_hash, None)
            pattern_ids = self._by_pattern_hash.get(proof.pattern_hash, [])
            if proof_id in pattern_ids:
                pattern_ids.remove(proof_id)
    
    def _evict_oldest(self) -> None:
        """Evict oldest proofs by last_used_at."""
        if not self._proofs:
            return
        
        # Sort by last used and evict bottom 10%
        sorted_proofs = sorted(
            self._proofs.values(),
            key=lambda p: p.last_used_at,
        )
        evict_count = max(1, len(sorted_proofs) // 10)
        
        for proof in sorted_proofs[:evict_count]:
            self._proofs.pop(proof.id, None)
            self._by_code_hash.pop(proof.code_hash, None)


# =============================================================================
# Knowledge Graph
# =============================================================================

class GraphNodeType(str, Enum):
    """Type of node in the knowledge graph."""
    
    PROOF = "proof"
    PATTERN = "pattern"
    FUNCTION = "function"
    CLASS = "class"
    FILE = "file"
    REPOSITORY = "repository"
    CONSTRAINT = "constraint"
    INVARIANT = "invariant"


class GraphEdgeType(str, Enum):
    """Type of edge in the knowledge graph."""
    
    PROVES = "proves"  # Proof -> Pattern
    CONTAINS = "contains"  # Class -> Function
    DEPENDS_ON = "depends_on"  # Function -> Function
    SIMILAR_TO = "similar_to"  # Pattern -> Pattern
    IMPLIES = "implies"  # Constraint -> Constraint
    DERIVED_FROM = "derived_from"  # Proof -> Proof


@dataclass
class GraphNode:
    """Node in the knowledge graph."""
    
    id: str
    node_type: GraphNodeType
    data: dict[str, Any] = field(default_factory=dict)
    embedding: list[float] | None = None
    created_at: float = field(default_factory=time.time)


@dataclass
class GraphEdge:
    """Edge in the knowledge graph."""
    
    id: str
    edge_type: GraphEdgeType
    source_id: str
    target_id: str
    weight: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)


class VerificationKnowledgeGraph:
    """Knowledge graph of verification artifacts.
    
    Enables:
    - Proof lookup by pattern similarity
    - Invariant propagation across call graphs
    - Cross-project proof reuse
    - Verification confidence scoring
    """
    
    def __init__(self, storage: ProofStorageBackend | None = None):
        self.storage = storage or InMemoryProofStorage()
        
        self._nodes: dict[str, GraphNode] = {}
        self._edges: dict[str, GraphEdge] = {}
        self._adjacency: dict[str, list[str]] = defaultdict(list)  # node_id -> edge_ids
        self._reverse_adjacency: dict[str, list[str]] = defaultdict(list)
    
    def add_node(self, node: GraphNode) -> None:
        """Add a node to the graph."""
        self._nodes[node.id] = node
    
    def add_edge(self, edge: GraphEdge) -> None:
        """Add an edge to the graph."""
        self._edges[edge.id] = edge
        self._adjacency[edge.source_id].append(edge.id)
        self._reverse_adjacency[edge.target_id].append(edge.id)
    
    def get_node(self, node_id: str) -> GraphNode | None:
        """Get a node by ID."""
        return self._nodes.get(node_id)
    
    def get_neighbors(
        self,
        node_id: str,
        edge_type: GraphEdgeType | None = None,
    ) -> list[GraphNode]:
        """Get neighboring nodes."""
        neighbors = []
        for edge_id in self._adjacency.get(node_id, []):
            edge = self._edges.get(edge_id)
            if edge and (edge_type is None or edge.edge_type == edge_type):
                target_node = self._nodes.get(edge.target_id)
                if target_node:
                    neighbors.append(target_node)
        return neighbors
    
    def get_incoming(
        self,
        node_id: str,
        edge_type: GraphEdgeType | None = None,
    ) -> list[GraphNode]:
        """Get nodes pointing to this node."""
        incoming = []
        for edge_id in self._reverse_adjacency.get(node_id, []):
            edge = self._edges.get(edge_id)
            if edge and (edge_type is None or edge.edge_type == edge_type):
                source_node = self._nodes.get(edge.source_id)
                if source_node:
                    incoming.append(source_node)
        return incoming
    
    async def add_proof(self, proof: ProofArtifact) -> str:
        """Add a proof to the graph."""
        # Store proof
        await self.storage.store(proof)
        
        # Create proof node
        proof_node = GraphNode(
            id=f"proof:{proof.id}",
            node_type=GraphNodeType.PROOF,
            data={
                "proof_id": str(proof.id),
                "status": proof.status.value,
                "code_hash": proof.code_hash,
                "pattern_hash": proof.pattern_hash,
            },
        )
        self.add_node(proof_node)
        
        # Create pattern node if new
        pattern_node_id = f"pattern:{proof.pattern_hash}"
        if pattern_node_id not in self._nodes:
            pattern_node = GraphNode(
                id=pattern_node_id,
                node_type=GraphNodeType.PATTERN,
                data={"pattern_hash": proof.pattern_hash},
            )
            self.add_node(pattern_node)
        
        # Link proof to pattern
        edge = GraphEdge(
            id=f"edge:{proof.id}->pattern",
            edge_type=GraphEdgeType.PROVES,
            source_id=proof_node.id,
            target_id=pattern_node_id,
        )
        self.add_edge(edge)
        
        # Link to function/class if available
        if proof.function_name:
            func_node_id = f"function:{proof.function_name}"
            if func_node_id not in self._nodes:
                func_node = GraphNode(
                    id=func_node_id,
                    node_type=GraphNodeType.FUNCTION,
                    data={"name": proof.function_name},
                )
                self.add_node(func_node)
            
            edge = GraphEdge(
                id=f"edge:{proof.id}->func",
                edge_type=GraphEdgeType.PROVES,
                source_id=proof_node.id,
                target_id=func_node_id,
            )
            self.add_edge(edge)
        
        logger.info(
            "Added proof to knowledge graph",
            proof_id=str(proof.id),
            pattern_hash=proof.pattern_hash[:16],
        )
        
        return proof_node.id
    
    async def find_reusable_proof(
        self,
        code_hash: str,
        pattern_hash: str,
    ) -> ProofArtifact | None:
        """Find a reusable proof for the given code.
        
        First tries exact match, then similar patterns.
        """
        # Try exact match
        exact = await self.storage.get_by_code_hash(code_hash)
        if exact and exact.status == ProofStatus.VERIFIED:
            exact.mark_used()
            return exact
        
        # Try similar patterns
        similar = await self.storage.find_similar(pattern_hash, limit=5)
        for proof in similar:
            if proof.status == ProofStatus.VERIFIED:
                proof.mark_used()
                logger.info(
                    "Found similar proof for reuse",
                    proof_id=str(proof.id),
                    original_pattern=pattern_hash[:16],
                    matched_pattern=proof.pattern_hash[:16],
                )
                return proof
        
        return None
    
    async def invalidate_proofs_for_file(self, file_path: str) -> int:
        """Invalidate all proofs for a file that changed."""
        count = 0
        for node_id, node in list(self._nodes.items()):
            if node.node_type == GraphNodeType.PROOF:
                proof_id = UUID(node.data.get("proof_id", ""))
                proof = await self.storage.get(proof_id)
                if proof and proof.file_path == file_path:
                    await self.storage.invalidate(proof_id)
                    node.data["status"] = ProofStatus.STALE.value
                    count += 1
        
        logger.info("Invalidated proofs for file", file_path=file_path, count=count)
        return count
    
    def get_proof_confidence(self, pattern_hash: str) -> float:
        """Get confidence score for a pattern based on proof history.
        
        Higher confidence if:
        - Multiple successful proofs exist
        - Proofs have been reused successfully
        - No invalidations
        """
        pattern_node_id = f"pattern:{pattern_hash}"
        if pattern_node_id not in self._nodes:
            return 0.0
        
        proof_nodes = self.get_incoming(pattern_node_id, GraphEdgeType.PROVES)
        
        if not proof_nodes:
            return 0.0
        
        # Calculate confidence from proof statistics
        verified_count = 0
        total_uses = 0
        invalidations = 0
        
        for proof_node in proof_nodes:
            status = proof_node.data.get("status", "")
            if status == ProofStatus.VERIFIED.value:
                verified_count += 1
            elif status == ProofStatus.INVALID.value:
                invalidations += 1
        
        # Base confidence from verification ratio
        total = len(proof_nodes)
        confidence = verified_count / total if total > 0 else 0.0
        
        # Penalty for invalidations
        confidence *= max(0.5, 1 - invalidations * 0.1)
        
        return min(1.0, confidence)
    
    def propagate_invariants(
        self,
        source_node_id: str,
        max_depth: int = 3,
    ) -> list[tuple[str, dict[str, Any]]]:
        """Propagate invariants through the call graph.
        
        Returns list of (node_id, invariant_data) pairs.
        """
        propagated = []
        visited = {source_node_id}
        queue = [(source_node_id, 0)]
        
        while queue:
            node_id, depth = queue.pop(0)
            
            if depth >= max_depth:
                continue
            
            # Get dependent nodes
            dependents = self.get_neighbors(node_id, GraphEdgeType.DEPENDS_ON)
            
            for dep_node in dependents:
                if dep_node.id not in visited:
                    visited.add(dep_node.id)
                    queue.append((dep_node.id, depth + 1))
                    
                    # Create propagated invariant
                    invariant_data = {
                        "source": source_node_id,
                        "depth": depth + 1,
                        "propagated": True,
                    }
                    propagated.append((dep_node.id, invariant_data))
        
        return propagated
    
    def get_statistics(self) -> dict[str, Any]:
        """Get graph statistics."""
        node_counts = defaultdict(int)
        edge_counts = defaultdict(int)
        
        for node in self._nodes.values():
            node_counts[node.node_type.value] += 1
        
        for edge in self._edges.values():
            edge_counts[edge.edge_type.value] += 1
        
        return {
            "total_nodes": len(self._nodes),
            "total_edges": len(self._edges),
            "nodes_by_type": dict(node_counts),
            "edges_by_type": dict(edge_counts),
        }


# =============================================================================
# Cross-Project Learning
# =============================================================================

class CrossProjectLearner:
    """Learns verification patterns across projects.
    
    Enables:
    - Organization-wide proof sharing
    - Privacy-preserving aggregation
    - Pattern confidence scoring from usage
    """
    
    def __init__(self, graph: VerificationKnowledgeGraph):
        self.graph = graph
        self._organization_patterns: dict[str, set[str]] = defaultdict(set)  # org_id -> pattern_hashes
        self._pattern_success_rate: dict[str, tuple[int, int]] = {}  # pattern_hash -> (successes, failures)
    
    async def learn_from_verification(
        self,
        proof: ProofArtifact,
        organization_id: str | None = None,
    ) -> None:
        """Learn from a verification result."""
        pattern_hash = proof.pattern_hash
        
        # Track organization patterns
        if organization_id:
            self._organization_patterns[organization_id].add(pattern_hash)
        
        # Update success rate
        successes, failures = self._pattern_success_rate.get(pattern_hash, (0, 0))
        if proof.status == ProofStatus.VERIFIED:
            successes += 1
        else:
            failures += 1
        self._pattern_success_rate[pattern_hash] = (successes, failures)
        
        # Add to graph
        await self.graph.add_proof(proof)
    
    def get_pattern_confidence(self, pattern_hash: str) -> float:
        """Get learned confidence for a pattern."""
        stats = self._pattern_success_rate.get(pattern_hash)
        if not stats:
            return 0.5  # Unknown pattern
        
        successes, failures = stats
        total = successes + failures
        if total == 0:
            return 0.5
        
        # Bayesian estimate with prior
        alpha = 1  # Prior successes
        beta = 1  # Prior failures
        return (successes + alpha) / (total + alpha + beta)
    
    def suggest_similar_proofs(
        self,
        pattern_hash: str,
        organization_id: str | None = None,
        limit: int = 5,
    ) -> list[str]:
        """Suggest similar patterns that have been successfully verified.
        
        Prioritizes patterns from the same organization.
        """
        suggestions = []
        
        # Get organization patterns first
        if organization_id:
            org_patterns = self._organization_patterns.get(organization_id, set())
            for p in org_patterns:
                if p != pattern_hash:
                    confidence = self.get_pattern_confidence(p)
                    if confidence > 0.7:
                        suggestions.append((p, confidence, True))
        
        # Add global high-confidence patterns
        for p, (successes, failures) in self._pattern_success_rate.items():
            if p != pattern_hash and p not in [s[0] for s in suggestions]:
                confidence = self.get_pattern_confidence(p)
                if confidence > 0.8:
                    suggestions.append((p, confidence, False))
        
        # Sort by confidence (org patterns first)
        suggestions.sort(key=lambda x: (x[2], x[1]), reverse=True)
        
        return [s[0] for s in suggestions[:limit]]
    
    def get_organization_stats(self, organization_id: str) -> dict[str, Any]:
        """Get verification stats for an organization."""
        patterns = self._organization_patterns.get(organization_id, set())
        
        total_verifications = 0
        successful = 0
        
        for pattern in patterns:
            stats = self._pattern_success_rate.get(pattern, (0, 0))
            total_verifications += sum(stats)
            successful += stats[0]
        
        return {
            "unique_patterns": len(patterns),
            "total_verifications": total_verifications,
            "success_rate": successful / total_verifications if total_verifications > 0 else 0,
        }


# =============================================================================
# Helper Functions
# =============================================================================

def compute_code_hash(code: str) -> str:
    """Compute hash of code content."""
    # Normalize whitespace but preserve structure
    normalized = "\n".join(line.rstrip() for line in code.split("\n"))
    return hashlib.sha256(normalized.encode()).hexdigest()


def compute_pattern_hash(
    node_types: list[str],
    control_flow: str,
    data_flow: str,
) -> str:
    """Compute hash of abstract code pattern."""
    pattern_str = f"{','.join(sorted(node_types))}|{control_flow}|{data_flow}"
    return hashlib.md5(pattern_str.encode()).hexdigest()


def extract_pattern_fingerprint(code: str, language: str) -> PatternFingerprint:
    """Extract pattern fingerprint from code.
    
    This is a simplified version - production would use proper AST analysis.
    """
    lines = code.split("\n")
    
    # Simple analysis
    node_types = []
    loop_depth = 0
    branch_count = 0
    
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("def ") or stripped.startswith("async def "):
            node_types.append("function")
        elif stripped.startswith("class "):
            node_types.append("class")
        elif stripped.startswith("if ") or stripped.startswith("elif "):
            node_types.append("branch")
            branch_count += 1
        elif stripped.startswith("for ") or stripped.startswith("while "):
            node_types.append("loop")
            loop_depth += 1
        elif stripped.startswith("return "):
            node_types.append("return")
        elif "=" in stripped and "==" not in stripped:
            node_types.append("assignment")
    
    # Create simplified signatures
    control_flow = f"branches:{branch_count},loops:{loop_depth}"
    data_flow = f"assigns:{node_types.count('assignment')}"
    
    # Create similarity vector
    similarity_vector = [
        float(branch_count),
        float(loop_depth),
        float(len(node_types)),
        float(node_types.count("function")),
        float(node_types.count("return")),
    ]
    
    return PatternFingerprint(
        node_types=node_types,
        control_flow_signature=control_flow,
        data_flow_signature=data_flow,
        variable_count=node_types.count("assignment"),
        loop_depth=loop_depth,
        branch_count=branch_count,
        similarity_vector=similarity_vector,
    )


# Global instance
verification_memory_graph = VerificationKnowledgeGraph()
cross_project_learner = CrossProjectLearner(verification_memory_graph)
