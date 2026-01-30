"""Proof Artifact Repository - Searchable library of verification proofs."""

import hashlib
import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Iterator

import structlog

logger = structlog.get_logger()


class ProofStatus(str, Enum):
    """Status of a proof artifact."""
    VERIFIED = "verified"
    PARTIAL = "partial"
    FAILED = "failed"
    PENDING = "pending"


class ProofCategory(str, Enum):
    """Category of proof."""
    NULL_SAFETY = "null_safety"
    BOUNDS_CHECK = "bounds_check"
    TYPE_SAFETY = "type_safety"
    INVARIANT = "invariant"
    PRECONDITION = "precondition"
    POSTCONDITION = "postcondition"
    TERMINATION = "termination"
    CONCURRENCY = "concurrency"
    SECURITY = "security"
    CUSTOM = "custom"


@dataclass
class ProofArtifact:
    """A verified proof artifact that can be reused."""
    proof_id: str
    category: ProofCategory
    status: ProofStatus
    
    # Pattern identification
    pattern_name: str
    pattern_description: str
    language: str
    
    # Code representation
    code_template: str  # Abstracted code pattern
    code_hash: str  # Hash of original code
    
    # Proof details
    constraints: list[str] = field(default_factory=list)  # Z3/SMT constraints
    preconditions: list[str] = field(default_factory=list)
    postconditions: list[str] = field(default_factory=list)
    invariants: list[str] = field(default_factory=list)
    
    # Verification metadata
    z3_model: str | None = None  # Serialized Z3 model
    verification_time_ms: float = 0.0
    solver_stats: dict[str, Any] = field(default_factory=dict)
    
    # Reuse metadata
    created_at: datetime = field(default_factory=datetime.utcnow)
    created_by: str = "system"
    reuse_count: int = 0
    success_rate: float = 1.0
    
    # Searchability
    tags: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    
    # Community metadata
    is_community: bool = False
    organization_id: str | None = None
    upvotes: int = 0
    downvotes: int = 0


@dataclass
class ProofTemplate:
    """A reusable proof template for common patterns."""
    template_id: str
    name: str
    description: str
    category: ProofCategory
    language: str
    
    # Template definition
    code_pattern: str  # Regex or AST pattern to match
    constraint_template: str  # Template for generating constraints
    variables: list[str] = field(default_factory=list)  # Placeholder variables
    
    # Usage
    example_code: str = ""
    example_proof: str = ""
    
    # Stats
    usage_count: int = 0
    avg_verification_time_ms: float = 0.0


@dataclass
class SearchQuery:
    """Query for searching proof artifacts."""
    keywords: list[str] = field(default_factory=list)
    categories: list[ProofCategory] = field(default_factory=list)
    languages: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    min_success_rate: float = 0.0
    min_reuse_count: int = 0
    include_community: bool = True
    organization_id: str | None = None
    limit: int = 20
    offset: int = 0


@dataclass
class SearchResult:
    """Result from proof search."""
    proof: ProofArtifact
    relevance_score: float
    match_reasons: list[str] = field(default_factory=list)


class ProofStorage(ABC):
    """Abstract storage backend for proof artifacts."""
    
    @abstractmethod
    async def store(self, proof: ProofArtifact) -> bool:
        """Store a proof artifact."""
        pass
    
    @abstractmethod
    async def get(self, proof_id: str) -> ProofArtifact | None:
        """Get a proof by ID."""
        pass
    
    @abstractmethod
    async def search(self, query: SearchQuery) -> list[SearchResult]:
        """Search for proofs."""
        pass
    
    @abstractmethod
    async def delete(self, proof_id: str) -> bool:
        """Delete a proof."""
        pass
    
    @abstractmethod
    async def update_stats(
        self, proof_id: str, success: bool
    ) -> bool:
        """Update reuse statistics for a proof."""
        pass


class InMemoryProofStorage(ProofStorage):
    """In-memory storage implementation for proofs."""
    
    def __init__(self) -> None:
        self._proofs: dict[str, ProofArtifact] = {}
        self._index_by_category: dict[ProofCategory, set[str]] = {}
        self._index_by_language: dict[str, set[str]] = {}
        self._index_by_tag: dict[str, set[str]] = {}
    
    async def store(self, proof: ProofArtifact) -> bool:
        """Store a proof artifact."""
        self._proofs[proof.proof_id] = proof
        
        # Update indices
        if proof.category not in self._index_by_category:
            self._index_by_category[proof.category] = set()
        self._index_by_category[proof.category].add(proof.proof_id)
        
        if proof.language not in self._index_by_language:
            self._index_by_language[proof.language] = set()
        self._index_by_language[proof.language].add(proof.proof_id)
        
        for tag in proof.tags:
            if tag not in self._index_by_tag:
                self._index_by_tag[tag] = set()
            self._index_by_tag[tag].add(proof.proof_id)
        
        return True
    
    async def get(self, proof_id: str) -> ProofArtifact | None:
        """Get a proof by ID."""
        return self._proofs.get(proof_id)
    
    async def search(self, query: SearchQuery) -> list[SearchResult]:
        """Search for proofs."""
        candidates = set(self._proofs.keys())
        
        # Filter by category
        if query.categories:
            cat_matches = set()
            for cat in query.categories:
                cat_matches.update(self._index_by_category.get(cat, set()))
            candidates &= cat_matches
        
        # Filter by language
        if query.languages:
            lang_matches = set()
            for lang in query.languages:
                lang_matches.update(self._index_by_language.get(lang, set()))
            candidates &= lang_matches
        
        # Filter by tags
        if query.tags:
            tag_matches = set()
            for tag in query.tags:
                tag_matches.update(self._index_by_tag.get(tag, set()))
            candidates &= tag_matches
        
        # Score and filter remaining candidates
        results = []
        for proof_id in candidates:
            proof = self._proofs[proof_id]
            
            # Apply filters
            if proof.success_rate < query.min_success_rate:
                continue
            if proof.reuse_count < query.min_reuse_count:
                continue
            if not query.include_community and proof.is_community:
                continue
            if query.organization_id and proof.organization_id != query.organization_id:
                if not proof.is_community:
                    continue
            
            # Score relevance
            score, reasons = self._score_relevance(proof, query)
            if score > 0:
                results.append(SearchResult(
                    proof=proof,
                    relevance_score=score,
                    match_reasons=reasons,
                ))
        
        # Sort by relevance and apply pagination
        results.sort(key=lambda r: r.relevance_score, reverse=True)
        return results[query.offset:query.offset + query.limit]
    
    def _score_relevance(
        self, proof: ProofArtifact, query: SearchQuery
    ) -> tuple[float, list[str]]:
        """Score how relevant a proof is to a query."""
        score = 0.0
        reasons = []
        
        # Keyword matching
        if query.keywords:
            all_text = " ".join([
                proof.pattern_name,
                proof.pattern_description,
                " ".join(proof.keywords),
            ]).lower()
            
            for keyword in query.keywords:
                if keyword.lower() in all_text:
                    score += 10
                    reasons.append(f"Matches keyword: {keyword}")
        
        # Boost by success rate
        score += proof.success_rate * 5
        if proof.success_rate >= 0.9:
            reasons.append("High success rate")
        
        # Boost by reuse count (logarithmic)
        import math
        if proof.reuse_count > 0:
            score += math.log(proof.reuse_count + 1) * 2
            reasons.append(f"Reused {proof.reuse_count} times")
        
        # Community proofs get slight boost
        if proof.is_community and proof.upvotes > proof.downvotes:
            score += (proof.upvotes - proof.downvotes) * 0.5
            reasons.append("Community verified")
        
        return score, reasons
    
    async def delete(self, proof_id: str) -> bool:
        """Delete a proof."""
        if proof_id not in self._proofs:
            return False
        
        proof = self._proofs.pop(proof_id)
        
        # Update indices
        if proof.category in self._index_by_category:
            self._index_by_category[proof.category].discard(proof_id)
        if proof.language in self._index_by_language:
            self._index_by_language[proof.language].discard(proof_id)
        for tag in proof.tags:
            if tag in self._index_by_tag:
                self._index_by_tag[tag].discard(proof_id)
        
        return True
    
    async def update_stats(self, proof_id: str, success: bool) -> bool:
        """Update reuse statistics."""
        proof = self._proofs.get(proof_id)
        if not proof:
            return False
        
        proof.reuse_count += 1
        
        # Update success rate (exponential moving average)
        alpha = 0.1
        proof.success_rate = (
            alpha * (1.0 if success else 0.0) +
            (1 - alpha) * proof.success_rate
        )
        
        return True


class ProofArtifactRepository:
    """
    Main repository for managing verification proofs.
    
    Provides searchable library of proofs with support for:
    - Pattern-based proof reuse
    - Community-contributed templates
    - Organization-level sharing
    - Proof versioning and validation
    """
    
    def __init__(
        self,
        storage: ProofStorage | None = None,
    ) -> None:
        """Initialize the proof repository."""
        self.storage = storage or InMemoryProofStorage()
        self._templates: dict[str, ProofTemplate] = {}
        
        # Initialize with common templates
        self._load_builtin_templates()
    
    def _load_builtin_templates(self) -> None:
        """Load built-in proof templates for common patterns."""
        templates = [
            ProofTemplate(
                template_id="null_check_before_use",
                name="Null Check Before Use",
                description="Verifies that a variable is checked for null before dereferencing",
                category=ProofCategory.NULL_SAFETY,
                language="python",
                code_pattern=r"if\s+(\w+)\s+is\s+not\s+None:\s*\n\s+.*\1\.",
                constraint_template="(implies (not (= {var} null)) (valid-access {var}))",
                variables=["var"],
                example_code="if x is not None:\n    x.method()",
                example_proof="(assert (not (= x null))) => safe",
            ),
            ProofTemplate(
                template_id="array_bounds_loop",
                name="Array Bounds in Loop",
                description="Verifies array access within loop is always in bounds",
                category=ProofCategory.BOUNDS_CHECK,
                language="python",
                code_pattern=r"for\s+(\w+)\s+in\s+range\(.*len\((\w+)\).*\):",
                constraint_template="(forall ((i Int)) (=> (and (>= i 0) (< i (len {arr}))) (valid-index {arr} i)))",
                variables=["arr"],
                example_code="for i in range(len(arr)):\n    x = arr[i]",
                example_proof="(assert (and (>= i 0) (< i (len arr)))) => valid",
            ),
            ProofTemplate(
                template_id="division_by_zero",
                name="Division by Zero Guard",
                description="Verifies divisor is checked before division",
                category=ProofCategory.PRECONDITION,
                language="python",
                code_pattern=r"if\s+(\w+)\s*!=\s*0:\s*\n\s+.*\/\s*\1",
                constraint_template="(implies (not (= {divisor} 0)) (safe-division))",
                variables=["divisor"],
                example_code="if y != 0:\n    result = x / y",
                example_proof="(assert (not (= y 0))) => safe-division",
            ),
        ]
        
        for template in templates:
            self._templates[template.template_id] = template
    
    async def store_proof(
        self,
        proof: ProofArtifact,
        validate: bool = True,
    ) -> str:
        """
        Store a new proof artifact.
        
        Args:
            proof: The proof artifact to store
            validate: Whether to validate the proof before storing
            
        Returns:
            The proof_id of the stored proof
        """
        if validate:
            is_valid = await self._validate_proof(proof)
            if not is_valid:
                raise ValueError("Proof validation failed")
        
        # Generate ID if not provided
        if not proof.proof_id:
            proof.proof_id = self._generate_proof_id(proof)
        
        await self.storage.store(proof)
        
        logger.info(
            "Proof stored",
            proof_id=proof.proof_id,
            category=proof.category.value,
            pattern=proof.pattern_name,
        )
        
        return proof.proof_id
    
    async def get_proof(self, proof_id: str) -> ProofArtifact | None:
        """Get a proof by ID."""
        return await self.storage.get(proof_id)
    
    async def search_proofs(
        self,
        keywords: list[str] | None = None,
        categories: list[ProofCategory] | None = None,
        language: str | None = None,
        tags: list[str] | None = None,
        min_success_rate: float = 0.0,
        include_community: bool = True,
        limit: int = 20,
    ) -> list[SearchResult]:
        """
        Search for proof artifacts.
        
        Args:
            keywords: Keywords to search for
            categories: Filter by proof categories
            language: Filter by programming language
            tags: Filter by tags
            min_success_rate: Minimum success rate (0.0 to 1.0)
            include_community: Include community proofs
            limit: Maximum results to return
            
        Returns:
            List of matching proofs with relevance scores
        """
        query = SearchQuery(
            keywords=keywords or [],
            categories=categories or [],
            languages=[language] if language else [],
            tags=tags or [],
            min_success_rate=min_success_rate,
            include_community=include_community,
            limit=limit,
        )
        
        return await self.storage.search(query)
    
    async def find_applicable_proofs(
        self,
        code: str,
        language: str,
        context: dict[str, Any] | None = None,
    ) -> list[tuple[ProofArtifact, float]]:
        """
        Find proofs that may apply to the given code.
        
        Args:
            code: The code to find proofs for
            language: Programming language
            context: Additional context (function name, file path, etc.)
            
        Returns:
            List of (proof, applicability_score) tuples
        """
        import re
        
        applicable = []
        
        # Check templates first
        for template in self._templates.values():
            if template.language != language:
                continue
            
            if re.search(template.code_pattern, code):
                # Template matches - search for related proofs
                results = await self.search_proofs(
                    categories=[template.category],
                    language=language,
                    limit=5,
                )
                for result in results:
                    applicable.append((result.proof, result.relevance_score))
        
        # Also do semantic search
        code_keywords = self._extract_keywords(code)
        if code_keywords:
            results = await self.search_proofs(
                keywords=code_keywords,
                language=language,
                limit=10,
            )
            for result in results:
                if result.proof not in [p for p, _ in applicable]:
                    applicable.append((result.proof, result.relevance_score * 0.5))
        
        # Sort by score
        applicable.sort(key=lambda x: x[1], reverse=True)
        return applicable[:10]
    
    def _extract_keywords(self, code: str) -> list[str]:
        """Extract keywords from code for search."""
        import re
        
        # Extract function names
        func_matches = re.findall(r'def\s+(\w+)|function\s+(\w+)', code)
        keywords = [m[0] or m[1] for m in func_matches]
        
        # Extract common patterns
        patterns = [
            (r'\bif\b.*\bis\s+not\s+None\b', 'null_check'),
            (r'\bfor\b.*\brange\b.*\blen\b', 'bounds_check'),
            (r'\btry\b', 'error_handling'),
            (r'\/[^\n\/]*\w+', 'division'),
            (r'\[\w+\]', 'array_access'),
        ]
        
        for pattern, keyword in patterns:
            if re.search(pattern, code):
                keywords.append(keyword)
        
        return keywords
    
    async def record_reuse(
        self,
        proof_id: str,
        success: bool,
        context: dict[str, Any] | None = None,
    ) -> None:
        """Record that a proof was reused."""
        await self.storage.update_stats(proof_id, success)
        
        logger.info(
            "Proof reuse recorded",
            proof_id=proof_id,
            success=success,
        )
    
    async def create_proof_from_verification(
        self,
        code: str,
        constraints: list[str],
        verification_result: dict[str, Any],
        language: str,
        pattern_name: str | None = None,
    ) -> ProofArtifact:
        """
        Create a proof artifact from a successful verification.
        
        Args:
            code: The verified code
            constraints: Z3/SMT constraints used
            verification_result: Result from the verifier
            language: Programming language
            pattern_name: Optional name for the pattern
            
        Returns:
            Created ProofArtifact
        """
        # Generate abstracted code template
        code_template = self._abstract_code(code, language)
        
        # Determine category from constraints
        category = self._infer_category(constraints)
        
        # Extract metadata
        keywords = self._extract_keywords(code)
        
        proof = ProofArtifact(
            proof_id="",  # Will be generated
            category=category,
            status=ProofStatus.VERIFIED,
            pattern_name=pattern_name or f"verified_{category.value}",
            pattern_description=f"Auto-generated proof for {category.value} verification",
            language=language,
            code_template=code_template,
            code_hash=hashlib.sha256(code.encode()).hexdigest(),
            constraints=constraints,
            preconditions=verification_result.get("preconditions", []),
            postconditions=verification_result.get("postconditions", []),
            invariants=verification_result.get("invariants", []),
            z3_model=verification_result.get("z3_model"),
            verification_time_ms=verification_result.get("time_ms", 0),
            solver_stats=verification_result.get("solver_stats", {}),
            keywords=keywords,
            tags=verification_result.get("tags", []),
        )
        
        proof.proof_id = self._generate_proof_id(proof)
        
        return proof
    
    def _abstract_code(self, code: str, language: str) -> str:
        """Abstract code into a reusable template."""
        import re
        
        # Replace specific identifiers with placeholders
        abstracted = code
        
        # Replace string literals
        abstracted = re.sub(r'"[^"]*"', '"<STRING>"', abstracted)
        abstracted = re.sub(r"'[^']*'", "'<STRING>'", abstracted)
        
        # Replace numbers
        abstracted = re.sub(r'\b\d+\b', '<NUM>', abstracted)
        
        return abstracted
    
    def _infer_category(self, constraints: list[str]) -> ProofCategory:
        """Infer proof category from constraints."""
        constraint_text = " ".join(constraints).lower()
        
        if "null" in constraint_text or "none" in constraint_text:
            return ProofCategory.NULL_SAFETY
        if "bounds" in constraint_text or "len" in constraint_text:
            return ProofCategory.BOUNDS_CHECK
        if "type" in constraint_text:
            return ProofCategory.TYPE_SAFETY
        if "invariant" in constraint_text:
            return ProofCategory.INVARIANT
        if "pre" in constraint_text:
            return ProofCategory.PRECONDITION
        if "post" in constraint_text:
            return ProofCategory.POSTCONDITION
        
        return ProofCategory.CUSTOM
    
    def _generate_proof_id(self, proof: ProofArtifact) -> str:
        """Generate a unique ID for a proof."""
        content = f"{proof.category.value}:{proof.code_hash}:{proof.pattern_name}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    async def _validate_proof(self, proof: ProofArtifact) -> bool:
        """Validate a proof artifact."""
        # Basic validation
        if not proof.constraints and not proof.z3_model:
            return False
        
        if not proof.pattern_name:
            return False
        
        # Future enhancement: Re-run Z3 verification to validate proof is still valid.
        # This would catch proofs that have become stale due to code changes.
        # Implementation would involve: 1) Parse constraints 2) Run Z3 solver
        # 3) Compare result with stored proof status
        return True
    
    def get_template(self, template_id: str) -> ProofTemplate | None:
        """Get a proof template by ID."""
        return self._templates.get(template_id)
    
    def list_templates(
        self,
        category: ProofCategory | None = None,
        language: str | None = None,
    ) -> list[ProofTemplate]:
        """List available proof templates."""
        templates = list(self._templates.values())
        
        if category:
            templates = [t for t in templates if t.category == category]
        
        if language:
            templates = [t for t in templates if t.language == language]
        
        return templates
    
    async def contribute_proof(
        self,
        proof: ProofArtifact,
        organization_id: str | None = None,
        make_community: bool = False,
    ) -> str:
        """
        Contribute a proof to the repository.
        
        Args:
            proof: The proof to contribute
            organization_id: Limit sharing to organization
            make_community: Share with entire community
            
        Returns:
            The proof_id
        """
        proof.is_community = make_community
        proof.organization_id = organization_id
        
        return await self.store_proof(proof)
    
    async def vote_on_proof(
        self,
        proof_id: str,
        upvote: bool,
    ) -> bool:
        """Vote on a community proof."""
        proof = await self.storage.get(proof_id)
        if not proof or not proof.is_community:
            return False
        
        if upvote:
            proof.upvotes += 1
        else:
            proof.downvotes += 1
        
        await self.storage.store(proof)
        return True
    
    async def get_repository_stats(self) -> dict[str, Any]:
        """Get statistics about the proof repository."""
        # Query all proofs (in production, use aggregation queries)
        all_results = await self.storage.search(SearchQuery(limit=10000))
        
        proofs = [r.proof for r in all_results]
        
        return {
            "total_proofs": len(proofs),
            "by_category": self._count_by_field(proofs, "category"),
            "by_language": self._count_by_field(proofs, "language"),
            "by_status": self._count_by_field(proofs, "status"),
            "community_proofs": sum(1 for p in proofs if p.is_community),
            "total_reuses": sum(p.reuse_count for p in proofs),
            "avg_success_rate": (
                sum(p.success_rate for p in proofs) / len(proofs)
                if proofs else 0
            ),
            "templates_count": len(self._templates),
        }
    
    def _count_by_field(
        self, proofs: list[ProofArtifact], field: str
    ) -> dict[str, int]:
        """Count proofs by a field value."""
        counts: dict[str, int] = {}
        for proof in proofs:
            value = getattr(proof, field)
            if hasattr(value, "value"):
                value = value.value
            counts[value] = counts.get(value, 0) + 1
        return counts
