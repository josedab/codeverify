"""Smart Code Search - Semantic code search using embeddings.

This module provides semantic code search capabilities using embeddings,
enabling developers to find similar code patterns, potential duplicates,
and related implementations across the codebase.

Features:
- Code embedding generation
- Semantic similarity search
- Pattern clustering
- Duplicate detection
- Related code discovery
"""

import hashlib
import json
import math
import re
import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

import structlog

from .base import AgentConfig, AgentResult, BaseAgent

logger = structlog.get_logger()


# =============================================================================
# Enums and Data Classes
# =============================================================================


class SearchMode(str, Enum):
    """Search modes."""

    SEMANTIC = "semantic"  # Use embeddings
    STRUCTURAL = "structural"  # AST-based
    HYBRID = "hybrid"  # Both


class CodeType(str, Enum):
    """Types of code units."""

    FUNCTION = "function"
    CLASS = "class"
    METHOD = "method"
    MODULE = "module"
    SNIPPET = "snippet"


@dataclass
class CodeUnit:
    """A searchable unit of code."""

    id: str
    code: str
    code_type: CodeType
    name: str | None = None

    # Location
    file_path: str | None = None
    line_start: int | None = None
    line_end: int | None = None

    # Metadata
    language: str = "python"
    docstring: str | None = None
    signature: str | None = None

    # Embedding (computed)
    embedding: list[float] | None = None

    # Structural features (computed)
    structural_hash: str | None = None
    complexity: float = 0.0
    token_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "code": self.code[:500] + "..." if len(self.code) > 500 else self.code,
            "code_type": self.code_type.value,
            "name": self.name,
            "file_path": self.file_path,
            "line_start": self.line_start,
            "line_end": self.line_end,
            "language": self.language,
            "docstring": self.docstring,
            "signature": self.signature,
            "has_embedding": self.embedding is not None,
            "structural_hash": self.structural_hash,
            "complexity": self.complexity,
            "token_count": self.token_count,
        }


@dataclass
class SearchResult:
    """A search result."""

    code_unit: CodeUnit
    similarity_score: float
    match_type: str  # "semantic", "structural", "hybrid"

    # Explanation
    match_reason: str = ""
    highlighted_code: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "code_unit": self.code_unit.to_dict(),
            "similarity_score": self.similarity_score,
            "match_type": self.match_type,
            "match_reason": self.match_reason,
            "highlighted_code": self.highlighted_code,
        }


@dataclass
class SearchQuery:
    """A search query."""

    query: str
    mode: SearchMode = SearchMode.HYBRID
    language: str | None = None
    code_type: CodeType | None = None
    file_pattern: str | None = None
    max_results: int = 10
    min_similarity: float = 0.5

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "query": self.query,
            "mode": self.mode.value,
            "language": self.language,
            "code_type": self.code_type.value if self.code_type else None,
            "file_pattern": self.file_pattern,
            "max_results": self.max_results,
            "min_similarity": self.min_similarity,
        }


@dataclass
class DuplicateGroup:
    """A group of duplicate or near-duplicate code."""

    group_id: str
    code_units: list[CodeUnit]
    similarity: float
    duplicate_type: str  # "exact", "near", "semantic"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "group_id": self.group_id,
            "code_units": [cu.to_dict() for cu in self.code_units],
            "similarity": self.similarity,
            "duplicate_type": self.duplicate_type,
            "count": len(self.code_units),
        }


@dataclass
class CodeCluster:
    """A cluster of related code."""

    cluster_id: str
    label: str
    code_units: list[CodeUnit]
    centroid: list[float] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "cluster_id": self.cluster_id,
            "label": self.label,
            "code_units": [cu.to_dict() for cu in self.code_units],
            "count": len(self.code_units),
        }


# =============================================================================
# Smart Code Search Agent
# =============================================================================


class SmartCodeSearch(BaseAgent):
    """Semantic code search engine using embeddings.

    Enables finding similar code patterns, potential duplicates,
    and related implementations across a codebase.

    Example usage:
        search = SmartCodeSearch()
        search.index_code(code_units)
        results = await search.search("function that validates email")
        duplicates = search.find_duplicates()
    """

    EMBEDDING_PROMPT = """Generate a semantic summary for code search.

Code:
{code}

Summarize the purpose, functionality, and key patterns in this code.
Focus on what the code does, not implementation details."""

    def __init__(
        self,
        config: AgentConfig | None = None,
        embedding_dim: int = 384,
    ) -> None:
        """Initialize the smart code search engine."""
        super().__init__(config)
        self.embedding_dim = embedding_dim

        # Index storage
        self._code_units: dict[str, CodeUnit] = {}
        self._embeddings_index: dict[str, list[float]] = {}
        self._structural_index: dict[str, list[str]] = {}  # hash -> ids

        # Inverted index for keyword search
        self._keyword_index: dict[str, set[str]] = defaultdict(set)

    async def analyze(self, code: str, context: dict[str, Any]) -> AgentResult:
        """Analyze code for search indexing."""
        start_time = time.time()

        try:
            query = context.get("query", code)
            results = await self.search(SearchQuery(query=query))

            return AgentResult(
                success=True,
                data={
                    "results": [r.to_dict() for r in results],
                    "count": len(results),
                },
                latency_ms=(time.time() - start_time) * 1000,
            )

        except Exception as e:
            logger.error("Code search failed", error=str(e))
            return AgentResult(
                success=False,
                error=str(e),
                latency_ms=(time.time() - start_time) * 1000,
            )

    def index_code(self, code_units: list[CodeUnit]) -> int:
        """Index code units for searching."""
        indexed = 0

        for unit in code_units:
            # Compute structural features
            unit.structural_hash = self._compute_structural_hash(unit.code)
            unit.token_count = len(unit.code.split())
            unit.complexity = self._compute_complexity(unit.code)

            # Store in indices
            self._code_units[unit.id] = unit

            # Structural index
            if unit.structural_hash not in self._structural_index:
                self._structural_index[unit.structural_hash] = []
            self._structural_index[unit.structural_hash].append(unit.id)

            # Keyword index
            keywords = self._extract_keywords(unit)
            for kw in keywords:
                self._keyword_index[kw].add(unit.id)

            indexed += 1

        logger.info(f"Indexed {indexed} code units")
        return indexed

    async def index_with_embeddings(self, code_units: list[CodeUnit]) -> int:
        """Index code units with semantic embeddings."""
        indexed = self.index_code(code_units)

        # Generate embeddings
        for unit in code_units:
            embedding = await self._generate_embedding(unit)
            if embedding:
                unit.embedding = embedding
                self._embeddings_index[unit.id] = embedding

        logger.info(f"Generated embeddings for {len(self._embeddings_index)} units")
        return indexed

    async def search(self, query: SearchQuery) -> list[SearchResult]:
        """Search for code matching the query."""
        results = []

        if query.mode in (SearchMode.SEMANTIC, SearchMode.HYBRID):
            semantic_results = await self._semantic_search(query)
            results.extend(semantic_results)

        if query.mode in (SearchMode.STRUCTURAL, SearchMode.HYBRID):
            structural_results = self._structural_search(query)
            results.extend(structural_results)

        # Keyword fallback
        if not results:
            keyword_results = self._keyword_search(query)
            results.extend(keyword_results)

        # Merge and deduplicate
        results = self._merge_results(results)

        # Apply filters
        results = self._apply_filters(results, query)

        # Sort by similarity
        results.sort(key=lambda r: r.similarity_score, reverse=True)

        # Limit results
        return results[:query.max_results]

    async def _semantic_search(self, query: SearchQuery) -> list[SearchResult]:
        """Search using semantic embeddings."""
        results = []

        # Generate query embedding
        query_embedding = await self._generate_query_embedding(query.query)
        if not query_embedding:
            return results

        # Find similar embeddings
        for unit_id, embedding in self._embeddings_index.items():
            similarity = self._cosine_similarity(query_embedding, embedding)

            if similarity >= query.min_similarity:
                unit = self._code_units.get(unit_id)
                if unit:
                    results.append(SearchResult(
                        code_unit=unit,
                        similarity_score=similarity,
                        match_type="semantic",
                        match_reason=f"Semantic similarity: {similarity:.2f}",
                    ))

        return results

    def _structural_search(self, query: SearchQuery) -> list[SearchResult]:
        """Search using structural similarity."""
        results = []

        # Compute structural hash of query
        query_hash = self._compute_structural_hash(query.query)

        # Find exact structural matches
        if query_hash in self._structural_index:
            for unit_id in self._structural_index[query_hash]:
                unit = self._code_units.get(unit_id)
                if unit:
                    results.append(SearchResult(
                        code_unit=unit,
                        similarity_score=1.0,
                        match_type="structural",
                        match_reason="Structural match (same AST pattern)",
                    ))

        # Find similar structures
        for hash_key, unit_ids in self._structural_index.items():
            if hash_key != query_hash:
                similarity = self._hash_similarity(query_hash, hash_key)
                if similarity >= query.min_similarity:
                    for unit_id in unit_ids:
                        unit = self._code_units.get(unit_id)
                        if unit and not any(r.code_unit.id == unit_id for r in results):
                            results.append(SearchResult(
                                code_unit=unit,
                                similarity_score=similarity,
                                match_type="structural",
                                match_reason=f"Structural similarity: {similarity:.2f}",
                            ))

        return results

    def _keyword_search(self, query: SearchQuery) -> list[SearchResult]:
        """Search using keyword matching."""
        results = []

        # Extract keywords from query
        keywords = set(re.findall(r'\b\w+\b', query.query.lower()))

        # Score each code unit
        unit_scores: dict[str, float] = defaultdict(float)

        for keyword in keywords:
            if keyword in self._keyword_index:
                for unit_id in self._keyword_index[keyword]:
                    unit_scores[unit_id] += 1.0

        # Normalize scores
        max_score = len(keywords)
        for unit_id, score in unit_scores.items():
            similarity = score / max_score if max_score > 0 else 0

            if similarity >= query.min_similarity * 0.5:  # Lower threshold for keyword
                unit = self._code_units.get(unit_id)
                if unit:
                    results.append(SearchResult(
                        code_unit=unit,
                        similarity_score=similarity,
                        match_type="keyword",
                        match_reason=f"Keyword match: {int(score)}/{len(keywords)} keywords",
                    ))

        return results

    def find_duplicates(
        self,
        threshold: float = 0.9,
    ) -> list[DuplicateGroup]:
        """Find duplicate or near-duplicate code."""
        groups = []
        processed = set()

        # Exact duplicates (same structural hash)
        for hash_key, unit_ids in self._structural_index.items():
            if len(unit_ids) > 1:
                units = [self._code_units[uid] for uid in unit_ids if uid in self._code_units]
                if len(units) > 1:
                    groups.append(DuplicateGroup(
                        group_id=self._generate_id(),
                        code_units=units,
                        similarity=1.0,
                        duplicate_type="exact",
                    ))
                    processed.update(unit_ids)

        # Semantic duplicates (high embedding similarity)
        embedding_items = list(self._embeddings_index.items())

        for i, (id1, emb1) in enumerate(embedding_items):
            if id1 in processed:
                continue

            similar = [id1]

            for j, (id2, emb2) in enumerate(embedding_items[i + 1:], i + 1):
                if id2 in processed:
                    continue

                similarity = self._cosine_similarity(emb1, emb2)
                if similarity >= threshold:
                    similar.append(id2)

            if len(similar) > 1:
                units = [self._code_units[uid] for uid in similar if uid in self._code_units]
                if len(units) > 1:
                    groups.append(DuplicateGroup(
                        group_id=self._generate_id(),
                        code_units=units,
                        similarity=threshold,
                        duplicate_type="semantic" if threshold < 1.0 else "near",
                    ))
                    processed.update(similar)

        return groups

    def find_related(self, code_unit: CodeUnit, max_results: int = 5) -> list[SearchResult]:
        """Find code related to a given code unit."""
        results = []

        if code_unit.embedding:
            # Find by embedding similarity
            for unit_id, embedding in self._embeddings_index.items():
                if unit_id == code_unit.id:
                    continue

                similarity = self._cosine_similarity(code_unit.embedding, embedding)
                if similarity > 0.5:
                    unit = self._code_units.get(unit_id)
                    if unit:
                        results.append(SearchResult(
                            code_unit=unit,
                            similarity_score=similarity,
                            match_type="semantic",
                            match_reason=f"Related code (similarity: {similarity:.2f})",
                        ))

        # Add structural matches
        if code_unit.structural_hash and code_unit.structural_hash in self._structural_index:
            for unit_id in self._structural_index[code_unit.structural_hash]:
                if unit_id != code_unit.id and not any(r.code_unit.id == unit_id for r in results):
                    unit = self._code_units.get(unit_id)
                    if unit:
                        results.append(SearchResult(
                            code_unit=unit,
                            similarity_score=0.9,
                            match_type="structural",
                            match_reason="Same structural pattern",
                        ))

        results.sort(key=lambda r: r.similarity_score, reverse=True)
        return results[:max_results]

    def cluster_code(self, num_clusters: int = 10) -> list[CodeCluster]:
        """Cluster code by semantic similarity."""
        if not self._embeddings_index:
            return []

        # Simple k-means clustering
        embeddings = list(self._embeddings_index.items())

        if len(embeddings) < num_clusters:
            num_clusters = max(1, len(embeddings) // 2)

        # Initialize centroids randomly
        import random
        centroid_indices = random.sample(range(len(embeddings)), min(num_clusters, len(embeddings)))
        centroids = [embeddings[i][1] for i in centroid_indices]

        # Assign to clusters
        clusters: list[list[str]] = [[] for _ in range(len(centroids))]

        for unit_id, embedding in embeddings:
            best_cluster = 0
            best_similarity = -1

            for i, centroid in enumerate(centroids):
                similarity = self._cosine_similarity(embedding, centroid)
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_cluster = i

            clusters[best_cluster].append(unit_id)

        # Create cluster objects
        result = []
        for i, cluster_ids in enumerate(clusters):
            if cluster_ids:
                units = [self._code_units[uid] for uid in cluster_ids if uid in self._code_units]
                if units:
                    # Generate label from common patterns
                    label = self._generate_cluster_label(units)

                    result.append(CodeCluster(
                        cluster_id=self._generate_id(),
                        label=label,
                        code_units=units,
                        centroid=centroids[i] if i < len(centroids) else None,
                    ))

        return result

    async def _generate_embedding(self, unit: CodeUnit) -> list[float] | None:
        """Generate embedding for a code unit."""
        try:
            # Try LLM-based embedding
            prompt = f"Summarize this code's purpose: {unit.code[:1000]}"
            response = await self._call_llm(
                "You are a code summarization expert.",
                prompt,
            )

            summary = response.get("content", "")

            # Convert summary to simple embedding (hash-based for now)
            return self._text_to_embedding(summary + unit.code)

        except Exception:
            # Fallback to code-based embedding
            return self._text_to_embedding(unit.code)

    async def _generate_query_embedding(self, query: str) -> list[float] | None:
        """Generate embedding for a search query."""
        return self._text_to_embedding(query)

    def _text_to_embedding(self, text: str) -> list[float]:
        """Convert text to a simple embedding using character n-grams."""
        # Simple bag-of-ngrams embedding
        ngram_size = 3
        ngrams: dict[str, int] = defaultdict(int)

        text_lower = text.lower()
        for i in range(len(text_lower) - ngram_size + 1):
            ngram = text_lower[i:i + ngram_size]
            ngrams[ngram] += 1

        # Hash ngrams to fixed-size vector
        embedding = [0.0] * self.embedding_dim

        for ngram, count in ngrams.items():
            idx = hash(ngram) % self.embedding_dim
            embedding[idx] += count

        # Normalize
        magnitude = math.sqrt(sum(x * x for x in embedding))
        if magnitude > 0:
            embedding = [x / magnitude for x in embedding]

        return embedding

    def _compute_structural_hash(self, code: str) -> str:
        """Compute structural hash of code."""
        # Normalize code
        normalized = self._normalize_code(code)
        return hashlib.sha256(normalized.encode()).hexdigest()[:16]

    def _normalize_code(self, code: str) -> str:
        """Normalize code for structural comparison."""
        # Remove comments
        code = re.sub(r'#.*$', '', code, flags=re.MULTILINE)
        code = re.sub(r'//.*$', '', code, flags=re.MULTILINE)
        code = re.sub(r'/\*.*?\*/', '', code, flags=re.DOTALL)

        # Remove string literals
        code = re.sub(r'"[^"]*"', '""', code)
        code = re.sub(r"'[^']*'", "''", code)

        # Normalize whitespace
        code = re.sub(r'\s+', ' ', code)

        # Normalize variable names (simplified)
        code = re.sub(r'\b[a-z_][a-z0-9_]*\b', 'VAR', code)

        return code.strip()

    def _compute_complexity(self, code: str) -> float:
        """Compute cyclomatic complexity estimate."""
        # Count decision points
        decision_keywords = ['if', 'elif', 'else', 'for', 'while', 'and', 'or', 'try', 'except', 'case']
        complexity = 1.0

        for keyword in decision_keywords:
            complexity += len(re.findall(rf'\b{keyword}\b', code))

        return complexity

    def _extract_keywords(self, unit: CodeUnit) -> set[str]:
        """Extract keywords from a code unit."""
        keywords = set()

        # Extract identifiers
        identifiers = re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', unit.code)
        for ident in identifiers:
            # Split camelCase and snake_case
            parts = re.findall(r'[A-Z]?[a-z]+|[A-Z]+(?=[A-Z]|$)', ident)
            for part in parts:
                if len(part) > 2:
                    keywords.add(part.lower())

        # Add name if present
        if unit.name:
            keywords.add(unit.name.lower())

        # Remove common words
        common = {'def', 'class', 'return', 'self', 'none', 'true', 'false', 'and', 'or', 'not', 'if', 'else', 'for', 'in', 'while'}
        keywords -= common

        return keywords

    def _cosine_similarity(self, vec1: list[float], vec2: list[float]) -> float:
        """Compute cosine similarity between two vectors."""
        if len(vec1) != len(vec2):
            return 0.0

        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = math.sqrt(sum(a * a for a in vec1))
        norm2 = math.sqrt(sum(b * b for b in vec2))

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)

    def _hash_similarity(self, hash1: str, hash2: str) -> float:
        """Compute similarity between two hashes."""
        # Simple character-based similarity
        if hash1 == hash2:
            return 1.0

        matches = sum(c1 == c2 for c1, c2 in zip(hash1, hash2))
        return matches / max(len(hash1), len(hash2))

    def _merge_results(self, results: list[SearchResult]) -> list[SearchResult]:
        """Merge and deduplicate results."""
        merged: dict[str, SearchResult] = {}

        for result in results:
            unit_id = result.code_unit.id

            if unit_id in merged:
                # Keep the higher score
                if result.similarity_score > merged[unit_id].similarity_score:
                    merged[unit_id] = result
            else:
                merged[unit_id] = result

        return list(merged.values())

    def _apply_filters(
        self,
        results: list[SearchResult],
        query: SearchQuery,
    ) -> list[SearchResult]:
        """Apply query filters to results."""
        filtered = []

        for result in results:
            unit = result.code_unit

            # Language filter
            if query.language and unit.language != query.language:
                continue

            # Code type filter
            if query.code_type and unit.code_type != query.code_type:
                continue

            # File pattern filter
            if query.file_pattern and unit.file_path:
                if not re.search(query.file_pattern, unit.file_path):
                    continue

            # Similarity filter
            if result.similarity_score < query.min_similarity:
                continue

            filtered.append(result)

        return filtered

    def _generate_cluster_label(self, units: list[CodeUnit]) -> str:
        """Generate a label for a cluster of code units."""
        # Find common keywords
        all_keywords: dict[str, int] = defaultdict(int)

        for unit in units:
            keywords = self._extract_keywords(unit)
            for kw in keywords:
                all_keywords[kw] += 1

        # Sort by frequency
        sorted_keywords = sorted(all_keywords.items(), key=lambda x: x[1], reverse=True)

        # Take top keywords
        top = [kw for kw, _ in sorted_keywords[:3]]

        if top:
            return " / ".join(top)
        else:
            return f"Cluster ({len(units)} items)"

    def _generate_id(self) -> str:
        """Generate a unique ID."""
        return hashlib.sha256(
            f"{time.time()}{len(self._code_units)}".encode()
        ).hexdigest()[:12]

    def get_statistics(self) -> dict[str, Any]:
        """Get search engine statistics."""
        return {
            "indexed_units": len(self._code_units),
            "units_with_embeddings": len(self._embeddings_index),
            "unique_structures": len(self._structural_index),
            "keyword_index_size": len(self._keyword_index),
        }

    def clear_index(self) -> None:
        """Clear all indices."""
        self._code_units.clear()
        self._embeddings_index.clear()
        self._structural_index.clear()
        self._keyword_index.clear()
