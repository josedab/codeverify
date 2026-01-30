"""
Historical Regression Learning

ML model that learns from organization's past bugs and reverts to predict
similar patterns in new code:
- Bug pattern learning from history
- Personalized prediction model
- Revert pattern detection
- False positive reduction through organization-specific training

Enterprise stickiness through customized, improving predictions.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple


# =============================================================================
# Data Models
# =============================================================================

class BugType(str, Enum):
    """Types of bugs for classification."""
    NULL_POINTER = "null_pointer"
    ARRAY_BOUNDS = "array_bounds"
    TYPE_ERROR = "type_error"
    LOGIC_ERROR = "logic_error"
    CONCURRENCY = "concurrency"
    RESOURCE_LEAK = "resource_leak"
    SECURITY = "security"
    PERFORMANCE = "performance"
    API_MISUSE = "api_misuse"
    OTHER = "other"


class RevertReason(str, Enum):
    """Reasons for code reverts."""
    BUG = "bug"
    PERFORMANCE = "performance"
    BREAKING_CHANGE = "breaking_change"
    INCORRECT_LOGIC = "incorrect_logic"
    TEST_FAILURE = "test_failure"
    SECURITY = "security"
    OTHER = "other"


@dataclass
class BugPattern:
    """A learned bug pattern."""
    
    pattern_id: str
    pattern_type: BugType
    
    # Pattern definition
    description: str
    code_pattern: Optional[str] = None  # Regex pattern
    ast_pattern: Optional[Dict[str, Any]] = None
    
    # Statistics
    occurrences: int = 0
    last_seen: float = 0
    
    # Context
    file_patterns: List[str] = field(default_factory=list)  # File paths where often seen
    languages: List[str] = field(default_factory=list)
    
    # Detection
    confidence_threshold: float = 0.7
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "pattern_id": self.pattern_id,
            "pattern_type": self.pattern_type.value,
            "description": self.description,
            "code_pattern": self.code_pattern,
            "ast_pattern": self.ast_pattern,
            "occurrences": self.occurrences,
            "last_seen": self.last_seen,
            "file_patterns": self.file_patterns,
            "languages": self.languages,
            "confidence_threshold": self.confidence_threshold,
        }


@dataclass
class RevertPattern:
    """A learned revert pattern."""
    
    pattern_id: str
    reason: RevertReason
    
    # Context
    description: str
    original_commit_patterns: List[str] = field(default_factory=list)
    
    # Statistics
    occurrences: int = 0
    avg_time_to_revert_hours: float = 24.0
    
    # Related bugs
    related_bug_patterns: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "pattern_id": self.pattern_id,
            "reason": self.reason.value,
            "description": self.description,
            "original_commit_patterns": self.original_commit_patterns,
            "occurrences": self.occurrences,
            "avg_time_to_revert_hours": self.avg_time_to_revert_hours,
            "related_bug_patterns": self.related_bug_patterns,
        }


@dataclass
class HistoricalBug:
    """A bug from the organization's history."""
    
    bug_id: str
    repository: str
    
    # Commit info
    commit_sha: str
    commit_message: str
    commit_author: str
    commit_date: float
    
    # Bug info
    bug_type: BugType
    file_path: str
    line_number: Optional[int] = None
    
    # Code snippets
    buggy_code: Optional[str] = None
    fixed_code: Optional[str] = None
    
    # Resolution
    fix_commit_sha: Optional[str] = None
    time_to_fix_hours: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "bug_id": self.bug_id,
            "repository": self.repository,
            "commit_sha": self.commit_sha,
            "commit_message": self.commit_message,
            "commit_author": self.commit_author,
            "commit_date": self.commit_date,
            "bug_type": self.bug_type.value,
            "file_path": self.file_path,
            "line_number": self.line_number,
            "buggy_code": self.buggy_code,
            "fixed_code": self.fixed_code,
            "fix_commit_sha": self.fix_commit_sha,
            "time_to_fix_hours": self.time_to_fix_hours,
        }


@dataclass
class PredictionResult:
    """Result of bug prediction."""
    
    has_potential_bug: bool
    confidence: float
    
    # Matched patterns
    matched_patterns: List[str] = field(default_factory=list)
    
    # Details
    predictions: List[Dict[str, Any]] = field(default_factory=list)
    
    # Recommendations
    recommendations: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "has_potential_bug": self.has_potential_bug,
            "confidence": self.confidence,
            "matched_patterns": self.matched_patterns,
            "predictions": self.predictions,
            "recommendations": self.recommendations,
        }


# =============================================================================
# Feature Extraction
# =============================================================================

class CodeFeatureExtractor:
    """Extracts features from code for ML analysis."""
    
    def extract_features(
        self,
        code: str,
        language: str = "python",
    ) -> Dict[str, Any]:
        """Extract features from code."""
        features = {
            "language": language,
            "line_count": len(code.split('\n')),
            "char_count": len(code),
        }
        
        # Structural features
        features.update(self._extract_structural_features(code, language))
        
        # Complexity features
        features.update(self._extract_complexity_features(code))
        
        # Pattern features
        features.update(self._extract_pattern_features(code))
        
        return features
    
    def _extract_structural_features(
        self,
        code: str,
        language: str,
    ) -> Dict[str, Any]:
        """Extract structural features."""
        features = {}
        
        if language == "python":
            features["function_count"] = len(re.findall(r'^\s*def\s+', code, re.M))
            features["class_count"] = len(re.findall(r'^\s*class\s+', code, re.M))
            features["import_count"] = len(re.findall(r'^\s*(?:import|from)\s+', code, re.M))
            features["try_except_count"] = len(re.findall(r'^\s*try\s*:', code, re.M))
            features["assert_count"] = len(re.findall(r'^\s*assert\s+', code, re.M))
        
        elif language in ("typescript", "javascript"):
            features["function_count"] = len(re.findall(r'function\s+\w+', code))
            features["class_count"] = len(re.findall(r'class\s+\w+', code))
            features["import_count"] = len(re.findall(r'import\s+', code))
            features["try_catch_count"] = len(re.findall(r'try\s*{', code))
        
        return features
    
    def _extract_complexity_features(self, code: str) -> Dict[str, Any]:
        """Extract complexity features."""
        features = {}
        
        # Nesting depth estimation
        max_indent = 0
        for line in code.split('\n'):
            indent = len(line) - len(line.lstrip())
            max_indent = max(max_indent, indent)
        
        features["max_nesting"] = max_indent // 4  # Assuming 4-space indent
        
        # Cyclomatic complexity estimation
        decision_points = 0
        decision_points += len(re.findall(r'\bif\b', code))
        decision_points += len(re.findall(r'\belif\b', code))
        decision_points += len(re.findall(r'\belse\b', code))
        decision_points += len(re.findall(r'\bfor\b', code))
        decision_points += len(re.findall(r'\bwhile\b', code))
        decision_points += len(re.findall(r'\band\b', code))
        decision_points += len(re.findall(r'\bor\b', code))
        
        features["cyclomatic_complexity"] = decision_points + 1
        
        return features
    
    def _extract_pattern_features(self, code: str) -> Dict[str, Any]:
        """Extract pattern-based features."""
        features = {}
        
        # Risk patterns
        features["has_eval"] = "eval(" in code
        features["has_exec"] = "exec(" in code
        features["has_sql_concat"] = bool(re.search(r'f"[^"]*SELECT', code, re.I))
        features["has_file_open"] = "open(" in code
        features["has_subprocess"] = "subprocess" in code
        
        # Common bug indicators
        features["has_none_return"] = "return None" in code
        features["has_bare_except"] = bool(re.search(r'except\s*:', code))
        features["has_mutable_default"] = bool(re.search(r'def\s+\w+\([^)]*=\s*\[\]', code))
        
        return features


# =============================================================================
# Pattern Learning
# =============================================================================

class PatternLearner:
    """Learns bug patterns from historical data."""
    
    def __init__(self):
        self.bug_patterns: Dict[str, BugPattern] = {}
        self.revert_patterns: Dict[str, RevertPattern] = {}
        self.feature_extractor = CodeFeatureExtractor()
        
        # Feature statistics for normalization
        self._feature_stats: Dict[str, Dict[str, float]] = {}
        
        # Co-occurrence matrix
        self._pattern_cooccurrence: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    
    def learn_from_bug(self, bug: HistoricalBug) -> Optional[BugPattern]:
        """Learn pattern from a historical bug."""
        # Extract features from buggy code
        if not bug.buggy_code:
            return None
        
        features = self.feature_extractor.extract_features(bug.buggy_code)
        
        # Generate or update pattern
        pattern_key = self._generate_pattern_key(bug, features)
        
        if pattern_key in self.bug_patterns:
            # Update existing pattern
            pattern = self.bug_patterns[pattern_key]
            pattern.occurrences += 1
            pattern.last_seen = time.time()
            
            if bug.file_path not in pattern.file_patterns:
                pattern.file_patterns.append(bug.file_path)
        else:
            # Create new pattern
            pattern = BugPattern(
                pattern_id=pattern_key,
                pattern_type=bug.bug_type,
                description=self._generate_pattern_description(bug, features),
                code_pattern=self._extract_code_pattern(bug),
                occurrences=1,
                last_seen=time.time(),
                file_patterns=[bug.file_path],
                languages=[self._detect_language(bug.file_path)],
            )
            self.bug_patterns[pattern_key] = pattern
        
        return pattern
    
    def learn_from_revert(
        self,
        original_commit: Dict[str, Any],
        revert_commit: Dict[str, Any],
        reason: RevertReason,
    ) -> RevertPattern:
        """Learn pattern from a code revert."""
        # Extract key features from original commit
        commit_patterns = self._extract_commit_patterns(original_commit)
        
        pattern_key = hashlib.sha256(
            f"{reason.value}:{','.join(commit_patterns)}".encode()
        ).hexdigest()[:12]
        
        if pattern_key in self.revert_patterns:
            pattern = self.revert_patterns[pattern_key]
            pattern.occurrences += 1
            
            # Update average time to revert
            original_time = original_commit.get("timestamp", time.time())
            revert_time = revert_commit.get("timestamp", time.time())
            time_diff = (revert_time - original_time) / 3600  # hours
            
            pattern.avg_time_to_revert_hours = (
                (pattern.avg_time_to_revert_hours * (pattern.occurrences - 1) + time_diff)
                / pattern.occurrences
            )
        else:
            pattern = RevertPattern(
                pattern_id=pattern_key,
                reason=reason,
                description=self._generate_revert_description(original_commit, reason),
                original_commit_patterns=commit_patterns,
                occurrences=1,
            )
            self.revert_patterns[pattern_key] = pattern
        
        return pattern
    
    def _generate_pattern_key(
        self,
        bug: HistoricalBug,
        features: Dict[str, Any],
    ) -> str:
        """Generate unique key for a pattern."""
        key_parts = [
            bug.bug_type.value,
            self._detect_language(bug.file_path),
            str(features.get("function_count", 0) > 0),
            str(features.get("has_none_return", False)),
            str(features.get("has_bare_except", False)),
        ]
        
        return hashlib.sha256(":".join(key_parts).encode()).hexdigest()[:12]
    
    def _generate_pattern_description(
        self,
        bug: HistoricalBug,
        features: Dict[str, Any],
    ) -> str:
        """Generate human-readable pattern description."""
        parts = [f"{bug.bug_type.value.replace('_', ' ').title()} bug pattern"]
        
        if features.get("has_none_return"):
            parts.append("with potential null return")
        
        if features.get("has_bare_except"):
            parts.append("with bare except clause")
        
        if features.get("max_nesting", 0) > 4:
            parts.append("in deeply nested code")
        
        return " ".join(parts)
    
    def _extract_code_pattern(self, bug: HistoricalBug) -> Optional[str]:
        """Extract regex pattern from buggy code."""
        if not bug.buggy_code:
            return None
        
        # Look for common problematic patterns
        patterns = [
            (r"return\s+None\b", "null_return"),
            (r"except\s*:", "bare_except"),
            (r"==\s*None\b", "none_comparison"),
            (r"\[\w+\]", "array_access"),
            (r"/\s*\w+", "division"),
        ]
        
        for regex, _ in patterns:
            if re.search(regex, bug.buggy_code):
                return regex
        
        return None
    
    def _extract_commit_patterns(self, commit: Dict[str, Any]) -> List[str]:
        """Extract patterns from a commit."""
        patterns = []
        
        message = commit.get("message", "").lower()
        
        # Commit type indicators
        if "fix" in message:
            patterns.append("fix_commit")
        if "feat" in message or "feature" in message:
            patterns.append("feature_commit")
        if "refactor" in message:
            patterns.append("refactor_commit")
        
        # File patterns
        files = commit.get("files", [])
        if any("test" in f.lower() for f in files):
            patterns.append("modifies_tests")
        if any("config" in f.lower() for f in files):
            patterns.append("modifies_config")
        
        return patterns
    
    def _generate_revert_description(
        self,
        commit: Dict[str, Any],
        reason: RevertReason,
    ) -> str:
        """Generate description for revert pattern."""
        return f"Revert due to {reason.value.replace('_', ' ')}: {commit.get('message', '')[:50]}"
    
    def _detect_language(self, file_path: str) -> str:
        """Detect language from file path."""
        ext_map = {
            ".py": "python",
            ".js": "javascript",
            ".ts": "typescript",
            ".java": "java",
            ".go": "go",
            ".rs": "rust",
            ".rb": "ruby",
        }
        
        for ext, lang in ext_map.items():
            if file_path.endswith(ext):
                return lang
        
        return "unknown"
    
    def get_patterns_summary(self) -> Dict[str, Any]:
        """Get summary of learned patterns."""
        return {
            "bug_patterns": {
                "total": len(self.bug_patterns),
                "by_type": self._count_by_type(),
            },
            "revert_patterns": {
                "total": len(self.revert_patterns),
                "by_reason": self._count_reverts_by_reason(),
            },
        }
    
    def _count_by_type(self) -> Dict[str, int]:
        """Count bug patterns by type."""
        counts: Dict[str, int] = defaultdict(int)
        for pattern in self.bug_patterns.values():
            counts[pattern.pattern_type.value] += pattern.occurrences
        return dict(counts)
    
    def _count_reverts_by_reason(self) -> Dict[str, int]:
        """Count revert patterns by reason."""
        counts: Dict[str, int] = defaultdict(int)
        for pattern in self.revert_patterns.values():
            counts[pattern.reason.value] += pattern.occurrences
        return dict(counts)


# =============================================================================
# Prediction Engine
# =============================================================================

class RegressionPredictor:
    """Predicts bugs based on learned patterns."""
    
    def __init__(self, learner: PatternLearner):
        self.learner = learner
        self.feature_extractor = CodeFeatureExtractor()
    
    def predict(
        self,
        code: str,
        file_path: str,
        language: str = "python",
    ) -> PredictionResult:
        """Predict potential bugs in code."""
        result = PredictionResult(
            has_potential_bug=False,
            confidence=0.0,
        )
        
        # Extract features
        features = self.feature_extractor.extract_features(code, language)
        
        # Check against all learned patterns
        for pattern_id, pattern in self.learner.bug_patterns.items():
            match_score = self._match_pattern(code, features, pattern, file_path)
            
            if match_score >= pattern.confidence_threshold:
                result.matched_patterns.append(pattern_id)
                result.predictions.append({
                    "pattern_id": pattern_id,
                    "pattern_type": pattern.pattern_type.value,
                    "description": pattern.description,
                    "confidence": match_score,
                    "occurrences_in_history": pattern.occurrences,
                })
        
        if result.predictions:
            result.has_potential_bug = True
            result.confidence = max(p["confidence"] for p in result.predictions)
            
            # Generate recommendations
            result.recommendations = self._generate_recommendations(
                result.predictions
            )
        
        return result
    
    def _match_pattern(
        self,
        code: str,
        features: Dict[str, Any],
        pattern: BugPattern,
        file_path: str,
    ) -> float:
        """Calculate match score for a pattern."""
        scores: List[float] = []
        
        # Code pattern match
        if pattern.code_pattern:
            if re.search(pattern.code_pattern, code):
                scores.append(0.8)
        
        # File path similarity
        if pattern.file_patterns:
            for fp in pattern.file_patterns:
                if self._path_similarity(file_path, fp) > 0.5:
                    scores.append(0.6)
                    break
        
        # Language match
        detected_lang = self.learner._detect_language(file_path)
        if detected_lang in pattern.languages:
            scores.append(0.3)
        
        # Recency bonus
        days_since_seen = (time.time() - pattern.last_seen) / 86400
        if days_since_seen < 30:
            scores.append(0.2)
        
        # Occurrence frequency bonus
        if pattern.occurrences > 10:
            scores.append(0.3)
        elif pattern.occurrences > 5:
            scores.append(0.2)
        elif pattern.occurrences > 1:
            scores.append(0.1)
        
        if not scores:
            return 0.0
        
        # Weighted combination
        return min(1.0, sum(scores))
    
    def _path_similarity(self, path1: str, path2: str) -> float:
        """Calculate similarity between file paths."""
        parts1 = path1.split('/')
        parts2 = path2.split('/')
        
        common = len(set(parts1) & set(parts2))
        total = max(len(parts1), len(parts2))
        
        return common / total if total > 0 else 0.0
    
    def _generate_recommendations(
        self,
        predictions: List[Dict[str, Any]],
    ) -> List[str]:
        """Generate recommendations based on predictions."""
        recommendations = []
        
        # Group by pattern type
        by_type: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for pred in predictions:
            by_type[pred["pattern_type"]].append(pred)
        
        if "null_pointer" in by_type:
            recommendations.append(
                "Add null/None checks before accessing values"
            )
        
        if "array_bounds" in by_type:
            recommendations.append(
                "Add bounds checking for array/list access"
            )
        
        if "type_error" in by_type:
            recommendations.append(
                "Verify type compatibility and add type guards"
            )
        
        if "logic_error" in by_type:
            recommendations.append(
                "Review logic flow and add additional test cases"
            )
        
        if "security" in by_type:
            recommendations.append(
                "Review for security vulnerabilities and sanitize inputs"
            )
        
        # General recommendations
        if len(predictions) > 2:
            recommendations.append(
                "Consider breaking this code into smaller, tested functions"
            )
        
        return recommendations


# =============================================================================
# Model Persistence
# =============================================================================

class ModelStorage:
    """Handles persistence of learned models."""
    
    def __init__(self, storage_path: str = ""):
        self.storage_path = storage_path
    
    def save_learner(self, learner: PatternLearner) -> Dict[str, Any]:
        """Save learner state."""
        return {
            "bug_patterns": {
                k: v.to_dict() for k, v in learner.bug_patterns.items()
            },
            "revert_patterns": {
                k: v.to_dict() for k, v in learner.revert_patterns.items()
            },
            "feature_stats": learner._feature_stats,
            "saved_at": time.time(),
        }
    
    def load_learner(self, data: Dict[str, Any]) -> PatternLearner:
        """Load learner from saved state."""
        learner = PatternLearner()
        
        for pid, pdata in data.get("bug_patterns", {}).items():
            learner.bug_patterns[pid] = BugPattern(
                pattern_id=pdata["pattern_id"],
                pattern_type=BugType(pdata["pattern_type"]),
                description=pdata["description"],
                code_pattern=pdata.get("code_pattern"),
                ast_pattern=pdata.get("ast_pattern"),
                occurrences=pdata.get("occurrences", 0),
                last_seen=pdata.get("last_seen", 0),
                file_patterns=pdata.get("file_patterns", []),
                languages=pdata.get("languages", []),
                confidence_threshold=pdata.get("confidence_threshold", 0.7),
            )
        
        for pid, pdata in data.get("revert_patterns", {}).items():
            learner.revert_patterns[pid] = RevertPattern(
                pattern_id=pdata["pattern_id"],
                reason=RevertReason(pdata["reason"]),
                description=pdata["description"],
                original_commit_patterns=pdata.get("original_commit_patterns", []),
                occurrences=pdata.get("occurrences", 0),
                avg_time_to_revert_hours=pdata.get("avg_time_to_revert_hours", 24.0),
                related_bug_patterns=pdata.get("related_bug_patterns", []),
            )
        
        learner._feature_stats = data.get("feature_stats", {})
        
        return learner


# =============================================================================
# Organization-specific Model
# =============================================================================

class OrganizationModel:
    """Model customized for a specific organization."""
    
    def __init__(self, org_id: str):
        self.org_id = org_id
        self.learner = PatternLearner()
        self.predictor = RegressionPredictor(self.learner)
        self.storage = ModelStorage()
        
        # Training statistics
        self.training_stats = {
            "total_bugs_learned": 0,
            "total_reverts_learned": 0,
            "last_training": None,
            "accuracy_history": [],
        }
    
    def train_on_bug(self, bug: HistoricalBug) -> Optional[BugPattern]:
        """Train model on a historical bug."""
        pattern = self.learner.learn_from_bug(bug)
        if pattern:
            self.training_stats["total_bugs_learned"] += 1
            self.training_stats["last_training"] = time.time()
        return pattern
    
    def train_on_revert(
        self,
        original_commit: Dict[str, Any],
        revert_commit: Dict[str, Any],
        reason: RevertReason,
    ) -> RevertPattern:
        """Train model on a code revert."""
        pattern = self.learner.learn_from_revert(
            original_commit, revert_commit, reason
        )
        self.training_stats["total_reverts_learned"] += 1
        self.training_stats["last_training"] = time.time()
        return pattern
    
    def predict(
        self,
        code: str,
        file_path: str,
        language: str = "python",
    ) -> PredictionResult:
        """Predict bugs in code using learned patterns."""
        return self.predictor.predict(code, file_path, language)
    
    def export_model(self) -> Dict[str, Any]:
        """Export model state."""
        return {
            "org_id": self.org_id,
            "model": self.storage.save_learner(self.learner),
            "training_stats": self.training_stats,
        }
    
    def import_model(self, data: Dict[str, Any]) -> None:
        """Import model state."""
        self.learner = self.storage.load_learner(data.get("model", {}))
        self.predictor = RegressionPredictor(self.learner)
        self.training_stats = data.get("training_stats", self.training_stats)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get model statistics."""
        return {
            "org_id": self.org_id,
            "training_stats": self.training_stats,
            "patterns_summary": self.learner.get_patterns_summary(),
        }
