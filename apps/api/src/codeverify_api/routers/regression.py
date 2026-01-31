"""
Historical Regression Learning API Router

Provides REST API endpoints for regression learning:
- Train model on bugs and reverts
- Predict bugs in new code
- Model management per organization
"""

from __future__ import annotations

import hashlib
import time
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field


router = APIRouter(prefix="/api/v1/regression", tags=["regression-learning"])


# =============================================================================
# Request/Response Models
# =============================================================================

class BugTrainingRequest(BaseModel):
    """Request to train on a bug."""
    repository: str = Field(..., description="Repository name")
    commit_sha: str = Field(..., description="Commit SHA that introduced bug")
    commit_message: str = Field("", description="Commit message")
    commit_author: str = Field("", description="Commit author")
    bug_type: str = Field(..., description="Type of bug")
    file_path: str = Field(..., description="File path")
    line_number: Optional[int] = Field(None, description="Line number")
    buggy_code: str = Field(..., description="Code that contains bug")
    fixed_code: Optional[str] = Field(None, description="Fixed code")


class RevertTrainingRequest(BaseModel):
    """Request to train on a revert."""
    original_commit_sha: str = Field(..., description="Original commit SHA")
    original_commit_message: str = Field("", description="Original commit message")
    original_files: List[str] = Field(default_factory=list, description="Files changed")
    revert_commit_sha: str = Field(..., description="Revert commit SHA")
    reason: str = Field(..., description="Reason for revert")


class PredictRequest(BaseModel):
    """Request for bug prediction."""
    code: str = Field(..., description="Code to analyze")
    file_path: str = Field(..., description="File path")
    language: str = Field("python", description="Programming language")


class PredictionResponse(BaseModel):
    """Response with prediction results."""
    has_potential_bug: bool
    confidence: float
    matched_patterns: List[str]
    predictions: List[Dict[str, Any]]
    recommendations: List[str]


class ModelExportResponse(BaseModel):
    """Response with exported model."""
    org_id: str
    model_data: Dict[str, Any]
    exported_at: float


# =============================================================================
# In-Memory State (would use database in production)
# =============================================================================

# Organization models
_org_models: Dict[str, Dict[str, Any]] = {}

# Bug types enum values
BUG_TYPES = [
    "null_pointer", "array_bounds", "type_error", "logic_error",
    "concurrency", "resource_leak", "security", "performance",
    "api_misuse", "other"
]

# Revert reasons
REVERT_REASONS = [
    "bug", "performance", "breaking_change", "incorrect_logic",
    "test_failure", "security", "other"
]


def _get_or_create_model(org_id: str) -> Dict[str, Any]:
    """Get or create organization model."""
    if org_id not in _org_models:
        _org_models[org_id] = {
            "org_id": org_id,
            "bug_patterns": {},
            "revert_patterns": {},
            "training_stats": {
                "total_bugs_learned": 0,
                "total_reverts_learned": 0,
                "last_training": None,
            },
            "created_at": time.time(),
        }
    return _org_models[org_id]


# =============================================================================
# API Endpoints
# =============================================================================

@router.post(
    "/train/bug",
    summary="Train on Bug",
    description="Train the model on a historical bug"
)
async def train_on_bug(
    org_id: str,
    request: BugTrainingRequest,
) -> Dict[str, Any]:
    """Train the model on a historical bug."""
    if request.bug_type not in BUG_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid bug type. Must be one of: {BUG_TYPES}"
        )
    
    model = _get_or_create_model(org_id)
    
    # Create pattern from bug
    pattern_id = _create_bug_pattern(model, request)
    
    model["training_stats"]["total_bugs_learned"] += 1
    model["training_stats"]["last_training"] = time.time()
    
    return {
        "trained": True,
        "pattern_id": pattern_id,
        "org_id": org_id,
        "total_bugs_learned": model["training_stats"]["total_bugs_learned"],
    }


@router.post(
    "/train/revert",
    summary="Train on Revert",
    description="Train the model on a code revert"
)
async def train_on_revert(
    org_id: str,
    request: RevertTrainingRequest,
) -> Dict[str, Any]:
    """Train the model on a code revert."""
    if request.reason not in REVERT_REASONS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid reason. Must be one of: {REVERT_REASONS}"
        )
    
    model = _get_or_create_model(org_id)
    
    # Create pattern from revert
    pattern_id = _create_revert_pattern(model, request)
    
    model["training_stats"]["total_reverts_learned"] += 1
    model["training_stats"]["last_training"] = time.time()
    
    return {
        "trained": True,
        "pattern_id": pattern_id,
        "org_id": org_id,
        "total_reverts_learned": model["training_stats"]["total_reverts_learned"],
    }


@router.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Predict Bugs",
    description="Predict potential bugs in code based on learned patterns"
)
async def predict_bugs(
    org_id: str,
    request: PredictRequest,
) -> PredictionResponse:
    """Predict potential bugs in code."""
    model = _get_or_create_model(org_id)
    
    result = _predict_bugs(model, request)
    
    return PredictionResponse(
        has_potential_bug=result["has_potential_bug"],
        confidence=result["confidence"],
        matched_patterns=result["matched_patterns"],
        predictions=result["predictions"],
        recommendations=result["recommendations"],
    )


@router.get(
    "/model/{org_id}",
    summary="Get Model",
    description="Get organization model details"
)
async def get_model(org_id: str) -> Dict[str, Any]:
    """Get organization model details."""
    if org_id not in _org_models:
        raise HTTPException(status_code=404, detail="Model not found")
    
    model = _org_models[org_id]
    
    return {
        "org_id": org_id,
        "training_stats": model["training_stats"],
        "bug_patterns_count": len(model["bug_patterns"]),
        "revert_patterns_count": len(model["revert_patterns"]),
        "created_at": model["created_at"],
    }


@router.get(
    "/model/{org_id}/patterns",
    summary="List Patterns",
    description="List learned patterns for organization"
)
async def list_patterns(
    org_id: str,
    pattern_type: Optional[str] = None,
) -> Dict[str, Any]:
    """List learned patterns."""
    if org_id not in _org_models:
        raise HTTPException(status_code=404, detail="Model not found")
    
    model = _org_models[org_id]
    
    result = {
        "org_id": org_id,
        "bug_patterns": [],
        "revert_patterns": [],
    }
    
    if not pattern_type or pattern_type == "bug":
        result["bug_patterns"] = list(model["bug_patterns"].values())
    
    if not pattern_type or pattern_type == "revert":
        result["revert_patterns"] = list(model["revert_patterns"].values())
    
    return result


@router.post(
    "/model/{org_id}/export",
    response_model=ModelExportResponse,
    summary="Export Model",
    description="Export organization model for backup or transfer"
)
async def export_model(org_id: str) -> ModelExportResponse:
    """Export organization model."""
    if org_id not in _org_models:
        raise HTTPException(status_code=404, detail="Model not found")
    
    model = _org_models[org_id]
    
    return ModelExportResponse(
        org_id=org_id,
        model_data=model,
        exported_at=time.time(),
    )


@router.post(
    "/model/{org_id}/import",
    summary="Import Model",
    description="Import a previously exported model"
)
async def import_model(
    org_id: str,
    model_data: Dict[str, Any],
) -> Dict[str, Any]:
    """Import a model."""
    _org_models[org_id] = {
        "org_id": org_id,
        "bug_patterns": model_data.get("bug_patterns", {}),
        "revert_patterns": model_data.get("revert_patterns", {}),
        "training_stats": model_data.get("training_stats", {
            "total_bugs_learned": 0,
            "total_reverts_learned": 0,
            "last_training": None,
        }),
        "created_at": model_data.get("created_at", time.time()),
    }
    
    return {
        "imported": True,
        "org_id": org_id,
    }


@router.delete(
    "/model/{org_id}",
    summary="Delete Model",
    description="Delete organization model"
)
async def delete_model(org_id: str) -> Dict[str, Any]:
    """Delete organization model."""
    if org_id not in _org_models:
        raise HTTPException(status_code=404, detail="Model not found")
    
    del _org_models[org_id]
    
    return {"deleted": True, "org_id": org_id}


@router.get(
    "/model/{org_id}/stats",
    summary="Get Statistics",
    description="Get model training statistics"
)
async def get_stats(org_id: str) -> Dict[str, Any]:
    """Get model statistics."""
    if org_id not in _org_models:
        raise HTTPException(status_code=404, detail="Model not found")
    
    model = _org_models[org_id]
    
    # Calculate pattern statistics
    bug_by_type: Dict[str, int] = {}
    for pattern in model["bug_patterns"].values():
        ptype = pattern.get("pattern_type", "other")
        bug_by_type[ptype] = bug_by_type.get(ptype, 0) + pattern.get("occurrences", 1)
    
    revert_by_reason: Dict[str, int] = {}
    for pattern in model["revert_patterns"].values():
        reason = pattern.get("reason", "other")
        revert_by_reason[reason] = revert_by_reason.get(reason, 0) + pattern.get("occurrences", 1)
    
    return {
        "org_id": org_id,
        "training_stats": model["training_stats"],
        "bug_patterns": {
            "total": len(model["bug_patterns"]),
            "by_type": bug_by_type,
        },
        "revert_patterns": {
            "total": len(model["revert_patterns"]),
            "by_reason": revert_by_reason,
        },
    }


@router.get(
    "/bug-types",
    summary="List Bug Types",
    description="Get list of valid bug types"
)
async def list_bug_types() -> Dict[str, Any]:
    """List valid bug types."""
    return {"bug_types": BUG_TYPES}


@router.get(
    "/revert-reasons",
    summary="List Revert Reasons",
    description="Get list of valid revert reasons"
)
async def list_revert_reasons() -> Dict[str, Any]:
    """List valid revert reasons."""
    return {"revert_reasons": REVERT_REASONS}


# =============================================================================
# Helper Functions
# =============================================================================

def _create_bug_pattern(
    model: Dict[str, Any],
    request: BugTrainingRequest,
) -> str:
    """Create or update bug pattern from training request."""
    import re
    
    # Extract features
    features = _extract_features(request.buggy_code)
    
    # Generate pattern key
    key_parts = [
        request.bug_type,
        _detect_language(request.file_path),
        str(features.get("has_none_return", False)),
        str(features.get("has_bare_except", False)),
    ]
    pattern_id = hashlib.sha256(":".join(key_parts).encode()).hexdigest()[:12]
    
    if pattern_id in model["bug_patterns"]:
        # Update existing
        pattern = model["bug_patterns"][pattern_id]
        pattern["occurrences"] += 1
        pattern["last_seen"] = time.time()
        
        if request.file_path not in pattern.get("file_patterns", []):
            pattern.setdefault("file_patterns", []).append(request.file_path)
    else:
        # Create new
        model["bug_patterns"][pattern_id] = {
            "pattern_id": pattern_id,
            "pattern_type": request.bug_type,
            "description": _generate_description(request, features),
            "code_pattern": _extract_code_pattern(request.buggy_code),
            "occurrences": 1,
            "last_seen": time.time(),
            "file_patterns": [request.file_path],
            "languages": [_detect_language(request.file_path)],
            "confidence_threshold": 0.7,
        }
    
    return pattern_id


def _create_revert_pattern(
    model: Dict[str, Any],
    request: RevertTrainingRequest,
) -> str:
    """Create or update revert pattern from training request."""
    # Extract commit patterns
    commit_patterns = []
    msg = request.original_commit_message.lower()
    
    if "fix" in msg:
        commit_patterns.append("fix_commit")
    if "feat" in msg or "feature" in msg:
        commit_patterns.append("feature_commit")
    if "refactor" in msg:
        commit_patterns.append("refactor_commit")
    
    if any("test" in f.lower() for f in request.original_files):
        commit_patterns.append("modifies_tests")
    
    # Generate pattern key
    pattern_id = hashlib.sha256(
        f"{request.reason}:{','.join(commit_patterns)}".encode()
    ).hexdigest()[:12]
    
    if pattern_id in model["revert_patterns"]:
        pattern = model["revert_patterns"][pattern_id]
        pattern["occurrences"] += 1
    else:
        model["revert_patterns"][pattern_id] = {
            "pattern_id": pattern_id,
            "reason": request.reason,
            "description": f"Revert due to {request.reason}: {request.original_commit_message[:50]}",
            "original_commit_patterns": commit_patterns,
            "occurrences": 1,
            "avg_time_to_revert_hours": 24.0,
        }
    
    return pattern_id


def _predict_bugs(
    model: Dict[str, Any],
    request: PredictRequest,
) -> Dict[str, Any]:
    """Predict bugs using learned patterns."""
    import re
    
    result = {
        "has_potential_bug": False,
        "confidence": 0.0,
        "matched_patterns": [],
        "predictions": [],
        "recommendations": [],
    }
    
    features = _extract_features(request.code)
    detected_lang = _detect_language(request.file_path)
    
    for pattern_id, pattern in model["bug_patterns"].items():
        match_score = _calculate_match_score(
            request.code,
            features,
            pattern,
            request.file_path,
            detected_lang,
        )
        
        threshold = pattern.get("confidence_threshold", 0.7)
        if match_score >= threshold:
            result["matched_patterns"].append(pattern_id)
            result["predictions"].append({
                "pattern_id": pattern_id,
                "pattern_type": pattern.get("pattern_type", "other"),
                "description": pattern.get("description", ""),
                "confidence": match_score,
                "occurrences_in_history": pattern.get("occurrences", 0),
            })
    
    if result["predictions"]:
        result["has_potential_bug"] = True
        result["confidence"] = max(p["confidence"] for p in result["predictions"])
        result["recommendations"] = _generate_recommendations(result["predictions"])
    
    return result


def _extract_features(code: str) -> Dict[str, Any]:
    """Extract features from code."""
    import re
    
    return {
        "line_count": len(code.split('\n')),
        "function_count": len(re.findall(r'^\s*def\s+', code, re.M)),
        "class_count": len(re.findall(r'^\s*class\s+', code, re.M)),
        "has_none_return": "return None" in code,
        "has_bare_except": bool(re.search(r'except\s*:', code)),
        "has_eval": "eval(" in code,
        "has_file_open": "open(" in code,
    }


def _detect_language(file_path: str) -> str:
    """Detect language from file path."""
    ext_map = {
        ".py": "python",
        ".js": "javascript",
        ".ts": "typescript",
        ".java": "java",
        ".go": "go",
    }
    
    for ext, lang in ext_map.items():
        if file_path.endswith(ext):
            return lang
    
    return "unknown"


def _generate_description(
    request: BugTrainingRequest,
    features: Dict[str, Any],
) -> str:
    """Generate pattern description."""
    parts = [f"{request.bug_type.replace('_', ' ').title()} bug pattern"]
    
    if features.get("has_none_return"):
        parts.append("with potential null return")
    
    if features.get("has_bare_except"):
        parts.append("with bare except clause")
    
    return " ".join(parts)


def _extract_code_pattern(code: str) -> Optional[str]:
    """Extract regex pattern from code."""
    import re
    
    patterns = [
        (r"return\s+None\b", "null_return"),
        (r"except\s*:", "bare_except"),
        (r"\[\w+\]", "array_access"),
        (r"/\s*\w+", "division"),
    ]
    
    for regex, _ in patterns:
        if re.search(regex, code):
            return regex
    
    return None


def _calculate_match_score(
    code: str,
    features: Dict[str, Any],
    pattern: Dict[str, Any],
    file_path: str,
    detected_lang: str,
) -> float:
    """Calculate match score for a pattern."""
    import re
    
    scores: List[float] = []
    
    # Code pattern match
    code_pattern = pattern.get("code_pattern")
    if code_pattern:
        try:
            if re.search(code_pattern, code):
                scores.append(0.8)
        except re.error:
            pass
    
    # Language match
    if detected_lang in pattern.get("languages", []):
        scores.append(0.3)
    
    # Recency bonus
    last_seen = pattern.get("last_seen", 0)
    if last_seen:
        days_since = (time.time() - last_seen) / 86400
        if days_since < 30:
            scores.append(0.2)
    
    # Occurrence frequency bonus
    occurrences = pattern.get("occurrences", 0)
    if occurrences > 10:
        scores.append(0.3)
    elif occurrences > 5:
        scores.append(0.2)
    elif occurrences > 1:
        scores.append(0.1)
    
    return min(1.0, sum(scores)) if scores else 0.0


def _generate_recommendations(
    predictions: List[Dict[str, Any]],
) -> List[str]:
    """Generate recommendations from predictions."""
    recommendations = []
    
    types = {p["pattern_type"] for p in predictions}
    
    if "null_pointer" in types:
        recommendations.append("Add null/None checks before accessing values")
    
    if "array_bounds" in types:
        recommendations.append("Add bounds checking for array/list access")
    
    if "type_error" in types:
        recommendations.append("Verify type compatibility and add type guards")
    
    if "logic_error" in types:
        recommendations.append("Review logic flow and add additional test cases")
    
    if "security" in types:
        recommendations.append("Review for security vulnerabilities")
    
    if len(predictions) > 2:
        recommendations.append("Consider breaking this code into smaller functions")
    
    return recommendations
