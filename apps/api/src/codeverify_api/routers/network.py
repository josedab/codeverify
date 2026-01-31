"""
Distributed Verification Network API Router

Provides REST API endpoints for distributed verification:
- Node management
- Task distribution
- Network monitoring
"""

from __future__ import annotations

import hashlib
import time
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field


router = APIRouter(prefix="/api/v1/network", tags=["distributed-network"])


# =============================================================================
# Request/Response Models
# =============================================================================

class RegisterNodeRequest(BaseModel):
    """Request to register a node."""
    address: str = Field(..., description="Node address")
    port: int = Field(..., description="Node port")
    max_concurrent_tasks: int = Field(4, description="Max concurrent tasks")
    supported_languages: List[str] = Field(
        default_factory=lambda: ["python", "typescript"],
        description="Supported languages"
    )
    has_z3: bool = Field(True, description="Has Z3 solver")
    has_gpu: bool = Field(False, description="Has GPU acceleration")


class SubmitTaskRequest(BaseModel):
    """Request to submit a verification task."""
    code: str = Field(..., description="Code to verify")
    language: str = Field("python", description="Programming language")
    verification_type: str = Field("standard", description="Verification type")
    constraints: List[str] = Field(default_factory=list, description="Constraints")
    timeout_ms: int = Field(30000, description="Timeout in ms")
    priority: str = Field("normal", description="Priority level")
    privacy_level: str = Field("obfuscated", description="Privacy level")


class TaskResultRequest(BaseModel):
    """Request to submit task result."""
    task_id: str = Field(..., description="Task ID")
    verified: bool = Field(..., description="Verification passed")
    issues: List[Dict[str, Any]] = Field(default_factory=list, description="Issues found")
    proofs: List[str] = Field(default_factory=list, description="Proofs generated")
    execution_time_ms: float = Field(0.0, description="Execution time")


class NodeResponse(BaseModel):
    """Node information response."""
    node_id: str
    address: str
    port: int
    status: str
    current_load: float
    active_tasks: int
    reliability_score: float


class TaskResponse(BaseModel):
    """Task information response."""
    task_id: str
    status: str
    assigned_node: Optional[str]
    created_at: float
    verification_type: str


class NetworkStatsResponse(BaseModel):
    """Network statistics response."""
    total_nodes: int
    online_nodes: int
    total_capacity: int
    current_load: int
    pending_tasks: int
    tasks_distributed: int
    tasks_completed: int


# =============================================================================
# In-Memory State
# =============================================================================

# Nodes
_nodes: Dict[str, Dict[str, Any]] = {}

# Tasks
_tasks: Dict[str, Dict[str, Any]] = {}
_pending_tasks: List[str] = []

# Coordinator state
_coordinator_id = "coordinator-001"

# Statistics
_stats = {
    "tasks_distributed": 0,
    "tasks_completed": 0,
    "tasks_failed": 0,
    "total_verification_time_ms": 0.0,
}

# Node statuses
NODE_STATUSES = ["online", "offline", "busy", "maintenance"]

# Task statuses
TASK_STATUSES = ["pending", "assigned", "running", "completed", "failed", "timeout"]

# Task priorities
TASK_PRIORITIES = ["low", "normal", "high", "urgent"]

# Privacy levels
PRIVACY_LEVELS = ["none", "obfuscated", "encrypted", "split"]


# =============================================================================
# API Endpoints
# =============================================================================

@router.post(
    "/nodes",
    response_model=NodeResponse,
    summary="Register Node",
    description="Register a new node in the network"
)
async def register_node(request: RegisterNodeRequest) -> NodeResponse:
    """Register a new node."""
    node_id = hashlib.sha256(
        f"{request.address}:{request.port}:{time.time()}".encode()
    ).hexdigest()[:16]
    
    node = {
        "node_id": node_id,
        "address": request.address,
        "port": request.port,
        "max_concurrent_tasks": request.max_concurrent_tasks,
        "supported_languages": request.supported_languages,
        "has_z3": request.has_z3,
        "has_gpu": request.has_gpu,
        "status": "online",
        "current_load": 0.0,
        "active_tasks": 0,
        "tasks_completed": 0,
        "avg_task_time_ms": 0.0,
        "reliability_score": 1.0,
        "joined_at": time.time(),
        "last_heartbeat": time.time(),
    }
    
    _nodes[node_id] = node
    
    return NodeResponse(
        node_id=node_id,
        address=request.address,
        port=request.port,
        status="online",
        current_load=0.0,
        active_tasks=0,
        reliability_score=1.0,
    )


@router.get(
    "/nodes",
    summary="List Nodes",
    description="List all nodes in the network"
)
async def list_nodes(
    status: Optional[str] = None,
    has_capacity: bool = False,
) -> Dict[str, Any]:
    """List all nodes."""
    nodes = list(_nodes.values())
    
    if status:
        nodes = [n for n in nodes if n["status"] == status]
    
    if has_capacity:
        nodes = [
            n for n in nodes
            if n["active_tasks"] < n["max_concurrent_tasks"]
        ]
    
    return {
        "nodes": [
            {
                "node_id": n["node_id"],
                "address": n["address"],
                "port": n["port"],
                "status": n["status"],
                "current_load": n["current_load"],
                "active_tasks": n["active_tasks"],
                "max_concurrent_tasks": n["max_concurrent_tasks"],
                "reliability_score": n["reliability_score"],
                "supported_languages": n["supported_languages"],
            }
            for n in nodes
        ],
        "total": len(nodes),
    }


@router.get(
    "/nodes/{node_id}",
    response_model=NodeResponse,
    summary="Get Node",
    description="Get node details"
)
async def get_node(node_id: str) -> NodeResponse:
    """Get node details."""
    if node_id not in _nodes:
        raise HTTPException(status_code=404, detail="Node not found")
    
    node = _nodes[node_id]
    
    return NodeResponse(
        node_id=node["node_id"],
        address=node["address"],
        port=node["port"],
        status=node["status"],
        current_load=node["current_load"],
        active_tasks=node["active_tasks"],
        reliability_score=node["reliability_score"],
    )


@router.post(
    "/nodes/{node_id}/heartbeat",
    summary="Node Heartbeat",
    description="Send heartbeat from node"
)
async def node_heartbeat(node_id: str) -> Dict[str, Any]:
    """Update node heartbeat."""
    if node_id not in _nodes:
        raise HTTPException(status_code=404, detail="Node not found")
    
    _nodes[node_id]["last_heartbeat"] = time.time()
    
    # Return any pending tasks for this node
    pending_for_node = [
        t for t in _tasks.values()
        if t["assigned_node"] == node_id and t["status"] == "assigned"
    ]
    
    return {
        "acknowledged": True,
        "pending_tasks": len(pending_for_node),
    }


@router.delete(
    "/nodes/{node_id}",
    summary="Remove Node",
    description="Remove a node from the network"
)
async def remove_node(node_id: str) -> Dict[str, Any]:
    """Remove a node."""
    if node_id not in _nodes:
        raise HTTPException(status_code=404, detail="Node not found")
    
    # Reassign tasks
    for task in _tasks.values():
        if task["assigned_node"] == node_id and task["status"] in ("assigned", "running"):
            task["status"] = "pending"
            task["assigned_node"] = None
            _pending_tasks.append(task["task_id"])
    
    del _nodes[node_id]
    
    return {"removed": True, "node_id": node_id}


@router.post(
    "/tasks",
    response_model=TaskResponse,
    summary="Submit Task",
    description="Submit a verification task for distributed execution"
)
async def submit_task(request: SubmitTaskRequest) -> TaskResponse:
    """Submit a verification task."""
    task_id = hashlib.sha256(
        f"{time.time()}-{request.code[:50]}".encode()
    ).hexdigest()[:16]
    
    # Apply privacy (obfuscation)
    code = request.code
    if request.privacy_level == "obfuscated":
        code = _obfuscate_code(request.code)
    
    task = {
        "task_id": task_id,
        "code": code,
        "original_code": request.code,
        "language": request.language,
        "verification_type": request.verification_type,
        "constraints": request.constraints,
        "timeout_ms": request.timeout_ms,
        "priority": request.priority,
        "privacy_level": request.privacy_level,
        "assigned_node": None,
        "status": "pending",
        "result": None,
        "error": None,
        "created_at": time.time(),
        "assigned_at": None,
        "started_at": None,
        "completed_at": None,
    }
    
    _tasks[task_id] = task
    _pending_tasks.append(task_id)
    _stats["tasks_distributed"] += 1
    
    # Try to assign
    _try_assign_task(task_id)
    
    return TaskResponse(
        task_id=task_id,
        status=task["status"],
        assigned_node=task["assigned_node"],
        created_at=task["created_at"],
        verification_type=request.verification_type,
    )


@router.get(
    "/tasks/{task_id}",
    summary="Get Task",
    description="Get task status and result"
)
async def get_task(task_id: str) -> Dict[str, Any]:
    """Get task details."""
    if task_id not in _tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    
    task = _tasks[task_id]
    
    return {
        "task_id": task["task_id"],
        "status": task["status"],
        "assigned_node": task["assigned_node"],
        "verification_type": task["verification_type"],
        "created_at": task["created_at"],
        "started_at": task.get("started_at"),
        "completed_at": task.get("completed_at"),
        "result": task.get("result"),
        "error": task.get("error"),
    }


@router.post(
    "/tasks/{task_id}/start",
    summary="Start Task",
    description="Mark task as started (called by worker node)"
)
async def start_task(task_id: str, node_id: str) -> Dict[str, Any]:
    """Mark task as started."""
    if task_id not in _tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    
    task = _tasks[task_id]
    
    if task["assigned_node"] != node_id:
        raise HTTPException(status_code=400, detail="Task not assigned to this node")
    
    task["status"] = "running"
    task["started_at"] = time.time()
    
    return {"started": True, "task_id": task_id}


@router.post(
    "/tasks/{task_id}/complete",
    summary="Complete Task",
    description="Submit task completion with results"
)
async def complete_task(
    task_id: str,
    request: TaskResultRequest,
) -> Dict[str, Any]:
    """Complete a task with results."""
    if task_id not in _tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    
    task = _tasks[task_id]
    
    task["status"] = "completed"
    task["completed_at"] = time.time()
    task["result"] = {
        "verified": request.verified,
        "issues": request.issues,
        "proofs": request.proofs,
        "execution_time_ms": request.execution_time_ms,
    }
    
    # Update node stats
    if task["assigned_node"] and task["assigned_node"] in _nodes:
        node = _nodes[task["assigned_node"]]
        node["active_tasks"] = max(0, node["active_tasks"] - 1)
        node["current_load"] = node["active_tasks"] / node["max_concurrent_tasks"]
        node["tasks_completed"] += 1
    
    _stats["tasks_completed"] += 1
    _stats["total_verification_time_ms"] += request.execution_time_ms
    
    return {"completed": True, "task_id": task_id}


@router.post(
    "/tasks/{task_id}/fail",
    summary="Fail Task",
    description="Mark task as failed"
)
async def fail_task(task_id: str, error: str) -> Dict[str, Any]:
    """Mark task as failed."""
    if task_id not in _tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    
    task = _tasks[task_id]
    
    task["status"] = "failed"
    task["error"] = error
    task["completed_at"] = time.time()
    
    # Update node reliability
    if task["assigned_node"] and task["assigned_node"] in _nodes:
        node = _nodes[task["assigned_node"]]
        node["active_tasks"] = max(0, node["active_tasks"] - 1)
        node["current_load"] = node["active_tasks"] / node["max_concurrent_tasks"]
        node["reliability_score"] = max(0.1, node["reliability_score"] - 0.1)
    
    _stats["tasks_failed"] += 1
    
    return {"failed": True, "task_id": task_id}


@router.get(
    "/tasks",
    summary="List Tasks",
    description="List tasks with optional filters"
)
async def list_tasks(
    status: Optional[str] = None,
    node_id: Optional[str] = None,
    limit: int = 50,
) -> Dict[str, Any]:
    """List tasks."""
    tasks = list(_tasks.values())
    
    if status:
        tasks = [t for t in tasks if t["status"] == status]
    
    if node_id:
        tasks = [t for t in tasks if t["assigned_node"] == node_id]
    
    # Sort by creation time descending
    tasks.sort(key=lambda t: t["created_at"], reverse=True)
    
    return {
        "tasks": [
            {
                "task_id": t["task_id"],
                "status": t["status"],
                "assigned_node": t["assigned_node"],
                "verification_type": t["verification_type"],
                "created_at": t["created_at"],
            }
            for t in tasks[:limit]
        ],
        "total": len(tasks),
    }


@router.get(
    "/stats",
    response_model=NetworkStatsResponse,
    summary="Get Network Stats",
    description="Get network statistics"
)
async def get_stats() -> NetworkStatsResponse:
    """Get network statistics."""
    online_nodes = [n for n in _nodes.values() if n["status"] == "online"]
    
    return NetworkStatsResponse(
        total_nodes=len(_nodes),
        online_nodes=len(online_nodes),
        total_capacity=sum(n["max_concurrent_tasks"] for n in online_nodes),
        current_load=sum(n["active_tasks"] for n in online_nodes),
        pending_tasks=len(_pending_tasks),
        tasks_distributed=_stats["tasks_distributed"],
        tasks_completed=_stats["tasks_completed"],
    )


@router.post(
    "/process-pending",
    summary="Process Pending Tasks",
    description="Attempt to assign pending tasks to available nodes"
)
async def process_pending() -> Dict[str, Any]:
    """Process pending tasks."""
    assigned = 0
    
    for task_id in list(_pending_tasks):
        if _try_assign_task(task_id):
            assigned += 1
    
    return {
        "processed": True,
        "assigned": assigned,
        "remaining_pending": len(_pending_tasks),
    }


@router.get(
    "/health",
    summary="Network Health",
    description="Check network health"
)
async def health_check() -> Dict[str, Any]:
    """Check network health."""
    now = time.time()
    stale_threshold = 90  # 90 seconds
    
    stale_nodes = [
        n["node_id"] for n in _nodes.values()
        if n["status"] == "online" and now - n["last_heartbeat"] > stale_threshold
    ]
    
    # Mark stale nodes offline
    for node_id in stale_nodes:
        _nodes[node_id]["status"] = "offline"
    
    online = sum(1 for n in _nodes.values() if n["status"] == "online")
    
    return {
        "healthy": online > 0,
        "online_nodes": online,
        "stale_nodes_marked": len(stale_nodes),
        "pending_tasks": len(_pending_tasks),
    }


# =============================================================================
# Helper Functions
# =============================================================================

def _try_assign_task(task_id: str) -> bool:
    """Try to assign a task to a node."""
    task = _tasks.get(task_id)
    if not task or task["status"] != "pending":
        return False
    
    # Find available node
    available = [
        n for n in _nodes.values()
        if n["status"] == "online" and
        n["active_tasks"] < n["max_concurrent_tasks"] and
        task["language"] in n["supported_languages"]
    ]
    
    if not available:
        return False
    
    # Select node with lowest load
    selected = min(available, key=lambda n: n["current_load"])
    
    # Assign
    task["assigned_node"] = selected["node_id"]
    task["assigned_at"] = time.time()
    task["status"] = "assigned"
    
    selected["active_tasks"] += 1
    selected["current_load"] = selected["active_tasks"] / selected["max_concurrent_tasks"]
    
    if task_id in _pending_tasks:
        _pending_tasks.remove(task_id)
    
    return True


def _obfuscate_code(code: str) -> str:
    """Obfuscate variable names in code."""
    import re
    
    obfuscation_map: Dict[str, str] = {}
    counter = [0]
    
    keywords = {
        "def", "class", "if", "else", "elif", "for", "while", "return",
        "import", "from", "True", "False", "None", "and", "or", "not",
        "in", "is", "try", "except", "finally", "raise", "with", "as",
        "async", "await", "pass", "break", "continue", "lambda",
        "int", "str", "float", "bool", "list", "dict", "set", "tuple",
        "print", "len", "range", "enumerate", "zip", "map", "filter",
    }
    
    def replace(match):
        var = match.group(1)
        if var in keywords:
            return var
        
        if var not in obfuscation_map:
            obfuscation_map[var] = f"v{counter[0]:04d}"
            counter[0] += 1
        
        return obfuscation_map[var]
    
    return re.sub(r'\b([a-zA-Z_][a-zA-Z0-9_]*)\b', replace, code)
