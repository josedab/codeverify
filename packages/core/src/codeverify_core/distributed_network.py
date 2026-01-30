"""
Distributed Verification Network

Peer-to-peer network for sharing verification compute:
- P2P protocol for verification distribution
- Work distribution and load balancing
- Privacy-preserving verification
- Resource pooling for large codebases

Reduces cost and enables on-prem verification sharing.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import random
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple


# =============================================================================
# Data Models
# =============================================================================

class NodeStatus(str, Enum):
    """Status of a network node."""
    ONLINE = "online"
    OFFLINE = "offline"
    BUSY = "busy"
    MAINTENANCE = "maintenance"


class TaskStatus(str, Enum):
    """Status of a verification task."""
    PENDING = "pending"
    ASSIGNED = "assigned"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"


class TaskPriority(str, Enum):
    """Priority levels for tasks."""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"


@dataclass
class NetworkNode:
    """A node in the verification network."""
    
    node_id: str
    address: str
    port: int
    
    # Capabilities
    max_concurrent_tasks: int = 4
    supported_languages: List[str] = field(default_factory=lambda: ["python", "typescript"])
    has_z3: bool = True
    has_gpu: bool = False
    
    # Status
    status: NodeStatus = NodeStatus.OFFLINE
    current_load: float = 0.0
    active_tasks: int = 0
    
    # Statistics
    tasks_completed: int = 0
    avg_task_time_ms: float = 0.0
    reliability_score: float = 1.0
    
    # Metadata
    joined_at: float = 0.0
    last_heartbeat: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "node_id": self.node_id,
            "address": self.address,
            "port": self.port,
            "max_concurrent_tasks": self.max_concurrent_tasks,
            "supported_languages": self.supported_languages,
            "has_z3": self.has_z3,
            "has_gpu": self.has_gpu,
            "status": self.status.value,
            "current_load": self.current_load,
            "active_tasks": self.active_tasks,
            "tasks_completed": self.tasks_completed,
            "avg_task_time_ms": self.avg_task_time_ms,
            "reliability_score": self.reliability_score,
            "joined_at": self.joined_at,
            "last_heartbeat": self.last_heartbeat,
        }
    
    @property
    def available_capacity(self) -> int:
        """Get available task capacity."""
        return max(0, self.max_concurrent_tasks - self.active_tasks)
    
    @property
    def is_available(self) -> bool:
        """Check if node is available for tasks."""
        return (
            self.status == NodeStatus.ONLINE and
            self.available_capacity > 0
        )


@dataclass
class VerificationTask:
    """A verification task to be distributed."""
    
    task_id: str
    
    # Task content
    code: str
    language: str
    verification_type: str = "full"  # "quick", "standard", "full", "deep"
    
    # Configuration
    constraints: List[str] = field(default_factory=list)
    timeout_ms: int = 30000
    priority: TaskPriority = TaskPriority.NORMAL
    
    # Assignment
    assigned_node: Optional[str] = None
    status: TaskStatus = TaskStatus.PENDING
    
    # Results
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    
    # Timing
    created_at: float = 0.0
    assigned_at: Optional[float] = None
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    
    # Privacy
    encrypted: bool = False
    requester_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "task_id": self.task_id,
            "code_hash": hashlib.sha256(self.code.encode()).hexdigest()[:16],
            "language": self.language,
            "verification_type": self.verification_type,
            "constraints": self.constraints,
            "timeout_ms": self.timeout_ms,
            "priority": self.priority.value,
            "assigned_node": self.assigned_node,
            "status": self.status.value,
            "result": self.result,
            "error": self.error,
            "created_at": self.created_at,
            "assigned_at": self.assigned_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "encrypted": self.encrypted,
        }


@dataclass
class TaskResult:
    """Result of a verification task."""
    
    task_id: str
    verified: bool
    
    issues: List[Dict[str, Any]] = field(default_factory=list)
    proofs: List[str] = field(default_factory=list)
    
    execution_time_ms: float = 0.0
    node_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "task_id": self.task_id,
            "verified": self.verified,
            "issues": self.issues,
            "proofs": self.proofs,
            "execution_time_ms": self.execution_time_ms,
            "node_id": self.node_id,
        }


# =============================================================================
# Network Protocol
# =============================================================================

class MessageType(str, Enum):
    """Types of network messages."""
    # Node discovery
    JOIN = "join"
    LEAVE = "leave"
    HEARTBEAT = "heartbeat"
    
    # Task distribution
    TASK_SUBMIT = "task_submit"
    TASK_ASSIGN = "task_assign"
    TASK_ACCEPT = "task_accept"
    TASK_REJECT = "task_reject"
    TASK_RESULT = "task_result"
    TASK_CANCEL = "task_cancel"
    
    # Coordination
    STATUS_REQUEST = "status_request"
    STATUS_RESPONSE = "status_response"
    LOAD_REPORT = "load_report"


@dataclass
class NetworkMessage:
    """A message in the network protocol."""
    
    message_id: str
    message_type: MessageType
    sender_id: str
    
    # Content
    payload: Dict[str, Any] = field(default_factory=dict)
    
    # Routing
    recipient_id: Optional[str] = None  # None = broadcast
    
    # Metadata
    timestamp: float = 0.0
    ttl: int = 3  # Time-to-live for broadcast
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "message_id": self.message_id,
            "message_type": self.message_type.value,
            "sender_id": self.sender_id,
            "payload": self.payload,
            "recipient_id": self.recipient_id,
            "timestamp": self.timestamp,
            "ttl": self.ttl,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> NetworkMessage:
        """Create from dictionary."""
        return cls(
            message_id=data["message_id"],
            message_type=MessageType(data["message_type"]),
            sender_id=data["sender_id"],
            payload=data.get("payload", {}),
            recipient_id=data.get("recipient_id"),
            timestamp=data.get("timestamp", 0.0),
            ttl=data.get("ttl", 3),
        )


# =============================================================================
# Load Balancer
# =============================================================================

class LoadBalancingStrategy(str, Enum):
    """Load balancing strategies."""
    ROUND_ROBIN = "round_robin"
    LEAST_LOADED = "least_loaded"
    CAPABILITY_MATCH = "capability_match"
    WEIGHTED = "weighted"


class LoadBalancer:
    """Distributes tasks across network nodes."""
    
    def __init__(self, strategy: LoadBalancingStrategy = LoadBalancingStrategy.WEIGHTED):
        self.strategy = strategy
        self._round_robin_index = 0
    
    def select_node(
        self,
        nodes: List[NetworkNode],
        task: VerificationTask,
    ) -> Optional[NetworkNode]:
        """Select the best node for a task."""
        available = [n for n in nodes if n.is_available]
        
        if not available:
            return None
        
        # Filter by capability
        capable = [
            n for n in available
            if task.language in n.supported_languages
        ]
        
        if not capable:
            return None
        
        if self.strategy == LoadBalancingStrategy.ROUND_ROBIN:
            return self._round_robin(capable)
        
        elif self.strategy == LoadBalancingStrategy.LEAST_LOADED:
            return self._least_loaded(capable)
        
        elif self.strategy == LoadBalancingStrategy.CAPABILITY_MATCH:
            return self._capability_match(capable, task)
        
        elif self.strategy == LoadBalancingStrategy.WEIGHTED:
            return self._weighted(capable, task)
        
        return capable[0]
    
    def _round_robin(self, nodes: List[NetworkNode]) -> NetworkNode:
        """Simple round-robin selection."""
        self._round_robin_index = (self._round_robin_index + 1) % len(nodes)
        return nodes[self._round_robin_index]
    
    def _least_loaded(self, nodes: List[NetworkNode]) -> NetworkNode:
        """Select node with lowest load."""
        return min(nodes, key=lambda n: n.current_load)
    
    def _capability_match(
        self,
        nodes: List[NetworkNode],
        task: VerificationTask,
    ) -> NetworkNode:
        """Select node with best capability match."""
        def score(node: NetworkNode) -> float:
            s = 0.0
            if node.has_z3:
                s += 1.0
            if task.verification_type == "deep" and node.has_gpu:
                s += 0.5
            return s
        
        return max(nodes, key=score)
    
    def _weighted(
        self,
        nodes: List[NetworkNode],
        task: VerificationTask,
    ) -> NetworkNode:
        """Weighted selection considering multiple factors."""
        def score(node: NetworkNode) -> float:
            s = 0.0
            
            # Available capacity (higher is better)
            s += node.available_capacity * 0.3
            
            # Reliability (higher is better)
            s += node.reliability_score * 0.3
            
            # Low load (lower is better, so invert)
            s += (1 - node.current_load) * 0.2
            
            # Capability match
            if node.has_z3:
                s += 0.1
            if task.verification_type == "deep" and node.has_gpu:
                s += 0.1
            
            return s
        
        return max(nodes, key=score)


# =============================================================================
# Privacy Manager
# =============================================================================

class PrivacyLevel(str, Enum):
    """Privacy levels for verification."""
    NONE = "none"                   # Code sent as-is
    OBFUSCATED = "obfuscated"       # Variable names obfuscated
    ENCRYPTED = "encrypted"          # Full encryption
    SPLIT = "split"                  # Code split across nodes


class PrivacyManager:
    """Manages privacy for distributed verification."""
    
    def __init__(self, level: PrivacyLevel = PrivacyLevel.OBFUSCATED):
        self.level = level
        self._obfuscation_map: Dict[str, str] = {}
    
    def prepare_task(self, task: VerificationTask) -> VerificationTask:
        """Prepare task for distribution with privacy protection."""
        if self.level == PrivacyLevel.NONE:
            return task
        
        elif self.level == PrivacyLevel.OBFUSCATED:
            return self._obfuscate_task(task)
        
        elif self.level == PrivacyLevel.ENCRYPTED:
            return self._encrypt_task(task)
        
        elif self.level == PrivacyLevel.SPLIT:
            # Splitting is handled at a higher level
            return task
        
        return task
    
    def restore_result(self, result: TaskResult) -> TaskResult:
        """Restore result with deobfuscation."""
        if self.level == PrivacyLevel.OBFUSCATED:
            return self._deobfuscate_result(result)
        
        elif self.level == PrivacyLevel.ENCRYPTED:
            return self._decrypt_result(result)
        
        return result
    
    def _obfuscate_task(self, task: VerificationTask) -> VerificationTask:
        """Obfuscate variable names in code."""
        import re
        
        code = task.code
        
        # Find and replace variable names
        var_pattern = r'\b([a-zA-Z_][a-zA-Z0-9_]*)\b'
        
        # Keywords to preserve
        keywords = {
            "def", "class", "if", "else", "elif", "for", "while", "return",
            "import", "from", "True", "False", "None", "and", "or", "not",
            "in", "is", "try", "except", "finally", "raise", "with", "as",
            "async", "await", "pass", "break", "continue", "lambda",
            "int", "str", "float", "bool", "list", "dict", "set", "tuple",
            "print", "len", "range", "enumerate", "zip", "map", "filter",
        }
        
        def replace_var(match):
            var = match.group(1)
            if var in keywords:
                return var
            
            if var not in self._obfuscation_map:
                self._obfuscation_map[var] = f"v{len(self._obfuscation_map):04d}"
            
            return self._obfuscation_map[var]
        
        obfuscated_code = re.sub(var_pattern, replace_var, code)
        
        # Create new task with obfuscated code
        obfuscated_task = VerificationTask(
            task_id=task.task_id,
            code=obfuscated_code,
            language=task.language,
            verification_type=task.verification_type,
            constraints=task.constraints,
            timeout_ms=task.timeout_ms,
            priority=task.priority,
            created_at=task.created_at,
            encrypted=True,
            requester_id=task.requester_id,
        )
        
        return obfuscated_task
    
    def _encrypt_task(self, task: VerificationTask) -> VerificationTask:
        """Encrypt task content."""
        # In production, would use actual encryption
        import base64
        
        encrypted_code = base64.b64encode(task.code.encode()).decode()
        
        encrypted_task = VerificationTask(
            task_id=task.task_id,
            code=encrypted_code,
            language=task.language,
            verification_type=task.verification_type,
            constraints=task.constraints,
            timeout_ms=task.timeout_ms,
            priority=task.priority,
            created_at=task.created_at,
            encrypted=True,
            requester_id=task.requester_id,
        )
        
        return encrypted_task
    
    def _deobfuscate_result(self, result: TaskResult) -> TaskResult:
        """Restore original variable names in result."""
        # Create reverse map
        reverse_map = {v: k for k, v in self._obfuscation_map.items()}
        
        def restore_names(obj):
            if isinstance(obj, str):
                for obf, orig in reverse_map.items():
                    obj = obj.replace(obf, orig)
                return obj
            elif isinstance(obj, dict):
                return {k: restore_names(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [restore_names(item) for item in obj]
            return obj
        
        restored_issues = restore_names(result.issues)
        restored_proofs = restore_names(result.proofs)
        
        return TaskResult(
            task_id=result.task_id,
            verified=result.verified,
            issues=restored_issues,
            proofs=restored_proofs,
            execution_time_ms=result.execution_time_ms,
            node_id=result.node_id,
        )
    
    def _decrypt_result(self, result: TaskResult) -> TaskResult:
        """Decrypt result."""
        # In production, would use actual decryption
        return result


# =============================================================================
# Network Coordinator
# =============================================================================

class NetworkCoordinator:
    """
    Coordinates the distributed verification network.
    
    Manages nodes, distributes tasks, and collects results.
    """
    
    def __init__(
        self,
        node_id: str,
        load_balancer: Optional[LoadBalancer] = None,
        privacy_manager: Optional[PrivacyManager] = None,
    ):
        self.node_id = node_id
        self.load_balancer = load_balancer or LoadBalancer()
        self.privacy_manager = privacy_manager or PrivacyManager()
        
        # Network state
        self.nodes: Dict[str, NetworkNode] = {}
        self.tasks: Dict[str, VerificationTask] = {}
        self.pending_tasks: List[str] = []
        
        # Configuration
        self.heartbeat_interval = 30.0  # seconds
        self.task_timeout = 60.0  # seconds
        
        # Statistics
        self.stats = {
            "tasks_distributed": 0,
            "tasks_completed": 0,
            "tasks_failed": 0,
            "total_verification_time_ms": 0.0,
        }
    
    def register_node(self, node: NetworkNode) -> None:
        """Register a node in the network."""
        node.joined_at = time.time()
        node.last_heartbeat = time.time()
        node.status = NodeStatus.ONLINE
        
        self.nodes[node.node_id] = node
    
    def remove_node(self, node_id: str) -> None:
        """Remove a node from the network."""
        if node_id in self.nodes:
            self.nodes[node_id].status = NodeStatus.OFFLINE
            del self.nodes[node_id]
            
            # Reassign tasks from removed node
            for task in self.tasks.values():
                if task.assigned_node == node_id and task.status == TaskStatus.RUNNING:
                    task.status = TaskStatus.PENDING
                    task.assigned_node = None
                    self.pending_tasks.append(task.task_id)
    
    def update_node_heartbeat(self, node_id: str) -> None:
        """Update node heartbeat timestamp."""
        if node_id in self.nodes:
            self.nodes[node_id].last_heartbeat = time.time()
    
    def submit_task(self, task: VerificationTask) -> str:
        """Submit a task for distributed verification."""
        task.created_at = time.time()
        task.status = TaskStatus.PENDING
        
        # Apply privacy protection
        protected_task = self.privacy_manager.prepare_task(task)
        
        self.tasks[task.task_id] = protected_task
        self.pending_tasks.append(task.task_id)
        self.stats["tasks_distributed"] += 1
        
        # Try to assign immediately
        self._try_assign_task(task.task_id)
        
        return task.task_id
    
    def _try_assign_task(self, task_id: str) -> bool:
        """Try to assign a task to a node."""
        task = self.tasks.get(task_id)
        if not task or task.status != TaskStatus.PENDING:
            return False
        
        # Select best node
        available_nodes = list(self.nodes.values())
        selected = self.load_balancer.select_node(available_nodes, task)
        
        if not selected:
            return False
        
        # Assign task
        task.assigned_node = selected.node_id
        task.assigned_at = time.time()
        task.status = TaskStatus.ASSIGNED
        
        selected.active_tasks += 1
        selected.current_load = selected.active_tasks / selected.max_concurrent_tasks
        
        if task_id in self.pending_tasks:
            self.pending_tasks.remove(task_id)
        
        return True
    
    def start_task(self, task_id: str) -> bool:
        """Mark a task as started."""
        task = self.tasks.get(task_id)
        if not task or task.status != TaskStatus.ASSIGNED:
            return False
        
        task.started_at = time.time()
        task.status = TaskStatus.RUNNING
        return True
    
    def complete_task(
        self,
        task_id: str,
        result: TaskResult,
    ) -> bool:
        """Complete a task with results."""
        task = self.tasks.get(task_id)
        if not task:
            return False
        
        task.completed_at = time.time()
        task.status = TaskStatus.COMPLETED
        
        # Restore privacy
        restored_result = self.privacy_manager.restore_result(result)
        task.result = restored_result.to_dict()
        
        # Update node stats
        if task.assigned_node and task.assigned_node in self.nodes:
            node = self.nodes[task.assigned_node]
            node.active_tasks = max(0, node.active_tasks - 1)
            node.current_load = node.active_tasks / node.max_concurrent_tasks
            node.tasks_completed += 1
            
            # Update average task time
            exec_time = result.execution_time_ms
            prev_avg = node.avg_task_time_ms
            total = node.tasks_completed
            node.avg_task_time_ms = (prev_avg * (total - 1) + exec_time) / total
        
        self.stats["tasks_completed"] += 1
        self.stats["total_verification_time_ms"] += result.execution_time_ms
        
        return True
    
    def fail_task(self, task_id: str, error: str) -> bool:
        """Mark a task as failed."""
        task = self.tasks.get(task_id)
        if not task:
            return False
        
        task.status = TaskStatus.FAILED
        task.error = error
        task.completed_at = time.time()
        
        # Update node reliability
        if task.assigned_node and task.assigned_node in self.nodes:
            node = self.nodes[task.assigned_node]
            node.active_tasks = max(0, node.active_tasks - 1)
            node.current_load = node.active_tasks / node.max_concurrent_tasks
            node.reliability_score = max(0.1, node.reliability_score - 0.1)
        
        self.stats["tasks_failed"] += 1
        
        return True
    
    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get task status."""
        task = self.tasks.get(task_id)
        if task:
            return task.to_dict()
        return None
    
    def get_network_stats(self) -> Dict[str, Any]:
        """Get network statistics."""
        online_nodes = [n for n in self.nodes.values() if n.status == NodeStatus.ONLINE]
        
        return {
            "total_nodes": len(self.nodes),
            "online_nodes": len(online_nodes),
            "total_capacity": sum(n.max_concurrent_tasks for n in online_nodes),
            "current_load": sum(n.active_tasks for n in online_nodes),
            "pending_tasks": len(self.pending_tasks),
            "stats": self.stats,
        }
    
    def process_pending_tasks(self) -> int:
        """Process pending tasks queue."""
        assigned = 0
        
        for task_id in list(self.pending_tasks):
            if self._try_assign_task(task_id):
                assigned += 1
        
        return assigned
    
    def check_timeouts(self) -> List[str]:
        """Check for timed out tasks."""
        timed_out = []
        now = time.time()
        
        for task_id, task in self.tasks.items():
            if task.status == TaskStatus.RUNNING:
                if task.started_at and (now - task.started_at) * 1000 > task.timeout_ms:
                    task.status = TaskStatus.TIMEOUT
                    task.error = "Task timed out"
                    timed_out.append(task_id)
        
        return timed_out
    
    def check_stale_nodes(self) -> List[str]:
        """Check for nodes with stale heartbeats."""
        stale = []
        now = time.time()
        
        for node_id, node in self.nodes.items():
            if node.status == NodeStatus.ONLINE:
                if now - node.last_heartbeat > self.heartbeat_interval * 3:
                    node.status = NodeStatus.OFFLINE
                    stale.append(node_id)
        
        return stale
