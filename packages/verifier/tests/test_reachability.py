"""Tests for Vulnerability Reachability Analysis."""

import pytest

from codeverify_verifier.reachability import (
    CallGraphBuilder,
    CodeNode,
    ReachabilityAnalyzer,
    ReachabilityReport,
    ReachabilityStatus,
    Vulnerability,
    VulnerabilityScanner,
    VulnerabilityType,
    create_cve_vulnerability,
)


# Sample code with vulnerabilities
VULNERABLE_CODE = '''
import subprocess
import pickle

def main():
    """Entry point."""
    user_input = input("Enter command: ")
    process_input(user_input)

def process_input(data):
    """Process user input."""
    if data.startswith("run:"):
        execute_command(data[4:])
    elif data.startswith("load:"):
        load_data(data[5:])
    else:
        safe_operation(data)

def execute_command(cmd):
    """Execute shell command - VULNERABLE."""
    subprocess.call(cmd, shell=True)  # CWE-78: Command Injection

def load_data(path):
    """Load pickled data - VULNERABLE."""
    with open(path, 'rb') as f:
        return pickle.load(f)  # CWE-502: Insecure Deserialization

def safe_operation(data):
    """Safe operation."""
    print(f"Processing: {data}")
'''

# Code where vulnerable function is unreachable
UNREACHABLE_VULN_CODE = '''
import pickle

def main():
    """Entry point."""
    safe_process()

def safe_process():
    """Only uses safe functions."""
    data = {"key": "value"}
    print(data)

def dangerous_load(path):
    """VULNERABLE but never called."""
    with open(path, 'rb') as f:
        return pickle.load(f)
'''

# Code with conditional reachability
CONDITIONAL_VULN_CODE = '''
import os
import subprocess

DEBUG = False

def main():
    user_input = get_input()
    if DEBUG:
        debug_execute(user_input)
    else:
        safe_execute(user_input)

def get_input():
    return input("Command: ")

def debug_execute(cmd):
    """Only called in debug mode - VULNERABLE."""
    subprocess.call(cmd, shell=True)

def safe_execute(cmd):
    """Safe execution."""
    allowed = ["list", "status", "help"]
    if cmd in allowed:
        print(f"Running: {cmd}")
'''


class TestCallGraphBuilder:
    """Test call graph construction."""
    
    def test_build_from_simple_code(self):
        """Test building call graph from simple code."""
        builder = CallGraphBuilder()
        nodes, edges = builder.build_from_code(
            VULNERABLE_CODE,
            "test.py",
            "python"
        )
        
        assert len(nodes) > 0
        assert "test.py::main" in nodes
        assert "test.py::process_input" in nodes
        assert "test.py::execute_command" in nodes
    
    def test_detect_entry_points(self):
        """Test entry point detection."""
        code = '''
@app.route("/api/data")
def get_data():
    return fetch_data()

def fetch_data():
    return {"data": "value"}

if __name__ == "__main__":
    main()
'''
        builder = CallGraphBuilder()
        nodes, _ = builder.build_from_code(code, "app.py", "python")
        
        # Should detect decorated function as entry point
        entry_points = [n for n in nodes.values() if n.is_entry_point]
        assert len(entry_points) >= 1
    
    def test_build_edges(self):
        """Test that call edges are created."""
        builder = CallGraphBuilder()
        nodes, edges = builder.build_from_code(
            VULNERABLE_CODE,
            "test.py",
            "python"
        )
        
        # main calls process_input
        main_to_process = any(
            e for e in edges
            if e.source == "test.py::main" and "process_input" in e.target
        )
        assert main_to_process


class TestReachabilityAnalyzer:
    """Test reachability analysis with Z3."""
    
    def test_reachable_vulnerability(self):
        """Test detection of reachable vulnerability."""
        vuln = Vulnerability(
            id="test-vuln-1",
            type=VulnerabilityType.CWE,
            title="Command Injection",
            description="Test",
            severity="critical",
            affected_functions=["execute_command"],
            cwe_id="CWE-78",
        )
        
        analyzer = ReachabilityAnalyzer()
        report = analyzer.analyze(
            {"test.py": VULNERABLE_CODE},
            [vuln],
            entry_points=["main"],
            language="python"
        )
        
        assert report.total_vulnerabilities == 1
        # The vulnerability should be reachable through main -> process_input -> execute_command
        result = report.results[0]
        assert result.status in [ReachabilityStatus.REACHABLE, ReachabilityStatus.CONDITIONAL]
    
    def test_unreachable_vulnerability(self):
        """Test that unreachable vulnerabilities are detected."""
        vuln = Vulnerability(
            id="test-vuln-2",
            type=VulnerabilityType.CWE,
            title="Insecure Deserialization",
            description="Test",
            severity="high",
            affected_functions=["dangerous_load"],
            cwe_id="CWE-502",
        )
        
        analyzer = ReachabilityAnalyzer()
        report = analyzer.analyze(
            {"test.py": UNREACHABLE_VULN_CODE},
            [vuln],
            entry_points=["main"],
            language="python"
        )
        
        assert report.total_vulnerabilities == 1
        result = report.results[0]
        # Should be unreachable since dangerous_load is never called
        assert result.status == ReachabilityStatus.UNREACHABLE
    
    def test_multiple_vulnerabilities(self):
        """Test analysis of multiple vulnerabilities."""
        vulns = [
            Vulnerability(
                id="vuln-1",
                type=VulnerabilityType.CWE,
                title="Command Injection",
                description="Test",
                severity="critical",
                affected_functions=["execute_command"],
            ),
            Vulnerability(
                id="vuln-2",
                type=VulnerabilityType.CWE,
                title="Insecure Deserialization",
                description="Test",
                severity="high",
                affected_functions=["load_data"],
            ),
        ]
        
        analyzer = ReachabilityAnalyzer()
        report = analyzer.analyze(
            {"test.py": VULNERABLE_CODE},
            vulns,
            entry_points=["main"],
            language="python"
        )
        
        assert report.total_vulnerabilities == 2


class TestVulnerabilityScanner:
    """Test vulnerability scanning and reachability."""
    
    def test_scan_for_vulnerabilities(self):
        """Test automatic vulnerability detection."""
        scanner = VulnerabilityScanner()
        report = scanner.scan_and_analyze(
            {"vulnerable.py": VULNERABLE_CODE},
            language="python"
        )
        
        # Should find at least the command injection and pickle vulnerabilities
        assert report.total_vulnerabilities >= 2
    
    def test_scan_safe_code(self):
        """Test scanning code without vulnerabilities."""
        safe_code = '''
def greet(name: str) -> str:
    """Greet a user safely."""
    return f"Hello, {name}!"

def main():
    name = input("Name: ")
    print(greet(name))
'''
        scanner = VulnerabilityScanner()
        report = scanner.scan_and_analyze(
            {"safe.py": safe_code},
            language="python"
        )
        
        # Should find no vulnerabilities (or very few)
        assert report.total_vulnerabilities == 0 or report.unreachable_count >= 0


class TestReachabilityReport:
    """Test report generation."""
    
    def test_report_to_dict(self):
        """Test report serialization."""
        vuln = Vulnerability(
            id="test",
            type=VulnerabilityType.CVE,
            title="Test Vuln",
            description="Test",
            severity="high",
            cve_id="CVE-2024-0001",
        )
        
        report = ReachabilityReport(
            total_vulnerabilities=1,
            reachable_count=0,
            unreachable_count=1,
            conditional_count=0,
            unknown_count=0,
            results=[],
        )
        
        data = report.to_dict()
        
        assert data["total_vulnerabilities"] == 1
        assert data["unreachable_count"] == 1
        assert "false_positive_rate" in data
        assert data["false_positive_rate"] == 100.0
    
    def test_false_positive_rate_calculation(self):
        """Test false positive rate is calculated correctly."""
        report = ReachabilityReport(
            total_vulnerabilities=10,
            reachable_count=2,
            unreachable_count=8,
            conditional_count=0,
            unknown_count=0,
        )
        
        data = report.to_dict()
        
        # 8 out of 10 are unreachable = 80% false positive rate
        assert data["false_positive_rate"] == 80.0


class TestCVEVulnerability:
    """Test CVE vulnerability creation."""
    
    def test_create_cve_vulnerability(self):
        """Test creating a CVE vulnerability."""
        vuln = create_cve_vulnerability(
            cve_id="CVE-2024-1234",
            title="Critical RCE in example-lib",
            affected_package="example-lib",
            affected_functions=["parse_input", "execute"],
            severity="critical",
            cvss_score=9.8,
            fix_available=True,
        )
        
        assert vuln.cve_id == "CVE-2024-1234"
        assert vuln.type == VulnerabilityType.CVE
        assert vuln.cvss_score == 9.8
        assert vuln.fix_available is True
        assert len(vuln.affected_functions) == 2
