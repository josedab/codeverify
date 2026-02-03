#!/usr/bin/env python3
"""
CodeVerify Benchmark Suite

Runs performance benchmarks and generates metrics for documentation.
Results are output in JSON format for easy parsing and comparison.

Usage:
    python scripts/benchmark.py [--output results.json] [--iterations 3]
"""

import argparse
import json
import os
import statistics
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class BenchmarkResult:
    """Single benchmark result."""
    name: str
    mean_time_ms: float
    min_time_ms: float
    max_time_ms: float
    std_dev_ms: float
    iterations: int
    metadata: dict = field(default_factory=dict)


@dataclass
class BenchmarkReport:
    """Complete benchmark report."""
    timestamp: str
    python_version: str
    codeverify_version: str
    results: list[BenchmarkResult]
    system_info: dict


def get_codeverify_version() -> str:
    """Get installed CodeVerify version."""
    try:
        result = subprocess.run(
            ["codeverify", "--version"],
            capture_output=True,
            text=True,
            timeout=10
        )
        return result.stdout.strip() or "unknown"
    except Exception:
        return "not installed"


def get_system_info() -> dict:
    """Collect system information."""
    import platform
    return {
        "os": platform.system(),
        "os_version": platform.version(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python_implementation": platform.python_implementation(),
    }


def create_test_file(lines: int, complexity: str = "simple") -> str:
    """Create a test Python file with specified characteristics."""
    
    if complexity == "simple":
        # Simple functions with basic operations
        code = '''"""Test module for benchmarking."""

def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b

def subtract(a: int, b: int) -> int:
    """Subtract two numbers."""
    return a - b

'''
        # Repeat to reach desired line count
        repeat_block = '''
def function_{i}(x: int) -> int:
    """Function {i}."""
    if x > 0:
        return x * 2
    return x
'''
    elif complexity == "moderate":
        code = '''"""Test module with moderate complexity."""
from typing import Optional, List

def find_element(items: List[int], target: int) -> Optional[int]:
    """Find element index."""
    for i, item in enumerate(items):
        if item == target:
            return i
    return None

'''
        repeat_block = '''
def process_{i}(data: List[int]) -> int:
    """Process data {i}."""
    result = 0
    for item in data:
        if item > 0:
            result += item
        elif item < 0:
            result -= item
    return result
'''
    else:  # complex
        code = '''"""Test module with complex code."""
from typing import Optional, List, Dict, Any

class DataProcessor:
    """Process data with various transformations."""
    
    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.cache: Dict[str, Any] = {}
    
    def process(self, data: List[int]) -> List[int]:
        """Process the data."""
        result = []
        for item in data:
            if item in self.cache:
                result.append(self.cache[item])
            else:
                processed = self._transform(item)
                self.cache[item] = processed
                result.append(processed)
        return result
    
    def _transform(self, value: int) -> int:
        """Transform a single value."""
        if value < 0:
            return -value
        return value * 2

'''
        repeat_block = '''
def analyze_{i}(items: List[int], threshold: int) -> Dict[str, int]:
    """Analyze items {i}."""
    above = below = equal = 0
    for item in items:
        if item > threshold:
            above += 1
        elif item < threshold:
            below += 1
        else:
            equal += 1
    return {{"above": above, "below": below, "equal": equal}}
'''
    
    # Calculate how many functions to add
    current_lines = len(code.split('\n'))
    lines_per_func = len(repeat_block.split('\n'))
    funcs_needed = max(0, (lines - current_lines) // lines_per_func)
    
    for i in range(funcs_needed):
        code += repeat_block.format(i=i)
    
    return code


def run_benchmark(
    name: str,
    code: str,
    iterations: int = 3,
    checks: Optional[list[str]] = None
) -> BenchmarkResult:
    """Run a single benchmark."""
    
    times = []
    
    with tempfile.NamedTemporaryFile(
        mode='w', 
        suffix='.py', 
        delete=False
    ) as f:
        f.write(code)
        temp_path = f.name
    
    try:
        for _ in range(iterations):
            cmd = ["codeverify", "analyze", temp_path, "--format", "json"]
            if checks:
                cmd.extend(["--checks", ",".join(checks)])
            
            start = time.perf_counter()
            try:
                subprocess.run(
                    cmd,
                    capture_output=True,
                    timeout=120
                )
            except subprocess.TimeoutExpired:
                times.append(120000)  # 120s timeout
                continue
            end = time.perf_counter()
            
            times.append((end - start) * 1000)  # Convert to ms
    finally:
        os.unlink(temp_path)
    
    return BenchmarkResult(
        name=name,
        mean_time_ms=statistics.mean(times),
        min_time_ms=min(times),
        max_time_ms=max(times),
        std_dev_ms=statistics.stdev(times) if len(times) > 1 else 0,
        iterations=iterations,
        metadata={"checks": checks or ["all"]}
    )


def run_all_benchmarks(iterations: int = 3) -> list[BenchmarkResult]:
    """Run the complete benchmark suite."""
    
    results = []
    
    print("Running CodeVerify Benchmark Suite")
    print("=" * 50)
    
    # Check if codeverify is installed
    version = get_codeverify_version()
    if version == "not installed":
        print("ERROR: codeverify is not installed")
        print("Install with: pip install codeverify")
        sys.exit(1)
    
    print(f"CodeVerify version: {version}")
    print(f"Iterations per benchmark: {iterations}")
    print()
    
    # Benchmark 1: Single file analysis (varying sizes)
    for lines in [50, 200, 500, 1000]:
        print(f"  Benchmarking {lines} LOC file...", end=" ", flush=True)
        code = create_test_file(lines, "simple")
        result = run_benchmark(
            name=f"single_file_{lines}_loc",
            code=code,
            iterations=iterations
        )
        results.append(result)
        print(f"{result.mean_time_ms:.0f}ms")
    
    # Benchmark 2: Individual check types
    print("\n  Benchmarking individual checks...")
    code = create_test_file(200, "moderate")
    for check in ["null_safety", "division_by_zero", "array_bounds", "integer_overflow"]:
        print(f"    {check}...", end=" ", flush=True)
        result = run_benchmark(
            name=f"check_{check}",
            code=code,
            iterations=iterations,
            checks=[check]
        )
        results.append(result)
        print(f"{result.mean_time_ms:.0f}ms")
    
    # Benchmark 3: Complexity levels
    print("\n  Benchmarking complexity levels...")
    for complexity in ["simple", "moderate", "complex"]:
        print(f"    {complexity}...", end=" ", flush=True)
        code = create_test_file(200, complexity)
        result = run_benchmark(
            name=f"complexity_{complexity}",
            code=code,
            iterations=iterations
        )
        results.append(result)
        print(f"{result.mean_time_ms:.0f}ms")
    
    print("\n" + "=" * 50)
    print("Benchmark complete!")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="CodeVerify Benchmark Suite")
    parser.add_argument(
        "--output", "-o",
        help="Output file for JSON results",
        default="benchmark-results.json"
    )
    parser.add_argument(
        "--iterations", "-n",
        type=int,
        default=3,
        help="Number of iterations per benchmark"
    )
    parser.add_argument(
        "--format",
        choices=["json", "markdown"],
        default="json",
        help="Output format"
    )
    
    args = parser.parse_args()
    
    results = run_all_benchmarks(iterations=args.iterations)
    
    report = BenchmarkReport(
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        python_version=sys.version,
        codeverify_version=get_codeverify_version(),
        results=results,
        system_info=get_system_info()
    )
    
    if args.format == "json":
        output = {
            "timestamp": report.timestamp,
            "python_version": report.python_version,
            "codeverify_version": report.codeverify_version,
            "system_info": report.system_info,
            "results": [asdict(r) for r in report.results]
        }
        
        with open(args.output, "w") as f:
            json.dump(output, f, indent=2)
        
        print(f"\nResults written to {args.output}")
    
    elif args.format == "markdown":
        print("\n## Benchmark Results\n")
        print("| Benchmark | Mean (ms) | Min (ms) | Max (ms) | Std Dev |")
        print("|-----------|-----------|----------|----------|---------|")
        for r in results:
            print(f"| {r.name} | {r.mean_time_ms:.1f} | {r.min_time_ms:.1f} | {r.max_time_ms:.1f} | {r.std_dev_ms:.1f} |")


if __name__ == "__main__":
    main()
