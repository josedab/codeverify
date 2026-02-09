#!/usr/bin/env python3
"""Verify a code snippet with CodeVerify in 15 lines.

No API keys needed — uses the Z3 formal verifier locally.

Run:
    python examples/quickstart.py
"""

from codeverify_core import LocalZ3Verifier

code = """\
def get_element(items, index):
    return items[index]  # No bounds check — could crash!

def average(values):
    total = sum(values)
    return total / len(values)  # Division by zero if values is empty!
"""

verifier = LocalZ3Verifier()
print(f"Z3 solver available: {verifier.is_available}\n")

print("Analyzing code for potential issues...")
print("-" * 50)

# Run all verification checks (null safety, bounds, division, overflow)
results = verifier.verify_all(code, "python")

print(f"Status: {results['status']}")
print(f"Findings: {results['total_findings']}")

if results["findings"]:
    for f in results["findings"]:
        print(f"  ⚠️  {f}")
else:
    print("  ✅ No issues found by Z3 (install z3-solver for deeper analysis)")

# You can also run individual checks:
print("\n--- Individual checks ---")
for check_name in ("verify_null_safety", "verify_bounds", "verify_division"):
    check_fn = getattr(verifier, check_name)
    result = check_fn(code, "python")
    status = "✅" if result.get("status") == "success" else "⚠️"
    print(f"  {status} {check_name}: {result.get('status', 'unknown')}")
