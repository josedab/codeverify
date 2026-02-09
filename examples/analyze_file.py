#!/usr/bin/env python3
"""Analyze a local file with CodeVerify's verification and rules engine.

Run:
    python examples/analyze_file.py path/to/file.py
    python examples/analyze_file.py examples/quickstart.py  # analyze itself!
"""

import sys
from pathlib import Path

from codeverify_core import LocalZ3Verifier, RuleEvaluator, get_builtin_rules


def analyze_file(file_path: str) -> int:
    path = Path(file_path)
    if not path.exists():
        print(f"❌ File not found: {file_path}")
        return 1

    code = path.read_text()
    language = {"py": "python", "ts": "typescript", "js": "javascript"}.get(
        path.suffix.lstrip("."), "python"
    )

    print(f"📄 Analyzing {path.name} ({language}, {len(code)} chars)")
    print("=" * 50)

    total_issues = 0

    # --- Formal Verification ---
    print("\n🔬 Formal Verification (Z3)")
    print("-" * 30)
    verifier = LocalZ3Verifier()
    results = verifier.verify_all(code, language)
    z3_findings = results.get("findings", [])
    if z3_findings:
        for f in z3_findings:
            print(f"  ⚠️  {f}")
        total_issues += len(z3_findings)
    else:
        note = "" if verifier.is_available else " (install z3-solver for deeper analysis)"
        print(f"  ✅ No formal verification issues{note}")

    # --- Rules Engine ---
    print("\n📋 Rules Engine")
    print("-" * 30)
    rules = list(get_builtin_rules().values())
    evaluator = RuleEvaluator(rules=rules)
    violations = evaluator.evaluate(code, str(path))
    if violations:
        for v in violations:
            print(f"  ⚠️  Line {v['line']}: [{v['severity']}] {v['message']}")
        total_issues += len(violations)
    else:
        print("  ✅ No rule violations")

    # --- Summary ---
    print(f"\n{'='*50}")
    status = "⚠️" if total_issues > 0 else "✅"
    print(f"{status} Total issues: {total_issues}")
    return 1 if total_issues > 0 else 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python examples/analyze_file.py <file>")
        print("       python examples/analyze_file.py examples/quickstart.py")
        sys.exit(1)
    sys.exit(analyze_file(sys.argv[1]))
