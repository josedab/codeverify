#!/usr/bin/env python3
"""Create and apply custom lint rules with CodeVerify.

Demonstrates the built-in rules engine and how to define your own.

Run:
    python examples/custom_rules.py
"""

from codeverify_core import CustomRule, RuleEvaluator, get_builtin_rules

# --- 1. Explore built-in rules ---

builtin = get_builtin_rules()
print(f"📋 {len(builtin)} built-in rules available:\n")
for name, rule in builtin.items():
    print(f"  [{rule.severity.value:>8}] {rule.name}")
    print(f"             {rule.description}")

# --- 2. Evaluate built-in rules against sample code ---

sample_code = """\
import os

password = "hunter2"          # hardcoded secret!
eval(user_input)              # dangerous eval
print("debug:", password)     # print instead of logging
"""

print(f"\n{'='*50}")
print("🔍 Scanning sample code with built-in rules...\n")

evaluator = RuleEvaluator(rules=list(builtin.values()))
violations = evaluator.evaluate(sample_code, "sample.py")

if violations:
    for v in violations:
        print(f"  ⚠️  Line {v['line']}: [{v['severity']}] {v['message']}")
        print(f"       Rule: {v['rule_name']}")
else:
    print("  ✅ No violations found")

print(f"\nTotal violations: {len(violations)}")
