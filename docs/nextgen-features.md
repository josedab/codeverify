# Next-Generation Features Guide

This guide covers the 10 next-generation features added in CodeVerify v0.3.0.

## Table of Contents

1. [Monorepo Intelligence](#monorepo-intelligence)
2. [AI Regression Test Generator](#ai-regression-test-generator)
3. [Proof-Carrying PRs](#proof-carrying-prs)
4. [IDE-Native Copilot Interceptor](#ide-native-copilot-interceptor)
5. [Natural Language Invariant Specs](#natural-language-invariant-specs)
6. [Semantic Diff Visualization](#semantic-diff-visualization)
7. [Verification Budget Optimizer](#verification-budget-optimizer)
8. [Team Learning Mode](#team-learning-mode)
9. [Competing Model Arbitration](#competing-model-arbitration)
10. [Gradual Verification Ramp](#gradual-verification-ramp)

---

## Monorepo Intelligence

Cross-package dependency analysis for modern monorepo setups.

### Supported Tools

- **Nx** (`nx.json`)
- **Turborepo** (`turbo.json`)
- **Lerna** (`lerna.json`)
- **pnpm** (`pnpm-workspace.yaml`)
- **Yarn Workspaces** (`package.json` with `workspaces`)

### CLI Usage

```bash
# Analyze monorepo structure
codeverify monorepo analyze

# Get affected packages from changed files
codeverify monorepo affected packages/core/src/index.ts

# JSON output for CI integration
codeverify monorepo analyze -f json
```

### API Usage

```python
from codeverify_core.monorepo import MonorepoAnalyzer

analyzer = MonorepoAnalyzer("/path/to/monorepo")
print(f"Type: {analyzer.monorepo_type}")

# Get all packages
packages = analyzer.discover_packages()

# Build dependency graph
graph = analyzer.build_dependency_graph()

# Check for circular dependencies
cycles = graph.detect_cycles()

# Get affected packages
affected = analyzer.get_affected_packages(["packages/core/src/index.ts"])
```

### Configuration

```yaml
# .codeverify.yml
monorepo:
  enabled: true
  affected_analysis: true
  cycle_detection: true
```

---

## AI Regression Test Generator

Automatically generates regression tests from Z3 verification counterexamples.

### Supported Frameworks

| Language | Frameworks |
|----------|------------|
| Python | pytest, unittest |
| TypeScript/JavaScript | Jest, Vitest |
| Go | go test |

### CLI Usage

```bash
# Generate tests from verification results
codeverify generate-tests src/math.py

# Specify framework
codeverify generate-tests src/utils.ts -f jest

# Output to file
codeverify generate-tests src/calc.py -o tests/test_calc.py
```

### API Usage

```python
from codeverify_agents.test_generator import (
    TestGeneratorAgent,
    CounterexampleToTest,
    TestFramework,
)

agent = TestGeneratorAgent()

# Create counterexample
counterexample = CounterexampleToTest(
    function_name="divide",
    input_values={"a": 10, "b": 0},
    expected_behavior="raise ZeroDivisionError",
    verification_type="division_by_zero",
)

# Generate test
test = agent.generate_test(counterexample, TestFramework.PYTEST)
print(test.test_code)
```

### Example Output

```python
# Generated test for divide function
def test_divide_division_by_zero():
    """
    Regression test generated from Z3 counterexample.
    Verification type: division_by_zero
    """
    with pytest.raises(ZeroDivisionError):
        divide(a=10, b=0)
```

---

## Proof-Carrying PRs

Cryptographic attestations for verification results that travel with your code.

### How It Works

1. CodeVerify runs verification on your PR
2. Creates a cryptographically signed proof
3. Embeds proof in commit message or PR comment
4. Downstream consumers can verify the proof

### CLI Usage

```bash
# Create attestation
codeverify attest PR#123

# Verify existing attestation
codeverify attest --verify attestation.json

# Export attestation
codeverify attest commit:abc123 -o proof.json
```

### API Usage

```python
from codeverify_core.proof_carrying import (
    ProofCarryingManager,
    VerificationProof,
)

# Initialize with secret key
manager = ProofCarryingManager(secret_key="your-secret-key")

# Create proof
proof = manager.create_proof(
    code_hash="abc123",
    verification_type="formal",
    result="passed",
)

# Sign proof
attestation = manager.sign_proof(proof)

# Verify attestation
is_valid = manager.verify_attestation(attestation)

# Embed in commit message
message = manager.embed_in_commit_message(
    attestation,
    "feat: add new feature"
)
```

### Commit Message Format

```
feat: add user authentication

CodeVerify-Attestation: eyJ0eXBlIjoiYXR0ZXN0...
```

---

## IDE-Native Copilot Interceptor

Real-time verification of GitHub Copilot suggestions in VS Code.

### Features

- Intercepts Copilot suggestions before insertion
- Real-time verification with inline decorations
- Trust score display
- Warning/error indicators

### Installation

1. Install CodeVerify VS Code extension
2. Enable Copilot interception in settings:

```json
{
  "codeverify.copilotInterception.enabled": true,
  "codeverify.copilotInterception.verificationLevel": "standard"
}
```

### Verification Levels

| Level | Description | Speed |
|-------|-------------|-------|
| `fast` | Pattern matching only | <100ms |
| `standard` | Static + AI analysis | <2s |
| `strict` | Full formal verification | <5s |

### Status Indicators

| Icon | Status | Meaning |
|------|--------|---------|
| ⏳ | Verifying | Analysis in progress |
| ✅ | Passed | No issues found |
| ⚠️ | Warning | Potential issues |
| ❌ | Failed | Critical issues |

---

## Natural Language Invariant Specs

Write specifications in English, get Z3 assertions.

### CLI Usage

```bash
# Compile invariants
codeverify invariants specs/balance.txt

# Output SMT-LIB format
codeverify invariants specs/user.md -o smt
```

### Supported Patterns

```
# Basic constraints
x must be positive
count must be non-negative
balance must be greater than 0

# Ranges
age must be between 0 and 150
percentage must be between 0 and 100 inclusive

# Null checks
name must not be null
user must not be empty

# Comparisons
end_date must be after start_date
max must be greater than or equal to min
```

### API Usage

```python
from codeverify_agents.nl_invariants import NaturalLanguageInvariantsAgent

agent = NaturalLanguageInvariantsAgent()

# Compile specification
spec = agent.compile_spec("""
balance must be non-negative
withdrawal must be less than or equal to balance
""")

for assertion in spec.assertions:
    print(f"Z3: {assertion.z3_code}")
    print(f"SMT: {assertion.smt_lib}")
```

---

## Semantic Diff Visualization

Understand behavioral changes between code versions.

### CLI Usage

```bash
# Compare files
codeverify semantic-diff old.py new.py

# Mermaid diagram output
codeverify semantic-diff v1.ts v2.ts -f mermaid

# HTML report
codeverify semantic-diff main.py feature.py -f html -o diff.html
```

### Change Types Detected

- **Signature Changes**: Parameters, return types
- **Behavior Changes**: Logic, control flow
- **Exception Changes**: Error handling modifications
- **Side Effect Changes**: I/O, state mutations

### API Usage

```python
from codeverify_agents.semantic_diff import SemanticDiffAgent

agent = SemanticDiffAgent()

result = await agent.analyze(
    old_code="def greet(name): return f'Hello {name}'",
    new_code="def greet(name, title=''): return f'Hello {title} {name}'",
    language="python",
)

for change in result.changes:
    print(f"{change.change_type}: {change.impact}")

# Get Mermaid diagram
print(result.mermaid_diagram)
```

---

## Verification Budget Optimizer

Intelligent routing of verification based on risk and cost.

### CLI Usage

```bash
# Estimate verification cost
codeverify budget estimate src/*.py

# Estimate with premium tier
codeverify budget estimate src/ --tier premium

# View usage report
codeverify budget report
```

### Verification Depths

| Depth | Cost | Time | Use Case |
|-------|------|------|----------|
| Pattern | $0.01 | 50ms | Low-risk, trivial changes |
| Static | $0.05 | 200ms | Medium complexity |
| AI | $0.10 | 2s | Complex logic |
| Formal | $0.25 | 5s | Critical paths |
| Full | $0.41 | 7s | Security-sensitive |

### API Usage

```python
from codeverify_core.budget_optimizer import (
    VerificationBudgetOptimizer,
    Budget,
    RiskFactors,
)

optimizer = VerificationBudgetOptimizer()

decision = optimizer.optimize_file(
    file_path="src/auth.py",
    file_size_lines=200,
    factors=RiskFactors(
        file_complexity=0.8,
        historical_bug_rate=0.3,
        has_security_patterns=True,
    ),
    budget=Budget(tier="standard", max_cost_per_pr=5.0),
)

print(f"Depth: {decision.depth}")
print(f"Estimated cost: ${decision.estimated_cost:.2f}")
```

---

## Team Learning Mode

Organization-wide pattern detection and training recommendations.

### CLI Usage

```bash
# Generate team report
codeverify team-report

# Export as markdown
codeverify team-report -o report.md -f markdown

# JSON output
codeverify team-report -f json
```

### Features

- Systemic pattern identification
- Team-specific metrics
- Training recommendations
- Trend analysis

### API Usage

```python
from codeverify_agents.team_learning import TeamLearningAgent

agent = TeamLearningAgent()

# Configure team mappings
agent.configure_teams({
    "alice": "frontend",
    "bob": "backend",
})

# Record findings
agent.record_findings(findings, "my-repo", "alice")

# Generate report
report = agent.generate_org_health_report()

print(f"Total findings: {report.total_findings}")
print(f"Top patterns: {report.systemic_patterns[:3]}")

# Export markdown
markdown = agent.export_report_markdown(report)
```

---

## Competing Model Arbitration

Multi-LLM consensus for higher-confidence findings.

### Voting Methods

| Method | Description |
|--------|-------------|
| `approval` | Simple majority vote |
| `borda_count` | Points-based ranking |
| `ranked_choice` | Instant runoff |
| `confidence_weighted` | Weight by confidence scores |
| `specialization` | Weight by model expertise |

### API Usage

```python
from codeverify_agents.model_arbitrator import (
    CompetingModelArbitrator,
    VotingMethod,
)

arbitrator = CompetingModelArbitrator(
    voting_method=VotingMethod.CONFIDENCE_WEIGHTED,
    enable_debate=True,
)

result = await arbitrator.analyze(code, context)

for finding in result.data["findings"]:
    print(f"{finding['title']}: {finding['verdict']} ({finding['confidence']:.0%})")
```

---

## Gradual Verification Ramp

Warnings-only onboarding mode for new repositories.

### CLI Usage

```bash
# Start ramp
codeverify ramp start myorg/myrepo

# Custom schedule
codeverify ramp start myrepo --baseline-days 3 --observation-days 7

# Check status
codeverify ramp status myorg/myrepo

# Pause/resume
codeverify ramp pause myorg/myrepo
codeverify ramp resume myorg/myrepo

# End early
codeverify ramp end myorg/myrepo --confirm
```

### Phases

| Phase | Duration | Behavior |
|-------|----------|----------|
| Baseline | 7 days | Silent observation |
| Observation | 14 days | Warnings only |
| Transition | 14 days | Progressive enforcement |
| Enforcing | Ongoing | Full enforcement |

### API Usage

```python
from codeverify_core.gradual_ramp import (
    GradualVerificationRamp,
    RampSchedule,
)

ramp = GradualVerificationRamp()

# Start with custom schedule
schedule = RampSchedule(
    baseline_days=3,
    observation_days=7,
    transition_days=7,
)

state = ramp.start_ramp("myorg/myrepo", schedule=schedule)

# Evaluate PR
decision = ramp.evaluate_enforcement("myorg/myrepo", findings)

if decision.should_block:
    print("PR blocked")
else:
    print(f"Warnings: {len(decision.warning_findings)}")
    if decision.days_until_enforcement:
        print(f"Enforcement in {decision.days_until_enforcement} days")
```

### PR Comment Example

```markdown
## CodeVerify Onboarding Status

👀 **Phase:** Observation
**Enforcement Level:** Warn

### ⚠️ Warnings (3)
*These findings will become blocking in future phases.*

- **ERROR**: Potential null reference
- **WARNING**: Unused variable
- **WARNING**: Missing type annotation

📅 *Some warnings will become blocking in 14 days.*

---
*CodeVerify is gradually ramping up verification for this repository.*
```

---

## Configuration Reference

All features can be configured in `.codeverify.yml`:

```yaml
version: "1"

# Monorepo Intelligence
monorepo:
  enabled: true
  affected_analysis: true

# Budget Optimizer
budget:
  tier: standard
  max_cost_per_pr: 5.0
  max_cost_per_file: 1.0

# Team Learning
team_learning:
  enabled: true
  report_schedule: weekly

# Model Arbitration
arbitration:
  voting_method: confidence_weighted
  enable_debate: true
  models:
    - openai_gpt5
    - anthropic_claude

# Gradual Ramp
ramp:
  enabled: true
  baseline_days: 7
  observation_days: 14
  transition_days: 14

# Proof Attestations
attestation:
  enabled: true
  embed_in_commits: true
```
