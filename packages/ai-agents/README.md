# @codeverify/ai-agents

LLM-powered analysis agents for CodeVerify.

## Installation

```bash
pip install codeverify-ai-agents

# Or from source
pip install -e packages/ai-agents
```

**Requirements:**
- Python 3.11+
- OpenAI API key and/or Anthropic API key

## Overview

This package provides AI-powered agents for code analysis:

- **Semantic Agent**: Understands code intent and logic
- **Security Agent**: Detects vulnerabilities (OWASP Top 10)
- **Synthesis Agent**: Consolidates findings and generates fixes
- **Trust Score Agent**: Assesses AI-generated code quality
- **Diff Summarizer**: Auto-generates PR descriptions

## Quick Start

```python
from codeverify_agents import SemanticAgent, SecurityAgent

# Initialize agents
semantic = SemanticAgent(model="gpt-4")
security = SecurityAgent(model="claude-3-sonnet")

# Analyze code
code = """
def login(username, password):
    query = f"SELECT * FROM users WHERE name='{username}'"
    return db.execute(query)
"""

semantic_result = await semantic.analyze(code, context="authentication module")
security_result = await security.analyze(code)

print(security_result.findings[0].title)  # "SQL Injection vulnerability"
```

## Agents

### SemanticAgent

Understands code intent and extracts contracts:

```python
from codeverify_agents import SemanticAgent

agent = SemanticAgent(model="gpt-4")

result = await agent.analyze(
    code=code,
    context="This function processes user payments",
    extract_contracts=True,
)

print(result.intent)  # "Processes payment and updates balance"
print(result.contracts)  # Pre/post conditions
print(result.potential_issues)  # Logic concerns
```

### SecurityAgent

Detects security vulnerabilities:

```python
from codeverify_agents import SecurityAgent

agent = SecurityAgent(model="claude-3-sonnet")

result = await agent.analyze(
    code=code,
    context="API endpoint",
    check_owasp=True,
)

for finding in result.findings:
    print(f"{finding.severity}: {finding.title}")
    print(f"  OWASP: {finding.owasp_category}")
    print(f"  CWE: {finding.cwe_id}")
    print(f"  Fix: {finding.fix_suggestion}")
```

### SynthesisAgent

Consolidates findings and generates fixes:

```python
from codeverify_agents import SynthesisAgent

agent = SynthesisAgent()

result = await agent.synthesize(
    code=code,
    semantic_findings=semantic_result.findings,
    security_findings=security_result.findings,
    verification_findings=z3_result.findings,
    generate_fixes=True,
)

print(result.summary)
print(result.prioritized_findings)
print(result.suggested_fixes)
```

### TrustScoreAgent

Assesses AI-generated code quality:

```python
from codeverify_agents import TrustScoreAgent

agent = TrustScoreAgent()

result = await agent.calculate_score(
    code=code,
    metadata={
        "generated_by": "copilot",
        "prompt": "Create a login function",
    },
)

print(f"Trust Score: {result.score}/100")
print(f"Risk Level: {result.risk_level}")  # low/medium/high/critical
print(f"AI Probability: {result.ai_probability}")
print(f"Factors: {result.factors}")
```

### DiffSummarizerAgent

Auto-generates PR descriptions:

```python
from codeverify_agents import DiffSummarizerAgent

agent = DiffSummarizerAgent()

result = await agent.summarize(
    diff=pr_diff,
    commit_messages=["Add user authentication", "Fix SQL injection"],
)

print(result.title)
print(result.description)
print(result.changelog_entry)
print(result.suggested_reviewers)
```

### TestGeneratorAgent

Generates tests from verification counterexamples:

```python
from codeverify_agents import TestGeneratorAgent, TestFramework

agent = TestGeneratorAgent()

tests = await agent.generate_tests(
    counterexamples=[
        {"function": "divide", "inputs": {"a": 10, "b": 0}},
    ],
    framework=TestFramework.PYTEST,
)

print(tests[0].test_code)
```

### NaturalLanguageInvariantsAgent

Converts English specs to Z3:

```python
from codeverify_agents import NaturalLanguageInvariantsAgent

agent = NaturalLanguageInvariantsAgent()

result = await agent.compile_spec("""
balance must be non-negative
withdrawal must be less than or equal to balance
""")

for assertion in result.assertions:
    print(f"NL: {assertion.natural_language}")
    print(f"Z3: {assertion.z3_code}")
```

### MultiModelConsensus

Query multiple LLMs for higher confidence:

```python
from codeverify_agents import MultiModelConsensus, VotingStrategy

consensus = MultiModelConsensus(
    models=["gpt-4", "claude-3-sonnet", "gpt-4-turbo"],
    voting_strategy=VotingStrategy.CONFIDENCE_WEIGHTED,
)

result = await consensus.analyze(code, context)

for finding in result.findings:
    print(f"{finding.title}: {finding.confidence}% confidence")
    print(f"  Agreed by: {finding.agreeing_models}")
```

## Base Agent

All agents inherit from `BaseAgent`:

```python
from codeverify_agents.base import BaseAgent

class CustomAgent(BaseAgent):
    def __init__(self, model: str = "gpt-4"):
        super().__init__(model=model)
    
    async def analyze(self, code: str, **kwargs) -> AnalysisResult:
        prompt = self.build_prompt(code, kwargs)
        response = await self.llm.complete(prompt)
        return self.parse_response(response)
```

## Configuration

### Environment Variables

```bash
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
CODEVERIFY_DEFAULT_MODEL=gpt-4
CODEVERIFY_TEMPERATURE=0.1
CODEVERIFY_MAX_TOKENS=4096
```

### Agent Configuration

```python
from codeverify_agents import AgentConfig

config = AgentConfig(
    model="gpt-4",
    temperature=0.1,
    max_tokens=4096,
    timeout_seconds=60,
    retry_attempts=3,
)

agent = SemanticAgent(config=config)
```

## Development

```bash
# Install with dev dependencies
pip install -e "packages/ai-agents[dev]"

# Run tests
pytest packages/ai-agents/tests -v

# Run with mocked LLM
CODEVERIFY_MOCK_LLM=true pytest packages/ai-agents/tests

# Type checking
mypy packages/ai-agents/src
```

## Further Reading

- [ADR-0003: Multi-Agent Architecture](../../docs/adr/0003-multi-agent-architecture.md)
- [Architecture Overview](../../docs/architecture/overview.md)
