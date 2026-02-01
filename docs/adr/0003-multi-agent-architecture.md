# ADR-0003: Multi-Agent AI Architecture

## Status

Accepted

## Context

CodeVerify uses AI to analyze code for bugs, security issues, and quality problems. We needed to decide how to structure the AI components:

1. **Single monolithic agent**: One LLM call analyzes everything
2. **Pipeline of prompts**: Sequential prompts with handoffs
3. **Multi-agent system**: Specialized agents that can work independently or together

Key requirements:
- Different analysis types need different prompts and reasoning
- Some analyses benefit from multiple perspectives (consensus)
- System should be extensible for new analysis types
- Cost and latency should be manageable

## Decision

We adopted a **multi-agent architecture** with specialized agents:

### Core Agents
- **SemanticAgent**: Understands code intent, extracts contracts, identifies logic errors
- **SecurityAgent**: Finds vulnerabilities, OWASP issues, security anti-patterns
- **SynthesisAgent**: Consolidates findings, generates fixes, creates PR comments

### Advanced Agents
- **TrustScoreAgent**: ML-powered scoring of AI-generated code risk
- **DiffSummarizerAgent**: Generates PR descriptions and changelogs
- **ThreatModelingAgent**: STRIDE threat categorization
- **SpecificationGeneratorAgent**: Infers pre/post conditions
- **RegressionOracle**: Predicts bug-prone changes
- **MultiModelConsensus**: Queries multiple LLMs to reduce false positives

### Architecture Pattern
```
                    ┌─────────────────┐
                    │ Analysis Task   │
                    └────────┬────────┘
                             │
              ┌──────────────┼──────────────┐
              ▼              ▼              ▼
        ┌──────────┐  ┌──────────┐  ┌──────────┐
        │ Semantic │  │ Formal   │  │ Security │
        │  Agent   │  │ Verifier │  │  Agent   │
        └────┬─────┘  └────┬─────┘  └────┬─────┘
             │             │             │
             └─────────────┼─────────────┘
                           ▼
                    ┌──────────────┐
                    │  Synthesis   │
                    │    Agent     │
                    └──────────────┘
```

### Implementation
- `BaseAgent` class with provider abstraction (OpenAI, Anthropic)
- `AgentConfig` for model, temperature, timeout settings
- `AgentResult` standardized output format
- Lazy client initialization to minimize API calls

## Consequences

### Positive
- **Specialization**: Each agent optimized for its task
- **Parallel execution**: Independent agents can run concurrently
- **Extensibility**: New agents can be added without modifying existing ones
- **Testability**: Agents can be tested in isolation
- **Consensus**: Multiple agents can validate findings
- **Cost optimization**: Use cheaper models for simpler tasks

### Negative
- **Complexity**: More components to maintain
- **Coordination overhead**: Need orchestration logic
- **Potential redundancy**: Agents might duplicate analysis
- **Cost multiplication**: Multiple API calls per analysis

### Neutral
- Requires clear agent boundaries and responsibilities
- Need monitoring for per-agent performance

## Alternatives Considered

### Single Monolithic Agent
- **Rejected because**: Single prompt becomes unwieldy; harder to optimize for specific tasks

### Fine-tuned Models
- **Future consideration**: Could fine-tune models for specific agents; deferred until we have sufficient training data

### Local Models (Ollama, llama.cpp)
- **Complementary**: Supported for self-hosted deployments; API providers used for cloud

## References

- [AutoGPT Agent Architecture](https://github.com/Significant-Gravitas/AutoGPT)
- [LangChain Agents](https://python.langchain.com/docs/modules/agents/)
- [Multi-Agent Systems in AI](https://arxiv.org/abs/2308.08155)
