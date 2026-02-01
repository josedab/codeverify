---
sidebar_position: 2
---

# Frequently Asked Questions

Common questions about CodeVerify.

## General

### What is CodeVerify?

CodeVerify is an AI-powered code review tool that combines large language models (LLMs) with Z3 formal verification to find bugs before they reach production. It provides mathematical proofs that certain classes of bugs don't exist in your code.

### How is this different from linters?

Linters check for style issues and common patterns. CodeVerify uses formal verification to mathematically prove properties about your code. For example, a linter might warn about missing null checks based on patterns, but CodeVerify can prove whether a null dereference is actually possible.

### What languages are supported?

- **Python** — Full support
- **TypeScript/JavaScript** — Full support
- **Go** — Beta support

More languages coming soon.

### Is my code sent to the cloud?

It depends on your setup:

- **Cloud (default):** Code is sent to our servers for analysis
- **Self-hosted:** Code never leaves your infrastructure
- **CLI-only:** Verification runs locally, AI features require API calls

### Is CodeVerify open source?

Yes, the core verification engine is open source under the Apache 2.0 license. Enterprise features and the hosted service are commercial.

## Verification

### What can CodeVerify verify?

- **Null safety** — No null/undefined dereferences
- **Array bounds** — No out-of-bounds access
- **Division by zero** — No divide-by-zero errors
- **Integer overflow** — No arithmetic overflow

### Can CodeVerify find all bugs?

No. CodeVerify focuses on specific, mathematically provable properties. It cannot find:

- Business logic errors
- Performance issues
- UI/UX problems
- Race conditions (coming soon)

### What is a false positive?

A false positive is when CodeVerify reports an issue that isn't actually a bug. This can happen when:

- External invariants exist that CodeVerify can't see
- The code is unreachable in practice
- Complex control flow limits analysis

False positive rate is typically below 10%.

### How do I suppress false positives?

```python
# Inline suppression
value = data[index]  # codeverify-disable-line array_bounds

# Config suppression
ignore:
  - finding_id: "f_abc123"
    reason: "External validation ensures this is safe"
```

### What is Z3?

Z3 is a high-performance SMT (Satisfiability Modulo Theories) solver developed by Microsoft Research. It's used in many verification tools and can reason about mathematical formulas involving integers, arrays, and more.

## Trust Score

### What is the Copilot Trust Score?

The Trust Score (0-100) indicates confidence in code correctness, combining:
- Formal verification results (40%)
- AI analysis confidence (30%)
- Pattern coverage (20%)
- Historical accuracy (10%)

### What's a good Trust Score?

- **90-100:** Excellent, high confidence
- **70-89:** Good, minor concerns
- **50-69:** Fair, review recommended
- **Below 50:** Poor, significant issues

### How do I improve my Trust Score?

1. Fix reported findings
2. Add type annotations
3. Add explicit null checks
4. Simplify complex functions
5. Add tests (historical accuracy)

## Integration

### Does CodeVerify work with GitHub Copilot?

Yes! The Copilot Interceptor feature analyzes Copilot suggestions in real-time and shows Trust Scores before you accept code.

### Can I use CodeVerify without GitHub?

Yes. CodeVerify supports:
- Local CLI usage
- GitLab CI/CD
- Bitbucket Pipelines
- Any CI system via Docker

### How long does analysis take?

Typical times:
- Small file (under 100 lines): 2-5 seconds
- Medium project (1000 lines): 30-60 seconds
- Large project (10000+ lines): 2-5 minutes

Factors affecting speed:
- Code complexity
- Number of checks enabled
- AI features enabled

## Pricing

### Is there a free tier?

Yes. The free tier includes:
- 100 analyses per month
- Public repositories
- Community support

### What's included in Pro?

- Unlimited analyses
- Private repositories
- Priority support
- Team features
- Advanced integrations

### What's included in Enterprise?

Everything in Pro, plus:
- Self-hosting option
- SSO/SAML
- Dedicated support
- SLA guarantees
- Custom integrations

### Can I try before buying?

Yes. All plans have a 14-day free trial with full features.

## Self-Hosting

### What are the system requirements?

Minimum:
- 4 CPU cores
- 8 GB RAM
- 50 GB storage

Recommended for production:
- 8+ CPU cores
- 16+ GB RAM
- 100+ GB SSD
- PostgreSQL 14+

### Can I run CodeVerify air-gapped?

Yes, with limitations:
- Formal verification works fully offline
- AI features require API access to OpenAI/Anthropic
- You can use local LLMs (Ollama) for offline AI

### How do I update self-hosted deployments?

```bash
# Docker Compose
docker compose pull
docker compose up -d

# Kubernetes
helm repo update
helm upgrade codeverify codeverify/codeverify
```

## Security

### Is my code secure?

Yes. We follow security best practices:
- TLS encryption in transit
- Encryption at rest
- SOC 2 Type II certified (in progress)
- No code storage after analysis
- Self-hosting option for full control

### What data is collected?

- Code snippets (for analysis, not stored long-term)
- Findings and metadata
- Usage analytics (can be disabled)

We never collect:
- Credentials or secrets
- Personal information
- Unrelated code

### Can I get a security assessment?

Yes. Enterprise customers can request:
- Security questionnaire responses
- Penetration test results
- SOC 2 report
- Data processing agreement (DPA)

## Support

### How do I get help?

- **Documentation:** You're here!
- **GitHub Issues:** Bug reports and feature requests
- **Discord:** Community support
- **Email:** Pro/Enterprise support

### How do I report a bug?

1. Check existing issues on GitHub
2. Create a new issue with:
   - CodeVerify version
   - Reproduction steps
   - Expected vs actual behavior
   - Relevant code (if possible)

### How do I request a feature?

Create a GitHub issue with the "feature request" label, describing:
- Use case
- Proposed solution
- Alternatives considered
