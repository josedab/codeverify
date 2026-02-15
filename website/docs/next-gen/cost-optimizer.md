---
sidebar_position: 4
---

# LLM Cost Optimizer

Intelligent model routing to minimize LLM costs while maintaining quality.

## Overview

The Cost Optimizer classifies code complexity and routes analysis to the most cost-effective model. Simple linting goes to local/small models, while complex verification uses premium cloud models — reducing costs by 40-60% without sacrificing accuracy.

## Routing Strategy

| Complexity | Model Tier    | Example Use Case                  | Relative Cost |
|------------|---------------|-----------------------------------|:-------------:|
| Low        | Local / Small | Style checks, simple patterns     | $0.001        |
| Medium     | Standard      | Type analysis, common bugs        | $0.01         |
| High       | Premium       | Formal verification, deep logic   | $0.10         |

## Usage

```python
from codeverify_core.llm_cost_optimizer import (
    CostOptimizer,
    ComplexityClassifier,
)

optimizer = CostOptimizer(monthly_budget=100.0)
classifier = ComplexityClassifier()

# Classify code complexity
complexity = classifier.classify(source_code)

# Get recommended model
model = optimizer.recommend_model(complexity)

# Record usage for budget tracking
optimizer.record_usage(
    model=model.name,
    tokens=1500,
    cost=model.cost_per_1k * 1.5,
)

# Check budget status
status = optimizer.get_budget_status()
print(f"Spent: ${status.spent:.2f} / ${status.budget:.2f}")
```

## Budget Tracking with Redis

For persistent budget tracking across deployments:

```python
from codeverify_core.redis_backends import RedisCostStore

store = RedisCostStore(redis_url="redis://localhost:6379/2")

# Record a call
store.record_call(model="gpt-4", tokens=2000, cost=0.06, complexity="high")

# Check spending
daily = store.get_daily_spend()
monthly = store.get_monthly_spend()
over_budget = store.is_over_budget()
```

## Configuration

```yaml
# .codeverify.yml
cost_optimizer:
  enabled: true
  monthly_budget: 100.00
  
  routing:
    low_complexity: local     # Use local model
    medium_complexity: gpt-3.5-turbo
    high_complexity: gpt-4
  
  fallback_on_budget_exceeded: local
```

:::note
When the monthly budget is exceeded, the optimizer automatically falls back to local models to prevent unexpected charges.
:::
