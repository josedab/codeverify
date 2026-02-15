---
sidebar_position: 8
---

# Plugin Marketplace & SDK

Extend CodeVerify with community plugins for custom rules, integrations, and analyzers.

## Overview

The Plugin Marketplace provides a framework for publishing, discovering, and installing CodeVerify extensions. The SDK makes it simple to create plugins that integrate with the verification pipeline.

## Plugin Types

| Type         | Description                         | Example                        |
|-------------|-------------------------------------|--------------------------------|
| Rule         | Custom verification rules           | Company coding standards       |
| Integration  | External tool connectors            | Jira ticket creation           |
| Analyzer     | Language-specific analyzers         | Kotlin null-safety checker     |
| Reporter     | Custom output formats               | PDF compliance reports         |

## Creating a Plugin

### 1. Define the Manifest

```python
from codeverify_core.plugin_marketplace import PluginManifest, PluginType

manifest = PluginManifest(
    name="my-custom-rules",
    version="1.0.0",
    description="Custom verification rules for our team",
    author="your-github-handle",
    plugin_type=PluginType.RULE,
    entry_point="my_rules:register",
    tags=["security", "python"],
)
```

### 2. Implement the Plugin

```python
# my_rules.py
from codeverify_core.plugin_marketplace import PluginSDK

sdk = PluginSDK()

@sdk.rule("no-print-statements")
def check_no_prints(source: str, path: str) -> list[dict]:
    """Disallow print() in production code."""
    issues = []
    for i, line in enumerate(source.splitlines(), 1):
        if "print(" in line and not path.endswith("_test.py"):
            issues.append({
                "line": i,
                "message": "Use logging instead of print()",
                "severity": "medium",
            })
    return issues

def register(registry):
    sdk.register_all(registry)
```

### 3. Publish

```python
from codeverify_core.plugin_marketplace import get_plugin_registry

registry = get_plugin_registry()
registry.publish(manifest, package_url="https://github.com/you/my-custom-rules")
```

## Installing Plugins

```python
registry = get_plugin_registry()

# Search for plugins
results = registry.search("security python")

# Install a plugin
registry.install("my-custom-rules", version="1.0.0")
```

## Plugin Reviews

```python
from codeverify_core.plugin_marketplace import PluginReview

review = PluginReview(
    plugin_id="my-custom-rules",
    author="reviewer-handle",
    rating=5,
    comment="Great rules, caught several issues in our codebase.",
)
registry.add_review(review)
```

## Redis-Backed Registry

For production deployments with persistent plugin storage:

```python
from codeverify_core.redis_backends import RedisPluginStore

store = RedisPluginStore(redis_url="redis://localhost:6379/2")
store.save_plugin("my-custom-rules", manifest.to_dict())
```

:::note
Plugins run in the same process as CodeVerify. Always review plugin code before installing, especially plugins from untrusted sources.
:::
