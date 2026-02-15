---
sidebar_position: 1
---

# Language Expansion Engine

Extend CodeVerify to any programming language with pluggable adapters.

## Overview

CodeVerify v0.8.0 introduces a **pluggable language adapter framework** that makes it straightforward to add verification support for new languages. Built-in adapters ship for Python, TypeScript, Go, Java, and Rust.

## Architecture

The adapter framework separates language-specific concerns from verification logic:

```
┌──────────────────┐     ┌──────────────────┐
│  LanguageAdapter │────▶│  LanguageAdapter  │
│    Registry      │     │   (ABC)           │
└──────────────────┘     └──────────────────┘
         │                       ▲
         │          ┌────────────┼────────────┐
         │          │            │            │
         ▼          ▼            ▼            ▼
   ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐
   │  Python  │ │TypeScript│ │   Rust   │ │  Custom  │
   │ Adapter  │ │ Adapter  │ │ Adapter  │ │ Adapter  │
   └──────────┘ └──────────┘ └──────────┘ └──────────┘
```

## Built-in Adapters

| Language   | File Extensions     | Key Checks                            |
|------------|---------------------|---------------------------------------|
| Python     | `.py`               | Type hints, exception handling        |
| TypeScript | `.ts`, `.tsx`       | Type assertions, null checks          |
| Go         | `.go`               | Error returns, goroutine safety       |
| Java       | `.java`             | Null checks, resource management      |
| Rust       | `.rs`               | Unsafe blocks, ownership patterns     |

## Usage

```python
from codeverify_core.language_adapter import (
    get_adapter_registry,
    LanguageAdapter,
)
from codeverify_core.language_support import SupportedLanguage

# Get the global registry
registry = get_adapter_registry()

# Analyze a file
result = registry.analyze_file(source_code, SupportedLanguage.RUST)

# Get adapter-specific prompt context
adapter = registry.get(SupportedLanguage.RUST)
prompt = adapter.agent_prompt_template(source_code, "main.rs")
```

## Creating a Custom Adapter

Implement the `LanguageAdapter` abstract base class:

```python
from codeverify_core.language_adapter import LanguageAdapter, get_adapter_registry
from codeverify_core.language_support import SupportedLanguage

class KotlinAdapter(LanguageAdapter):
    @property
    def language(self) -> SupportedLanguage:
        return SupportedLanguage.KOTLIN  # Add to enum first

    @property
    def file_extensions(self) -> list[str]:
        return [".kt", ".kts"]

    def parse_functions(self, source: str) -> list[dict]:
        # Extract function signatures
        ...

    def extract_imports(self, source: str) -> list[str]:
        # Parse import statements
        ...

    def agent_prompt_template(self, source: str, path: str) -> str:
        return f"Analyze this Kotlin file for null safety and coroutine issues: {path}"

# Register it
registry = get_adapter_registry()
registry.register(KotlinAdapter())
```

:::note
Custom adapters are automatically picked up by the worker analysis pipeline when registered before analysis starts.
:::

## Rust Support Details

Rust verification includes checks for:

- **Unsafe blocks** — Flags `unsafe {}` usage with context
- **Unwrap panics** — Detects `.unwrap()` calls that may panic
- **Ownership patterns** — Validates borrow checker compliance hints
- **Error handling** — Ensures `Result<T, E>` is properly propagated
