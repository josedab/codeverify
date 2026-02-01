---
sidebar_position: 1
---

# Installation

Get CodeVerify installed and ready to analyze your code.

## Prerequisites

Before installing CodeVerify, ensure you have:

- **Python 3.11+** (for CLI and core packages)
- **Node.js 20+** (for dashboard and GitHub App, optional)
- **Docker** (for self-hosted deployment, optional)

## Installation Methods

### PyPI (Recommended)

The fastest way to get started:

```bash
pip install codeverify
```

This installs the CLI and core verification engine.

### With All Extras

For full functionality including all AI agents:

```bash
pip install "codeverify[all]"
```

### From Source

For development or the latest features:

```bash
git clone https://github.com/codeverify/codeverify.git
cd codeverify

# Install core packages
pip install -e "packages/core" \
            -e "packages/verifier" \
            -e "packages/ai-agents" \
            -e "packages/cli"
```

### Docker

Run CodeVerify without installing Python:

```bash
docker pull codeverify/codeverify:latest

# Analyze a file
docker run -v $(pwd):/code codeverify/codeverify analyze /code/src/main.py
```

## Verify Installation

Check that CodeVerify is installed correctly:

```bash
codeverify --version
# CodeVerify v0.3.0

codeverify doctor
# ✅ Python 3.11.4
# ✅ Z3 solver available
# ✅ OpenAI API key configured
# ✅ All systems operational
```

## Configure API Keys

CodeVerify uses LLM APIs for AI-powered analysis. Set at least one:

```bash
# OpenAI (recommended)
export OPENAI_API_KEY="sk-..."

# Or Anthropic
export ANTHROPIC_API_KEY="sk-ant-..."
```

You can also create a `.env` file in your project root:

```bash title=".env"
OPENAI_API_KEY=sk-...
# or
ANTHROPIC_API_KEY=sk-ant-...
```

:::tip No API Key?
CodeVerify works without an API key—you'll still get formal verification with Z3. AI-powered semantic and security analysis requires an LLM API key.
:::

## Shell Completion

Enable tab completion for your shell:

```bash
# Bash
codeverify completion bash >> ~/.bashrc

# Zsh
codeverify completion zsh >> ~/.zshrc

# Fish
codeverify completion fish > ~/.config/fish/completions/codeverify.fish
```

## Troubleshooting Installation

### pip install fails with "z3-solver" error

Z3 requires a C++ compiler. On macOS:

```bash
xcode-select --install
pip install codeverify
```

On Ubuntu/Debian:

```bash
sudo apt-get install build-essential
pip install codeverify
```

### "Permission denied" errors

Use a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install codeverify
```

### Slow installation

Z3 compilation takes time. Use the pre-built wheel:

```bash
pip install --prefer-binary codeverify
```

## Next Steps

Now that CodeVerify is installed:

- **[Quick Start](./quick-start)** — Run your first analysis in 2 minutes
- **[First Analysis](./first-analysis)** — Detailed walkthrough with examples
- **[GitHub Integration](/docs/integrations/github)** — Set up automatic PR checks
