#!/bin/bash
# Development setup script for CodeVerify

set -e

echo "🚀 Setting up CodeVerify development environment..."

# Check prerequisites
command -v node >/dev/null 2>&1 || { echo "❌ Node.js is required but not installed."; exit 1; }
command -v docker >/dev/null 2>&1 || { echo "❌ Docker is required but not installed."; exit 1; }

# Find the best available Python 3.11+
PYTHON=$(command -v python3.12 2>/dev/null || command -v python3.11 2>/dev/null || command -v python3 2>/dev/null)
if [ -z "$PYTHON" ]; then
    echo "❌ Python 3 is required but not installed."
    exit 1
fi
if ! "$PYTHON" -c "import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)" 2>/dev/null; then
    echo "❌ Python 3.11+ required. Found: $($PYTHON --version 2>&1)"
    echo "   Install with: brew install python@3.12  (macOS)"
    echo "                  sudo apt install python3.12  (Ubuntu/Debian)"
    exit 1
fi
echo "   Using $($PYTHON --version 2>&1) at $PYTHON"

# Create virtual environment
echo "📦 Creating Python virtual environment..."
"$PYTHON" -m venv .venv
source .venv/bin/activate

# Install Python packages
echo "📦 Installing Python packages..."
pip install --upgrade pip
pip install -e "packages/core[dev]"
pip install -e "packages/verifier[dev]"
pip install -e "packages/ai-agents[dev]"
pip install -e "packages/z3-mcp"
pip install -e "apps/api[dev]"
pip install -e "apps/worker[dev]"

# Set up pre-commit hooks
echo "🔧 Installing pre-commit hooks..."
pip install pre-commit
pre-commit install

# Install Node.js packages
echo "📦 Installing Node.js packages..."
cd apps/github-app && npm install && cd ../..
cd apps/web && npm install && cd ../..

# Copy environment file
if [ ! -f .env ]; then
    echo "📋 Creating .env file from minimal template..."
    cp .env.minimal .env
    echo "⚠️  Created .env with local defaults. For AI features, see .env.example"
fi

# Start infrastructure
echo "🐳 Starting Docker infrastructure..."
docker compose up -d postgres redis

# Wait for services
echo "⏳ Waiting for services to be ready..."
sleep 5

echo ""
echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "  1. Edit .env with your API keys"
echo "  2. Run 'source .venv/bin/activate' to activate the virtual environment"
echo "  3. Start the API: uvicorn codeverify_api.main:app --reload --port 8000"
echo "  4. Start the worker: celery -A codeverify_worker.main worker --loglevel=info"
echo "  5. Start the GitHub App: cd apps/github-app && npm run dev"
echo "  6. Start the Web UI: cd apps/web && npm run dev"
echo ""
