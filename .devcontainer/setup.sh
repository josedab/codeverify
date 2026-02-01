#!/bin/bash
set -e

echo "🚀 Setting up CodeVerify development environment..."

# Copy environment file if it doesn't exist
if [ ! -f .env ]; then
    echo "📋 Creating .env from template..."
    cp .env.example .env
fi

# Install Python packages in editable mode
echo "🐍 Installing Python packages..."
pip install --upgrade pip
pip install -e "packages/core[dev]" \
            -e "packages/verifier[dev]" \
            -e "packages/ai-agents[dev]" \
            -e "apps/api[dev]" \
            -e "apps/worker[dev]"

# Install Node.js packages
echo "📦 Installing Node.js packages..."
cd apps/web && npm install && cd ../..
cd apps/github-app && npm install && cd ../..

# Wait for PostgreSQL to be ready
echo "⏳ Waiting for PostgreSQL..."
while ! pg_isready -h postgres -p 5432 -U codeverify; do
    sleep 1
done

# Run database migrations
echo "🗄️ Running database migrations..."
cd apps/api && alembic upgrade head && cd ../..

# Validate environment
echo "✅ Validating environment..."
python scripts/validate_env.py || true

echo ""
echo "✨ Development environment ready!"
echo ""
echo "Available commands:"
echo "  uvicorn codeverify_api.main:app --reload     # Start API on :8000"
echo "  celery -A codeverify_worker.main worker      # Start worker"
echo "  cd apps/web && npm run dev                   # Start dashboard on :3000"
echo "  cd apps/github-app && npm run dev            # Start GitHub App on :3001"
echo ""
