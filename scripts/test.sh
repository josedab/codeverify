#!/bin/bash
# Run all tests.
# Prefer using Makefile targets instead:
#   make test-core   — core package only (no infrastructure needed)
#   make test-fast   — all packages (no infrastructure needed)
#   make test        — everything including apps (needs Postgres + Redis)

set -e

echo "🧪 Running CodeVerify tests..."

# Activate virtual environment if exists
if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

# Run the complete Python suite.
echo ""
echo "📦 Running Python tests..."
pytest packages/ apps/ tests/ -v --tb=short

# Run Node.js tests if they exist
echo ""
if [ -f "apps/github-app/package.json" ] && grep -q '"test"' apps/github-app/package.json 2>/dev/null; then
    echo "🔗 Running GitHub App tests..."
    cd apps/github-app && npm test && cd ../..
else
    echo "⏭️  Skipping GitHub App tests (no test script found)"
fi

echo ""
if [ -f "apps/web/package.json" ] && grep -q '"test"' apps/web/package.json 2>/dev/null; then
    echo "🖥️ Running Web unit tests..."
    cd apps/web && npm test && cd ../..
else
    echo "⏭️  Skipping Web tests (no test script found)"
fi

echo ""
echo "✅ All tests completed!"

# Optional: Run with coverage
if [ "$1" = "--coverage" ]; then
    echo ""
    echo "📊 Generating coverage report..."
    pytest packages/ apps/ tests/ \
        --cov=codeverify \
        --cov-report=html \
        --cov-report=term-missing
    echo "Coverage report: htmlcov/index.html"
fi

# Optional: Run E2E tests
if [ "$1" = "--e2e" ]; then
    echo ""
    echo "🎭 Running E2E tests (Playwright)..."
    cd apps/web && npm run test:e2e
    cd ../..
fi

# Optional: Run load tests
if [ "$1" = "--load" ]; then
    echo ""
    echo "📈 Running load tests (Locust)..."
    locust -f tests/load/locustfile.py --host=http://localhost:8000 --users=10 --spawn-rate=2 --run-time=1m --headless --only-summary
fi
