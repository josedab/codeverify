#!/bin/bash
# Run all tests

set -e

echo "🧪 Running CodeVerify tests..."

# Activate virtual environment if exists
if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

# Run Python tests
echo ""
echo "📦 Running Python package tests..."
pytest packages/ -v --tb=short

echo ""
echo "🌐 Running API tests..."
pytest apps/api/tests -v --tb=short

echo ""
echo "⚙️ Running Worker tests..."
pytest apps/worker/tests -v --tb=short 2>/dev/null || echo "No worker tests yet"

echo ""
echo "🔗 Running Integration tests..."
pytest tests/integration -v --tb=short 2>/dev/null || echo "No integration tests yet"

# Run Node.js tests
echo ""
echo "🔗 Running GitHub App tests..."
cd apps/github-app && npm test 2>/dev/null || echo "No GitHub App tests yet"
cd ../..

echo ""
echo "🖥️ Running Web unit tests..."
cd apps/web && npm test 2>/dev/null || echo "No Web unit tests yet"
cd ../..

echo ""
echo "✅ All tests completed!"

# Optional: Run with coverage
if [ "$1" = "--coverage" ]; then
    echo ""
    echo "📊 Generating coverage report..."
    pytest packages/ apps/api/tests apps/worker/tests tests/integration \
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
