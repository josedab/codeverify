#!/usr/bin/env python3
"""Environment validation script for CodeVerify.

Run this script to check if all dependencies and services are properly configured.
"""
import os
import subprocess
import sys
from typing import Callable


def check(name: str, fn: Callable[[], bool], required: bool = True) -> bool:
    """Run a check and print result."""
    try:
        result = fn()
        status = "✅" if result else ("⚠️" if not required else "❌")
        print(f"{status} {name}")
        return result
    except Exception as e:
        status = "⚠️" if not required else "❌"
        print(f"{status} {name}: {e}")
        return False


def check_python_version() -> bool:
    """Check Python version >= 3.11."""
    return sys.version_info >= (3, 11)


def check_z3() -> bool:
    """Check if Z3 is installed."""
    try:
        import z3
        return True
    except ImportError:
        return False


def check_postgres() -> bool:
    """Check PostgreSQL connection."""
    import asyncpg
    import asyncio
    
    async def _check():
        conn = await asyncpg.connect(os.getenv(
            "DATABASE_URL",
            "postgresql://codeverify:codeverify@localhost:5432/codeverify"
        ))
        await conn.close()
        return True
    
    return asyncio.run(_check())


def check_redis() -> bool:
    """Check Redis connection."""
    import redis
    r = redis.from_url(os.getenv("REDIS_URL", "redis://localhost:6379/0"))
    return r.ping()


def check_env_var(name: str) -> bool:
    """Check if environment variable is set."""
    return bool(os.getenv(name))


def check_node() -> bool:
    """Check Node.js version."""
    result = subprocess.run(
        ["node", "--version"],
        capture_output=True,
        text=True
    )
    if result.returncode != 0:
        return False
    version = result.stdout.strip().lstrip("v")
    major = int(version.split(".")[0])
    return major >= 18


def check_docker() -> bool:
    """Check Docker is running."""
    result = subprocess.run(
        ["docker", "info"],
        capture_output=True,
        text=True
    )
    return result.returncode == 0


def main():
    """Run all environment checks."""
    print("=" * 50)
    print("CodeVerify Environment Validation")
    print("=" * 50)
    print()
    
    print("📦 Core Dependencies")
    print("-" * 30)
    
    checks = [
        ("Python 3.11+", check_python_version, True),
        ("Z3 SMT Solver", check_z3, True),
        ("Node.js 18+", check_node, True),
        ("Docker", check_docker, False),
    ]
    
    for name, fn, required in checks:
        check(name, fn, required)
    
    print()
    print("🔌 Services")
    print("-" * 30)
    
    service_checks = [
        ("PostgreSQL", check_postgres, True),
        ("Redis", check_redis, True),
    ]
    
    for name, fn, required in service_checks:
        check(name, fn, required)
    
    print()
    print("🔑 Environment Variables")
    print("-" * 30)
    
    env_vars = [
        ("DATABASE_URL", False),
        ("REDIS_URL", False),
        ("GITHUB_APP_ID", True),
        ("GITHUB_APP_PRIVATE_KEY", True),
        ("GITHUB_WEBHOOK_SECRET", True),
        ("JWT_SECRET", True),
        ("OPENAI_API_KEY", False),
        ("ANTHROPIC_API_KEY", False),
    ]
    
    for name, required in env_vars:
        check(f"${name}", lambda n=name: check_env_var(n), required)
    
    print()
    print("=" * 50)
    print()
    
    # Summary
    missing_required = [
        name for name, required in env_vars 
        if required and not check_env_var(name)
    ]
    
    if missing_required:
        print("⚠️  Missing required environment variables:")
        for var in missing_required:
            print(f"   - {var}")
        print()
        print("Copy .env.example to .env and fill in the values:")
        print("   cp .env.example .env")
        print()
    else:
        print("✅ All required checks passed!")
        print()
        print("Start the services with:")
        print("   docker compose up -d")
        print()
        print("Then run the API:")
        print("   cd apps/api && uvicorn codeverify_api.main:app --reload")
        print()


if __name__ == "__main__":
    main()
