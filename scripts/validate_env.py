#!/usr/bin/env python3
"""Environment validation script for CodeVerify.

Run this script to check if all dependencies and services are properly configured.
"""

import importlib.util
import os
import subprocess
import sys
from collections.abc import Callable


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
    return importlib.util.find_spec("z3") is not None


def check_postgres() -> bool:
    """Check PostgreSQL connection."""
    try:
        import asyncpg
    except ImportError:
        raise RuntimeError("asyncpg not installed (pip install asyncpg)") from None
    import asyncio

    async def _check():
        conn = await asyncpg.connect(
            os.getenv(
                "DATABASE_URL", "postgresql://codeverify:codeverify@localhost:5432/codeverify"
            )
        )
        await conn.close()
        return True

    return asyncio.run(_check())


def check_redis() -> bool:
    """Check Redis connection."""
    try:
        import redis
    except ImportError:
        raise RuntimeError("redis not installed (pip install redis)") from None
    r = redis.from_url(os.getenv("REDIS_URL", "redis://localhost:6379/0"))
    return r.ping()


def check_env_var(name: str) -> bool:
    """Check if environment variable is set."""
    return bool(os.getenv(name))


def check_node() -> bool:
    """Check Node.js version."""
    result = subprocess.run(["node", "--version"], capture_output=True, text=True)
    if result.returncode != 0:
        return False
    version = result.stdout.strip().lstrip("v")
    major = int(version.split(".")[0])
    return major >= 18


def check_docker() -> bool:
    """Check Docker is running."""
    result = subprocess.run(["docker", "info"], capture_output=True, text=True)
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
        ("GITHUB_APP_ID", False),
        ("GITHUB_APP_PRIVATE_KEY", False),
        ("GITHUB_WEBHOOK_SECRET", False),
        ("JWT_SECRET", False),
        ("OPENAI_API_KEY", False),
        ("ANTHROPIC_API_KEY", False),
    ]

    for name, required in env_vars:
        check(f"${name}", lambda n=name: check_env_var(n), required)

    print()
    print("=" * 50)
    print()

    # Summary
    github_set = all(
        check_env_var(v)
        for v in ["GITHUB_APP_ID", "GITHUB_APP_PRIVATE_KEY", "GITHUB_WEBHOOK_SECRET"]
    )
    llm_set = check_env_var("OPENAI_API_KEY") or check_env_var("ANTHROPIC_API_KEY")

    print("📊 Summary")
    print("-" * 30)
    print(f"  {'✅' if True else '❌'} Core verification:     Ready (no API keys needed)")
    print(
        f"  {'✅' if llm_set else '⚠️'} AI analysis:           {'Ready' if llm_set else 'Set OPENAI_API_KEY or ANTHROPIC_API_KEY'}"
    )
    print(
        f"  {'✅' if github_set else '⚠️'} GitHub PR integration: {'Ready' if github_set else 'Set GITHUB_APP_* vars (optional)'}"
    )
    print()

    if not llm_set:
        print("💡 Tip: For verification-only mode (no AI), you're all set!")
        print("   To enable AI analysis, add an API key to .env:")
        print("     OPENAI_API_KEY=sk-...")
        print("     # or")
        print("     ANTHROPIC_API_KEY=sk-ant-...")
        print()
    else:
        print("✅ All checks passed!")
        print()
        print("Start the services with:")
        print("   make dev")
        print()


if __name__ == "__main__":
    main()
