"""Tests for Language Support module."""

import pytest

from codeverify_core.language_support import (
    LanguageConfig,
    LanguageFeature,
    LanguageParser,
    LanguageRule,
    LanguageRuleRegistry,
    SupportedLanguage,
    Z3ConstraintGenerator,
    get_language_registry,
    reset_language_registry,
)


class TestSupportedLanguage:
    """Tests for SupportedLanguage enum."""

    def test_all_languages_exist(self):
        """All expected languages exist."""
        assert SupportedLanguage.PYTHON.value == "python"
        assert SupportedLanguage.TYPESCRIPT.value == "typescript"
        assert SupportedLanguage.GO.value == "go"
        assert SupportedLanguage.JAVA.value == "java"


class TestLanguageFeature:
    """Tests for LanguageFeature enum."""

    def test_feature_values(self):
        """All expected features exist."""
        assert LanguageFeature.NULL_SAFETY.value == "null_safety"
        assert LanguageFeature.BOUNDS_CHECK.value == "bounds_check"
        assert LanguageFeature.OVERFLOW.value == "overflow"
        assert LanguageFeature.CONCURRENCY.value == "concurrency"
        assert LanguageFeature.MEMORY_SAFETY.value == "memory_safety"
        assert LanguageFeature.ERROR_HANDLING.value == "error_handling"
        assert LanguageFeature.TYPE_SAFETY.value == "type_safety"


class TestLanguageConfig:
    """Tests for LanguageConfig dataclass."""

    def test_creation(self):
        """Can create a LanguageConfig."""
        config = LanguageConfig(
            language=SupportedLanguage.PYTHON,
            file_extensions=[".py"],
            comment_styles={"line": "#"},
            type_system="dynamic",
        )
        assert config.language == SupportedLanguage.PYTHON
        assert config.type_system == "dynamic"
        assert config.supports_generics is True


class TestLanguageRule:
    """Tests for LanguageRule dataclass."""

    def test_creation_with_fix_template(self):
        """Can create a LanguageRule with fix_template."""
        rule = LanguageRule(
            id="test_rule",
            language=SupportedLanguage.GO,
            pattern=r"\bpanic\s*\(",
            message="Avoid panic",
            severity="error",
            category="error_handling",
            fix_template="return fmt.Errorf(...)",
        )
        assert rule.id == "test_rule"
        assert rule.fix_template is not None


class TestLanguageParser:
    """Tests for LanguageParser."""

    def test_detect_language_python(self):
        """Detects Python from .py extension."""
        parser = LanguageParser()
        assert parser.detect_language("main.py") == SupportedLanguage.PYTHON

    def test_detect_language_typescript(self):
        """Detects TypeScript from .ts extension."""
        parser = LanguageParser()
        assert parser.detect_language("index.ts") == SupportedLanguage.TYPESCRIPT

    def test_detect_language_go(self):
        """Detects Go from .go extension."""
        parser = LanguageParser()
        assert parser.detect_language("main.go") == SupportedLanguage.GO

    def test_detect_language_java(self):
        """Detects Java from .java extension."""
        parser = LanguageParser()
        assert parser.detect_language("App.java") == SupportedLanguage.JAVA

    def test_detect_language_unknown(self):
        """Returns None for unknown extension."""
        parser = LanguageParser()
        assert parser.detect_language("README.md") is None

    def test_parse_functions_python(self):
        """Extracts Python function signatures."""
        parser = LanguageParser()
        code = (
            "def hello(name: str) -> str:\n"
            "    return f'Hello {name}'\n"
            "\n"
            "async def fetch(url: str) -> bytes:\n"
            "    pass\n"
        )
        funcs = parser.parse_functions(code, SupportedLanguage.PYTHON)
        assert len(funcs) == 2
        assert funcs[0]["name"] == "hello"
        assert funcs[1]["name"] == "fetch"

    def test_parse_imports_python(self):
        """Extracts Python imports."""
        parser = LanguageParser()
        code = (
            "import os\n"
            "from pathlib import Path\n"
            "import json\n"
        )
        imports = parser.parse_imports(code, SupportedLanguage.PYTHON)
        assert len(imports) >= 2
        assert any("os" in i for i in imports)


class TestZ3ConstraintGenerator:
    """Tests for Z3ConstraintGenerator."""

    def test_generate_null_check(self):
        """Generates null check constraint."""
        gen = Z3ConstraintGenerator()
        result = gen.generate_null_check("x", SupportedLanguage.PYTHON)
        assert "x" in result
        assert "null check" in result
        assert "None" in result

    def test_generate_bounds_check(self):
        """Generates bounds check constraint."""
        gen = Z3ConstraintGenerator()
        result = gen.generate_bounds_check("arr", "i", SupportedLanguage.GO)
        assert "arr" in result
        assert "check-sat" in result

    def test_generate_overflow_check(self):
        """Generates overflow check constraint."""
        gen = Z3ConstraintGenerator()
        result = gen.generate_overflow_check("val", 32, SupportedLanguage.JAVA)
        assert "val" in result
        assert "overflow check" in result
        assert "check-sat" in result


class TestLanguageRuleRegistry:
    """Tests for LanguageRuleRegistry."""

    def test_pre_registered_go_rules(self):
        """Registry has pre-registered Go rules."""
        registry = LanguageRuleRegistry()
        go_rules = registry.get_rules(SupportedLanguage.GO)
        assert len(go_rules) > 0
        assert any(r.id == "go_error_ignored" for r in go_rules)

    def test_pre_registered_java_rules(self):
        """Registry has pre-registered Java rules."""
        registry = LanguageRuleRegistry()
        java_rules = registry.get_rules(SupportedLanguage.JAVA)
        assert len(java_rules) > 0
        assert any(r.id == "java_empty_catch" for r in java_rules)


class TestLanguageRegistrySingletons:
    """Tests for module-level singletons."""

    def test_get_and_reset_language_registry(self):
        """get/reset language registry singletons."""
        reset_language_registry()
        r1 = get_language_registry()
        r2 = get_language_registry()
        assert r1 is r2
        reset_language_registry()
        r3 = get_language_registry()
        assert r3 is not r1
        reset_language_registry()
