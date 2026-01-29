"""Pre-built rule templates for common code quality checks.

This module provides ready-to-use rules for common patterns like
detecting print statements, hardcoded secrets, and unsafe eval usage.
"""

from codeverify_core.rules.builder import RuleBuilder
from codeverify_core.rules.models import CustomRule, RuleSeverity


# Pre-built rule templates
RULE_TEMPLATES: dict[str, CustomRule] = {
    "no-print": RuleBuilder()
        .name("No Print Statements")
        .description("Use logging instead of print statements")
        .severity(RuleSeverity.LOW)
        .pattern(r"\bprint\s*\(")
        .action("Replace print() with proper logging", fix_template="logger.info({args})")
        .for_languages("python")
        .with_tags("style", "logging")
        .build(),

    "no-hardcoded-secrets": RuleBuilder()
        .name("No Hardcoded Secrets")
        .description("Detect hardcoded passwords and API keys")
        .severity(RuleSeverity.CRITICAL)
        .pattern(r"(?i)(password|api_key|secret|token)\s*=\s*['\"][^'\"]+['\"]")
        .action("Use environment variables for sensitive data")
        .with_tags("security", "secrets")
        .build(),

    "no-eval": RuleBuilder()
        .name("No Eval Usage")
        .description("Avoid using eval() which can execute arbitrary code")
        .severity(RuleSeverity.HIGH)
        .pattern(r"\beval\s*\(")
        .action("Replace eval() with safer alternatives like ast.literal_eval()")
        .for_languages("python")
        .with_tags("security")
        .build(),

    "require-type-hints": RuleBuilder()
        .name("Require Type Hints")
        .description("Function parameters should have type hints")
        .severity(RuleSeverity.LOW)
        .pattern(r"def\s+\w+\s*\([^)]*[a-zA-Z_]\w*\s*[,)]")
        .action("Add type hints to function parameters")
        .for_languages("python")
        .with_tags("style", "typing")
        .build(),

    "no-console-log": RuleBuilder()
        .name("No Console.log")
        .description("Use proper logging instead of console.log")
        .severity(RuleSeverity.LOW)
        .pattern(r"\bconsole\s*\.\s*log\s*\(")
        .action("Replace console.log() with proper logging")
        .for_languages("javascript", "typescript")
        .with_tags("style", "logging")
        .build(),

    "no-debugger": RuleBuilder()
        .name("No Debugger Statements")
        .description("Remove debugger statements before committing")
        .severity(RuleSeverity.MEDIUM)
        .pattern(r"\bdebugger\b")
        .action("Remove debugger statement")
        .for_languages("javascript", "typescript")
        .with_tags("debugging")
        .build(),

    "no-todo-in-code": RuleBuilder()
        .name("No TODO Comments")
        .description("Track TODO items in issue tracker, not code")
        .severity(RuleSeverity.INFO)
        .pattern(r"(?i)#\s*TODO|//\s*TODO|/\*\s*TODO")
        .action("Create an issue for this TODO item")
        .with_tags("style", "documentation")
        .build(),

    "no-sql-injection": RuleBuilder()
        .name("Potential SQL Injection")
        .description("Detect string concatenation in SQL queries")
        .severity(RuleSeverity.CRITICAL)
        .pattern(r"(?i)(execute|cursor\.execute)\s*\(\s*[\"'].*\s*\+")
        .action("Use parameterized queries instead of string concatenation")
        .for_languages("python")
        .with_tags("security", "sql")
        .build(),

    "no-exec": RuleBuilder()
        .name("No Exec Usage")
        .description("Avoid using exec() which can execute arbitrary code")
        .severity(RuleSeverity.HIGH)
        .pattern(r"\bexec\s*\(")
        .action("Replace exec() with safer alternatives")
        .for_languages("python")
        .with_tags("security")
        .build(),

    "require-docstrings": RuleBuilder()
        .name("Require Docstrings")
        .description("Functions should have docstrings")
        .severity(RuleSeverity.LOW)
        .pattern(r'def\s+\w+\([^)]*\)\s*:\s*\n\s*(?![\'\"]{3})')
        .action("Add a docstring to this function")
        .for_languages("python")
        .with_tags("documentation")
        .build(),
}


def get_builtin_rules() -> dict[str, CustomRule]:
    """Get all built-in rule templates.
    
    Returns:
        Dictionary mapping rule IDs to CustomRule instances.
    """
    return RULE_TEMPLATES.copy()


def get_rule_by_name(name: str) -> CustomRule | None:
    """Get a built-in rule by name.
    
    Args:
        name: Rule template name (e.g., "no-print", "no-eval")
        
    Returns:
        CustomRule instance or None if not found
    """
    return RULE_TEMPLATES.get(name)


def get_rules_by_tag(tag: str) -> list[CustomRule]:
    """Get all built-in rules with a specific tag.
    
    Args:
        tag: Tag to filter by (e.g., "security", "style")
        
    Returns:
        List of matching CustomRule instances
    """
    return [rule for rule in RULE_TEMPLATES.values() if tag in rule.tags]


def get_security_rules() -> list[CustomRule]:
    """Get all built-in security-related rules.
    
    Returns:
        List of security rules
    """
    return get_rules_by_tag("security")


def get_style_rules() -> list[CustomRule]:
    """Get all built-in style-related rules.
    
    Returns:
        List of style rules
    """
    return get_rules_by_tag("style")
