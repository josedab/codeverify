"""AI Regression Test Generator - Auto-generate tests from Z3 counterexamples.

Converts verification findings with counterexamples into executable test cases
that can be committed to the repository to prevent regressions.
"""

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import structlog

from codeverify_agents.base import AgentConfig, AgentResult, BaseAgent

logger = structlog.get_logger()


class TestFramework(str, Enum):
    """Supported test frameworks."""
    PYTEST = "pytest"
    UNITTEST = "unittest"
    JEST = "jest"
    MOCHA = "mocha"
    VITEST = "vitest"
    GO_TEST = "go_test"
    JUNIT = "junit"


class Language(str, Enum):
    """Programming languages for test generation."""
    PYTHON = "python"
    TYPESCRIPT = "typescript"
    JAVASCRIPT = "javascript"
    GO = "go"
    JAVA = "java"


@dataclass
class Counterexample:
    """A counterexample from verification."""
    variables: dict[str, Any]
    expected_behavior: str
    actual_behavior: str | None = None
    verification_type: str = "formal"  # formal, ai, pattern


@dataclass
class GeneratedTest:
    """A generated test case."""
    name: str
    description: str
    code: str
    language: Language
    framework: TestFramework
    file_name: str
    target_function: str
    counterexample: Counterexample
    setup_code: str | None = None
    teardown_code: str | None = None
    imports: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)


@dataclass
class TestGenerationResult:
    """Result of test generation."""
    tests: list[GeneratedTest]
    coverage_delta: float | None = None
    suggestions: list[str] = field(default_factory=list)


TEST_TEMPLATES = {
    (Language.PYTHON, TestFramework.PYTEST): '''
{imports}


{setup}
def test_{name}():
    """
    {description}
    
    Generated from verification counterexample.
    Counterexample: {counterexample}
    """
    {test_body}
''',
    (Language.PYTHON, TestFramework.UNITTEST): '''
{imports}


class Test{class_name}(unittest.TestCase):
    """Generated tests for {target}."""
    
    {setup}
    
    def test_{name}(self):
        """
        {description}
        
        Generated from verification counterexample.
        """
        {test_body}
''',
    (Language.TYPESCRIPT, TestFramework.JEST): '''
{imports}

describe('{target}', () => {{
    {setup}
    
    test('{description}', () => {{
        {test_body}
    }});
}});
''',
    (Language.TYPESCRIPT, TestFramework.VITEST): '''
{imports}

describe('{target}', () => {{
    {setup}
    
    it('{description}', () => {{
        {test_body}
    }});
}});
''',
    (Language.GO, TestFramework.GO_TEST): '''
package {package}

{imports}

func Test{name}(t *testing.T) {{
    // {description}
    // Generated from verification counterexample
    {test_body}
}}
''',
}


class TestGeneratorAgent(BaseAgent):
    """
    Agent for generating regression tests from verification counterexamples.
    
    Takes verification results with counterexamples and produces idiomatic
    test cases in the appropriate test framework.
    """

    def __init__(
        self,
        config: AgentConfig | None = None,
        default_frameworks: dict[Language, TestFramework] | None = None,
    ) -> None:
        """Initialize test generator agent."""
        super().__init__(config)
        self.default_frameworks = default_frameworks or {
            Language.PYTHON: TestFramework.PYTEST,
            Language.TYPESCRIPT: TestFramework.JEST,
            Language.JAVASCRIPT: TestFramework.JEST,
            Language.GO: TestFramework.GO_TEST,
            Language.JAVA: TestFramework.JUNIT,
        }

    async def analyze(self, code: str, context: dict[str, Any]) -> AgentResult:
        """
        Generate tests from verification results.
        
        Args:
            code: The source code that was verified
            context: Additional context including:
                - verification_results: Results from formal verification
                - file_path: Path to the source file
                - language: Programming language
                - framework: Optional test framework override
                
        Returns:
            AgentResult with generated tests
        """
        try:
            language = self._detect_language(
                context.get("language"),
                context.get("file_path", ""),
            )
            
            framework = context.get("framework") or self.default_frameworks.get(language)
            if not framework:
                return AgentResult(
                    success=False,
                    error=f"No test framework configured for {language}",
                )
            
            verification_results = context.get("verification_results", {})
            counterexamples = self._extract_counterexamples(verification_results)
            
            if not counterexamples:
                return AgentResult(
                    success=True,
                    data={"tests": [], "message": "No counterexamples to generate tests from"},
                )
            
            result = await self.generate_tests(
                code=code,
                counterexamples=counterexamples,
                language=language,
                framework=framework,
                context=context,
            )
            
            return AgentResult(
                success=True,
                data={
                    "tests": [self._test_to_dict(t) for t in result.tests],
                    "coverage_delta": result.coverage_delta,
                    "suggestions": result.suggestions,
                },
            )
            
        except Exception as e:
            logger.error("Test generation failed", error=str(e))
            return AgentResult(success=False, error=str(e))

    def _detect_language(self, hint: str | None, file_path: str) -> Language:
        """Detect programming language from hints or file path."""
        if hint:
            hint_lower = hint.lower()
            for lang in Language:
                if lang.value == hint_lower:
                    return lang
        
        if file_path.endswith(".py"):
            return Language.PYTHON
        elif file_path.endswith((".ts", ".tsx")):
            return Language.TYPESCRIPT
        elif file_path.endswith((".js", ".jsx")):
            return Language.JAVASCRIPT
        elif file_path.endswith(".go"):
            return Language.GO
        elif file_path.endswith(".java"):
            return Language.JAVA
        
        return Language.PYTHON

    def _extract_counterexamples(
        self,
        verification_results: dict[str, Any],
    ) -> list[tuple[str, Counterexample]]:
        """Extract counterexamples from verification results."""
        counterexamples = []
        
        # Handle various result formats
        results = verification_results.get("results", [])
        if isinstance(results, dict):
            results = [results]
        
        for result in results:
            if result.get("satisfiable") and result.get("counterexample"):
                ce = Counterexample(
                    variables=result["counterexample"],
                    expected_behavior=result.get("message", "No violation expected"),
                    verification_type=result.get("verification_type", "formal"),
                )
                target = result.get("target_function", result.get("var_name", "unknown"))
                counterexamples.append((target, ce))
        
        # Also check findings
        for finding in verification_results.get("findings", []):
            if finding.get("counterexample"):
                ce = Counterexample(
                    variables=finding["counterexample"],
                    expected_behavior=finding.get("title", "Expected no issue"),
                    verification_type=finding.get("verification_type", "formal"),
                )
                target = finding.get("target_function", "unknown")
                counterexamples.append((target, ce))
        
        return counterexamples

    async def generate_tests(
        self,
        code: str,
        counterexamples: list[tuple[str, Counterexample]],
        language: Language,
        framework: TestFramework,
        context: dict[str, Any],
    ) -> TestGenerationResult:
        """Generate tests from counterexamples."""
        tests: list[GeneratedTest] = []
        suggestions: list[str] = []
        
        for target_function, counterexample in counterexamples:
            test = await self._generate_single_test(
                code=code,
                target_function=target_function,
                counterexample=counterexample,
                language=language,
                framework=framework,
                context=context,
            )
            if test:
                tests.append(test)
        
        # Add edge case variants
        edge_case_tests = await self._generate_edge_cases(
            tests, language, framework
        )
        tests.extend(edge_case_tests)
        
        if not tests:
            suggestions.append(
                "No tests could be generated - counterexamples may need manual review"
            )
        
        return TestGenerationResult(
            tests=tests,
            suggestions=suggestions,
        )

    async def _generate_single_test(
        self,
        code: str,
        target_function: str,
        counterexample: Counterexample,
        language: Language,
        framework: TestFramework,
        context: dict[str, Any],
    ) -> GeneratedTest | None:
        """Generate a single test from a counterexample."""
        try:
            # Extract function signature from code
            signature = self._extract_function_signature(code, target_function, language)
            
            # Generate test name
            test_name = self._generate_test_name(target_function, counterexample)
            
            # Generate test body based on verification type
            test_body = self._generate_test_body(
                target_function,
                counterexample,
                signature,
                language,
                framework,
            )
            
            # Generate imports
            imports = self._generate_imports(language, framework, context)
            
            # Apply template
            template = TEST_TEMPLATES.get((language, framework))
            if not template:
                template = self._get_fallback_template(language)
            
            test_code = template.format(
                imports="\n".join(imports),
                name=test_name,
                class_name=self._to_class_name(target_function),
                target=target_function,
                description=f"Regression test for {counterexample.expected_behavior}",
                counterexample=str(counterexample.variables),
                test_body=test_body,
                setup="",
                package=context.get("package", "main"),
            )
            
            # Determine test file name
            source_file = context.get("file_path", "unknown")
            test_file = self._generate_test_filename(source_file, language, framework)
            
            return GeneratedTest(
                name=test_name,
                description=f"Regression test: {counterexample.expected_behavior}",
                code=test_code.strip(),
                language=language,
                framework=framework,
                file_name=test_file,
                target_function=target_function,
                counterexample=counterexample,
                imports=imports,
                tags=["generated", "regression", counterexample.verification_type],
            )
            
        except Exception as e:
            logger.warning(
                "Failed to generate test",
                target=target_function,
                error=str(e),
            )
            return None

    def _extract_function_signature(
        self,
        code: str,
        function_name: str,
        language: Language,
    ) -> dict[str, Any]:
        """Extract function signature from source code."""
        signature = {
            "name": function_name,
            "parameters": [],
            "return_type": None,
        }
        
        if language == Language.PYTHON:
            pattern = rf'def\s+{function_name}\s*\(([^)]*)\)\s*(?:->\s*([^:]+))?:'
            match = re.search(pattern, code)
            if match:
                params_str = match.group(1)
                signature["return_type"] = match.group(2).strip() if match.group(2) else None
                signature["parameters"] = self._parse_python_params(params_str)
                
        elif language in (Language.TYPESCRIPT, Language.JAVASCRIPT):
            pattern = rf'(?:function\s+{function_name}|{function_name}\s*=\s*(?:async\s+)?\([^)]*\)\s*=>|{function_name}\s*\([^)]*\))\s*(?::\s*([^{{]+))?'
            match = re.search(pattern, code)
            if match:
                signature["return_type"] = match.group(1).strip() if match.group(1) else None
        
        elif language == Language.GO:
            pattern = rf'func\s+{function_name}\s*\(([^)]*)\)\s*(?:\(([^)]*)\)|(\w+))?'
            match = re.search(pattern, code)
            if match:
                params_str = match.group(1)
                signature["return_type"] = match.group(2) or match.group(3)
        
        return signature

    def _parse_python_params(self, params_str: str) -> list[dict[str, Any]]:
        """Parse Python function parameters."""
        params = []
        if not params_str.strip():
            return params
        
        for param in params_str.split(","):
            param = param.strip()
            if not param or param == "self":
                continue
            
            if ":" in param:
                name, type_hint = param.split(":", 1)
                name = name.strip().split("=")[0].strip()
                type_hint = type_hint.split("=")[0].strip()
                params.append({"name": name, "type": type_hint})
            else:
                name = param.split("=")[0].strip()
                params.append({"name": name, "type": "Any"})
        
        return params

    def _generate_test_name(
        self,
        target_function: str,
        counterexample: Counterexample,
    ) -> str:
        """Generate a descriptive test name."""
        # Create name from counterexample type
        ce_type = counterexample.expected_behavior.lower()
        
        if "overflow" in ce_type:
            suffix = "overflow_detected"
        elif "bounds" in ce_type or "index" in ce_type:
            suffix = "bounds_violation"
        elif "null" in ce_type or "none" in ce_type:
            suffix = "null_safety"
        elif "division" in ce_type or "zero" in ce_type:
            suffix = "division_by_zero"
        else:
            suffix = "regression"
        
        # Sanitize function name
        safe_name = re.sub(r'[^a-zA-Z0-9_]', '_', target_function)
        
        return f"{safe_name}_{suffix}"

    def _generate_test_body(
        self,
        target_function: str,
        counterexample: Counterexample,
        signature: dict[str, Any],
        language: Language,
        framework: TestFramework,
    ) -> str:
        """Generate the test body based on language and framework."""
        variables = counterexample.variables
        expected = counterexample.expected_behavior.lower()
        
        if language == Language.PYTHON:
            return self._generate_python_test_body(
                target_function, variables, expected, framework
            )
        elif language in (Language.TYPESCRIPT, Language.JAVASCRIPT):
            return self._generate_ts_test_body(
                target_function, variables, expected, framework
            )
        elif language == Language.GO:
            return self._generate_go_test_body(
                target_function, variables, expected
            )
        
        return f"# Test for {target_function} with inputs {variables}"

    def _generate_python_test_body(
        self,
        target: str,
        variables: dict[str, Any],
        expected: str,
        framework: TestFramework,
    ) -> str:
        """Generate Python test body."""
        lines = []
        
        # Setup variables
        for name, value in variables.items():
            if isinstance(value, str) and value != "null":
                lines.append(f'    {name} = "{value}"')
            elif value == "null":
                lines.append(f'    {name} = None')
            else:
                lines.append(f'    {name} = {value}')
        
        # Generate assertion based on expected behavior
        if "overflow" in expected:
            lines.append(f'    # Verify overflow is handled')
            lines.append(f'    with pytest.raises((OverflowError, ValueError)):')
            args = ", ".join(variables.keys())
            lines.append(f'        {target}({args})')
        elif "bounds" in expected or "index" in expected:
            lines.append(f'    # Verify bounds checking')
            lines.append(f'    with pytest.raises(IndexError):')
            args = ", ".join(variables.keys())
            lines.append(f'        {target}({args})')
        elif "null" in expected or "none" in expected:
            lines.append(f'    # Verify null handling')
            lines.append(f'    with pytest.raises((TypeError, AttributeError)):')
            args = ", ".join(variables.keys())
            lines.append(f'        {target}({args})')
        elif "division" in expected or "zero" in expected:
            lines.append(f'    # Verify division by zero handling')
            lines.append(f'    with pytest.raises(ZeroDivisionError):')
            args = ", ".join(variables.keys())
            lines.append(f'        {target}({args})')
        else:
            # Generic test
            args = ", ".join(variables.keys())
            lines.append(f'    result = {target}({args})')
            lines.append(f'    # Verify result is valid')
            lines.append(f'    assert result is not None')
        
        return "\n".join(lines)

    def _generate_ts_test_body(
        self,
        target: str,
        variables: dict[str, Any],
        expected: str,
        framework: TestFramework,
    ) -> str:
        """Generate TypeScript/JavaScript test body."""
        lines = []
        
        # Setup variables
        for name, value in variables.items():
            if isinstance(value, str) and value != "null":
                lines.append(f'        const {name} = "{value}";')
            elif value == "null":
                lines.append(f'        const {name} = null;')
            else:
                lines.append(f'        const {name} = {value};')
        
        # Generate assertion
        args = ", ".join(variables.keys())
        
        if any(word in expected for word in ["overflow", "bounds", "null", "zero"]):
            lines.append(f'        expect(() => {target}({args})).toThrow();')
        else:
            lines.append(f'        const result = {target}({args});')
            lines.append(f'        expect(result).toBeDefined();')
        
        return "\n".join(lines)

    def _generate_go_test_body(
        self,
        target: str,
        variables: dict[str, Any],
        expected: str,
    ) -> str:
        """Generate Go test body."""
        lines = []
        
        # Setup variables
        for name, value in variables.items():
            if isinstance(value, str) and value != "null":
                lines.append(f'    {name} := "{value}"')
            elif value == "null":
                lines.append(f'    var {name} interface{{}} = nil')
            else:
                lines.append(f'    {name} := {value}')
        
        # Add test logic (Go uses panic/recover or error returns)
        args = ", ".join(variables.keys())
        lines.append(f'    defer func() {{')
        lines.append(f'        if r := recover(); r == nil {{')
        lines.append(f'            t.Errorf("{target} should have panicked")')
        lines.append(f'        }}')
        lines.append(f'    }}()')
        lines.append(f'    {target}({args})')
        
        return "\n".join(lines)

    def _generate_imports(
        self,
        language: Language,
        framework: TestFramework,
        context: dict[str, Any],
    ) -> list[str]:
        """Generate import statements."""
        imports = []
        
        if language == Language.PYTHON:
            if framework == TestFramework.PYTEST:
                imports.append("import pytest")
            else:
                imports.append("import unittest")
            
            # Add import for the module under test
            module = context.get("module_name")
            if module:
                imports.append(f"from {module} import *")
                
        elif language in (Language.TYPESCRIPT, Language.JAVASCRIPT):
            source_file = context.get("file_path", "")
            if source_file:
                # Convert to relative import
                module_name = source_file.replace(".ts", "").replace(".js", "")
                imports.append(f"import {{ * }} from '{module_name}';")
            
            if framework == TestFramework.VITEST:
                imports.append("import { describe, it, expect } from 'vitest';")
                
        elif language == Language.GO:
            imports.append('import "testing"')
        
        return imports

    def _generate_test_filename(
        self,
        source_file: str,
        language: Language,
        framework: TestFramework,
    ) -> str:
        """Generate appropriate test file name."""
        if language == Language.PYTHON:
            base = source_file.replace(".py", "")
            return f"test_{base.split('/')[-1]}.py"
        elif language in (Language.TYPESCRIPT, Language.JAVASCRIPT):
            ext = ".test.ts" if language == Language.TYPESCRIPT else ".test.js"
            base = source_file.replace(".ts", "").replace(".tsx", "").replace(".js", "").replace(".jsx", "")
            return f"{base.split('/')[-1]}{ext}"
        elif language == Language.GO:
            base = source_file.replace(".go", "")
            return f"{base.split('/')[-1]}_test.go"
        
        return f"test_generated.{language.value}"

    def _to_class_name(self, name: str) -> str:
        """Convert function name to class name."""
        parts = re.split(r'[_\-]', name)
        return "".join(part.capitalize() for part in parts)

    def _get_fallback_template(self, language: Language) -> str:
        """Get a fallback template for unsupported combinations."""
        if language == Language.PYTHON:
            return '''
{imports}

def test_{name}():
    """
    {description}
    Counterexample: {counterexample}
    """
    {test_body}
'''
        return '''
// {description}
// Counterexample: {counterexample}
{test_body}
'''

    async def _generate_edge_cases(
        self,
        base_tests: list[GeneratedTest],
        language: Language,
        framework: TestFramework,
    ) -> list[GeneratedTest]:
        """Generate additional edge case tests from base tests."""
        edge_cases: list[GeneratedTest] = []
        
        for test in base_tests:
            # Generate boundary variants
            variants = self._generate_boundary_variants(test.counterexample)
            
            for i, variant in enumerate(variants):
                variant_test = GeneratedTest(
                    name=f"{test.name}_variant_{i}",
                    description=f"Edge case variant: {variant.expected_behavior}",
                    code="",  # Will be regenerated
                    language=language,
                    framework=framework,
                    file_name=test.file_name,
                    target_function=test.target_function,
                    counterexample=variant,
                    tags=["generated", "edge_case"],
                )
                
                # Regenerate test body for variant
                test_body = self._generate_test_body(
                    test.target_function,
                    variant,
                    {"name": test.target_function, "parameters": []},
                    language,
                    framework,
                )
                
                template = TEST_TEMPLATES.get((language, framework)) or self._get_fallback_template(language)
                variant_test.code = template.format(
                    imports="\n".join(test.imports),
                    name=variant_test.name,
                    class_name=self._to_class_name(test.target_function),
                    target=test.target_function,
                    description=variant_test.description,
                    counterexample=str(variant.variables),
                    test_body=test_body,
                    setup="",
                    package="main",
                ).strip()
                
                edge_cases.append(variant_test)
        
        return edge_cases

    def _generate_boundary_variants(
        self,
        counterexample: Counterexample,
    ) -> list[Counterexample]:
        """Generate boundary value variants from a counterexample."""
        variants = []
        
        for var_name, value in counterexample.variables.items():
            if isinstance(value, int):
                # Generate boundary values
                for boundary in [0, -1, 1, value - 1, value + 1]:
                    if boundary != value:
                        new_vars = counterexample.variables.copy()
                        new_vars[var_name] = boundary
                        variants.append(Counterexample(
                            variables=new_vars,
                            expected_behavior=f"Boundary test: {var_name}={boundary}",
                            verification_type="edge_case",
                        ))
        
        # Limit variants
        return variants[:3]

    def _test_to_dict(self, test: GeneratedTest) -> dict[str, Any]:
        """Convert GeneratedTest to dictionary."""
        return {
            "name": test.name,
            "description": test.description,
            "code": test.code,
            "language": test.language.value,
            "framework": test.framework.value,
            "file_name": test.file_name,
            "target_function": test.target_function,
            "counterexample": {
                "variables": test.counterexample.variables,
                "expected_behavior": test.counterexample.expected_behavior,
            },
            "imports": test.imports,
            "tags": test.tags,
        }
