"""Tests for enhanced Go and Java language support."""

import pytest

from codeverify_core.language_support import (
    AdvancedLanguageAnalyzer,
    LanguageParser,
    SupportedLanguage,
    Z3ConstraintGenerator,
    _bit_width_for_type,
    get_language_registry,
    reset_language_registry,
)


class TestEnhancedGoRules:
    @pytest.fixture(autouse=True)
    def _reset(self):
        reset_language_registry()
        yield
        reset_language_registry()

    def test_go_rules_registered(self):
        registry = get_language_registry()
        rules = registry.get_rules(SupportedLanguage.GO)
        rule_ids = {r.id for r in rules}
        assert "go_error_ignored" in rule_ids
        assert "go_nil_map_write" in rule_ids
        assert "go_context_missing" in rule_ids
        assert "go_mutex_copy" in rule_ids
        assert len(rules) >= 8

    def test_go_nil_map_detection(self):
        analyzer = AdvancedLanguageAnalyzer()
        code = 'var m map[string]int\nm["key"] = 1'
        findings = analyzer.analyze(code, SupportedLanguage.GO)
        rule_ids = {f["rule_id"] for f in findings}
        assert "go_nil_map_write" in rule_ids

    def test_go_goroutine_no_sync(self):
        analyzer = AdvancedLanguageAnalyzer()
        code = """
func doWork() {
    go processItem(item)
}
"""
        findings = analyzer.analyze(code, SupportedLanguage.GO)
        rule_ids = {f["rule_id"] for f in findings}
        assert "go_goroutine_no_sync" in rule_ids

    def test_go_goroutine_with_context_ok(self):
        analyzer = AdvancedLanguageAnalyzer()
        code = """
func doWork(ctx context.Context) {
    go processItem(ctx, item)
}
"""
        findings = analyzer.analyze(code, SupportedLanguage.GO)
        rule_ids = {f["rule_id"] for f in findings}
        assert "go_goroutine_no_sync" not in rule_ids


class TestEnhancedJavaRules:
    @pytest.fixture(autouse=True)
    def _reset(self):
        reset_language_registry()
        yield
        reset_language_registry()

    def test_java_rules_registered(self):
        registry = get_language_registry()
        rules = registry.get_rules(SupportedLanguage.JAVA)
        rule_ids = {r.id for r in rules}
        assert "java_empty_catch" in rule_ids
        assert "java_optional_get" in rule_ids
        assert "java_string_equals" in rule_ids
        assert "java_concurrent_modification" in rule_ids
        assert len(rules) >= 9

    def test_java_string_equals_detection(self):
        analyzer = AdvancedLanguageAnalyzer()
        code = 'if (name == "admin") { grant(); }'
        findings = analyzer.analyze(code, SupportedLanguage.JAVA)
        rule_ids = {f["rule_id"] for f in findings}
        assert "java_string_equals" in rule_ids

    def test_java_raw_collection_init(self):
        analyzer = AdvancedLanguageAnalyzer()
        code = "List items = new ArrayList();"
        findings = analyzer.analyze(code, SupportedLanguage.JAVA)
        rule_ids = {f["rule_id"] for f in findings}
        assert "java_raw_collection_init" in rule_ids

    def test_java_optional_returns_null(self):
        analyzer = AdvancedLanguageAnalyzer()
        code = """
    Optional<String> findUser(int id) {
        if (id < 0) return null;
        return Optional.of("user");
    }}
"""
        findings = analyzer.analyze(code, SupportedLanguage.JAVA)
        rule_ids = {f["rule_id"] for f in findings}
        assert "java_optional_returns_null" in rule_ids


class TestLanguageParser:
    def test_parse_go_functions(self):
        parser = LanguageParser()
        code = """
func HandleRequest(ctx context.Context, req *Request) (*Response, error) {
    return nil, nil
}

func (s *Server) Start() error {
    return nil
}
"""
        funcs = parser.parse_functions(code, SupportedLanguage.GO)
        names = [f["name"] for f in funcs]
        assert "HandleRequest" in names
        assert "Start" in names

    def test_parse_java_functions(self):
        parser = LanguageParser()
        code = """
public static <T> List<T> filter(List<T> items, Predicate<T> pred) {
    return items.stream().filter(pred).collect(toList());
}

private void processOrder(Order order) throws IOException {
    // ...
}
"""
        funcs = parser.parse_functions(code, SupportedLanguage.JAVA)
        names = [f["name"] for f in funcs]
        assert "filter" in names
        assert "processOrder" in names


class TestZ3ConstraintGeneration:
    def test_go_error_handling_constraint(self):
        gen = Z3ConstraintGenerator()
        code = """
result, err := doWork()
if err != nil {
    return err
}
"""
        smt = gen.generate_error_handling_check(code, SupportedLanguage.GO)
        assert "err_checked" in smt
        assert "true" in smt

    def test_go_unhandled_error_constraint(self):
        gen = Z3ConstraintGenerator()
        code = "result, err := doWork()\nfmt.Println(result)"
        smt = gen.generate_error_handling_check(code, SupportedLanguage.GO)
        assert "UNHANDLED" in smt

    def test_java_exception_handling_constraint(self):
        gen = Z3ConstraintGenerator()
        code = "try { conn.open(); } catch (IOException e) { log.error(e); }"
        smt = gen.generate_error_handling_check(code, SupportedLanguage.JAVA)
        assert "exceptions_handled" in smt
        assert "true" in smt

    def test_overflow_check_go_int64(self):
        gen = Z3ConstraintGenerator()
        smt = gen.generate_overflow_check("counter", 64, SupportedLanguage.GO)
        assert "64-bit" in smt
        assert "counter" in smt


class TestAdvancedConstraintGeneration:
    def test_generate_constraints_for_go_func(self):
        analyzer = AdvancedLanguageAnalyzer()
        code = """
func ProcessData(data []byte, count int32) error {
    return nil
}
"""
        constraints = analyzer.generate_constraints(code, SupportedLanguage.GO)
        types = {c["type"] for c in constraints}
        assert "null_safety" in types

    def test_generate_constraints_for_java_func(self):
        analyzer = AdvancedLanguageAnalyzer()
        code = """
public int computeSum(int a, int b) {
    return a + b;
}
"""
        constraints = analyzer.generate_constraints(code, SupportedLanguage.JAVA)
        assert len(constraints) > 0


class TestBitWidthMapping:
    def test_known_types(self):
        assert _bit_width_for_type("int8", SupportedLanguage.GO) == 8
        assert _bit_width_for_type("int64", SupportedLanguage.GO) == 64
        assert _bit_width_for_type("long", SupportedLanguage.JAVA) == 64
        assert _bit_width_for_type("short", SupportedLanguage.JAVA) == 16

    def test_unknown_type_defaults_32(self):
        assert _bit_width_for_type("BigInteger", SupportedLanguage.JAVA) == 32


class TestPythonAndTypeScriptAnalysis:
    def test_python_mutable_default(self):
        analyzer = AdvancedLanguageAnalyzer()
        code = "def process(items=[]):\n    items.append(1)"
        findings = analyzer.analyze(code, SupportedLanguage.PYTHON)
        rule_ids = {f["rule_id"] for f in findings}
        assert "python_mutable_default" in rule_ids

    def test_typescript_any_type(self):
        analyzer = AdvancedLanguageAnalyzer()
        code = "function process(data: any): any { return data; }"
        findings = analyzer.analyze(code, SupportedLanguage.TYPESCRIPT)
        rule_ids = {f["rule_id"] for f in findings}
        assert "ts_any_type" in rule_ids
