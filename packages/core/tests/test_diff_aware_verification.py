"""Tests for diff_aware_verification module."""

from __future__ import annotations

from codeverify_core.diff_aware_verification import (
    FunctionScope,
    VerificationScope,
    compute_verification_scope,
    extract_functions,
    parse_unified_diff,
)

SIMPLE_DIFF = """\
diff --git a/src/utils.py b/src/utils.py
--- a/src/utils.py
+++ b/src/utils.py
@@ -10,6 +10,8 @@ def helper():
     return 1
 
 def process(data):
+    if data is None:
+        return []
     return sorted(data)
"""

MULTI_FILE_DIFF = """\
diff --git a/a.py b/a.py
@@ -1,3 +1,4 @@
 def foo():
+    print("added")
     pass

diff --git a/b.py b/b.py
@@ -5,3 +5,4 @@
 def bar():
+    print("added")
     pass
"""

NEW_FILE_DIFF = """\
diff --git a/new_module.py b/new_module.py
new file mode 100644
--- /dev/null
+++ b/new_module.py
@@ -0,0 +1,5 @@
+def brand_new():
+    pass
"""

DELETED_FILE_DIFF = """\
diff --git a/old.py b/old.py
deleted file mode 100644
--- a/old.py
+++ /dev/null
@@ -1,3 +0,0 @@
-def old_func():
-    pass
"""

RENAME_DIFF = """\
diff --git a/old_name.py b/new_name.py
rename from old_name.py
rename to new_name.py
@@ -1,3 +1,3 @@
 def func():
-    return 1
+    return 2
"""


class TestParseUnifiedDiff:
    def test_simple_diff(self):
        files = parse_unified_diff(SIMPLE_DIFF)
        assert len(files) == 1
        assert files[0].old_path == "src/utils.py"
        assert files[0].new_path == "src/utils.py"
        assert len(files[0].hunks) == 1

    def test_multi_file_diff(self):
        files = parse_unified_diff(MULTI_FILE_DIFF)
        assert len(files) == 2
        assert files[0].new_path == "a.py"
        assert files[1].new_path == "b.py"

    def test_new_file_detected(self):
        files = parse_unified_diff(NEW_FILE_DIFF)
        assert len(files) == 1
        assert files[0].is_new is True

    def test_deleted_file_detected(self):
        files = parse_unified_diff(DELETED_FILE_DIFF)
        assert len(files) == 1
        assert files[0].is_deleted is True

    def test_rename_detected(self):
        files = parse_unified_diff(RENAME_DIFF)
        assert len(files) == 1
        assert files[0].is_renamed is True

    def test_hunk_lines_captured(self):
        files = parse_unified_diff(SIMPLE_DIFF)
        hunk = files[0].hunks[0]
        additions = [ln for ln in hunk.lines if ln.startswith("+")]
        assert len(additions) == 2

    def test_empty_diff(self):
        assert parse_unified_diff("") == []


class TestFileDiff:
    def test_changed_lines_new(self):
        files = parse_unified_diff(SIMPLE_DIFF)
        changed = files[0].changed_lines_new
        assert len(changed) == 2
        assert all(isinstance(ln, int) for ln in changed)

    def test_total_additions(self):
        files = parse_unified_diff(SIMPLE_DIFF)
        assert files[0].total_additions == 2

    def test_total_deletions(self):
        files = parse_unified_diff(RENAME_DIFF)
        assert files[0].total_deletions == 1
        assert files[0].total_additions == 1


class TestExtractFunctions:
    def test_python_functions(self):
        src = "def foo():\n    pass\n\ndef bar(x):\n    return x\n"
        funcs = extract_functions(src, "python")
        assert len(funcs) == 2
        assert funcs[0].name == "foo"
        assert funcs[1].name == "bar"

    def test_python_async_function(self):
        src = "async def fetch():\n    pass\n"
        funcs = extract_functions(src, "python")
        assert len(funcs) == 1
        assert funcs[0].name == "fetch"

    def test_go_functions(self):
        src = "func main() {\n    fmt.Println()\n}\n\nfunc (s *Server) Handle() {\n}\n"
        funcs = extract_functions(src, "go")
        assert len(funcs) == 2
        assert funcs[0].name == "main"
        assert funcs[1].name == "Handle"

    def test_typescript_functions(self):
        src = "export async function fetchData() {\n  return 1;\n}\n"
        funcs = extract_functions(src, "typescript")
        assert len(funcs) == 1
        assert funcs[0].name == "fetchData"

    def test_java_methods(self):
        src = "    public void process(String data) {\n        return;\n    }\n"
        funcs = extract_functions(src, "java")
        assert len(funcs) == 1
        assert funcs[0].name == "process"

    def test_unknown_language(self):
        assert extract_functions("some code", "brainfuck") == []

    def test_function_line_ranges(self):
        src = "def a():\n    pass\n\ndef b():\n    x = 1\n    return x\n"
        funcs = extract_functions(src, "python")
        assert funcs[0].line_start == 1
        assert funcs[0].line_end <= funcs[1].line_start


class TestVerificationScope:
    def test_scope_reduction(self):
        scope = VerificationScope(
            affected_functions=[FunctionScope("f", 1, 5)],
            total_source_lines=100,
        )
        assert scope.scope_reduction_pct == 95.0

    def test_zero_source_lines(self):
        scope = VerificationScope(total_source_lines=0)
        assert scope.scope_reduction_pct == 0.0


class TestComputeVerificationScope:
    def test_identifies_affected_functions(self):
        # Build source so `process` starts at line 12, matching SIMPLE_DIFF hunk @@ -10,6 +10,8 @@
        lines = ["# padding\n"] * 11  # lines 1-11
        lines.append("def process(data):\n")  # line 12
        lines.append("    if data is None:\n")  # line 13 (added)
        lines.append("        return []\n")  # line 14 (added)
        lines.append("    return sorted(data)\n")  # line 15
        source = "".join(lines)

        scope = compute_verification_scope(
            diff_text=SIMPLE_DIFF,
            file_sources={"src/utils.py": source},
            file_languages={"src/utils.py": "python"},
        )
        names = [f.name for f in scope.affected_functions]
        assert "process" in names

    def test_new_files_tracked(self):
        scope = compute_verification_scope(
            diff_text=NEW_FILE_DIFF,
            file_sources={},
            file_languages={},
        )
        assert "new_module.py" in scope.new_files

    def test_deleted_files_tracked(self):
        scope = compute_verification_scope(
            diff_text=DELETED_FILE_DIFF,
            file_sources={},
            file_languages={},
        )
        assert "old.py" in scope.deleted_files

    def test_scope_reduction_on_large_file(self):
        lines = ["# line\n"] * 100
        lines[49] = "def target():\n"
        lines[50] = "    return 1\n"
        source = "".join(lines)

        diff = """\
diff --git a/big.py b/big.py
@@ -50,1 +50,2 @@
 def target():
+    x = 2
     return 1
"""
        scope = compute_verification_scope(
            diff_text=diff,
            file_sources={"big.py": source},
            file_languages={"big.py": "python"},
        )
        assert scope.scope_reduction_pct > 90.0

    def test_empty_diff(self):
        scope = compute_verification_scope("", {}, {})
        assert len(scope.affected_functions) == 0
