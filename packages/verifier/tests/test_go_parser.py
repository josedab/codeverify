"""Tests for Go parser."""

import pytest
from codeverify_verifier.parsers import GoParser


@pytest.fixture
def parser() -> GoParser:
    return GoParser()


class TestGoParser:
    """Test suite for Go parser."""

    def test_language(self, parser: GoParser) -> None:
        assert parser.language == "go"

    def test_file_extensions(self, parser: GoParser) -> None:
        assert ".go" in parser.file_extensions

    def test_can_parse(self, parser: GoParser) -> None:
        assert parser.can_parse("main.go")
        assert parser.can_parse("pkg/utils/helper.go")
        assert not parser.can_parse("main.py")
        assert not parser.can_parse("main.java")

    def test_parse_simple_function(self, parser: GoParser) -> None:
        code = """
package main

func add(a int, b int) int {
    return a + b
}
"""
        result = parser.parse(code, "main.go")
        
        assert result.language == "go"
        assert len(result.functions) == 1
        
        func = result.functions[0]
        assert func.name == "add"
        assert len(func.parameters) == 2
        assert func.parameters[0].name == "a"
        assert func.parameters[0].type_hint == "int"
        assert func.return_type == "int"

    def test_parse_function_multiple_params_same_type(self, parser: GoParser) -> None:
        code = """
func multiply(x, y, z int) int {
    return x * y * z
}
"""
        result = parser.parse(code, "math.go")
        
        assert len(result.functions) == 1
        func = result.functions[0]
        assert func.name == "multiply"
        # Parameters with shared types
        assert len(func.parameters) >= 1

    def test_parse_method_with_receiver(self, parser: GoParser) -> None:
        code = """
type Calculator struct {
    value int
}

func (c *Calculator) Add(n int) {
    c.value += n
}

func (c Calculator) GetValue() int {
    return c.value
}
"""
        result = parser.parse(code, "calculator.go")
        
        # Should find the struct
        assert len(result.classes) == 1
        assert result.classes[0].name == "Calculator"
        
        # Should find methods
        methods = [f for f in result.functions if f.decorators]
        assert len(methods) >= 1

    def test_parse_imports(self, parser: GoParser) -> None:
        code = """
package main

import "fmt"

import (
    "os"
    "strings"
    mylog "log"
)

func main() {
    fmt.Println("hello")
}
"""
        result = parser.parse(code, "main.go")
        
        assert len(result.imports) >= 2
        
        # Check for fmt import
        fmt_import = next((i for i in result.imports if i.module == "fmt"), None)
        assert fmt_import is not None
        
        # Check for aliased import
        log_import = next((i for i in result.imports if i.alias == "mylog"), None)
        assert log_import is not None

    def test_parse_struct(self, parser: GoParser) -> None:
        code = """
// User represents a user in the system
type User struct {
    ID       int
    Name     string
    Email    string
    IsActive bool
}
"""
        result = parser.parse(code, "models.go")
        
        assert len(result.classes) == 1
        struct = result.classes[0]
        assert struct.name == "User"
        assert struct.docstring is not None
        assert "User represents" in struct.docstring

    def test_parse_function_with_error_return(self, parser: GoParser) -> None:
        code = """
func divide(a, b float64) (float64, error) {
    if b == 0 {
        return 0, errors.New("division by zero")
    }
    return a / b, nil
}
"""
        result = parser.parse(code, "math.go")
        
        assert len(result.functions) == 1
        func = result.functions[0]
        assert func.name == "divide"
        assert "error" in func.return_type or "float64" in func.return_type

    def test_complexity_calculation(self, parser: GoParser) -> None:
        code = """
func complex(x int) int {
    if x > 0 {
        if x > 10 {
            return 100
        }
        return x * 2
    } else if x < 0 {
        return -x
    }
    
    for i := 0; i < 10; i++ {
        x += i
    }
    
    switch x {
    case 1:
        return 1
    case 2:
        return 2
    default:
        return 0
    }
}
"""
        result = parser.parse(code, "complex.go")
        
        assert len(result.functions) == 1
        func = result.functions[0]
        # Should have high complexity due to multiple branches
        assert func.complexity > 1

    def test_extract_function_calls(self, parser: GoParser) -> None:
        code = """
func process() {
    fmt.Println("starting")
    data := fetchData()
    result := transform(data)
    save(result)
}
"""
        result = parser.parse(code, "process.go")
        
        assert len(result.functions) == 1
        func = result.functions[0]
        
        # Should extract function calls
        assert "fmt.Println" in func.calls or "Println" in str(func.calls)

    def test_parse_global_vars(self, parser: GoParser) -> None:
        code = """
package main

var globalCounter int
var (
    name string
    age  int
)

const MaxSize = 100

func main() {}
"""
        result = parser.parse(code, "main.go")
        
        # Should find global variables
        assert len(result.global_variables) >= 1

    def test_parse_empty_function(self, parser: GoParser) -> None:
        code = """
func empty() {
}
"""
        result = parser.parse(code, "empty.go")
        
        assert len(result.functions) == 1
        assert result.functions[0].name == "empty"

    def test_parse_interface_method(self, parser: GoParser) -> None:
        code = """
type Reader interface {
    Read(p []byte) (n int, err error)
}
"""
        result = parser.parse(code, "interfaces.go")
        
        # Interface should be parsed as a class-like structure
        # (simplified parser may not fully support interfaces)
        assert len(result.errors) == 0
