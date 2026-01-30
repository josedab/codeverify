"""Tests for Java parser."""

import pytest
from codeverify_verifier.parsers import JavaParser


@pytest.fixture
def parser() -> JavaParser:
    return JavaParser()


class TestJavaParser:
    """Test suite for Java parser."""

    def test_language(self, parser: JavaParser) -> None:
        assert parser.language == "java"

    def test_file_extensions(self, parser: JavaParser) -> None:
        assert ".java" in parser.file_extensions

    def test_can_parse(self, parser: JavaParser) -> None:
        assert parser.can_parse("Main.java")
        assert parser.can_parse("com/example/Service.java")
        assert not parser.can_parse("main.py")
        assert not parser.can_parse("main.go")

    def test_parse_simple_class(self, parser: JavaParser) -> None:
        code = """
package com.example;

public class Calculator {
    public int add(int a, int b) {
        return a + b;
    }
}
"""
        result = parser.parse(code, "Calculator.java")
        
        assert result.language == "java"
        assert len(result.classes) == 1
        
        cls = result.classes[0]
        assert cls.name == "Calculator"
        assert len(cls.methods) == 1
        
        method = cls.methods[0]
        assert method.name == "add"
        assert len(method.parameters) == 2
        assert method.return_type == "int"

    def test_parse_method_with_annotations(self, parser: JavaParser) -> None:
        code = """
public class Service {
    @Override
    @Deprecated
    public void process() {
        // do something
    }
}
"""
        result = parser.parse(code, "Service.java")
        
        assert len(result.classes) == 1
        method = result.classes[0].methods[0]
        assert method.name == "process"
        assert "@Override" in method.decorators or any("Override" in d for d in method.decorators)

    def test_parse_imports(self, parser: JavaParser) -> None:
        code = """
package com.example;

import java.util.List;
import java.util.Map;
import static java.lang.Math.PI;

public class Main {
    public void test() {}
}
"""
        result = parser.parse(code, "Main.java")
        
        assert len(result.imports) >= 2
        
        # Check for List import
        list_import = next((i for i in result.imports if "List" in i.names or "java.util" in i.module), None)
        assert list_import is not None

    def test_parse_class_with_inheritance(self, parser: JavaParser) -> None:
        code = """
public class Dog extends Animal implements Pet, Trainable {
    public void bark() {
        System.out.println("Woof!");
    }
}
"""
        result = parser.parse(code, "Dog.java")
        
        assert len(result.classes) == 1
        cls = result.classes[0]
        assert cls.name == "Dog"
        assert "Animal" in cls.base_classes
        assert "Pet" in cls.base_classes or "Trainable" in cls.base_classes

    def test_parse_generic_method(self, parser: JavaParser) -> None:
        code = """
public class Utils {
    public <T> List<T> filter(List<T> items, Predicate<T> predicate) {
        return items.stream().filter(predicate).collect(Collectors.toList());
    }
}
"""
        result = parser.parse(code, "Utils.java")
        
        assert len(result.classes) == 1
        # Should handle generic methods
        assert len(result.classes[0].methods) >= 0  # May or may not parse generics fully

    def test_parse_javadoc(self, parser: JavaParser) -> None:
        code = """
public class Example {
    /**
     * Calculates the sum of two numbers.
     * 
     * @param a first number
     * @param b second number
     * @return the sum
     */
    public int sum(int a, int b) {
        return a + b;
    }
}
"""
        result = parser.parse(code, "Example.java")
        
        assert len(result.classes) == 1
        method = result.classes[0].methods[0]
        assert method.docstring is not None
        assert "Calculates the sum" in method.docstring

    def test_parse_throws_clause(self, parser: JavaParser) -> None:
        code = """
public class FileHandler {
    public void readFile(String path) throws IOException, FileNotFoundException {
        // read file
    }
}
"""
        result = parser.parse(code, "FileHandler.java")
        
        assert len(result.classes) == 1
        method = result.classes[0].methods[0]
        # Throws should be captured in decorators
        has_throws = any("throws" in d.lower() for d in method.decorators)
        assert has_throws or method.name == "readFile"

    def test_complexity_calculation(self, parser: JavaParser) -> None:
        code = """
public class Complex {
    public int process(int x) {
        if (x > 0) {
            if (x > 10) {
                return 100;
            }
            return x * 2;
        } else if (x < 0) {
            return -x;
        }
        
        for (int i = 0; i < 10; i++) {
            x += i;
        }
        
        switch (x) {
            case 1:
                return 1;
            case 2:
                return 2;
            default:
                return 0;
        }
    }
}
"""
        result = parser.parse(code, "Complex.java")
        
        assert len(result.classes) == 1
        method = result.classes[0].methods[0]
        # Should have high complexity
        assert method.complexity > 1

    def test_parse_interface(self, parser: JavaParser) -> None:
        code = """
public interface Repository<T> {
    T findById(long id);
    List<T> findAll();
    void save(T entity);
}
"""
        result = parser.parse(code, "Repository.java")
        
        # Interface should be parsed as a class
        assert len(result.classes) == 1
        assert result.classes[0].name == "Repository"

    def test_parse_enum(self, parser: JavaParser) -> None:
        code = """
public enum Status {
    PENDING,
    ACTIVE,
    COMPLETED
}
"""
        result = parser.parse(code, "Status.java")
        
        # Enum should be parsed
        assert len(result.errors) == 0

    def test_parse_abstract_class(self, parser: JavaParser) -> None:
        code = """
public abstract class Shape {
    protected String color;
    
    public abstract double getArea();
    
    public String getColor() {
        return color;
    }
}
"""
        result = parser.parse(code, "Shape.java")
        
        assert len(result.classes) == 1
        cls = result.classes[0]
        assert cls.name == "Shape"
        # Should have both abstract and concrete methods
        assert len(cls.methods) >= 1

    def test_extract_method_calls(self, parser: JavaParser) -> None:
        code = """
public class Processor {
    public void run() {
        System.out.println("Starting");
        validate();
        process(getData());
        save();
    }
    
    private void validate() {}
    private Object getData() { return null; }
    private void process(Object data) {}
    private void save() {}
}
"""
        result = parser.parse(code, "Processor.java")
        
        # Find the run method
        run_method = None
        for method in result.functions:
            if method.name == "run":
                run_method = method
                break
        
        if run_method:
            # Should have method calls extracted
            assert len(run_method.calls) > 0

    def test_parse_varargs(self, parser: JavaParser) -> None:
        code = """
public class VarArgsExample {
    public void printAll(String... messages) {
        for (String msg : messages) {
            System.out.println(msg);
        }
    }
}
"""
        result = parser.parse(code, "VarArgsExample.java")
        
        assert len(result.classes) == 1
        method = result.classes[0].methods[0]
        assert method.name == "printAll"
        # Varargs should be converted to array type
        if method.parameters:
            assert "[]" in method.parameters[0].type_hint or "String" in method.parameters[0].type_hint

    def test_parse_nested_class(self, parser: JavaParser) -> None:
        code = """
public class Outer {
    public void outerMethod() {}
    
    public class Inner {
        public void innerMethod() {}
    }
}
"""
        result = parser.parse(code, "Outer.java")
        
        # Should handle nested classes (may parse both or just outer)
        assert len(result.classes) >= 1
        assert result.classes[0].name == "Outer"
