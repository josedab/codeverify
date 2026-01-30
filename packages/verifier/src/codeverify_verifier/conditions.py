"""Verification condition generators for different code patterns."""

from typing import Any


def generate_null_check(var_name: str, var_type: str) -> str:
    """
    Generate SMT-LIB formula to check for null dereference.

    Args:
        var_name: Name of the variable being checked
        var_type: Type of the variable

    Returns:
        SMT-LIB formatted formula
    """
    return f"""
; Null check for {var_name} of type {var_type}
(declare-const {var_name}_is_null Bool)
(assert {var_name}_is_null)
(check-sat)
"""


def generate_bounds_check(
    index_expr: str,
    array_name: str,
    array_length: int | str,
) -> str:
    """
    Generate SMT-LIB formula to check array bounds.

    Args:
        index_expr: Expression used as array index
        array_name: Name of the array
        array_length: Length of the array (can be variable name or int)

    Returns:
        SMT-LIB formatted formula
    """
    length = str(array_length)
    return f"""
; Bounds check for {array_name}[{index_expr}]
(declare-const index Int)
(declare-const length Int)
(assert (= length {length}))
(assert (or (< index 0) (>= index length)))
(check-sat)
"""


def generate_overflow_check(
    operation: str,
    operand1: str,
    operand2: str,
    result_type: str = "int32",
) -> str:
    """
    Generate SMT-LIB formula to check for integer overflow.

    Args:
        operation: The operation ('+', '-', '*')
        operand1: First operand expression
        operand2: Second operand expression
        result_type: Type of the result (int32, int64, etc.)

    Returns:
        SMT-LIB formatted formula
    """
    # Define bounds based on type
    bounds = {
        "int8": (-128, 127),
        "int16": (-32768, 32767),
        "int32": (-2147483648, 2147483647),
        "int64": (-9223372036854775808, 9223372036854775807),
        "uint8": (0, 255),
        "uint16": (0, 65535),
        "uint32": (0, 4294967295),
    }

    min_val, max_val = bounds.get(result_type, bounds["int32"])

    op_map = {"+": "+", "-": "-", "*": "*"}
    smt_op = op_map.get(operation, "+")

    return f"""
; Overflow check for {operand1} {operation} {operand2}
(declare-const a Int)
(declare-const b Int)
(define-fun result () Int ({smt_op} a b))
(assert (or (< result {min_val}) (> result {max_val})))
(check-sat)
"""


def generate_division_check(dividend: str, divisor: str) -> str:
    """
    Generate SMT-LIB formula to check for division by zero.

    Args:
        dividend: Dividend expression
        divisor: Divisor expression

    Returns:
        SMT-LIB formatted formula
    """
    return f"""
; Division by zero check for {dividend} / {divisor}
(declare-const divisor Int)
(assert (= divisor 0))
(check-sat)
"""


def generate_loop_termination_check(
    loop_var: str,
    init_value: int,
    condition: str,
    increment: str,
) -> str:
    """
    Generate SMT-LIB formula to check loop termination.

    This is a simplified check that works for basic counting loops.

    Args:
        loop_var: Name of loop variable
        init_value: Initial value
        condition: Loop condition (e.g., "< 10")
        increment: How variable changes (e.g., "+1", "-1")

    Returns:
        SMT-LIB formatted formula
    """
    return f"""
; Loop termination check for {loop_var}
; Initial: {loop_var} = {init_value}
; Condition: {loop_var} {condition}
; Update: {loop_var} {increment}
(declare-const i Int)
(declare-const bound Int)
(assert (>= bound 0))
; Check if loop can run forever
(assert (forall ((n Int)) (=> (>= n 0) (< (+ {init_value} n) bound))))
(check-sat)
"""


def generate_precondition_check(
    function_name: str,
    preconditions: list[dict[str, Any]],
) -> str:
    """
    Generate SMT-LIB formula to check function preconditions.

    Args:
        function_name: Name of the function
        preconditions: List of precondition specifications

    Returns:
        SMT-LIB formatted formula
    """
    declarations = []
    assertions = []

    for i, pre in enumerate(preconditions):
        param = pre.get("param", f"param_{i}")
        condition = pre.get("condition", "true")
        declarations.append(f"(declare-const {param} Int)")
        assertions.append(f"(assert (not {condition}))")

    decl_str = "\n".join(declarations)
    assert_str = "\n".join(assertions)

    return f"""
; Precondition check for {function_name}
{decl_str}
{assert_str}
(check-sat)
"""
