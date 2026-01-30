"""Z3 SMT Solver based verifier."""

import time
from typing import Any

import structlog
from z3 import (
    And,
    ArithRef,
    BitVec,
    BitVecVal,
    Bool,
    BoolRef,
    Int,
    Not,
    Or,
    Solver,
    parse_smt2_string,
    sat,
    unknown,
    unsat,
)

logger = structlog.get_logger()


class Z3Verifier:
    """
    Formal verification engine using Z3 SMT solver.

    This class provides methods to verify various code properties
    using SMT (Satisfiability Modulo Theories) solving.
    """

    def __init__(self, timeout_ms: int = 60000) -> None:
        """
        Initialize the Z3 verifier.

        Args:
            timeout_ms: Timeout for SMT solving in milliseconds (default: 60s)
        """
        self.timeout_ms = timeout_ms

    def verify_condition(self, condition: str, description: str = "") -> dict[str, Any]:
        """
        Verify a condition expressed in SMT-LIB format.

        Args:
            condition: SMT-LIB formatted verification condition
            description: Human-readable description of what's being verified

        Returns:
            Dictionary with verification result:
            - satisfiable: True if counterexample found, False if proven, None if timeout
            - counterexample: Values that violate the condition (if satisfiable)
            - proof_time_ms: Time taken for verification
        """
        solver = Solver()
        solver.set("timeout", self.timeout_ms)

        start_time = time.time()

        try:
            # Parse SMT-LIB formula
            solver.from_string(condition)

            result = solver.check()
            elapsed_ms = (time.time() - start_time) * 1000

            if result == sat:
                model = solver.model()
                counterexample = {str(d): str(model[d]) for d in model.decls()}
                logger.info(
                    "Verification found counterexample",
                    description=description,
                    elapsed_ms=elapsed_ms,
                )
                return {
                    "satisfiable": True,
                    "counterexample": counterexample,
                    "proof_time_ms": elapsed_ms,
                }
            elif result == unsat:
                logger.info(
                    "Verification proven correct",
                    description=description,
                    elapsed_ms=elapsed_ms,
                )
                return {
                    "satisfiable": False,
                    "counterexample": None,
                    "proof_time_ms": elapsed_ms,
                }
            else:
                logger.warning(
                    "Verification timeout or unknown",
                    description=description,
                    elapsed_ms=elapsed_ms,
                )
                return {
                    "satisfiable": None,
                    "counterexample": None,
                    "proof_time_ms": elapsed_ms,
                    "reason": "timeout_or_unknown",
                }

        except Exception as e:
            elapsed_ms = (time.time() - start_time) * 1000
            logger.error("Verification error", error=str(e), description=description)
            return {
                "satisfiable": None,
                "counterexample": None,
                "proof_time_ms": elapsed_ms,
                "error": str(e),
            }

    def check_integer_overflow(
        self,
        var_name: str,
        operation: str,
        operand1_range: tuple[int, int],
        operand2_range: tuple[int, int] | None = None,
        bit_width: int = 32,
    ) -> dict[str, Any]:
        """
        Check if an integer operation can overflow.

        Args:
            var_name: Name of the variable for reporting
            operation: Type of operation ('add', 'mul', 'sub')
            operand1_range: (min, max) range of first operand
            operand2_range: (min, max) range of second operand (if applicable)
            bit_width: Bit width of integers (default: 32 for int32)

        Returns:
            Verification result with potential overflow values
        """
        solver = Solver()
        solver.set("timeout", self.timeout_ms)

        start_time = time.time()

        # Create bit-vector variables for precise overflow checking
        max_val = (1 << (bit_width - 1)) - 1  # Max signed value
        min_val = -(1 << (bit_width - 1))  # Min signed value

        a = Int("a")
        b = Int("b") if operand2_range else None

        # Constrain operand ranges
        constraints = [a >= operand1_range[0], a <= operand1_range[1]]
        if b is not None and operand2_range:
            constraints.extend([b >= operand2_range[0], b <= operand2_range[1]])

        # Define result based on operation
        if operation == "add" and b is not None:
            result = a + b
        elif operation == "mul" and b is not None:
            result = a * b
        elif operation == "sub" and b is not None:
            result = a - b
        else:
            return {
                "satisfiable": None,
                "error": f"Unsupported operation: {operation}",
                "proof_time_ms": 0,
            }

        # Check if result can be outside valid range
        overflow_condition = Or(result > max_val, result < min_val)

        solver.add(*constraints)
        solver.add(overflow_condition)

        check_result = solver.check()
        elapsed_ms = (time.time() - start_time) * 1000

        if check_result == sat:
            model = solver.model()
            counterexample = {
                "a": model[a].as_long() if model[a] else None,
            }
            if b is not None:
                counterexample["b"] = model[b].as_long() if model[b] else None
            counterexample["result"] = "overflow"

            logger.info(
                "Integer overflow possible",
                var_name=var_name,
                operation=operation,
                counterexample=counterexample,
            )
            return {
                "satisfiable": True,
                "counterexample": counterexample,
                "proof_time_ms": elapsed_ms,
                "message": f"Integer overflow possible in {var_name}",
            }
        elif check_result == unsat:
            logger.info(
                "No integer overflow possible",
                var_name=var_name,
                operation=operation,
            )
            return {
                "satisfiable": False,
                "counterexample": None,
                "proof_time_ms": elapsed_ms,
                "message": f"No overflow possible in {var_name}",
            }
        else:
            return {
                "satisfiable": None,
                "counterexample": None,
                "proof_time_ms": elapsed_ms,
                "reason": "timeout_or_unknown",
            }

    def check_array_bounds(
        self,
        index_var: str,
        index_range: tuple[int, int] | None,
        array_length: int,
    ) -> dict[str, Any]:
        """
        Check if array access can go out of bounds.

        Args:
            index_var: Name of the index variable
            index_range: (min, max) possible values of index, None if unknown
            array_length: Length of the array

        Returns:
            Verification result
        """
        solver = Solver()
        solver.set("timeout", self.timeout_ms)

        start_time = time.time()

        idx = Int("index")

        # Add constraints on index if known
        if index_range:
            solver.add(idx >= index_range[0])
            solver.add(idx <= index_range[1])

        # Check for out of bounds: index < 0 OR index >= length
        out_of_bounds = Or(idx < 0, idx >= array_length)
        solver.add(out_of_bounds)

        result = solver.check()
        elapsed_ms = (time.time() - start_time) * 1000

        if result == sat:
            model = solver.model()
            bad_index = model[idx].as_long() if model[idx] else "unknown"
            return {
                "satisfiable": True,
                "counterexample": {"index": bad_index, "array_length": array_length},
                "proof_time_ms": elapsed_ms,
                "message": f"Array bounds violation possible: index={bad_index}",
            }
        elif result == unsat:
            return {
                "satisfiable": False,
                "counterexample": None,
                "proof_time_ms": elapsed_ms,
                "message": "Array access is always within bounds",
            }
        else:
            return {
                "satisfiable": None,
                "counterexample": None,
                "proof_time_ms": elapsed_ms,
                "reason": "timeout_or_unknown",
            }

    def check_division_by_zero(
        self,
        divisor_var: str,
        divisor_range: tuple[int, int] | None,
    ) -> dict[str, Any]:
        """
        Check if division by zero is possible.

        Args:
            divisor_var: Name of the divisor variable
            divisor_range: (min, max) possible values, None if unknown

        Returns:
            Verification result
        """
        solver = Solver()
        solver.set("timeout", self.timeout_ms)

        start_time = time.time()

        divisor = Int("divisor")

        if divisor_range:
            solver.add(divisor >= divisor_range[0])
            solver.add(divisor <= divisor_range[1])

        # Check if divisor can be zero
        solver.add(divisor == 0)

        result = solver.check()
        elapsed_ms = (time.time() - start_time) * 1000

        if result == sat:
            return {
                "satisfiable": True,
                "counterexample": {"divisor": 0},
                "proof_time_ms": elapsed_ms,
                "message": f"Division by zero possible for {divisor_var}",
            }
        elif result == unsat:
            return {
                "satisfiable": False,
                "counterexample": None,
                "proof_time_ms": elapsed_ms,
                "message": "Division by zero not possible",
            }
        else:
            return {
                "satisfiable": None,
                "counterexample": None,
                "proof_time_ms": elapsed_ms,
                "reason": "timeout_or_unknown",
            }

    def check_null_dereference(
        self,
        var_name: str,
        can_be_null: bool,
        null_check_exists: bool,
    ) -> dict[str, Any]:
        """
        Check if null dereference is possible.

        Args:
            var_name: Name of the variable
            can_be_null: Whether the variable can potentially be null
            null_check_exists: Whether there's a null check before dereference

        Returns:
            Verification result
        """
        start_time = time.time()

        if not can_be_null:
            return {
                "satisfiable": False,
                "counterexample": None,
                "proof_time_ms": (time.time() - start_time) * 1000,
                "message": f"{var_name} cannot be null",
            }

        if null_check_exists:
            return {
                "satisfiable": False,
                "counterexample": None,
                "proof_time_ms": (time.time() - start_time) * 1000,
                "message": f"Null check exists for {var_name}",
            }

        # Variable can be null and no check exists
        return {
            "satisfiable": True,
            "counterexample": {var_name: "null"},
            "proof_time_ms": (time.time() - start_time) * 1000,
            "message": f"Null dereference possible for {var_name}",
        }
