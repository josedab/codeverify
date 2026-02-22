"""Python API backend — part of the monorepo example.

CodeVerify's polyglot bridge verifies that this API's contract
matches what the TypeScript frontend expects.
"""


def get_user(user_id: int) -> dict:
    """Returns a user dict. The frontend expects {name: string, age: number}."""
    # BUG: Returns 'years' instead of 'age' — contract mismatch
    # CodeVerify finds: cross-language contract violation
    return {"name": "Alice", "years": 30}


def calculate_discount(price: float, percent: float) -> float:
    """Calculate discount amount."""
    # BUG: Division by zero possible
    # CodeVerify finds: division by zero when percent is 100
    return price / (100 - percent)
