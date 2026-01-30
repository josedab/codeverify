# @codeverify/z3-mcp

Model Context Protocol (MCP) server for Z3 SMT solver.

This package allows AI agents to use Z3 formal verification as a tool through the MCP standard.

## Installation

```bash
pip install codeverify-z3-mcp

# Or from source
pip install -e packages/z3-mcp
```

## Overview

The Z3 MCP server exposes Z3 verification capabilities as MCP tools:

- `verify_null_safety` - Check for null dereferences
- `verify_array_bounds` - Check for out-of-bounds access
- `verify_integer_overflow` - Check for arithmetic overflow
- `verify_division_by_zero` - Check for divide-by-zero
- `verify_custom` - Run custom Z3 constraints

## Quick Start

### Starting the Server

```bash
# Start MCP server
python -m z3_mcp.server

# Or with specific port
python -m z3_mcp.server --port 8765
```

### Using with Claude/Copilot

Configure your MCP client to connect to the server:

```json
{
  "mcpServers": {
    "z3": {
      "command": "python",
      "args": ["-m", "z3_mcp.server"]
    }
  }
}
```

## Available Tools

### `verify_null_safety`

Check for potential null dereferences:

```json
{
  "tool": "verify_null_safety",
  "arguments": {
    "code": "def get_name(user): return user.name",
    "language": "python"
  }
}
```

**Response:**
```json
{
  "status": "vulnerable",
  "finding": {
    "title": "Potential null dereference",
    "line": 1,
    "description": "user could be None when accessing .name"
  },
  "counterexample": {"user": "None"}
}
```

### `verify_array_bounds`

Check for out-of-bounds array access:

```json
{
  "tool": "verify_array_bounds",
  "arguments": {
    "code": "def get_item(items, i): return items[i]",
    "language": "python"
  }
}
```

### `verify_integer_overflow`

Check for arithmetic overflow:

```json
{
  "tool": "verify_integer_overflow",
  "arguments": {
    "code": "def multiply(a: int, b: int) -> int: return a * b",
    "language": "python",
    "bit_width": 64
  }
}
```

### `verify_division_by_zero`

Check for divide-by-zero:

```json
{
  "tool": "verify_division_by_zero",
  "arguments": {
    "code": "def divide(a, b): return a / b",
    "language": "python"
  }
}
```

### `verify_custom`

Run custom Z3 constraints:

```json
{
  "tool": "verify_custom",
  "arguments": {
    "constraints": [
      "x > 0",
      "y > 0",
      "x + y < 0"
    ],
    "variables": {
      "x": "Int",
      "y": "Int"
    }
  }
}
```

## Templates

The server includes built-in verification templates:

```python
from z3_mcp.server import Z3MCPServer

server = Z3MCPServer()

# List available templates
templates = server.list_templates()
# ["null_check", "bounds_check", "overflow_check", "div_zero_check", ...]

# Use template
result = server.apply_template(
    template="bounds_check",
    code=code,
    parameters={"array_name": "items", "index_name": "i"},
)
```

## Python API

Use the server programmatically:

```python
from z3_mcp import Z3MCPServer

server = Z3MCPServer()

# Direct verification
result = server.verify(
    code="def f(x): return 1/x",
    checks=["division_by_zero"],
)

# Custom constraints
result = server.verify_custom(
    constraints=["x > 0", "x < 0"],
    variables={"x": "Int"},
)
print(result.status)  # "unsat" - constraints are contradictory
```

## Configuration

```python
from z3_mcp import Z3MCPConfig

config = Z3MCPConfig(
    timeout_seconds=30,
    max_path_depth=10,
    bit_width=64,
    enable_templates=True,
)

server = Z3MCPServer(config)
```

## Development

```bash
# Install dev dependencies
pip install -e "packages/z3-mcp[dev]"

# Run tests
pytest packages/z3-mcp/tests -v

# Run server locally
python -m z3_mcp.server --debug
```

## License

MIT License - This package is open source and can be used independently of CodeVerify.

## Further Reading

- [Model Context Protocol](https://modelcontextprotocol.io/)
- [Z3 Documentation](https://z3prover.github.io/api/html/index.html)
- [CodeVerify Verification Guide](../../docs/verification.md)
