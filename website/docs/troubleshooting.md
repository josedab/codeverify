---
sidebar_position: 1
---

# Troubleshooting

Common issues and solutions.

## Analysis Issues

### Analysis Times Out

Increase timeout in configuration:

```yaml
verification:
  timeout: 60  # seconds
```

### False Positives

Suppress with inline comments:

```python
result = a / b  # codeverify: ignore division_by_zero
```

## Installation Issues

### pip install fails

Try upgrading pip:

```bash
pip install --upgrade pip
pip install codeverify
```

## Need More Help?

- [GitHub Discussions](https://github.com/codeverify/codeverify/discussions)
- [Discord](https://discord.gg/codeverify)
