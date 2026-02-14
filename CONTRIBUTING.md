# Contributing to TetherAI

## Getting Started

1. Clone the repo:
   ```bash
   git clone https://github.com/tetherai/tetherai-python.git
   cd tetherai-python
   ```

2. Install dev dependencies:
   ```bash
   pip install -e ".[dev]"
   ```

3. Run tests to verify setup:
   ```bash
   pytest tests/unit/ -v
   ```

## Development Commands

Run tests:
```bash
pytest tests/unit/ -v
```

Run linting:
```bash
ruff check src/ tests/
```

Format code:
```bash
ruff format src/ tests/
```

Run type checking:
```bash
mypy src/tetherai/ --strict
```

## Pull Request Requirements

Before submitting a PR, ensure:
- All tests pass: `pytest tests/unit/ -v`
- Linting is clean: `ruff check src/ tests/`
- Code is formatted: `ruff format src/ tests/`
- Type checking passes: `mypy src/tetherai/ --strict`

## Finding Things to Work On

Check [GitHub Issues](https://github.com/tetherai/tetherai-python/issues) for good first issues and feature requests.
