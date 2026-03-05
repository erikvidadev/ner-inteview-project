# Development Guide

This project follows modern Python packaging standards using **Poetry** for dependency management and deterministic environment builds.

### Prerequisites
- **Python 3.12**: Recommended for optimal library compatibility.
- **Poetry**: Ensure you have Poetry installed (`pip install poetry`, 
`brew install poetry`).

---

### Environment Setup

Poetry handles virtual environment creation and dependency resolution in one step.

```bash
# 1. Install all dependencies
poetry env use python3.12
poetry install

# 2. Activate the virtual environment
poetry shell
```

### Run
```bash
# Running the Pipeline
poetry run python main.py
```

### Code Quality
```bash
# Check for linting issues
poetry run ruff check .

# Automatically fix formatting
poetry run ruff format .
```