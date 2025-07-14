# Contributing to KARA

Thank you for your interest in contributing to KARA! This document provides guidelines for contributing to the project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Branch Naming and Protection](#branch-naming-and-protection)
- [Development Workflow](#development-workflow)
- [Code Style and Quality](#code-style-and-quality)
- [Testing](#testing)
- [Documentation](#documentation)
- [Pull Request Process](#pull-request-process)
- [Release Process](#release-process)

## Code of Conduct

Please be respectful and professional in all interactions. We welcome contributions from everyone.

## Getting Started

### Prerequisites

- Python 3.9 or higher
- Git
- Basic understanding of RAG (Retrieval-Augmented Generation) concepts

### Development Setup

1. Fork the repository
2. Clone your fork:
   ```bash
   git clone https://github.com/YOUR_USERNAME/kara.git
   cd kara
   ```

3. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

4. Install development dependencies:
   ```bash
   pip install -e ".[dev]"
   ```

5. Install pre-commit hooks:
   ```bash
   pre-commit install
   ```

## Branch Naming and Protection

We follow strict branch naming conventions and have protection rules in place. Please read our [Branch Protection Guide](.github/BRANCH_PROTECTION.md) for complete details.

### Quick Reference

**Protected Branches:**
- `main` - Production branch (requires PR, reviews, passing tests)
- `dev` - Development branch (requires PR, reviews, passing tests)

**Feature Branches:**
- `feat/description-of-feature` - New features
- `hotfix/description-of-fix` - Critical bug fixes
- `bugfix/description-of-bug` - Non-critical bug fixes
- `docs/description-of-docs` - Documentation changes
- `refactor/description-of-refactor` - Code refactoring

### Branch Creation Examples:
```bash
# Feature branch
git checkout dev
git checkout -b feat/langchain-integration

# Hotfix branch
git checkout main
git checkout -b hotfix/memory-leak

# Documentation branch
git checkout dev
git checkout -b docs/api-reference
```

## Development Workflow

### 1. Choose the Right Base Branch
- **Features**: Branch from `dev`
- **Hotfixes**: Branch from `main`
- **Documentation**: Branch from `dev`
- **Bug fixes**: Branch from `dev`

### 2. Make Your Changes
- Keep commits small and focused
- Write clear commit messages following [Conventional Commits](https://www.conventionalcommits.org/)
- Test your changes thoroughly

### 3. Commit Message Format
```
type(scope): description

[optional body]

[optional footer]
```

**Types:**
- `feat` - New feature
- `fix` - Bug fix
- `docs` - Documentation changes
- `style` - Code style changes (formatting, etc.)
- `refactor` - Code refactoring
- `test` - Adding or updating tests
- `chore` - Maintenance tasks
- `perf` - Performance improvements
- `ci` - CI/CD changes
- `build` - Build system changes

**Examples:**
```
feat(core): implement token-based chunking algorithm
fix(splitters): resolve memory leak in recursive splitter
docs(api): add examples for KARAUpdater usage
test(integration): add langchain integration tests
```

### 4. Push and Create PR
```bash
git push origin feat/your-feature-name
```

Then create a Pull Request through GitHub's web interface.

## Code Style and Quality

We use several tools to maintain code quality:

### Ruff (Linting and Formatting)
```bash
# Check for linting issues
ruff check src tests examples

# Format code
ruff format src tests examples

# Fix auto-fixable issues
ruff check --fix src tests examples
```

### MyPy (Type Checking)
```bash
mypy src
```

### Pre-commit Hooks
Pre-commit hooks run automatically on each commit:
- Code formatting with ruff
- Linting with ruff
- Type checking with mypy
- Various file checks

## Testing

### Running Tests
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=kara --cov-report=html

# Run specific test file
pytest tests/test_core.py

# Run specific test
pytest tests/test_core.py::test_specific_function
```

### Writing Tests
- Place tests in the `tests/` directory
- Use descriptive test names
- Include both unit tests and integration tests
- Test edge cases and error conditions
- Maintain or improve test coverage

### Test Structure
```python
def test_function_name_should_behavior():
    """Test that function_name behaves correctly under specific conditions."""
    # Arrange
    input_data = "test data"
    expected_output = "expected result"
    
    # Act
    result = function_name(input_data)
    
    # Assert
    assert result == expected_output
```

## Documentation

### API Documentation
- Use clear docstrings following the NumPy style
- Include examples in docstrings
- Document parameters, returns, and raises

### Example Docstring:
```python
def chunk_text(text: str, chunk_size: int = 500) -> List[str]:
    """Split text into chunks of specified size.
    
    Parameters
    ----------
    text : str
        The input text to be chunked.
    chunk_size : int, default=500
        Maximum size of each chunk in characters.
        
    Returns
    -------
    List[str]
        List of text chunks.
        
    Raises
    ------
    ValueError
        If chunk_size is less than 1.
        
    Examples
    --------
    >>> chunks = chunk_text("Hello world", chunk_size=5)
    >>> len(chunks)
    2
    """
```

### Documentation Files
- Update relevant documentation when making changes
- Add examples to the `examples/` directory for new features
- Update README.md if adding new functionality

## Pull Request Process

### Before Submitting
1. **Sync with base branch:**
   ```bash
   git checkout dev  # or main for hotfixes
   git pull origin dev
   git checkout your-branch
   git rebase dev  # or merge if preferred
   ```

2. **Run all checks:**
   ```bash
   ruff check src tests examples
   ruff format --check src tests examples
   mypy src
   pytest
   ```

3. **Update documentation** if needed

### PR Title and Description
- Use a descriptive title following conventional commit format
- Provide a clear description of changes
- Link to any related issues
- Include testing instructions if applicable

### PR Template Example:
```markdown
## Description
Brief description of changes made.

## Type of Change
- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature (non-breaking change which adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update

## Testing
- [ ] Tests pass locally
- [ ] New tests added for new functionality
- [ ] Manual testing completed

## Checklist
- [ ] Code follows the style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] Tests added/updated
```

### Review Process
1. **Automated checks** must pass (CI, linting, tests)
2. **Code review** by at least one maintainer
3. **Address feedback** promptly and professionally
4. **Final approval** before merge

## Release Process

### Versioning
We follow [Semantic Versioning](https://semver.org/):
- **MAJOR** version for incompatible API changes
- **MINOR** version for backward-compatible functionality additions
- **PATCH** version for backward-compatible bug fixes

### Release Workflow
1. **Create release PR** from `dev` to `main`
2. **Update version** in relevant files
3. **Update CHANGELOG.md** with release notes
4. **Merge to main** after approval
5. **Tag release** on main branch
6. **GitHub Actions** automatically publishes to PyPI

### Changelog Format
```markdown
## [1.2.0] - 2024-01-15

### Added
- New token-based chunking algorithm
- LlamaIndex integration

### Changed
- Improved performance of recursive splitter

### Fixed
- Memory leak in core module

### Deprecated
- Old chunking method (will be removed in 2.0.0)
```

## Getting Help

- **Issues**: Check existing issues or create a new one
- **Discussions**: Use GitHub Discussions for questions
- **Documentation**: Refer to the official documentation
- **Code Examples**: Check the `examples/` directory

## Recognition

Contributors will be recognized in:
- CHANGELOG.md for significant contributions
- README.md contributors section
- Release notes for major features

Thank you for contributing to KARA! 🚀