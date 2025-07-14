# GitHub Scripts

This directory contains utility scripts for repository management and automation.

## validate-branch.py

A Python script to validate branch names against the repository's naming conventions.

### Usage

```bash
# Validate current branch
python .github/scripts/validate-branch.py

# Validate specific branch name
python .github/scripts/validate-branch.py feat/my-feature

# Make executable and run directly
chmod +x .github/scripts/validate-branch.py
./.github/scripts/validate-branch.py
```

### Integration Options

#### Pre-push Hook
Add to `.git/hooks/pre-push`:
```bash
#!/bin/sh
python .github/scripts/validate-branch.py
```

#### Manual Check
Run before creating a pull request:
```bash
./.github/scripts/validate-branch.py
```

### Supported Branch Patterns

- `main` - Production branch
- `dev` - Development branch
- `feat/description` - Feature branches
- `hotfix/description` - Hotfix branches
- `bugfix/description` - Bug fix branches
- `docs/description` - Documentation branches
- `refactor/description` - Refactoring branches

### Examples

Valid branch names:
- `feat/langchain-integration`
- `hotfix/memory-leak`
- `docs/api-reference`
- `bugfix/null-pointer`

Invalid branch names:
- `feature/my-feature` (use `feat/` instead)
- `feat/My-Feature` (use lowercase)
- `feat/-invalid` (don't start with hyphen)
- `feat/invalid-` (don't end with hyphen)