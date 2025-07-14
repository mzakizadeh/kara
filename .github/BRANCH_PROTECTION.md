# Branch Protection Rules and Naming Conventions

This document outlines the branch protection rules and naming conventions for the KARA repository to ensure code quality, maintain a clean git history, and enable efficient collaboration.

## Branch Naming Conventions

### Protected Branches
- `main` - Production branch containing stable, released code
- `dev` - Development branch for integrating features before release

### Feature Branches
- Pattern: `feat/description-of-feature`
- Examples:
  - `feat/langchain-integration`
  - `feat/token-based-chunking`
  - `feat/performance-optimization`

### Hotfix Branches
- Pattern: `hotfix/description-of-fix`
- Examples:
  - `hotfix/memory-leak`
  - `hotfix/security-patch`
  - `hotfix/critical-bug`

### Additional Supported Patterns
- `bugfix/description-of-bug` - For non-critical bug fixes
- `docs/description-of-docs` - For documentation-only changes
- `refactor/description-of-refactor` - For code refactoring

## Branch Protection Rules

### For `main` branch:
1. **Require pull request reviews before merging**
   - Required number of reviewers: 1
   - Dismiss stale reviews when new commits are pushed
   - Require review from code owners

2. **Require status checks to pass before merging**
   - Required status checks:
     - `test (3.9)`
     - `test (3.10)`
     - `test (3.11)`
     - `test (3.12)`
     - `test (3.13)`

3. **Require branches to be up to date before merging**

4. **Require linear history**

5. **Do not allow force pushes**

6. **Do not allow deletions**

### For `dev` branch:
1. **Require pull request reviews before merging**
   - Required number of reviewers: 1
   - Allow review dismissals

2. **Require status checks to pass before merging**
   - Required status checks:
     - `test (3.9)`
     - `test (3.11)`
     - `test (3.13)` (subset for faster development)

3. **Require branches to be up to date before merging**

4. **Do not allow force pushes**

## Workflow Rules

### Merging Strategy
- **main**: Squash and merge (clean history)
- **dev**: Merge commits allowed (preserves feature branch history)

### Automatic Branch Deletion
- Enable automatic deletion of head branches after pull requests are merged

## Implementation Steps

### 1. GitHub Repository Settings

Navigate to your repository settings and implement these rules:

```
Repository → Settings → Branches → Add Rule
```

#### Main Branch Protection Rule:
```yaml
Branch name pattern: main
Settings:
  ✅ Restrict pushes that create files larger than 100 MB
  ✅ Require a pull request before merging
    ✅ Require approvals (1)
    ✅ Dismiss stale pull request approvals when new commits are pushed
    ✅ Require review from code owners
  ✅ Require status checks to pass before merging
    ✅ Require branches to be up to date before merging
    Status checks:
      - test (3.9)
      - test (3.10) 
      - test (3.11)
      - test (3.12)
      - test (3.13)
  ✅ Require linear history
  ✅ Restrict pushes that create files larger than 100 MB
  ❌ Allow force pushes
  ❌ Allow deletions
```

#### Dev Branch Protection Rule:
```yaml
Branch name pattern: dev
Settings:
  ✅ Require a pull request before merging
    ✅ Require approvals (1)
    ❌ Dismiss stale pull request approvals when new commits are pushed
  ✅ Require status checks to pass before merging
    ✅ Require branches to be up to date before merging
    Status checks:
      - test (3.9)
      - test (3.11)
      - test (3.13)
  ❌ Require linear history
  ❌ Allow force pushes
  ❌ Allow deletions
```

### 2. Repository Rules (Modern Approach)

GitHub's newer "Repository Rules" provide more advanced protection. Navigate to:

```
Repository → Settings → Rules → Rulesets → New ruleset
```

#### Ruleset for Protected Branches:
```yaml
Name: "Protected Branches"
Enforcement: Active
Target: Include by pattern
  Patterns: main, dev

Rules:
  - Restrict creations: ✅
  - Restrict updates: ✅
    - Restrict pushes that create files larger than 100 MB
  - Restrict deletions: ✅
  - Require pull requests: ✅
    - Required approving review count: 1
    - Dismiss stale reviews: ✅ (main only)
    - Require code owner review: ✅ (main only)
  - Require status checks: ✅
    - Status checks: test (3.9), test (3.10), test (3.11), test (3.12), test (3.13)
    - Require branches up to date: ✅
  - Require linear history: ✅ (main only)
```

#### Ruleset for Branch Naming:
```yaml
Name: "Branch Naming Convention"
Enforcement: Active  
Target: Include all branches
Bypass: Allow repository administrators

Rules:
  - Restrict creations: ✅
    - Restrict creation of branches with patterns:
      - Deny: *
      - Allow: main
      - Allow: dev
      - Allow: feat/*
      - Allow: hotfix/*
      - Allow: bugfix/*
      - Allow: docs/*
      - Allow: refactor/*
```

### 3. CODEOWNERS File

Create a `.github/CODEOWNERS` file to automatically request reviews:

```
# Global owners
* @mzakizadeh

# Core algorithm files require extra review
/src/kara/core.py @mzakizadeh
/src/kara/splitters.py @mzakizadeh

# CI/CD and build configuration
/.github/ @mzakizadeh
/pyproject.toml @mzakizadeh
```

## Development Workflow

### Creating Feature Branches
```bash
# From dev branch
git checkout dev
git pull origin dev
git checkout -b feat/your-feature-name

# Work on your feature
git add .
git commit -m "feat: implement your feature"
git push origin feat/your-feature-name

# Create PR to dev branch
```

### Creating Hotfix Branches
```bash
# From main branch for critical fixes
git checkout main
git pull origin main
git checkout -b hotfix/critical-fix

# Work on your hotfix
git add .
git commit -m "fix: critical bug description"
git push origin hotfix/critical-fix

# Create PR to main branch
```

### Release Workflow
```bash
# When ready to release from dev
git checkout main
git pull origin main
git merge --no-ff dev
git tag v1.0.0
git push origin main --tags
```

## Enforcement Tools

### Pre-commit Hooks
The repository already includes pre-commit hooks. Ensure they include:
- Code formatting (ruff)
- Linting (ruff)
- Type checking (mypy)
- Commit message validation

### GitHub Actions
Current CI workflow supports the branch patterns and should be maintained to:
- Run on all protected branches and PR targets
- Enforce code quality checks
- Validate test coverage
- Check documentation builds

## Monitoring and Maintenance

### Regular Reviews
- Monthly review of branch protection effectiveness
- Quarterly review of naming conventions
- Update rules as the project grows

### Metrics to Track
- PR review time
- CI/CD success rates
- Branch naming compliance
- Code coverage trends

## Troubleshooting

### Common Issues
1. **Status checks failing**: Ensure all CI jobs complete successfully
2. **Force push blocked**: Use `git revert` instead of force push
3. **Linear history requirement**: Use squash merge or rebase before merge
4. **Review requirements**: Ensure reviewers are available and responsive

### Emergency Procedures
In case of critical issues:
1. Repository administrators can bypass rules temporarily
2. Create emergency hotfix branches following the hotfix pattern
3. Document the emergency procedure and review afterward