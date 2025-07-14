#!/usr/bin/env python3
"""
Branch name validation script for KARA repository.

This script validates branch names against the repository's naming conventions.
Can be used as a pre-push hook or manually to check branch names.
"""

import re
import sys
import subprocess
from typing import List, Tuple


def get_current_branch() -> str:
    """Get the current branch name."""
    try:
        result = subprocess.run(
            ['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError:
        print("Error: Not in a git repository or git not available")
        sys.exit(1)


def validate_branch_name(branch_name: str) -> Tuple[bool, str]:
    """
    Validate a branch name against allowed patterns.
    
    Args:
        branch_name: The branch name to validate
        
    Returns:
        Tuple of (is_valid, message)
    """
    # Define allowed patterns
    allowed_patterns = [
        (r'^main$', 'main'),
        (r'^dev$', 'dev'),
        (r'^feat/[a-z0-9]([a-z0-9-]*[a-z0-9])?$', 'feat/feature-name'),
        (r'^hotfix/[a-z0-9]([a-z0-9-]*[a-z0-9])?$', 'hotfix/fix-name'),
        (r'^bugfix/[a-z0-9]([a-z0-9-]*[a-z0-9])?$', 'bugfix/bug-name'),
        (r'^docs/[a-z0-9]([a-z0-9-]*[a-z0-9])?$', 'docs/doc-name'),
        (r'^refactor/[a-z0-9]([a-z0-9-]*[a-z0-9])?$', 'refactor/refactor-name'),
    ]
    
    # Check if branch name matches any allowed pattern
    for pattern, example in allowed_patterns:
        if re.match(pattern, branch_name):
            return True, f"✅ Valid branch name: '{branch_name}' matches pattern '{example}'"
    
    # If no pattern matches, return error with guidance
    error_msg = f"""❌ Invalid branch name: '{branch_name}'

Allowed branch name patterns:
  - main
  - dev
  - feat/feature-name
  - hotfix/fix-name
  - bugfix/bug-name
  - docs/doc-name
  - refactor/refactor-name

Branch naming rules:
  - Use lowercase letters, numbers, and hyphens only
  - Follow the pattern: type/description
  - Use descriptive names (e.g., feat/langchain-integration)
  - Start with a letter or number (not a hyphen)
  - No consecutive hyphens or trailing hyphens

Examples of valid branch names:
  - feat/token-based-chunking
  - hotfix/memory-leak
  - bugfix/null-pointer-fix
  - docs/api-reference
  - refactor/core-optimization
"""
    
    return False, error_msg


def validate_commit_message(message: str) -> Tuple[bool, str]:
    """
    Validate a commit message against conventional commits format.
    
    Args:
        message: The commit message to validate
        
    Returns:
        Tuple of (is_valid, message)
    """
    # Pattern for conventional commits
    pattern = r'^(feat|fix|docs|style|refactor|test|chore|perf|ci|build)(\(.+\))?: .+'
    
    if re.match(pattern, message):
        return True, f"✅ Valid commit message: '{message}'"
    
    error_msg = f"""❌ Invalid commit message: '{message}'

Commit messages should follow the conventional commits format:
  type(scope): description

Valid types:
  - feat: new feature
  - fix: bug fix
  - docs: documentation changes
  - style: code style changes
  - refactor: code refactoring
  - test: adding or updating tests
  - chore: maintenance tasks
  - perf: performance improvements
  - ci: CI/CD changes
  - build: build system changes

Examples:
  - feat: add new chunking algorithm
  - fix: resolve memory leak in core module
  - docs: update API documentation
  - test: add unit tests for splitters
  - feat(core): implement token-based chunking
"""
    
    return False, error_msg


def main():
    """Main function to run validations."""
    if len(sys.argv) > 1:
        # Validate provided branch name
        branch_name = sys.argv[1]
    else:
        # Validate current branch
        branch_name = get_current_branch()
    
    print(f"Validating branch name: {branch_name}")
    
    is_valid, message = validate_branch_name(branch_name)
    print(message)
    
    if not is_valid:
        print("\nTo rename your current branch:")
        print("  git branch -m new-branch-name")
        sys.exit(1)
    
    print("\n✅ Branch name validation passed!")


if __name__ == "__main__":
    main()