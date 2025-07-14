#!/usr/bin/env python3
"""
Test script for branch name validation to ensure it works correctly.
"""

import sys
import os
import subprocess

# Import the validation functions by running the script
def run_validation_script(branch_name):
    """Run the validation script and return the result."""
    script_path = os.path.join(os.path.dirname(__file__), 'validate-branch.py')
    try:
        result = subprocess.run(
            [sys.executable, script_path, branch_name],
            capture_output=True,
            text=True
        )
        return result.returncode == 0, result.stdout + result.stderr
    except Exception as e:
        return False, str(e)


def test_valid_branch_names():
    """Test that valid branch names pass validation."""
    valid_names = [
        'main',
        'dev',
        'feat/test-feature',
        'feat/langchain-integration',
        'hotfix/memory-leak',
        'hotfix/security-patch',
        'bugfix/null-pointer',
        'docs/api-reference',
        'refactor/core-optimization',
        'feat/token-based-chunking',
        'docs/contributing-guide'
    ]
    
    for name in valid_names:
        is_valid, message = run_validation_script(name)
        if not is_valid:
            print(f"❌ FAIL: {name} should be valid but was rejected")
            print(f"    {message}")
            return False
        else:
            print(f"✅ PASS: {name}")
    
    return True


def test_invalid_branch_names():
    """Test that invalid branch names fail validation."""
    invalid_names = [
        'feature/test',  # Should be feat/
        'Feature/test',  # Should be lowercase
        'feat/Test-Feature',  # Should be lowercase
        'feat/-invalid',  # Can't start with hyphen
        'feat/invalid-',  # Can't end with hyphen
        'feat/',  # No description
        'random-branch',  # No pattern match
        'fix/something',  # Should be hotfix/ or bugfix/
        'documentation/something',  # Should be docs/
    ]
    
    for name in invalid_names:
        is_valid, message = run_validation_script(name)
        if is_valid:
            print(f"❌ FAIL: {name} should be invalid but was accepted")
            return False
        else:
            print(f"✅ PASS: {name} correctly rejected")
    
    return True


def main():
    """Run all tests."""
    print("Testing Branch Name Validation...")
    print("=" * 50)
    
    print("\n1. Testing Valid Branch Names:")
    print("-" * 30)
    if not test_valid_branch_names():
        sys.exit(1)
    
    print("\n2. Testing Invalid Branch Names:")
    print("-" * 30)
    if not test_invalid_branch_names():
        sys.exit(1)
    
    print("\n" + "=" * 50)
    print("🎉 All tests passed! Validation script is working correctly.")


if __name__ == "__main__":
    main()