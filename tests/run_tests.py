#!/usr/bin/env python3
"""
Simple test runner script for the LBP package.

Usage:
    python tests/run_tests.py            # Run all tests
    python tests/run_tests.py --unit     # Run unit tests only
    python tests/run_tests.py --coverage # Run all tests with coverage
"""

import argparse
import subprocess
import sys
from pathlib import Path


def run_tests(test_type="all", coverage=False, verbose=False):
    """Run tests with specified options."""
    # This script is in tests/run_tests.py
    test_dir = Path(__file__).parent  # tests directory
    project_root = test_dir.parent     # project root directory
    
    # Build pytest command
    cmd = ["python", "-m", "pytest"]
    
    # Add test path based on type
    if test_type == "unit":
        cmd.append(str(test_dir / "unit"))
    elif test_type == "integration":
        cmd.append(str(test_dir / "integration"))
    elif test_type == "end_to_end":
        cmd.append(str(test_dir / "end_to_end"))
    else:  # all
        cmd.append(str(test_dir))
    
    # Add options
    if coverage:
        cmd.extend(["--cov=lbp_package", "--cov-report=term-missing"])
    
    if verbose:
        cmd.append("-v")
    else:
        cmd.append("-q")
    
    # Run command from project root
    print(f"Running from {project_root}: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, cwd=project_root)
        return result.returncode == 0
    except KeyboardInterrupt:
        print("\nTest execution interrupted.")
        return False


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Run LBP package tests")
    
    parser.add_argument("--unit", action="store_true", help="Run unit tests only")
    parser.add_argument("--integration", action="store_true", help="Run integration tests only")
    parser.add_argument("--end-to-end", action="store_true", help="Run end-to-end tests only")
    parser.add_argument("--coverage", action="store_true", help="Generate coverage report")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    # Determine test type
    if args.unit:
        test_type = "unit"
    elif args.integration:
        test_type = "integration"
    elif args.end_to_end:
        test_type = "end_to_end"
    else:
        test_type = "all"
    
    # Run tests
    success = run_tests(test_type, args.coverage, args.verbose)
    
    if success:
        print("✓ Tests completed successfully!")
        sys.exit(0)
    else:
        print("✗ Some tests failed.")
        sys.exit(1)


if __name__ == "__main__":
    main()