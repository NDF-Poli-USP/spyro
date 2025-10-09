#!/usr/bin/env python3
"""
Test runner for notebook tutorials.
This script runs the tutorial notebooks using nbval and pytest.
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path


def main():
    """Main entry point for the test runner."""
    parser = argparse.ArgumentParser(description="Run notebook tutorial tests")
    parser.add_argument("--notebook", 
                       choices=["simple_forward", "simple_forward_exercises", "all"],
                       default="all",
                       help="Which notebook(s) to test")
    parser.add_argument("--fast", action="store_true", 
                       help="Run only fast tests (skip slow notebook execution)")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Verbose output")
    
    args = parser.parse_args()
    
    # Change to the tests directory
    test_dir = Path(__file__).parent
    os.chdir(test_dir)
    
    # Build pytest command
    cmd = ["python3", "-m", "pytest"]
    
    if args.verbose:
        cmd.extend(["-v", "-s"])
    
    if args.fast:
        cmd.append("--skip-slow")
    
    # Add notebook-specific tests
    if args.notebook == "simple_forward":
        cmd.append("test_simple_forward.py")
        if not args.fast:
            cmd.append("../../notebook_tutorials/simple_forward.ipynb")
    elif args.notebook == "simple_forward_exercises":
        cmd.append("test_simple_forward_exercises.py")
        if not args.fast:
            cmd.append("../../notebook_tutorials/simple_forward_exercises_answers.ipynb")
    else:  # all
        cmd.append(".")
        if not args.fast:
            cmd.extend([
                "../../notebook_tutorials/simple_forward.ipynb",
                "../../notebook_tutorials/simple_forward_exercises_answers.ipynb"
            ])
    
    print(f"Running command: {' '.join(cmd)}")
    
    # Run the tests
    try:
        result = subprocess.run(cmd, check=True)
        print("All tests passed!")
        return 0
    except subprocess.CalledProcessError as e:
        print(f"Tests failed with return code: {e.returncode}")
        return e.returncode


if __name__ == "__main__":
    sys.exit(main())