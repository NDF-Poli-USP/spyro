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
    
    # Get paths relative to repository root
    test_dir = Path(__file__).parent
    repo_root = test_dir.parent.parent
    
    # Change to the repository root for consistent path resolution
    os.chdir(repo_root)
    
    # Debug information
    print(f"Repository root: {repo_root}")
    print(f"Current working directory: {os.getcwd()}")
    
    # Check if notebook files exist
    nb_files = [
        "notebook_tutorials/simple_forward.ipynb",
        "notebook_tutorials/simple_forward_exercises_answers.ipynb"
    ]
    for nb_file in nb_files:
        if os.path.exists(nb_file):
            print(f"✓ Found: {nb_file}")
        else:
            print(f"✗ Missing: {nb_file}")
    
    # Build pytest command
    cmd = ["python3", "-m", "pytest"]
    
    if args.verbose:
        cmd.extend(["-v", "-s"])
    
    if args.fast:
        cmd.append("--skip-slow")
    
    # Add notebook-specific tests
    if args.notebook == "simple_forward":
        cmd.append("tests/tutorials/test_simple_forward.py")
        if not args.fast:
            nb_path = "notebook_tutorials/simple_forward.ipynb"
            if os.path.exists(nb_path):
                cmd.append(nb_path)
            else:
                print(f"Warning: Notebook {nb_path} not found, skipping notebook execution test")
    elif args.notebook == "simple_forward_exercises":
        cmd.append("tests/tutorials/test_simple_forward_exercises.py")
        if not args.fast:
            nb_path = "notebook_tutorials/simple_forward_exercises_answers.ipynb"
            if os.path.exists(nb_path):
                cmd.append(nb_path)
            else:
                print(f"Warning: Notebook {nb_path} not found, skipping notebook execution test")
    else:  # all
        cmd.append("tests/tutorials/")
        if not args.fast:
            notebook_paths = [
                "notebook_tutorials/simple_forward.ipynb",
                "notebook_tutorials/simple_forward_exercises_answers.ipynb"
            ]
            for nb_path in notebook_paths:
                if os.path.exists(nb_path):
                    cmd.append(nb_path)
                else:
                    print(f"Warning: Notebook {nb_path} not found, skipping notebook execution test")
    
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