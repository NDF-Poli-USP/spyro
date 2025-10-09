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
    
    # Determine which test files to run
    test_files = []
    notebook_files = []
    
    if args.notebook == "simple_forward":
        test_files.append("tests/tutorials/test_simple_forward.py")
        if not args.fast:
            notebook_files.append("notebook_tutorials/simple_forward.ipynb")
    elif args.notebook == "simple_forward_exercises":
        test_files.append("tests/tutorials/test_simple_forward_exercises.py")
        if not args.fast:
            notebook_files.append("notebook_tutorials/simple_forward_exercises_answers.ipynb")
    else:  # all
        test_files.append("tests/tutorials/")
        if not args.fast:
            notebook_files.extend([
                "notebook_tutorials/simple_forward.ipynb",
                "notebook_tutorials/simple_forward_exercises_answers.ipynb"
            ])
    
    # Run regular tests first
    cmd.extend(test_files)
    print(f"Running regular tests: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True)
        print("✓ Regular tests passed!")
    except subprocess.CalledProcessError as e:
        print(f"✗ Regular tests failed with return code: {e.returncode}")
        return e.returncode
    
    # Run notebook tests separately if not fast mode
    if notebook_files and not args.fast:
        print(f"\nRunning notebook execution tests...")
        
        # Filter existing notebooks
        existing_notebooks = []
        for nb_path in notebook_files:
            if os.path.exists(nb_path):
                existing_notebooks.append(nb_path)
            else:
                print(f"Warning: Notebook {nb_path} not found, skipping")
        
        if existing_notebooks:
            nb_cmd = ["python3", "-m", "pytest", "--nbval"]
            if args.verbose:
                nb_cmd.extend(["-v", "-s"])
            nb_cmd.extend(existing_notebooks)
            
            print(f"Running notebook tests: {' '.join(nb_cmd)}")
            try:
                result = subprocess.run(nb_cmd, check=True)
                print("✓ Notebook tests passed!")
            except subprocess.CalledProcessError as e:
                print(f"✗ Notebook tests failed with return code: {e.returncode}")
                return e.returncode
        else:
            print("Warning: No notebook files found for testing")
    
    print("🎉 All tests passed!")
    return 0


if __name__ == "__main__":
    sys.exit(main())