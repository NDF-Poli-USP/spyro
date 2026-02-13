#!/usr/bin/env python3
"""
Script to find all file saving/reading operations for HDF5, H5, and SEGY files in the spyro codebase.

This script searches the codebase for patterns related to:
- HDF5 file operations (*.hdf5, *.h5)
- SEGY file operations (*.segy)

Usage:
    python find_file_operations.py
"""

import os
import re
from pathlib import Path
from collections import defaultdict


def find_file_operations(root_dir="."):
    """
    Find all HDF5 and SEGY file operations in Python files.
    
    Parameters
    ----------
    root_dir : str
        Root directory to search from
        
    Returns
    -------
    dict
        Dictionary with file operations categorized by type
    """
    results = {
        'hdf5_read': [],
        'hdf5_write': [],
        'segy_read': [],
        'segy_write': [],
        'file_references': []
    }
    
    # Patterns to search for
    patterns = {
        'hdf5_read': [
            r'h5py\.File\s*\([^,]+,\s*["\']r["\']',  # h5py.File(fname, "r")
            r'with\s+h5py\.File\s*\([^,]+,\s*["\']r["\']',  # with h5py.File(fname, "r")
        ],
        'hdf5_write': [
            r'h5py\.File\s*\([^,]+,\s*["\']w["\']',  # h5py.File(fname, "w")
            r'with\s+h5py\.File\s*\([^,]+,\s*["\']w["\']',  # with h5py.File(fname, "w")
            r'h5py\.File\s*\([^,]+,\s*["\']a["\']',  # h5py.File(fname, "a") - append
        ],
        'segy_read': [
            r'segyio\.open\s*\(',  # segyio.open()
            r'with\s+segyio\.open\s*\(',  # with segyio.open()
        ],
        'segy_write': [
            r'segyio\.create\s*\(',  # segyio.create()
            r'with\s+segyio\.create\s*\(',  # with segyio.create()
        ],
        'file_references': [
            r'["\'][^"\']*\.hdf5["\']',  # "*.hdf5"
            r'["\'][^"\']*\.h5["\']',    # "*.h5"
            r'["\'][^"\']*\.segy["\']',  # "*.segy"
        ]
    }
    
    # Search through Python files
    for root, dirs, files in os.walk(root_dir):
        # Skip hidden directories and common non-source directories
        dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['__pycache__', 'build', 'dist', 'egg-info']]
        
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                rel_path = os.path.relpath(file_path, root_dir)
                
                # Skip this script itself
                if file == 'find_file_operations.py':
                    continue
                
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                        
                    for line_num, line in enumerate(lines, start=1):
                        # Check each pattern category
                        for category, pattern_list in patterns.items():
                            for pattern in pattern_list:
                                if re.search(pattern, line):
                                    # Get context (few lines before and after)
                                    start = max(0, line_num - 2)
                                    end = min(len(lines), line_num + 2)
                                    context = ''.join(lines[start:end])
                                    
                                    results[category].append({
                                        'file': rel_path,
                                        'line': line_num,
                                        'content': line.strip(),
                                        'context': context
                                    })
                                    break  # Only match once per line per category
                                    
                except (UnicodeDecodeError, IOError):
                    # Skip files that can't be read
                    pass
    
    return results


def print_results(results):
    """
    Print the results in a readable format.
    
    Parameters
    ----------
    results : dict
        Dictionary of file operations
    """
    categories = {
        'hdf5_read': 'HDF5 File Reading Operations',
        'hdf5_write': 'HDF5 File Writing Operations',
        'segy_read': 'SEGY File Reading Operations',
        'segy_write': 'SEGY File Writing Operations',
        'file_references': 'File Extension References (.hdf5, .h5, .segy)'
    }
    
    print("=" * 80)
    print("FILE OPERATIONS SEARCH RESULTS")
    print("Searching for HDF5 (*.hdf5, *.h5) and SEGY (*.segy) file operations")
    print("=" * 80)
    print()
    
    for category, title in categories.items():
        operations = results[category]
        if operations:
            print(f"\n{title}")
            print("-" * len(title))
            print(f"Found {len(operations)} occurrence(s)\n")
            
            # Group by file
            by_file = defaultdict(list)
            for op in operations:
                by_file[op['file']].append(op)
            
            for file_path in sorted(by_file.keys()):
                print(f"  File: {file_path}")
                for op in by_file[file_path]:
                    print(f"    Line {op['line']}: {op['content']}")
                print()
        else:
            print(f"\n{title}")
            print("-" * len(title))
            print("No occurrences found\n")
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"HDF5 Read operations:     {len(results['hdf5_read'])}")
    print(f"HDF5 Write operations:    {len(results['hdf5_write'])}")
    print(f"SEGY Read operations:     {len(results['segy_read'])}")
    print(f"SEGY Write operations:    {len(results['segy_write'])}")
    print(f"File extension references: {len(results['file_references'])}")
    print()
    
    # Key findings
    print("KEY FINDINGS:")
    print("-" * 80)
    if results['hdf5_write']:
        print("✓ HDF5 files are being WRITTEN")
    else:
        print("✗ HDF5 files are NOT being written (only read)")
    
    if results['segy_write']:
        print("✓ SEGY files are being WRITTEN")
    else:
        print("✗ SEGY files are NOT being written")
    
    if results['segy_read']:
        print("✓ SEGY files are being READ")
    else:
        print("✗ SEGY files are NOT being read")
    
    if results['hdf5_read']:
        print("✓ HDF5 files are being READ")
    else:
        print("✗ HDF5 files are NOT being read")
    print()


def save_results_to_file(results, output_file="file_operations_report.txt"):
    """
    Save detailed results to a text file.
    
    Parameters
    ----------
    results : dict
        Dictionary of file operations
    output_file : str
        Output file path
    """
    import sys
    from io import StringIO
    
    # Redirect stdout to string
    old_stdout = sys.stdout
    sys.stdout = StringIO()
    
    print_results(results)
    
    # Get the content and restore stdout
    content = sys.stdout.getvalue()
    sys.stdout = old_stdout
    
    # Write to file
    with open(output_file, 'w') as f:
        f.write(content)
    
    print(f"Results saved to: {output_file}")


if __name__ == "__main__":
    # Find the spyro package directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Search from the spyro subdirectory if it exists, otherwise current directory
    search_dir = os.path.join(script_dir, "spyro") if os.path.exists(os.path.join(script_dir, "spyro")) else script_dir
    
    if not os.path.exists(search_dir):
        search_dir = script_dir
    
    print(f"Searching in: {search_dir}\n")
    
    # Find operations
    results = find_file_operations(search_dir)
    
    # Print results
    print_results(results)
    
    # Optionally save to file
    # save_results_to_file(results, "file_operations_report.txt")
