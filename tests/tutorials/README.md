# Tutorial Notebook Testing

This directory contains tests for the Spyro tutorial notebooks using pytest-nbval.

## Overview

The testing framework validates that the tutorial notebooks execute without errors and produce expected outputs. This helps ensure that:

1. The tutorial notebooks remain functional as the codebase evolves
2. Dependencies are correctly installed and configured
3. Examples in tutorials work as intended for users

## Test Structure

### Test Files

- `test_simple_forward.py` - Tests for the simple_forward.ipynb notebook
- `test_simple_forward_exercises.py` - Tests for the exercises notebooks
- `run_tests.py` - Test runner script with various options

### Configuration Files

- `nbval_sanitize.cfg` - Output sanitization rules for environment differences
- Integration with existing `conftest.py` - Uses existing `--skip-slow` and `--only-slow` options

## Running Tests

### Local Testing

```bash
# Run all notebook tests
cd tests/tutorials
python run_tests.py

# Run only fast tests (skip notebook execution) 
python3 run_tests.py --fast

# Test specific notebook
python3 run_tests.py --notebook simple_forward

# Verbose output
python3 run_tests.py --verbose
```

### Manual pytest execution

```bash
# Test basic functionality only (uses existing conftest.py)
pytest --skip-slow

# Test only slow tests
pytest --only-slow

# Test notebook execution with nbval
pytest --nbval ../../notebook_tutorials/simple_forward.ipynb

# Test both basic functionality and notebook execution
pytest . ../../notebook_tutorials/simple_forward.ipynb ../../notebook_tutorials/simple_forward_exercises_answers.ipynb
```

## CI Integration

Tests run automatically:

- **Sundays at 2 AM UTC** - Full notebook execution tests
- **On pushes to main** - When tutorial-related files change
- **Manual trigger** - Via GitHub Actions workflow dispatch

The CI uses the same Firedrake container as other tests to ensure consistency.

## Notebooks Tested

1. **simple_forward.ipynb** - Basic forward modeling tutorial
2. **simple_forward_exercises_answers.ipynb** - Exercises with complete solutions

## Test Categories

### Fast Tests (`--skip-slow`)
- Import validation
- Basic functionality checks
- Configuration validation
- No actual forward modeling

### Slow Tests (`--only-slow` or default)
- Complete notebook execution
- Forward modeling runs
- Output validation
- File generation checks

## Output Sanitization

The `nbval_sanitize.cfg` file removes environment-specific content from notebook outputs:

- Memory addresses
- Timing information  
- File paths
- Warning messages
- Progress bars
- Numeric precision differences

This ensures tests are robust across different environments.

## Adding New Notebook Tests

1. Create test file: `test_<notebook_name>.py`
2. Add basic functionality tests
3. Add notebook path to CI workflow
4. Update this README

## Troubleshooting

### Common Issues

1. **Import errors** - Check that all dependencies are installed
2. **Timeout** - Increase timeout in CI for complex notebooks  
3. **Output differences** - Add sanitization rules to `nbval_sanitize.cfg`
4. **Memory issues** - Consider reducing problem size in tutorials

### Debugging

```bash
# Run with verbose output to see execution details
python run_tests.py --verbose

# Run single test to isolate issues
pytest -v test_simple_forward.py::test_spyro_basic_functionality

# Check notebook execution without test framework
jupyter nbconvert --to notebook --execute --inplace notebook.ipynb
```