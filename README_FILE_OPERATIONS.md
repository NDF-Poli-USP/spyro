# File Operations Finder - Quick Start Guide

This directory contains tools to find and document file saving operations for HDF5 and SEGY files in the spyro codebase.

## Files

1. **FILE_SAVING_LOCATIONS.md** - Human-readable documentation of all file operations
2. **find_file_operations.py** - Automated script to search for file operations
3. **README_FILE_OPERATIONS.md** - This file

## Quick Start

### View Documentation
```bash
# Read the comprehensive documentation
cat FILE_SAVING_LOCATIONS.md
```

### Run the Finder Script
```bash
# Search for file operations in the spyro package
python find_file_operations.py

# Search in the entire repository
python -c "
import find_file_operations
results = find_file_operations.find_file_operations('.')
find_file_operations.print_results(results)
"
```

## What the Script Finds

The script searches for:
- **HDF5 Read**: `h5py.File(fname, "r")` 
- **HDF5 Write**: `h5py.File(fname, "w")` or `"a"`
- **SEGY Read**: `segyio.open(filename)`
- **SEGY Write**: `segyio.create(filename)`
- **File References**: Strings containing `.hdf5`, `.h5`, or `.segy`

## Summary of Results

### Current Status (as of last check)
- **HDF5 Read**: 1 operation in `spyro/io/basicio.py`
- **HDF5 Write**: 0 operations (not implemented)
- **SEGY Read**: 2 operations in `spyro/tools/velocity_smoother.py`
- **SEGY Write**: 2 operations in `spyro/io/basicio.py` and `spyro/tools/velocity_smoother.py`

## Key Functions

### Reading HDF5 Files
```python
import spyro

# Read velocity model from HDF5
vp = spyro.io.interpolate(Wave_obj, "velocity.hdf5", function_space)
```
Located in: `spyro/io/basicio.py:443`

### Writing SEGY Files
```python
import spyro

# Write velocity model to SEGY
spyro.io.create_segy(vp_function, function_space, 0.01, "output.segy")
```
Located in: `spyro/io/basicio.py:295`

### Smoothing SEGY Files
```python
from spyro.tools.velocity_smoother import velocity_smoother

# Smooth a SEGY velocity model
velocity_smoother("input.segy", "output.segy", sigma=50.0)
```
Located in: `spyro/tools/velocity_smoother.py`

## Testing

To verify the file operations work correctly, run:
```bash
# Run I/O tests
pytest tests/on_one_core/test_io.py -v
```

## Notes

1. **HDF5 Format**: Used for storing velocity models, only reading is implemented
2. **SEGY Format**: Standard seismic data format, both reading and writing supported
3. **Libraries**: Requires `h5py` and `segyio` packages
4. **Coordinate System**: Uses (z, x) or (z, x, y) where z is depth (negative)

## Maintenance

To update the documentation after code changes:
1. Run `python find_file_operations.py` to get current operations
2. Update `FILE_SAVING_LOCATIONS.md` with any new findings
3. Verify with `pytest tests/on_one_core/test_io.py`

## See Also

- Main I/O module: `spyro/io/basicio.py`
- Tests: `tests/on_one_core/test_io.py`
- Velocity smoother: `spyro/tools/velocity_smoother.py`
