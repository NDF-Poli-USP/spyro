# File Saving Locations for HDF5, H5, and SEGY Files

This document lists all locations in the spyro codebase where files with extensions `.hdf5`, `.h5`, or `.segy` are saved or written.

## Summary

The spyro library primarily **reads** HDF5 files but does **write** SEGY files. Here are the key operations:

### HDF5/H5 Files
- **Reading Only**: The codebase reads HDF5 files containing velocity models but does not write to HDF5 format
- **Primary Function**: `interpolate()` in `spyro/io/basicio.py`

### SEGY Files
- **Writing**: The codebase can write velocity data to SEGY format
- **Primary Functions**: `create_segy()` in `spyro/io/basicio.py` and SEGY operations in `spyro/tools/velocity_smoother.py`

---

## Detailed Locations

### 1. Primary I/O Module: `spyro/io/basicio.py`

#### HDF5 Reading (Lines 443-484)
```python
def interpolate(Model, fname, V):
    """Read and interpolate a seismic velocity model stored
    in a HDF5 file onto the nodes of a finite element space.
    """
    with h5py.File(fname, "r") as f:
        Z = np.asarray(f.get("velocity_model")[()])
        # ... interpolation code ...
```
- **Operation**: READ
- **File Type**: `.hdf5` or `.h5`
- **Purpose**: Reads velocity model from HDF5 file and interpolates onto mesh nodes
- **Library Used**: `h5py`

#### SEGY Writing (Lines 267-298)
```python
def create_segy(function, V, grid_spacing, filename):
    """Write the velocity data into a segy file named filename"""
    velocity = write_function_to_grid(function, V, grid_spacing)
    spec = segyio.spec()
    # ... setup code ...
    with segyio.create(filename, spec) as f:
        for tr, il in enumerate(spec.ilines):
            f.trace[tr] = velocity[:, tr]
```
- **Operation**: WRITE
- **File Type**: `.segy`
- **Purpose**: Writes velocity data to SEGY format
- **Library Used**: `segyio`

---

### 2. Velocity Smoother Tool: `spyro/tools/velocity_smoother.py`

#### SEGY Reading (Line 35)
```python
with segyio.open(input_filename, ignore_geometry=True) as f:
    # ... reading code ...
```
- **Operation**: READ
- **File Type**: `.segy`
- **Library Used**: `segyio`

#### SEGY Writing (Lines 61-75)
```python
spec = segyio.spec()
# ... setup code ...
with segyio.create(output_filename, spec) as f:
    # ... writing code ...
```
- **Operation**: WRITE
- **File Type**: `.segy`
- **Purpose**: Smooths velocity models and writes to SEGY format
- **Library Used**: `segyio`

---

## Other File Saving Operations (Not HDF5/SEGY)

While not HDF5 or SEGY, the following locations save data to other formats:

### NumPy Files (.npy)
- `spyro/io/basicio.py` (lines 118-119): Temporary shot/receiver data
- `spyro/io/field_logger.py` (line 29, 93): Field logging data
- `spyro/solvers/inversion.py` (line 248): Control data
- `spyro/habc/habc.py` (lines 1715, 1723): Reference receiver data

### Pickle Files (.dat)
- `spyro/io/basicio.py` (line 320): Shot records via `save_shots()` function

### VTK/PVD Files
- Multiple locations throughout tests and paper scripts for visualization

---

## Test Files

### `tests/on_one_core/test_io.py`
Contains tests for:
- Reading and writing SEGY files (lines 15-82): `test_read_and_write_segy()`
- Saving and loading shot records (lines 84-97): `test_saving_and_loading_shot_record()`

---

## Usage Examples

### Reading HDF5 Velocity Model
```python
import spyro

# Create model and wave object
Wave_obj = spyro.AcousticWave(dictionary=model)

# Read and interpolate HDF5 velocity model
vp = spyro.io.interpolate(Wave_obj, "velocity_model.hdf5", Wave_obj.function_space)
```

### Writing SEGY File
```python
import spyro

# Assuming vp is a Firedrake function with velocity data
spyro.io.create_segy(vp, function_space, grid_spacing=0.01, filename="output.segy")
```

---

## Important Notes

1. **No HDF5 Writing**: The codebase currently does NOT write to HDF5 format, only reads from it
2. **SEGY Read/Write**: Full support for reading and writing SEGY files
3. **Libraries Required**:
   - `h5py`: For HDF5 file operations
   - `segyio`: For SEGY file operations
4. **Coordinate System**: Uses (z, x) or (z, x, y) coordinates where z is depth (negative values)

---

## File Naming Patterns

Files in the repository following these patterns:
- `velocity_models/*.hdf5` - Velocity model files
- `velocity_models/*.segy` - SEGY format velocity files
- Example files found:
  - `tests/inputfiles/velocity_models/Model1.hdf5`
  - `tests/inputfiles/velocity_models/foo3d.hdf5`
