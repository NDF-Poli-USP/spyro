from pathlib import Path
import importlib.util
import warnings

import h5py
import numpy as np
import pytest

from spyro.io.basicio import (
    read_bin_velocity_model,
    read_segy_velocity_model,
    write_velocity_model,
    _parse_axes_order,
)


AVENIR_SEGY = "tests/inputfiles/velocity_models/avenir.segy"
AVENIR3D_BIN = "tests/inputfiles/velocity_models/avenir3d.bin"

AVENIR3D_NZ = 41
AVENIR3D_NX = 21
AVENIR3D_NY = 21


def _require_file(path):
    path = Path(path)
    if not path.exists():
        pytest.skip(f"Missing test input file: {path}")
    return path


def _read_hdf5_velocity_model(filename):
    with h5py.File(filename, "r") as h5:
        data = h5["velocity_model"][:]
        attrs = dict(h5.attrs)
    return data, attrs


def _assert_hdf5_matches_expected(hdf5_file, expected):
    data, attrs = _read_hdf5_velocity_model(hdf5_file)

    assert data.shape == expected.shape
    assert data.dtype == np.float32
    assert np.array_equal(data, expected.astype(np.float32), equal_nan=True)

    assert "shape" in attrs
    assert "units" in attrs
    assert np.array_equal(attrs["shape"], expected.shape)
    assert attrs["units"] == "m/s"


def _axis_names_from_order(axes_order):
    """Independent axis parser for arrays in these tests."""
    axis_from_int = {0: "z", 1: "x", 2: "y"}

    if isinstance(axes_order, str):
        parts = axes_order.lower().replace(",", " ").split()
        if len(parts) == 1 and len(parts[0]) == 3:
            parts = list(parts[0])
        if all(part in {"0", "1", "2"} for part in parts):
            return tuple(axis_from_int[int(part)] for part in parts)
        return tuple(parts)

    if all(isinstance(axis, (int, np.integer)) for axis in axes_order):
        return tuple(axis_from_int[int(axis)] for axis in axes_order)

    return tuple(str(axis).lower().strip() for axis in axes_order)


def _expected_binary_model(
    filename,
    nz,
    nx,
    ny,
    byte_order,
    axes_order,
    axes_order_sort,
    dtype,
):
    """Build the expected ``(z, x, y)`` model from bytes."""
    dtype = np.dtype(dtype)
    endian = "<" if byte_order == "little" else ">"
    data = np.fromfile(filename, dtype=dtype.newbyteorder(endian))

    raw_axes = _axis_names_from_order(axes_order)
    sizes = {"z": nz, "x": nx, "y": ny}
    raw_shape = [sizes[axis] for axis in raw_axes]

    data = data.reshape(*raw_shape, order=axes_order_sort)

    final_axes = ("z", "x", "y")
    transpose_order = [raw_axes.index(axis) for axis in final_axes]
    return np.flipud(data.transpose(transpose_order))


def _expected_segy_model(filename):
    segyio = pytest.importorskip("segyio")

    with segyio.open(str(filename), "r", ignore_geometry=True) as segy:
        nx = len(segy.trace)
        nz = len(segy.samples)
        expected = np.zeros((nz, nx), dtype=np.float32)

        for index in range(nx):
            expected[:, index] = segy.trace[index]

    return np.flipud(expected)


@pytest.mark.parametrize(
    "axes_order, expected",
    [
        ("z x y", ("z", "x", "y")),
        ("zxy", ("z", "x", "y")),
        ("x z y", ("x", "z", "y")),
        ("1 0 2", ("x", "z", "y")),
        ("201", ("y", "z", "x")),
        ((0, 1, 2), ("z", "x", "y")),
        ([2, 1, 0], ("y", "x", "z")),
        (("x", "z", "y"), ("x", "z", "y")),
    ],
)
def test_parse_axes_order_valid_cases(axes_order, expected):
    assert _parse_axes_order(axes_order) == expected


@pytest.mark.parametrize(
    "axes_order, error_type",
    [
        ("z z y", ValueError),
        ("0 0 2", ValueError),
        ("abc", ValueError),
        ((0, 1), ValueError),
        ((0, 0, 2), ValueError),
        (("x", "x", "z"), ValueError),
        ((0, "x", 2), TypeError),
        (object(), TypeError),
    ],
)
def test_parse_axes_order_invalid_cases(axes_order, error_type):
    with pytest.raises(error_type):
        _parse_axes_order(axes_order)


def test_read_avenir3d_binary_switches_wrong_little_to_big():
    bin_file = _require_file(AVENIR3D_BIN)

    # The file is big-endian float32. Passing little should use the
    # NaN/Inf correction branch and switch to big.
    with pytest.warns(UserWarning, match="Using byte_order='big'"):
        vp, nz, nx, ny = read_bin_velocity_model(
            filename=str(bin_file),
            nz=AVENIR3D_NZ,
            nx=AVENIR3D_NX,
            ny=AVENIR3D_NY,
            byte_order="little",
            axes_order=(0, 1, 2),
            axes_order_sort="F",
            dtype="float32",
        )

    expected = _expected_binary_model(
        filename=bin_file,
        nz=AVENIR3D_NZ,
        nx=AVENIR3D_NX,
        ny=AVENIR3D_NY,
        byte_order="big",
        axes_order=(0, 1, 2),
        axes_order_sort="F",
        dtype="float32",
    )

    assert (nz, nx, ny) == (AVENIR3D_NZ, AVENIR3D_NX, AVENIR3D_NY)
    assert vp.shape == (AVENIR3D_NZ, AVENIR3D_NX, AVENIR3D_NY)
    assert np.all(np.isfinite(vp))
    assert np.array_equal(vp, expected)


def test_read_avenir3d_binary_correct_big_order_has_no_warning():
    bin_file = _require_file(AVENIR3D_BIN)

    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        vp, nz, nx, ny = read_bin_velocity_model(
            filename=str(bin_file),
            nz=AVENIR3D_NZ,
            nx=AVENIR3D_NX,
            ny=AVENIR3D_NY,
            byte_order="big",
            axes_order="z x y",
            axes_order_sort="F",
            dtype="float32",
        )

    assert not recorded
    assert (nz, nx, ny) == (AVENIR3D_NZ, AVENIR3D_NX, AVENIR3D_NY)
    assert vp.shape == (AVENIR3D_NZ, AVENIR3D_NX, AVENIR3D_NY)
    assert np.all(np.isfinite(vp))


def test_read_avenir3d_binary_corrects_wrong_dtype_by_file_size():
    bin_file = _require_file(AVENIR3D_BIN)

    with pytest.warns(UserWarning, match="Using dtype=float32"):
        vp, nz, nx, ny = read_bin_velocity_model(
            filename=str(bin_file),
            nz=AVENIR3D_NZ,
            nx=AVENIR3D_NX,
            ny=AVENIR3D_NY,
            byte_order="big",
            axes_order=(0, 1, 2),
            axes_order_sort="F",
            dtype="float64",
        )

    assert (nz, nx, ny) == (AVENIR3D_NZ, AVENIR3D_NX, AVENIR3D_NY)
    assert vp.dtype == np.dtype(">f4") or vp.dtype == np.float32
    assert vp.shape == (AVENIR3D_NZ, AVENIR3D_NX, AVENIR3D_NY)


def test_read_binary_keeps_selected_byte_order_when_other_is_not_better(tmp_path):
    # 0xffffffff is NaN as both little-endian and big-endian float32.
    filename = tmp_path / "all_nan_both_endians.bin"
    filename.write_bytes(b"\xff\xff\xff\xff")

    with pytest.warns(UserWarning, match="Keeping byte_order='little'"):
        vp, nz, nx, ny = read_bin_velocity_model(
            filename=str(filename),
            nz=1,
            nx=1,
            ny=1,
            byte_order="little",
            axes_order=(0, 1, 2),
            axes_order_sort="C",
            dtype="float32",
        )

    assert (nz, nx, ny) == (1, 1, 1)
    assert vp.shape == (1, 1, 1)
    assert np.isnan(vp[0, 0, 0])


def test_read_binary_axis_permutation_with_temporary_file(tmp_path):
    filename = tmp_path / "axis_permutation.bin"

    nz, nx, ny = 2, 3, 4
    raw_shape = (nx, nz, ny)  # axes_order=(1, 0, 2), i.e. x, z, y
    raw = np.arange(np.prod(raw_shape), dtype=np.float32).reshape(raw_shape, order="C")
    raw.tofile(filename)

    vp, _, _, _ = read_bin_velocity_model(
        filename=str(filename),
        nz=nz,
        nx=nx,
        ny=ny,
        byte_order="little",
        axes_order=(1, 0, 2),
        axes_order_sort="C",
        dtype="float32",
    )

    expected = np.flipud(raw.transpose(1, 0, 2))
    assert np.array_equal(vp, expected)


def test_read_binary_errors(tmp_path):
    filename = tmp_path / "tiny.bin"
    np.array([1.0], dtype=np.float32).tofile(filename)

    with pytest.raises(ValueError, match="grid points"):
        read_bin_velocity_model(str(filename), nz=None, nx=1, ny=1)

    with pytest.raises(ValueError, match="byte_order"):
        read_bin_velocity_model(str(filename), nz=1, nx=1, ny=1, byte_order="auto")

    with pytest.raises(ValueError, match="axes_order_sort"):
        read_bin_velocity_model(str(filename), nz=1, nx=1, ny=1, axes_order_sort="A")


def test_read_binary_file_size_mismatch_raises(tmp_path):
    filename = tmp_path / "bad_size.bin"
    filename.write_bytes(b"abc")

    with pytest.raises(ValueError, match="File size mismatch"):
        read_bin_velocity_model(
            filename=str(filename),
            nz=1,
            nx=1,
            ny=1,
            byte_order="little",
            axes_order=(0, 1, 2),
            axes_order_sort="C",
            dtype="float32",
        )


def test_write_velocity_model_avenir3d_binary_hdf5(tmp_path):
    bin_file = _require_file(AVENIR3D_BIN)
    output_stem = tmp_path / "avenir3d"

    with pytest.warns(UserWarning, match="Using byte_order='big'"):
        hdf5_file = write_velocity_model(
            filename=str(bin_file),
            ofname=str(output_stem),
            model_type="bin",
            nz=AVENIR3D_NZ,
            nx=AVENIR3D_NX,
            ny=AVENIR3D_NY,
            byte_order="little",
            axes_order=(0, 1, 2),
            axes_order_sort="F",
            dtype="float32",
        )

    expected = _expected_binary_model(
        filename=bin_file,
        nz=AVENIR3D_NZ,
        nx=AVENIR3D_NX,
        ny=AVENIR3D_NY,
        byte_order="big",
        axes_order=(0, 1, 2),
        axes_order_sort="F",
        dtype="float32",
    )

    assert hdf5_file == str(output_stem) + ".hdf5"
    _assert_hdf5_matches_expected(hdf5_file, expected)


@pytest.mark.skipif(not HAS_SEGYIO, reason="segyio is not installed")
def test_read_avenir_segy_velocity_model():
    segy_file = _require_file(AVENIR_SEGY)

    vp, nx, nz = read_segy_velocity_model(str(segy_file))
    expected = _expected_segy_model(segy_file)

    assert vp.shape == expected.shape
    assert nx == expected.shape[1]
    assert nz == expected.shape[0]
    assert np.array_equal(vp, expected)


def test_write_velocity_model_avenir_segy_hdf5(tmp_path):
    segy_file = _require_file(AVENIR_SEGY)
    output_stem = tmp_path / "avenir"

    hdf5_file = write_velocity_model(
        filename=str(segy_file),
        ofname=str(output_stem),
        model_type="segy",
    )

    expected = _expected_segy_model(segy_file)

    assert hdf5_file == str(output_stem) + ".hdf5"
    _assert_hdf5_matches_expected(hdf5_file, expected)


def test_write_velocity_model_invalid_model_type(tmp_path):
    filename = tmp_path / "tiny.bin"
    np.array([1.0], dtype=np.float32).tofile(filename)

    with pytest.raises(ValueError, match="model_type"):
        write_velocity_model(
            filename=str(filename),
            ofname=str(tmp_path / "out"),
            model_type="unknown",
        )


def test_write_velocity_model_default_output_name_warning(tmp_path):
    filename = tmp_path / "tiny.bin"
    np.array([1.0], dtype=np.float32).tofile(filename)

    with pytest.warns(UserWarning, match="No output filename specified"):
        hdf5_file = write_velocity_model(
            filename=str(filename),
            model_type="bin",
            nz=1,
            nx=1,
            ny=1,
            byte_order="little",
            axes_order=(0, 1, 2),
            axes_order_sort="C",
            dtype="float32",
        )

    expected = np.array([[[1.0]]], dtype=np.float32)
    assert hdf5_file == str(filename) + ".hdf5"
    _assert_hdf5_matches_expected(hdf5_file, expected)
