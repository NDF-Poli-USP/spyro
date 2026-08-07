from pathlib import Path

import numpy as np
from scipy.interpolate import RegularGridInterpolator

try:
    import SeismicMesh
except ImportError:
    SeismicMesh = None


def _read_velocity_binary3D(
    fname,
    nz,
    nx,
    ny,
    byte_order="big",
    axes_order=(0, 1, 2),
    axes_order_sort="F",
    dtype="float32",
):
    """Read the volumetric model using Spyro ``(z, x, y)`` convention."""
    path = Path(fname)
    if not path.exists():
        raise FileNotFoundError(f"Velocity model not found: {path}")

    if byte_order not in ("big", "little"):
        raise ValueError("byte_order must be 'big' or 'little'.")
    if axes_order_sort not in ("C", "F"):
        raise ValueError("axes_order_sort must be 'C' or 'F'.")
    if sorted(tuple(axes_order)) != [0, 1, 2]:  # noqa: C414
        raise ValueError("axes_order must be a permutation of (0, 1, 2).")

    dt = np.dtype(dtype).newbyteorder(">" if byte_order == "big" else "<")
    raw = np.fromfile(path, dtype=dt)
    expected = int(nz) * int(nx) * int(ny)
    if raw.size != expected:
        raise ValueError(
            f"Velocity file contains {raw.size} values; expected {expected} "
            f"for shape ({nz}, {nx}, {ny})."
        )

    canonical_shape = (int(nz), int(nx), int(ny))
    inverse = np.argsort(np.asarray(axes_order, dtype=int))
    raw_shape = tuple(canonical_shape[i] for i in inverse)
    values = raw.reshape(raw_shape, order=axes_order_sort)
    values = values.transpose(tuple(axes_order))
    return np.asarray(values, dtype=np.float64)


def create_sizing_function3D(
    fname,
    hmin,
    bbox,
    wl,
    freq,
    pad_type=None,
    pad_size_x=0.0,
    pad_size_y=0.0,
    pad_size_z=0.0,
    grade=0.15,
    vp_water=None,
    nz=None,
    nx=None,
    ny=None,
    byte_order="big",
    axes_order=(0, 1, 2),
    axes_order_sort="F",
    dtype="float32",
):
    """Create a finite, positive 3-D wavelength sizing function.

    Coordinates are accepted in Spyro's ``(z, x, y)`` order.
    """
    if any(value is None for value in (nz, nx, ny)):
        raise ValueError("3-D sizing requires nz, nx and ny.")
    if wl is None or float(wl) <= 0.0:
        raise ValueError("wl must be positive.")
    if freq is None or float(freq) <= 0.0:
        raise ValueError("freq must be positive.")

    zmin, zmax, xmin, xmax, ymin, ymax = map(float, bbox)
    if not (zmin < zmax and xmin < xmax and ymin < ymax):
        raise ValueError(
            "bbox must be ordered as "
            "(zmin, zmax, xmin, xmax, ymin, ymax)."
        )

    requested_hmin = None
    if hmin is not None and float(hmin) > 0.0:
        requested_hmin = float(hmin)

    if vp_water is not None and float(vp_water) > 0.0:
        physical_positive_floor = (
            float(vp_water) / (float(freq) * float(wl))
        )
    elif requested_hmin is not None:
        physical_positive_floor = requested_hmin
    else:
        physical_positive_floor = np.finfo(np.float64).eps

    if requested_hmin is not None:
        physical_positive_floor = max(
            physical_positive_floor,
            requested_hmin,
        )

    def edge_extended_callable(base_callable):
        """Wrap any core interpolator with true nearest-edge extension."""
        def evaluate(coordinates):
            points = np.asarray(coordinates, dtype=np.float64)
            scalar_input = points.ndim == 1

            if scalar_input:
                points = points.reshape(1, 3)

            if points.ndim != 2 or points.shape[1] != 3:
                raise ValueError(
                    "3-D sizing coordinates must have shape (N, 3) "
                    "in (z, x, y) order."
                )
            if not np.all(np.isfinite(points)):
                raise ValueError(
                    "3-D sizing coordinates contain NaN or infinity."
                )

            projected = points.copy()
            projected[:, 0] = np.clip(
                projected[:, 0], zmin, zmax
            )
            projected[:, 1] = np.clip(
                projected[:, 1], xmin, xmax
            )
            projected[:, 2] = np.clip(
                projected[:, 2], ymin, ymax
            )

            values = np.asarray(
                base_callable(projected),
                dtype=np.float64,
            ).reshape(-1)

            if values.size != projected.shape[0]:
                raise ValueError(
                    "The 3-D sizing callable returned an unexpected "
                    f"number of values: {values.size} for "
                    f"{projected.shape[0]} coordinates."
                )
            if not np.all(np.isfinite(values)):
                first = int(
                    np.flatnonzero(~np.isfinite(values))[0]
                )
                raise ValueError(
                    "The 3-D sizing function returned NaN or infinity "
                    "inside the velocity-model box. First projected "
                    f"coordinate (z, x, y)="
                    f"{projected[first].tolist()}."
                )
            if np.any(values < 0.0):
                first = int(np.flatnonzero(values < 0.0)[0])
                raise ValueError(
                    "The 3-D sizing function returned a negative size "
                    f"{values[first]} at projected coordinate "
                    f"(z, x, y)={projected[first].tolist()}."
                )

            values = np.maximum(
                values,
                physical_positive_floor,
            )

            if scalar_input:
                return values[0]
            return values

        return evaluate

    if SeismicMesh is not None:
        try:
            seismic_kwargs = {
                "vp_water": vp_water,
                "freq": float(freq),
                "wl": float(wl),
                "grade": float(grade),
                "domain_pad": max(
                    float(pad_size_x),
                    float(pad_size_y),
                    float(pad_size_z),
                    abs(xmax - xmin),
                    abs(ymax - ymin),
                    abs(zmax - zmin),
                ),
                "pad_style": "edge",
                "nz": int(nz),
                "nx": int(nx),
                "ny": int(ny),
                "byte_order": byte_order,
                "axes_order": tuple(axes_order),
                "axes_order_sort": axes_order_sort,
            }
            if requested_hmin is not None:
                seismic_kwargs["hmin"] = requested_hmin

            base_ef = SeismicMesh.get_sizing_function_from_segy(
                fname,
                bbox,
                **seismic_kwargs,
            )
            ef = edge_extended_callable(base_ef)

            return (
                ef,
                physical_positive_floor,
                None,
                int(nz),
                int(nx),
                int(ny),
            )
        except (TypeError, ValueError, RuntimeError):
            pass

    velocity = _read_velocity_binary3D(
        fname=fname,
        nz=nz,
        nx=nx,
        ny=ny,
        byte_order=byte_order,
        axes_order=axes_order,
        axes_order_sort=axes_order_sort,
        dtype=dtype,
    )

    if vp_water is not None:
        velocity = np.where(
            velocity == 0.0,
            float(vp_water),
            velocity,
        )

    if np.any(~np.isfinite(velocity)):
        raise ValueError(
            "The velocity model contains NaN or infinity."
        )
    if np.any(velocity <= 0.0):
        raise ValueError(
            "The velocity model contains non-positive values. Set vp_water "
            "or preprocess the model before constructing the sizing field."
        )

    sizes = velocity / (float(freq) * float(wl))
    if requested_hmin is not None:
        sizes = np.maximum(sizes, requested_hmin)

    sizes = np.maximum(sizes, physical_positive_floor)

    z_axis = np.linspace(zmax, zmin, int(nz))[::-1]
    x_axis = np.linspace(xmin, xmax, int(nx))
    y_axis = np.linspace(ymin, ymax, int(ny))
    sizes_for_interpolation = sizes[::-1, :, :]

    interpolator = RegularGridInterpolator(
        (z_axis, x_axis, y_axis),
        sizes_for_interpolation,
        method="linear",
        bounds_error=True,
    )

    ef = edge_extended_callable(interpolator)

    return (
        ef,
        float(np.min(sizes)),
        float(np.max(sizes)),
        int(nz),
        int(nx),
        int(ny),
    )
