"""Heterogeneous 2D isotropic elastic material model for the cpw calculator.

A pre-salt section: post-salt sediments over a salt body over a pre-salt
carbonate over the basement. Velocities are in km/s and densities in
g/cm^3; z is negative below the surface. Its P-wave velocity also serves
as a heterogeneous acoustic model.
"""

import numpy as np

# Salt properties
SALT_P_WAVE_VELOCITY = 4.5
SALT_S_WAVE_VELOCITY = 2.3
SALT_DENSITY = 2.20

# Basement properties, below this depth
BASEMENT_DEPTH = 3.8
BASEMENT_P_WAVE_VELOCITY = 6.2
BASEMENT_S_WAVE_VELOCITY = 3.5
BASEMENT_DENSITY = 2.85


def elastic_2d_material(
    z: np.ndarray, x: np.ndarray, length_x: float
) -> dict[str, np.ndarray]:
    """Evaluate the material of the pre-salt model.

    Parameters
    ----------
    z : np.ndarray
        Vertical coordinates, negative below the surface.
    x : np.ndarray
        Horizontal coordinates, broadcastable against ``z``.
    length_x : float
        Domain length in the x direction, the period of the salt
        interface undulation.

    Returns
    -------
    dict
        ``"p_wave_velocity"``, ``"s_wave_velocity"`` and ``"density"``
        arrays, of the broadcast shape of ``z`` and ``x``.
    """
    z, x = np.broadcast_arrays(np.asarray(z, dtype=float), np.asarray(x, dtype=float))
    depth = -z

    depth_top_salt = (
        1.0
        + 0.15 * np.sin(2.0 * np.pi * x / length_x)
        + 0.10 * np.exp(-(((x - 3.0) / 1.0) ** 2))
    )
    depth_bottom_salt = (
        2.0
        + 0.25 * np.exp(-(((x - 5.0) / 1.8) ** 2))
        + 0.15 * np.sin(2.0 * np.pi * x / length_x)
    )
    above_salt = depth < depth_top_salt
    inside_salt = ~above_salt & (depth <= depth_bottom_salt)
    above_basement = depth < BASEMENT_DEPTH

    def layered(sediment, salt, presalt, basement):
        """Select each layer's value, from the top down."""
        return np.where(
            above_salt,
            sediment,
            np.where(inside_salt, salt, np.where(above_basement, presalt, basement)),
        )

    return {
        "p_wave_velocity": layered(
            2.5 + 0.25 * depth,
            SALT_P_WAVE_VELOCITY,
            5.3 + 0.15 * depth,
            BASEMENT_P_WAVE_VELOCITY,
        ),
        "s_wave_velocity": layered(
            1.2 + 0.12 * depth,
            SALT_S_WAVE_VELOCITY,
            2.8 + 0.08 * depth,
            BASEMENT_S_WAVE_VELOCITY,
        ),
        "density": layered(
            2.10 + 0.05 * depth,
            SALT_DENSITY,
            2.55 + 0.02 * depth,
            BASEMENT_DENSITY,
        ),
    }
