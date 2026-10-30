"""Velocity grids, refinement densities and the adaptive mesh-size law.

Everything here lives on a regular grid over the padded domain, with arrays
indexed ``[iz, ix]`` and both coordinates increasing.

The adaptive law redistributes a budget of degrees of freedom according to a
density :math:`\\rho` (the squared, smoothed indicator):

.. math::

    h = \\operatorname{clip}\\left(\\alpha\\, h_k^{q/(q+2)}
        \\rho^{-1/(q+2)},\\ h_{\\min},\\ h_k\\right),

where :math:`h_k` is the local size of the current mesh, so a new mesh is
never coarser than the current one, :math:`q` is the observed convergence
order of the trace error and :math:`\\alpha` is set so that the new mesh
has the requested number of degrees of freedom. The floor
:math:`h_{\\min} = v_p / (c_{\\max} C_{eq} f)` keeps the time step of the
explicit scheme from collapsing, with :math:`C_{eq}` the cells per
wavelength of a uniform-CPW mesh with the same budget.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.ndimage import gaussian_filter


class VelocityGrid:
    """P-wave velocity sampled on a regular ``(z, x)`` grid.

    Values outside the grid are taken from its closest edge, so a model can be
    extended into an absorbing pad.

    Parameters
    ----------
    values : numpy.ndarray
        Velocities, shape ``(nz, nx)``, row ``i`` at depth ``z0 + i * dz``.
    z0 : float
        Depth of the first row (the deepest, since depths increase upwards).
    x0 : float
        Position of the first column.
    dz : float
        Row spacing.
    dx : float
        Column spacing.
    """

    def __init__(self, values: np.ndarray, z0: float, x0: float,
                 dz: float, dx: float) -> None:
        self.values = np.asarray(values, dtype=float)
        if self.values.ndim != 2 or min(self.values.shape) < 2:
            raise ValueError("The velocity grid must be 2D with at least 2x2 points.")
        self.z0 = float(z0)
        self.x0 = float(x0)
        self.dz = float(dz)
        self.dx = float(dx)

    def __call__(self, z: np.ndarray, x: np.ndarray) -> np.ndarray:
        """Bilinear interpolation of the velocity.

        Parameters
        ----------
        z : numpy.ndarray
            Depths.
        x : numpy.ndarray
            Horizontal positions, broadcastable with ``z``.

        Returns
        -------
        numpy.ndarray
            Velocities at the points.
        """
        nz, nx = self.values.shape
        fz = np.clip((np.asarray(z, dtype=float) - self.z0) / self.dz, 0, nz - 1.000001)
        fx = np.clip((np.asarray(x, dtype=float) - self.x0) / self.dx, 0, nx - 1.000001)
        i = fz.astype(int)
        j = fx.astype(int)
        tz = fz - i
        tx = fx - j
        v = self.values
        return ((1 - tz) * (1 - tx) * v[i, j] + tz * (1 - tx) * v[i + 1, j]
                + (1 - tz) * tx * v[i, j + 1] + tz * tx * v[i + 1, j + 1])


def gradient_density(velocity: np.ndarray, spacing: float, smoothing: float,
                     interior: np.ndarray) -> np.ndarray:
    """Refinement density of the velocity-gradient strategy.

    Parameters
    ----------
    velocity : numpy.ndarray
        Velocity on the grid, shape ``(nz, nx)``.
    spacing : float
        Grid spacing.
    smoothing : float
        Standard deviation of the Gaussian smoothing, in the units of
        ``spacing``.
    interior : numpy.ndarray
        Boolean mask of the region of interest; the density is zero outside.

    Returns
    -------
    numpy.ndarray
        :math:`(G_\\sigma * |\\nabla v_p|)^2` inside the region of interest.
    """
    gz, gx = np.gradient(velocity, spacing)
    rho = gaussian_filter(np.hypot(gz, gx), smoothing / spacing, mode="nearest") ** 2
    return np.where(interior, rho, 0.0)


def adjoint_density(centroids: np.ndarray, cell_indicators: np.ndarray,
                    shot_weights: np.ndarray, z: np.ndarray, x: np.ndarray,
                    smoothing: float) -> np.ndarray:
    """Refinement density of the adjoint (DWR) strategy.

    Each shot's absolute cell indicator is normalised to unit sum and weighted
    by ``shot_weights`` (the squared trace errors, so that the worst shot
    dominates). The result is deposited on the grid at the cell centroids,
    smoothed and divided by the grid-cell area.

    Parameters
    ----------
    centroids : numpy.ndarray
        Cell centroids ``(z, x)``, shape ``(n_cells, 2)``.
    cell_indicators : numpy.ndarray
        Indicator of each shot on each cell, shape ``(n_shots, n_cells)``.
    shot_weights : numpy.ndarray
        Weight of each shot, shape ``(n_shots,)``.
    z : numpy.ndarray
        Grid depths, equally spaced.
    x : numpy.ndarray
        Grid positions, with the same spacing.
    smoothing : float
        Standard deviation of the Gaussian smoothing.

    Returns
    -------
    numpy.ndarray
        Density on the grid, shape ``(len(z), len(x))``.
    """
    spacing = z[1] - z[0]
    combined = np.zeros(cell_indicators.shape[1])
    for weight, eta in zip(shot_weights, np.abs(cell_indicators)):
        total = eta.sum()
        if total > 0:
            combined += weight * eta / total
    iz = np.clip(np.round((centroids[:, 0] - z[0]) / spacing).astype(int), 0, len(z) - 1)
    ix = np.clip(np.round((centroids[:, 1] - x[0]) / spacing).astype(int), 0, len(x) - 1)
    deposit = np.zeros((len(z), len(x)))
    np.add.at(deposit, (iz, ix), combined)
    return gaussian_filter(deposit, smoothing / spacing, mode="nearest") / spacing**2


class AdaptiveSizeField:
    """Mesh-size field that redistributes a budget by a refinement density.

    Parameters
    ----------
    density : numpy.ndarray
        Refinement density on the grid, shape ``(nz, nx)``.
    current_size : numpy.ndarray
        Local size of the current mesh on the grid; the new mesh is never
        coarser.
    z : numpy.ndarray
        Grid depths, equally spaced.
    x : numpy.ndarray
        Grid positions, with the same spacing.
    order : float, optional
        Convergence order :math:`q` of the error. Default is ``2.7``.

    Attributes
    ----------
    h : numpy.ndarray or None
        Size on the grid for the last :meth:`set_scale`; ``None`` before.
    grid : VelocityGrid or None
        The same size as a grid that :func:`~spyro.adaptivity.generate_mesh`
        accepts; ``None`` before :meth:`set_scale`.
    h_min : numpy.ndarray
        Lower bound of the size, set by :meth:`set_floor`.
    """

    def __init__(self, density: np.ndarray, current_size: np.ndarray,
                 z: np.ndarray, x: np.ndarray, order: float = 2.7) -> None:
        if density.shape != current_size.shape or density.shape != (len(z), len(x)):
            raise ValueError("Density, current size and grid must have matching shapes.")
        rho = np.maximum(density, 1e-6 * max(float(density.max()), 1e-300))
        self.z = np.asarray(z)
        self.x = np.asarray(x)
        self.spacing = float(self.z[1] - self.z[0])
        self.h_max = np.asarray(current_size, dtype=float)
        self.h_min = np.zeros_like(self.h_max)
        self.shape = self.h_max ** (order / (order + 2)) * rho ** (-1.0 / (order + 2))
        self.h = None
        self.grid = None

    def set_floor(self, velocity: np.ndarray, frequency: float,
                  equivalent_cpw: float, cmax_ratio: float = 1.25) -> None:
        """Set :math:`h_{\\min} = v_p / (c_{\\max} C_{eq} f)`, capped by the current size.

        Parameters
        ----------
        velocity : numpy.ndarray
            Velocity on the grid.
        frequency : float
            Peak frequency of the source (Hz).
        equivalent_cpw : float
            Cells per wavelength of a uniform-CPW mesh with the target budget.
        cmax_ratio : float, optional
            Largest refinement relative to that mesh. Default ``1.25``.

        Returns
        -------
        None
            The floor is stored in :attr:`h_min`.
        """
        self.h_min = np.minimum(velocity / (cmax_ratio * equivalent_cpw * frequency),
                                self.h_max)

    def size(self, scale: float) -> np.ndarray:
        """Size on the grid for a given scale :math:`\\alpha`.

        Parameters
        ----------
        scale : float
            The factor :math:`\\alpha` of the law.

        Returns
        -------
        numpy.ndarray
            Clipped size on the grid.
        """
        return np.clip(scale * self.shape, self.h_min, self.h_max)

    def relative_cost(self, scale: float) -> float:
        """Integral of :math:`h^{-2}`, proportional to the number of DOFs.

        Parameters
        ----------
        scale : float
            The factor :math:`\\alpha` of the law.

        Returns
        -------
        float
            :math:`\\sum h^{-2} \\Delta^2` over the grid.
        """
        return float((self.size(scale) ** -2).sum() * self.spacing**2)

    def scale_for(self, target_cost: float) -> float:
        """Find the scale whose :meth:`relative_cost` matches a target.

        Parameters
        ----------
        target_cost : float
            Desired value of :meth:`relative_cost`.

        Returns
        -------
        float
            The scale, found by bisection in log space. Targets outside the
            reachable range return the closest bound.
        """
        lo, hi = 1e-8, 1e8
        for _ in range(200):
            mid = math.sqrt(lo * hi)
            if self.relative_cost(mid) > target_cost:
                lo = mid
            else:
                hi = mid
        return math.sqrt(lo * hi)

    def set_scale(self, scale: float) -> None:
        """Fix the scale of the size handed to the mesh generator.

        Parameters
        ----------
        scale : float
            The factor :math:`\\alpha` of the law.

        Returns
        -------
        None
            The size is stored in :attr:`h` and :attr:`grid`.
        """
        self.h = self.size(scale)
        self.grid = VelocityGrid(self.h, self.z[0], self.x[0],
                                 self.spacing, self.spacing)
