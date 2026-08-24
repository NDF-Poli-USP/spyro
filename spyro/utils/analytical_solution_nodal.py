"""Methods that calculate analytical solutions for implementation verification."""

import numpy as np
from scipy.integrate import quad
from scipy.special import hankel2

from ..sources import full_ricker_wavelet
from .typing import SourceType


def nodal_homogeneous_analytical(
    wave, offset: float, c_value: float, n_extra: int = 5000
):
    """Calculate the acoustic analytical solution for a homogeneous medium.

    The solution considers a single source and receiver separated by a given
    offset. The source time function is extended in time before computing the
    analytical solution to reduce artifacts associated with the Fourier
    transform. It is the analytical solution for the acoustic wave.

    Parameters
    ----------
    wave : spyro.Wave
        Wave object containing the time-step, final time, source frequency,
        source delay, and delay type.
    offset : float
        Distance between the source and receiver.
    c_value : float
        Wave propagation velocity in the homogeneous medium.
    n_extra : int, optional
        Factor used to extend the final simulation time when computing the
        analytical solution. The default is 5000.

    Returns
    -------
    numpy.ndarray
        Analytical wavefield evaluated at the time steps of ``wave``.
    """
    # Generating extended ricker wavelet
    dt = wave.dt
    final_time = wave.final_time
    num_t = int(final_time / dt) + 1

    extended_final_time = n_extra * final_time

    frequency = wave.frequency
    delay = wave.delay
    delay_type = wave.delay_type

    ricker_wavelet = full_ricker_wavelet(
        dt=dt,
        final_time=extended_final_time,
        frequency=frequency,
        delay=delay - dt,
        delay_type=delay_type,
    )

    full_u_analytical = _analytical_solution(
        ricker_wavelet, c_value, extended_final_time, offset
    )

    u_analytical = full_u_analytical[:num_t]

    return u_analytical


def _analytical_solution(
    ricker_wavelet: np.ndarray,
    c_value: float,
    final_time: float,
    offset: float,
):
    """Calculate the analytical solution for a homogeneous acoustic medium.

    The solution is computed in the frequency domain using the Hankel function
    of the second kind and then transformed back to the time domain.

    Parameters
    ----------
    ricker_wavelet : numpy.ndarray
        Source Ricker wavelet sampled in time.
    c_value : float
        Wave propagation velocity in the homogeneous medium.
    final_time : float
        Total duration of the time-domain signal.
    offset : float
        Distance between the source and receiver.

    Returns
    -------
    numpy.ndarray
        Analytical solution in the time domain.
    """
    num_t = len(ricker_wavelet)

    # Constantes de Fourier
    nf = int(num_t / 2 + 1)
    frequency_axis = (1.0 / final_time) * np.arange(nf)

    # FOurier tranform of ricker wavelet
    fft_rw = np.fft.fft(ricker_wavelet)
    fft_rw = fft_rw[0:nf]

    U_a = np.zeros((nf), dtype=complex)
    for a in range(1, nf - 1):
        k = 2 * np.pi * frequency_axis[a] / c_value
        tmp = k * offset
        U_a[a] = -1j * np.pi * hankel2(0.0, tmp) * fft_rw[a]

    U_t = 1.0 / (2.0 * np.pi) * np.real(np.fft.ifft(U_a[:], num_t))

    return np.real(U_t)


def analytical_solution_elastic(
    source_type: SourceType | str,
    offsets: np.ndarray | float,
    p_wave_velocity: float,
    s_wave_velocity: float,
    density: float,
    amplitude: float,
    frequency: float,
    time_delay: float,
    final_time: float,
    dt: float,
    force_direction: int = None,
    dimension: int = 3,
):
    """Calculate an analytical elastic-wave solution.

    The solution is computed for either a force source or an explosive source
    in a three-dimensional homogeneous elastic medium.

    Parameters
    ----------
    source_type : enum or str
        Type of source. Supported values are ``"force_source"`` and
        ``"explosive_source"``.
    offsets : numpy.ndarray
        Vector from the source to the receiver.
    p_wave_velocity : float
        P-wave velocity.
    s_wave_velocity : float
        S-wave velocity. Required for a force source.
    density : float
        Medium density.
    amplitude : float
        Source amplitude.
    frequency : float
        Source frequency.
    time_delay : float
        Source time delay.
    final_time : float
        Final time of the simulation.
    dt : float
        Time-step size.
    force_direction : int, optional
        Direction of the applied force, where ``0``, ``1``, and ``2`` represent
        the z-, x-, and y-directions, respectively. Required for a force
        source.
    dimension : int, optional
        Number of spatial dimensions. Only three-dimensional problems are
        currently supported. The default is 3.

    Returns
    -------
    tuple of numpy.ndarray
        Three displacement components ``(uz, ux, uy)``.

    Raises
    ------
    ValueError
        If ``dimension`` is not 3, if a force source is specified without a
        force direction, or if an unsupported source type is provided.
    """
    if isinstance(source_type, str):
        source_type = SourceType(source_type)
    if dimension not in [2, 3]:
        raise ValueError("Weird dimensions not yet supported")
    if force_direction is None and source_type == "force_source":
        raise ValueError(f"Can not use {source_type} with no force_direction")
    if dimension == 2 and source_type == "explosive_source":
        raise ValueError("Explosive source only supported for 3D simulations.")

    nt = int(final_time / dt + 1)
    final_time = dt * (nt - 1)
    time_vector = np.linspace(0.0, final_time, nt)
    if dimension == 2:
        u = np.zeros((nt, 2))
    elif dimension == 3:
        u = np.zeros((nt, 3))
    if source_type == SourceType.FORCE and dimension == 3:
        for i in range(dimension):
            u[:, i] = analytical_force_source_3d(
                offsets,
                time_vector,
                p_wave_velocity,
                s_wave_velocity,
                density,
                amplitude,
                frequency,
                time_delay,
                force_direction,
                i,
            )
    elif source_type == SourceType.EXPLOSIVE:
        for i in range(dimension):
            u[:, i] = analytical_explosive_source(
                offsets,
                time_vector,
                p_wave_velocity,
                density,
                amplitude,
                frequency,
                time_delay,
                i,
            )
    elif source_type == SourceType.FORCE and dimension == 2:
        for i in range(dimension):
            u[:, i] = analytical_force_source_2d(
                offsets,
                time_vector,
                p_wave_velocity,
                s_wave_velocity,
                density,
                amplitude,
                frequency,
                time_delay,
                force_direction,
                i,
            )
    else:
        raise ValueError(f"Source type of {source_type} not valid")

    if dimension == 2:
        return (u[:, 0], u[:, 1])
    else:
        return (u[:, 0], u[:, 1], u[:, 2])


def analytical_force_source_3d(
    offsets: float,
    time_vector: np.ndarray,
    p_wave_velocity: float,
    s_wave_velocity: float,
    density: float,
    amplitude: float,
    frequency: float,
    time_delay: float,
    force_direction: int,
    displacement_direction: int,
):
    r"""Calculate the analytical displacement solution for a force source.

    The solution includes near-field, P-wave far-field, and S-wave
    far-field contributions for a point force in a homogeneous isotropic
    elastic medium.

    Based on Aki and Richards (2002). #TODO: citation

    Parameters
    ----------
    offset : float
        Distance between source and receiver
    time_vector : numpy array
        Time vector
    p_wave_velocity : float
        P-wave velocity
    s_wave_velocity : float
        S-wave velocity
    density : float
        Density
    amplitude : float
        Source amplitude
    frequency : float
        Source frequency
    time_delay : float
        Source time delay
    force_direction : int
        Direction of the applied force, where ``0``, ``1``, and ``2``
        correspond to the z-, x-, and y-directions, respectively.
    displacement_direction : int
        Direction of the displacement component being calculated, where
        ``0``, ``1``, and ``2`` correspond to the z-, x-, and y-directions,
        respectively.

    Returns
    -------
    tuple of numpy arrays
        (ux, uy, uz) displacement components

    Notes
    -----
    The analytical solution is based on the displacement field generated by
    a point force in a homogeneous isotropic elastic medium. For displacement
    component :math:`u_i` and force direction :math:`j`, the solution is

    .. math::

        u_i =
        \frac{A}{4\pi\rho}
        (3\gamma_i\gamma_j - \delta_{ij})
        \frac{1}{r^3}
        \int_{r/\alpha}^{r/\beta}
        \tau X_0(t-\tau)\,d\tau

        + \frac{A}{4\pi\rho\alpha^2}
        \gamma_i\gamma_j
        \frac{1}{r}
        X_0\left(t-\frac{r}{\alpha}\right)

        - \frac{A}{4\pi\rho\beta^2}
        (\gamma_i\gamma_j-\delta_{ij})
        \frac{1}{r}
        X_0\left(t-\frac{r}{\beta}\right),

    where :math:`r` is the source-receiver distance,
    :math:`\gamma_i = x_i/r`, :math:`\delta_{ij}` is the Kronecker delta,
    :math:`\alpha` is the P-wave velocity, and :math:`\beta` is the
    S-wave velocity.

    The source time function used here is

    .. math::

        X_0(t) =
        \left(1 - 2a^2\right)e^{-a^2},

    with

    .. math::

        a = \pi f (t-t_0),

    where :math:`f` is the source frequency and :math:`t_0` is the time delay.
    """
    nt = len(time_vector)
    r = np.linalg.norm(offsets)
    i = displacement_direction
    j = force_direction

    gamma_i = offsets[i] / r
    gamma_j = offsets[j] / r
    delta_ij = 1 if i == j else 0

    def X0(t):
        """Source time function (Ricker wavelet derivative)."""
        a = np.pi * frequency * (t - time_delay)
        return (1 - 2 * a**2) * np.exp(-(a**2))

    # Initialize displacement components
    ui = np.zeros(nt)

    for k in range(nt):
        t = time_vector[k]

        # Near field contribution (convolution integral term)
        res = quad(
            lambda tau: tau * X0(t - tau), r / p_wave_velocity, r / s_wave_velocity
        )
        u_near = (
            amplitude
            * (1.0 / (4 * np.pi * density))
            * (3 * gamma_i * gamma_j - delta_ij)
            * (1.0 / r**3)
            * res[0]
        )

        # P-wave far-field
        P_far = (
            amplitude
            * (1.0 / (4 * np.pi * density * p_wave_velocity**2))
            * gamma_i
            * gamma_j
            * (1.0 / r)
            * X0(t - r / p_wave_velocity)
        )

        # S-wave far field
        S_far = (
            amplitude
            * (1.0 / (4 * np.pi * density * s_wave_velocity**2))
            * (gamma_i * gamma_j - delta_ij)
            * (1.0 / r)
            * X0(t - r / s_wave_velocity)
        )

        ui[k] = u_near + P_far - S_far

    return ui


def analytical_explosive_source(
    offsets: np.ndarray,
    time_vector: np.ndarray,
    p_wave_velocity: float,
    density: float,
    amplitude: float,
    frequency: float,
    time_delay: float,
    displacement_direction: int,
):
    r"""Calculate the analytical displacement solution for an explosive source.

    The solution includes the intermediate-field and far-field P-wave
    contributions from an explosive point source in a homogeneous isotropic
    elastic medium.

    Based on Aki and Richards (2002).

    #TODO: add citation

    Parameters
    ----------
    offset : numpy.ndarray
        Vector representing distance from the source to the receiver.
    time_vector : numpy array
        Time vector
    p_wave_velocity : float
        P-wave velocity
    density : float
        Density
    amplitude : float
        Source amplitude
    frequency : float
        Source frequency
    time_delay : float
        Source time delay
    displacement_direction : int
        Direction of the displacement component being calculated, where
        ``0``, ``1``, and ``2`` correspond to the z-, x-, and y-directions,
        respectively.

    Returns
    -------
    tuple of numpy arrays
        (ux, uy, uz) displacement components

    Notes
    -----
    For an explosive point source in a homogeneous isotropic elastic medium,
    the displacement component :math:`u_i` is given by

    .. math::

        u_i =
        \frac{A\gamma_i}{4\pi\rho\alpha^2 r^2}
        w\left(t-\frac{r}{\alpha}\right)
        +
        \frac{A\gamma_i}{4\pi\rho\alpha^3 r}
        \dot{w}\left(t-\frac{r}{\alpha}\right),

    where :math:`r` is the source-receiver distance and
    :math:`\gamma_i = x_i/r`.

    The source time function is

    .. math::

        w(t) = (t-t_0)
        \exp\left[-\left(\pi f(t-t_0)\right)^2\right],

    and its derivative is

    .. math::

        \dot{w}(t) =
        \left[1 - 2\left(\pi f(t-t_0)\right)^2\right]
        \exp\left[-\left(\pi f(t-t_0)\right)^2\right].
    """
    nt = len(time_vector)
    i = displacement_direction
    r = np.linalg.norm(offsets)
    gamma_i = offsets[i] / r

    def w(t):
        """Get source time function (integral of Ricker wavelet)."""
        a = np.pi * frequency * (t - time_delay)
        return (t - time_delay) * np.exp(-(a**2))

    def w_dot(t):
        """Get derivative of source time function (Ricker wavelet)."""
        a = np.pi * frequency * (t - time_delay)
        return (1 - 2 * a**2) * np.exp(-(a**2))

    # Initialize displacement components
    ui = np.zeros(nt)

    for k in range(nt):
        t = time_vector[k]

        # P wave intermediate field
        P_mid = (
            amplitude
            * (gamma_i / (4 * np.pi * density * p_wave_velocity**2))
            * (1.0 / r**2)
            * w(t - r / p_wave_velocity)
        )

        # P wave far field
        P_far = (
            amplitude
            * (gamma_i / (4 * np.pi * density * p_wave_velocity**3))
            * (1.0 / r)
            * w_dot(t - r / p_wave_velocity)
        )

        ui[k] = P_mid + P_far

    return ui


def analytical_force_source_2d(
    offsets: np.ndarray,
    time_vector: np.ndarray,
    alpha: float,
    beta: float,
    rho: float,
    amplitude: float,
    frequency: float,
    time_delay: float,
    force_direction: int,
    displacement_direction: int,
    n_extra: int = 10,
):
    r"""Calculate the 2D analytical displacement from a point force.

    The solution is the frequency-domain elastodynamic Green tensor for an
    unbounded, homogeneous, isotropic medium under plane-strain conditions.

    Parameters
    ----------
    offsets : numpy.ndarray
        Source-to-receiver offset vector.
    time_vector : numpy.ndarray
        Time samples.
    alpha : float
        P-wave velocity.
    beta : float
        S-wave velocity.
    rho : float
        Medium density.
    amplitude : float
        Force amplitude.
    frequency : float
        Source frequency.
    time_delay : float
        Source time delay.
    force_direction : int
        Direction of the applied force. Must be 0 or 1.
    displacement_direction : int
        Direction of the displacement component. Must be 0 or 1.
    n_extra: int = 10

    Returns
    -------
    numpy.ndarray
        Analytical displacement component.
    """
    if len(offsets) != 2:
        raise ValueError("2D offsets must contain exactly two components.")

    if force_direction not in (0, 1):
        raise ValueError("2D force_direction must be 0 or 1.")

    if displacement_direction not in (0, 1):
        raise ValueError("2D displacement_direction must be 0 or 1.")

    dt = time_vector[1] - time_vector[0]
    num_t = len(time_vector)

    extended_num_t = n_extra * num_t

    extended_time_vector = (
        np.arange(
            extended_num_t,
            dtype=float,
        )
        * dt
    )
    radius = np.linalg.norm(offsets)

    if radius == 0.0:
        raise ValueError("Source and receiver cannot have the same location.")

    # Direction cosines.
    gamma_i = offsets[displacement_direction] / radius
    gamma_j = offsets[force_direction] / radius

    delta_ij = float(displacement_direction == force_direction)

    # Number of time samples and corresponding postive frequencies.
    extended_num_t = len(extended_time_vector)

    num_frequencies = extended_num_t // 2 + 1
    frequency_axis = np.fft.rfftfreq(
        extended_num_t, d=extended_time_vector[1] - extended_time_vector[0]
    )

    # Source spectrum.
    source_wavelet = (
        1.0 - 2.0 * (np.pi * frequency * (extended_time_vector - time_delay)) ** 2
    ) * np.exp(-((np.pi * frequency * (extended_time_vector - time_delay)) ** 2))

    source_spectrum = np.fft.rfft(source_wavelet)

    displacement_spectrum = np.zeros(num_frequencies, dtype=complex)

    shear_modulus = rho * beta**2

    for frequency_index in range(1, num_frequencies):
        angular_frequency = 2.0 * np.pi * frequency_axis[frequency_index]

        k_p = angular_frequency / alpha
        k_s = angular_frequency / beta

        k_p_radius = k_p * radius
        k_s_radius = k_s * radius

        hankel_p = hankel2(0, k_p_radius)
        hankel_s = hankel2(0, k_s_radius)

        hankel_p_1 = hankel2(1, k_p_radius)
        hankel_s_1 = hankel2(1, k_s_radius)

        # Radial derivative of
        #
        # H_0^(2)(k_s r) - H_0^(2)(k_p r).
        radial_derivative = -k_s * hankel_s_1 + k_p * hankel_p_1

        second_derivative_radial = -(k_s**2) * hankel_s + k_p**2 * hankel_p

        green_tensor_component = (
            -1j
            / (4.0 * shear_modulus)
            * (
                hankel_s * delta_ij
                + (
                    second_derivative_radial * gamma_i * gamma_j
                    + (radial_derivative / radius) * delta_ij
                )
                / k_s**2
            )
        )

        displacement_spectrum[frequency_index] = (
            green_tensor_component * amplitude * source_spectrum[frequency_index]
        )

    displacement = np.fft.irfft(
        displacement_spectrum,
        n=extended_num_t,
    )

    return displacement[:num_t]
