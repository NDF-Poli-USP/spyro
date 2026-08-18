from math import pi as PI
import numpy as np
from scipy.integrate import quad
from scipy.special import hankel2
from ..sources import full_ricker_wavelet


def nodal_homogeneous_analytical(wave, offset, c_value, n_extra=5000):
    """
    This function calculates the analytical solution for an homogeneous
    medium with a single source and receiver.

    Parameters
    ----------
    wave: spyro.Wave
        Wave object
    offset: float
        Offset between source and receiver.
    c_value: float
        Velocity of the homogeneous medium.
    n_extra: int (optional)
        Multiplied factor for the final time.

    Returns
    -------
    u_analytical: numpy array
        Analytical solution for the wave equation.
    """

    # Generating extended ricker wavelet
    dt = wave.dt
    final_time = wave.final_time
    num_t = int(final_time / dt + 1)

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

    full_u_analytical = analytical_solution(
        ricker_wavelet, c_value, extended_final_time, offset
    )

    u_analytical = full_u_analytical[:num_t]

    return u_analytical


def analytical_solution(ricker_wavelet, c_value, final_time, offset):
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
    source_type,
    offsets,
    alpha,
    beta,
    rho,
    amplitude,
    frequency,
    time_delay,
    final_time,
    dt,
    force_direction=None,
    dimension=3,
):
    if dimension != 3:
        raise ValueError("2D or weird dimensions not yet supported")
    if force_direction is None and source_type == "force_source":
        raise ValueError(f"Can not use {source_type} with no force_direction")

    nt = int(final_time/dt + 1)
    final_time = dt*(nt-1)
    time_vector = np.linspace(0.0, final_time, nt)
    u = np.zeros((nt, 3))
    if source_type == "force_source":
        for i in range(dimension):
            u[:, i] = analytical_force_source(
                offsets,
                time_vector,
                alpha,
                beta,
                rho,
                amplitude,
                frequency,
                time_delay,
                force_direction,
                i,
            )
    elif source_type == "explosive_source":
        for i in range(dimension):
            u[:, i] = analytical_explosive_source(
                offsets,
                time_vector,
                alpha,
                rho,
                amplitude,
                frequency,
                time_delay,
                i,
            )
    else:
        raise ValueError(f"Source type of {source_type} not valid")

    return (u[:, 0], u[:, 1], u[:, 2])


def analytical_force_source(
    offsets,
    time_vector,
    alpha,
    beta,
    rho,
    amplitude,
    frequency,
    time_delay,
    force_direction,
    displacement_direction,
):
    """
    Analytical solution for force source based on Aki and Richards (2002)
    Returns displacement components (ux, uy, uz) for a force source.

    Parameters:
    ----------
    offset : float
        Distance between source and receiver
    time_vector : numpy array
        Time vector
    alpha : float
        P-wave velocity
    beta : float
        S-wave velocity
    rho : float
        Density
    amplitude : float
        Source amplitude
    frequency : float
        Source frequency
    time_delay : float
        Source time delay

    Returns:
    -------
    tuple of numpy arrays
        (ux, uy, uz) displacement components
    """
    nt = len(time_vector)
    r = np.linalg.norm(offsets)
    i = displacement_direction
    j = force_direction

    gamma_i = offsets[i]/r
    gamma_j = offsets[j]/r
    delta_ij = 1 if i == j else 0

    def X0(t):
        """Source time function (Ricker wavelet derivative)"""
        a = PI * frequency * (t - time_delay)
        return (1 - 2*a**2) * np.exp(-a**2)

    # Initialize displacement components
    ui = np.zeros(nt)

    for k in range(nt):
        t = time_vector[k]

        # Near field contribution (integral term)
        res = quad(lambda tau: tau*X0(t - tau), r/alpha, r/beta)
        u_near = amplitude * (1./(4*PI*rho)) * (3*gamma_i * gamma_j - delta_ij) * (1./r**3) * res[0]

        # P-wave far-field
        P_far = amplitude * (1./(4*PI*rho*alpha**2)) * gamma_i * gamma_j * (1./r) * X0(t - r/alpha)

        # S-wave far field
        S_far = amplitude * (1./(4*PI*rho*beta**2)) * (gamma_i*gamma_j - delta_ij) * (1./r) * X0(t - r/beta)

        ui[k] = u_near + P_far - S_far

    return ui


def analytical_explosive_source(
    offsets,
    time_vector,
    alpha,
    rho,
    amplitude,
    frequency,
    time_delay,
    displacement_direction,
):
    """
    Analytical solution for explosive source based on Aki and Richards (2002)
    Returns displacement components (ux, uy, uz) for an explosive source.

    Parameters:
    ----------
    offset : float
        Distance between source and receiver
    time_vector : numpy array
        Time vector
    alpha : float
        P-wave velocity
    rho : float
        Density
    amplitude : float
        Source amplitude
    frequency : float
        Source frequency
    time_delay : float
        Source time delay

    Returns:
    -------
    tuple of numpy arrays
        (ux, uy, uz) displacement components
    """
    nt = len(time_vector)
    i = displacement_direction
    r = np.linalg.norm(offsets)
    gamma_i = offsets[i]/r

    def w(t):
        """Source time function (integral of Ricker wavelet)"""
        a = PI * frequency * (t - time_delay)
        return (t - time_delay) * np.exp(-a**2)

    def w_dot(t):
        """Derivative of source time function (Ricker wavelet)"""
        a = PI * frequency * (t - time_delay)
        return (1 - 2*a**2) * np.exp(-a**2)

    # Initialize displacement components
    ui = np.zeros(nt)

    for k in range(nt):
        t = time_vector[k]

        # P wave intermediate field
        P_mid = amplitude * (gamma_i/(4*PI*rho*alpha**2)) * (1./r**2) * w(t - r/alpha)

        # P wave far field
        P_far = amplitude * (gamma_i/(4*PI*rho*alpha**3)) * (1./r) * w_dot(t - r/alpha)

        ui[k] = P_mid + P_far

    return ui
