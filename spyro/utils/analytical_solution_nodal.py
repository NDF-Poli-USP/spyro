from firedrake.petsc import PETSc
from math import pi as PI
from mpi4py import MPI
import numpy as np
from numpy.linalg import norm
import os
from scipy.integrate import quad
from scipy.special import hankel2
from ..sources import full_ricker_wavelet


def nodal_homogeneous_analytical(Wave_object, offset, c_value, n_extra=5000):
    """
    This function calculates the analytical solution for an homogeneous
    medium with a single source and receiver.

    Parameters
    ----------
    Wave_object: spyro.Wave
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
    dt = Wave_object.dt
    final_time = Wave_object.final_time
    num_t = int(final_time / dt + 1)

    extended_final_time = n_extra * final_time

    frequency = Wave_object.frequency
    delay = Wave_object.delay
    delay_type = Wave_object.delay_type

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
        offset,
        alpha,
        beta,
        rho,
        amplitude,
        frequency,
        time_delay,
        final_time,
        dt,
    ):
    result_tuple = None
    nt = int(final_time/dt + 1)
    final_time = dt*(nt-1)
    time_vector = np.linspace(0.0, final_time, nt)
    if source_type == "force_source":
        result_tuple = analytical_force_source(
            offset,
            time_vector,
            alpha,
            beta,
            rho,
            amplitude,
            frequency,
            time_delay,
        )
    elif source_type == "explosive_source":
        result_tuple = analytical_explosive_source(
            offset,
            time_vector,
            alpha,
            rho,
            amplitude,
            frequency,
            time_delay,
        )
    else:
        raise ValueError(f"Source type of {source_type} not valid")

    return result_tuple


def analytical_force_source(
        offset,
        time_vector,
        alpha,
        beta, 
        rho,
        amplitude,
        frequency,
        time_delay,
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
    r = offset
    
    # Assuming receiver is at (r, 0, 0) relative to source for simplicity
    # This gives gamma_x = 1, gamma_y = 0, gamma_z = 0
    
    def X0(t):
        """Source time function (Ricker wavelet derivative)"""
        a = PI * frequency * (t - time_delay)
        return (1 - 2*a**2) * np.exp(-a**2)
    
    # Initialize displacement components
    ux = np.zeros(nt)
    uy = np.zeros(nt) 
    uz = np.zeros(nt)
    
    # For a force source in x-direction, we compute u_x
    # Using i=0, j=0 (x-component): gamma_i*gamma_j = 1, delta_ij = 1
    for k in range(nt):
        t = time_vector[k]
        
        # Near field contribution (integral term)
        res = quad(lambda tau: tau*X0(t - tau), r/alpha, r/beta)
        u_near = amplitude * (1./(4*PI*rho)) * (3*1*1 - 1) * (1./r**3) * res[0]
        
        # P-wave far-field
        P_far = amplitude * (1./(4*PI*rho*alpha**2)) * 1 * 1 * (1./r) * X0(t - r/alpha)
        
        # S-wave far field  
        S_far = amplitude * (1./(4*PI*rho*beta**2)) * (1*1 - 1) * (1./r) * X0(t - r/beta)
        
        ux[k] = u_near + P_far - S_far
    
    # For y and z components with force in x-direction
    # Using i=1,j=0 and i=2,j=0: gamma_i*gamma_j = 0, delta_ij = 0
    for k in range(nt):
        t = time_vector[k]
        
        # Near field (no contribution since 3*0 - 0 = 0)
        u_near = 0.0
        
        # P-wave far-field (no contribution since gamma_i*gamma_j = 0)
        P_far = 0.0
        
        # S-wave far field (no contribution since 0 - 0 = 0)
        S_far = 0.0
        
        uy[k] = u_near + P_far - S_far  # = 0
        uz[k] = u_near + P_far - S_far  # = 0
    
    return (ux, uy, uz)


def analytical_explosive_source(
        offset,
        time_vector,
        alpha,
        rho,
        amplitude, 
        frequency,
        time_delay,
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
    r = offset
    
    # Assuming receiver is at (r, 0, 0) relative to source
    # This gives gamma_x = 1, gamma_y = 0, gamma_z = 0
    
    def w(t):
        """Source time function (integral of Ricker wavelet)"""
        a = PI * frequency * (t - time_delay)
        return (t - time_delay) * np.exp(-a**2)
    
    def w_dot(t):
        """Derivative of source time function (Ricker wavelet)"""
        a = PI * frequency * (t - time_delay)
        return (1 - 2*a**2) * np.exp(-a**2)
    
    # Initialize displacement components
    ux = np.zeros(nt)
    uy = np.zeros(nt)
    uz = np.zeros(nt)
    
    # For explosive source, only x-component has non-zero displacement
    # (assuming receiver at (r,0,0))
    for k in range(nt):
        t = time_vector[k]
        
        # P wave intermediate field
        P_mid = amplitude * (1/(4*PI*rho*alpha**2)) * (1./r**2) * w(t - r/alpha)
        
        # P wave far field
        P_far = amplitude * (1/(4*PI*rho*alpha**3)) * (1./r) * w_dot(t - r/alpha)
        
        ux[k] = P_mid + P_far
    
    # y and z components are zero for explosive source with receiver at (r,0,0)
    # (since gamma_y = gamma_z = 0)
    
    return (ux, uy, uz)

