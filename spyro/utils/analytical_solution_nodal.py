from firedrake.petsc import PETSc
from math import pi as PI
from mpi4py import MPI
import numpy as np
from numpy.linalg import norm
import os
from scipy.integrate import quad
from scipy.special import hankel2
import matplotlib.pyplot as plt
from spyro.sources import full_ricker_wavelet


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


def plot_analytical_displacement_components(
        time_vector,
        displacement_tuple,
        source_type="Unknown",
        save_plots=False,
        output_dir="."
    ):
    """
    Plot analytical displacement components (ux, uy, uz) over time.
    
    Parameters:
    ----------
    time_vector : numpy array
        Time vector
    displacement_tuple : tuple of numpy arrays
        (ux, uy, uz) displacement components
    source_type : str, optional
        Type of source ("force_source", "explosive_source", etc.)
    save_plots : bool, optional
        Whether to save plots to files
    output_dir : str, optional
        Directory to save plots if save_plots is True
        
    Returns:
    -------
    None
        Creates matplotlib figures
    """
    ux, uy, uz = displacement_tuple
    
    # Create the plot with separated subplots
    plt.figure(figsize=(12, 8))
    
    # Plot all three components
    plt.subplot(3, 1, 1)
    plt.plot(time_vector, ux, 'b-', linewidth=2, label='Ux (displacement in x)')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title(f'Displacement Component Ux - {source_type}')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.subplot(3, 1, 2)
    plt.plot(time_vector, uy, 'r-', linewidth=2, label='Uy (displacement in y)')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title(f'Displacement Component Uy - {source_type}')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.subplot(3, 1, 3)
    plt.plot(time_vector, uz, 'g-', linewidth=2, label='Uz (displacement in z)')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title(f'Displacement Component Uz - {source_type}')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    
    # Also create a combined plot
    plt.figure(figsize=(12, 6))
    plt.plot(time_vector, ux, 'b-', linewidth=2, label='Ux')
    plt.plot(time_vector, uy, 'r-', linewidth=2, label='Uy')
    plt.plot(time_vector, uz, 'g-', linewidth=2, label='Uz')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title(f'All Displacement Components - {source_type}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_plots:
        import os
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        basename = source_type.replace(" ", "_")
        
        # Save plots
        plt.figure(1)  # Select the first figure (subplots)
        plt.savefig(os.path.join(output_dir, f"analytical_{basename}_displacement_components_separated.png"), 
                   dpi=300, bbox_inches='tight')
        
        plt.figure(2)  # Select the second figure (combined)
        plt.savefig(os.path.join(output_dir, f"analytical_{basename}_displacement_components_combined.png"), 
                   dpi=300, bbox_inches='tight')
        
        print(f"Plots saved to {output_dir}")
    
    plt.show()


def demo_analytical_solutions():
    """
    Demonstration function to show how to use the analytical solutions
    and create plots for both force source and explosive source.
    Parameters match those from from_eduardos_Code.py
    """
    # Parameters matching from_eduardos_Code.py defaults
    offset = np.sqrt(100**2 + 0**2 + 100**2)  # Distance calculated from receiver position (100, 0, 100)
    alpha = 1500.0  # P-wave velocity in m/s (matches default)
    beta = 1000.0   # S-wave velocity in m/s (matches default)
    rho = 2000.0    # Density in kg/m³ (matches default)
    amplitude = 1e3 # Source amplitude (matches default)
    frequency = 20.0 # Source frequency in Hz (matches default)
    time_delay = 1/frequency # Time delay = 1/f0 (matches from_eduardos_Code.py)
    final_time = 0.3 # Final time in seconds (matches default)
    nt = 750 + 1
    dt = final_time/(nt - 1) # Time step calculated from final_time/nt
    
    print("Computing analytical solutions...")
    print(f"Parameters:")
    print(f"  Offset: {offset:.1f} m")
    print(f"  P-wave velocity (alpha): {alpha} m/s")
    print(f"  S-wave velocity (beta): {beta} m/s")
    print(f"  Density (rho): {rho} kg/m³")
    print(f"  Source amplitude: {amplitude}")
    print(f"  Frequency: {frequency} Hz")
    print(f"  Time delay: {time_delay:.3f} s")
    print(f"  Final time: {final_time} s")
    print(f"  Number of time steps: {nt}")
    print(f"  Time step (dt): {dt:.6f} s")
    
    # Force source
    print("\n1. Force Source:")
    force_result = analytical_solution_elastic(
        source_type="force_source",
        offset=offset,
        alpha=alpha,
        beta=beta,
        rho=rho,
        amplitude=amplitude,
        frequency=frequency,
        time_delay=time_delay,
        final_time=final_time,
        dt=dt
    )
    
    time_vector = np.linspace(0.0, final_time, nt)
    
    plot_analytical_displacement_components(
        time_vector,
        force_result,
        source_type="Force Source",
        save_plots=True,
        output_dir="analytical_plots"
    )
    
    # Explosive source
    print("\n2. Explosive Source:")
    explosive_result = analytical_solution_elastic(
        source_type="explosive_source",
        offset=offset,
        alpha=alpha,
        beta=beta,  # Not used for explosive source
        rho=rho,
        amplitude=amplitude,
        frequency=frequency,
        time_delay=time_delay,
        final_time=final_time,
        dt=dt
    )
    
    plot_analytical_displacement_components(
        time_vector,
        explosive_result,
        source_type="Explosive Source",
        save_plots=True,
        output_dir="analytical_plots"
    )
    
    print("\nDemo completed! Check the 'analytical_plots' directory for saved figures.")
    print("Parameters match those used in from_eduardos_Code.py")


if __name__ == "__main__":
    # Run the demo when script is executed directly
    demo_analytical_solutions()

