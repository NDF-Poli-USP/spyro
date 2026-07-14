from firedrake import dx


def acoustic_energy(wave):
    '''
    Calculates the acoustic energy as either the potential
    energy (if u is the pressure) or the kinetic energy (if
    u is the velocity).
    '''
    u = wave.get_function()
    velocity_model = wave.velocity_model

    return (0.5 * (u / velocity_model)**2) * dx
