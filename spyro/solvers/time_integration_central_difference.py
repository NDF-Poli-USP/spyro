import firedrake as fire

from firedrake import *

from . import helpers
from .. import utils


def central_difference(wave, source_ids=[0]):
    """
    Perform central difference time integration for wave propagation.

    Parameters:
    -----------
    wave: Spyro object
        The Wave object containing the necessary data and parameters.
    source_ids: list of ints (optional)
        The ID of the sources being propagated. Defaults to [0].

    Returns:
    --------
        tuple:
            A tuple containing the forward solution and the receiver output.
    """
    if wave.sources is not None:
        wave.sources.current_sources = source_ids
        rhs_forcing = fire.Cofunction(wave.function_space.dual())

    wave.field_logger.start_logging(source_ids)

    wave.comm.comm.barrier()

    t = wave.current_time
    nt = int(wave.final_time / wave.dt) + 1  # number of timesteps

    usol = [
        fire.Function(wave.function_space, name=wave.get_function_name())
        for t in range(nt)
        if t % wave.gradient_sampling_frequency == 0
    ]
    usol_recv = []
    save_step = 0
    for step in range(nt):
        # Basic way of applying sources
        wave.update_source_expression(t)
        fire.assemble(wave.rhs, tensor=wave.B)

        # More efficient way of applying sources
        if wave.sources is not None:
            f = wave.sources.apply_source(rhs_forcing, step)
            B0 = wave.rhs_no_pml()
            B0 += f

        wave.solver.solve(wave.next_vstate, wave.B)

        wave.prev_vstate = wave.vstate
        wave.vstate = wave.next_vstate
        
        if wave.viscoelastic:
            from .viscoelasticity_functions import (sigma_visco_kelvin, epsilon)
            if wave.visco_type == 'kelvin_voigt':
                wave.eps_old = epsilon(wave.vstate)
                
            elif wave.visco_type == "zener":    
                V = wave.function_space
                dte = wave.tau_epsilon / wave.dt
                dts = wave.tau_sigma / wave.dt
                eps_new = epsilon(wave.prev_vstate)
                dim = V.mesh().topological_dimension()
                I = Identity(dim)

                # Implicit elastic term (Backward Euler)
                elastic_part = wave.lmbda * tr(eps_new + dte * (eps_new - wave.eps_old)) * I \
                    + 2.0 * wave.mu * (eps_new + dte * (eps_new - wave.eps_old))

                # Stress update (implicit relaxation)
                sigma_new = (elastic_part + dts * wave.sigma_old) / (1.0 + dts)

                # Update of internal states
                wave.eps_old.assign(project(eps_new, wave.eps_old.function_space()))
                wave.sigma_old = (sigma_new)

            elif wave.visco_type == "gsls":
                V = wave.function_space
                W = wave.strain_space
                eps_new = epsilon(wave.prev_vstate)
                dim = V.mesh().topological_dimension()
                I = Identity(dim)
                n_branches = len(wave.tau_epsilons)
                lmbda_share = wave.lmbda / n_branches
                mu_share = wave.mu / n_branches

                for i in range(len(wave.tau_epsilons)):
                    dte = wave.tau_epsilons[i] / wave.dt
                    dts = wave.tau_sigmas[i] / wave.dt

                    # Background elastic term
                    elastic_term = lmbda_share * div(wave.prev_vstate) * I + 2 * mu_share * eps_new

                    eps_old = wave.eps_old_list[i]
                    sigma_old = wave.sigma_old_list[i]

                    # Implicit elasticity (Backward Euler)
                    viscous_term = dte * (eps_new - eps_old)
                    memory_term = dts * sigma_old

                    sigma_new = (elastic_term + viscous_term + memory_term) / (1.0 + dts)

                    # Update of internal states (stable form)
                    sigma_proj = Function(W)
                    eps_proj = Function(W)

                    sigma_proj.assign(project(sigma_new, W))
                    eps_proj.assign(project(eps_new, W))

                    # Update memory variables
                    wave.sigma_old_list[i].assign(sigma_proj)
                    wave.eps_old_list[i].assign(eps_proj)

            elif wave.visco_type == 'maxwell':
                    sigma_old = wave.sigma_old
                    eps_old   = wave.eps_old

                    V = wave.function_space
                    W = wave.strain_space

                    dt = Constant(wave.dt)
                    dim = V.mesh().topological_dimension()
                    I = Identity(dim)
                    eps = lambda w: 0.5*(grad(w) + grad(w).T)
                
                    # Strains at times n and n+1
                    eps_n   = project(eps(wave.prev_vstate), W)  # ensure it is in W
                    eps_np1 = project(eps(wave.vstate), W)
                    
                    lambda_m     = wave.lmbda
                    mu_m         = wave.mu
                    tau_eps = wave.tau_epsilon
                    tau_sig = wave.tau_sigma

                    sigma_old = wave.sigma_old
                    eps_old   = wave.eps_old

                    tau_e = Constant(tau_eps)
                    tau_s = Constant(tau_sig)

                    # Action of C_m on a tensor X: C_m:X = λ_m tr(X) I + 2 μ_m X
                    def C_m_action(X):
                        return lambda_m*tr(X)*Identity(dim) + 2.0*mu_m*X

                    num = sigma_old + C_m_action(eps_np1*(1.0 + dt/tau_e) - eps_n)
                    sigma_np1 = project(num / (1.0 + dt/tau_s), W)

                    sigma_old.assign(sigma_np1)
                    eps_old.assign(eps_np1)
                    
            elif wave.visco_type == 'maxwell_gsls':
                    sigma_old_list = wave.sigma_old_list
                    eps_old_list   = wave.eps_old_list

                    V = wave.function_space
                    W = wave.strain_space

                    dt = Constant(wave.dt)
                    dim = V.mesh().topological_dimension()
                    I = Identity(dim)
                    eps = lambda w: 0.5*(grad(w) + grad(w).T)
                
                    # Strains at times n and n+1
                    eps_n   = project(eps(wave.prev_vstate), W)  # ensure it is in W
                    eps_np1 = project(eps(wave.vstate), W)
                
                    lambda_m     = wave.lmbda_s
                    mu_m         = wave.mu_s
                    tau_eps_list = wave.tau_epsilons
                    tau_sig_list = wave.tau_sigmas

                    sigma_old_list = wave.sigma_old_list
                    eps_old_list   = wave.eps_old_list

                    # Per-branch update
                    for i in range(len(sigma_old_list)):
                        
                        tau_e = Constant(tau_eps_list[i])
                        tau_s = Constant(tau_sig_list[i])

                        def C_m_action(X):
                            return lambda_m[i]*tr(X)*Identity(dim) + 2.0*mu_m[i]*X

                        # Closed-form BE formula for GSLS
                        num = sigma_old_list[i] + C_m_action(eps_np1*(1.0 + dt/tau_e) - eps_n)
                        sigma_np1 = project(num / (1.0 + dt/tau_s), W)

                        sigma_old_list[i].assign(sigma_np1)

                        eps_old_list[i].assign(eps_np1)
                
        usol_recv.append(wave.get_receivers_output())

        if step % wave.gradient_sampling_frequency == 0:
            usol[save_step].assign(wave.get_function())
            save_step += 1

        if (step - 1) % wave.output_frequency == 0:
            assert (
                fire.norm(wave.get_function()) < 1
            ), "Numerical instability. Try reducing dt or building the " \
               "mesh differently"
            wave.field_logger.log(t)
            helpers.display_progress(wave.comm, t)
        
        

        t = step * float(wave.dt)

    wave.current_time = t
    helpers.display_progress(wave.comm, t)

    usol_recv = helpers.fill(
        usol_recv, wave.receivers.is_local, nt, wave.receivers.number_of_points
    )
    usol_recv = utils.utils.communicate(usol_recv, wave.comm)
    wave.receivers_output = usol_recv

    wave.forward_solution = usol
    wave.forward_solution_receivers = usol_recv

    wave.field_logger.stop_logging()
    

    return usol, usol_recv
