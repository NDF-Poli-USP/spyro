import firedrake as fire
from firedrake import Constant, dx, dot, grad

from .CG_acoustic import AcousticWave
from ..io.io import ensemble_propagator
from . import helpers
from .. import utils
from ..domains.quadrature import quadrature_rules

class AcousticNodalPropagation(AcousticWave):
    @ensemble_propagator
    def wave_propagator(self, dt = None, final_time = None, source_num = 0):
        """ Propagates the wave forward in time.
        Currently uses central differences.

        Parameters:
        -----------
        dt: Python 'float' (optional)
            Time step to be used explicitly. If not mentioned uses the default,
            that was estabilished in the model_parameters.
        final_time: Python 'float' (optional)
            Time which simulation ends. If not mentioned uses the default,
            that was estabilished in the model_parameters.
        """
        excitations = self.sources
        excitations.current_source = source_num
        receivers = self.receivers
        comm = self.comm
        temp_filename = self.forward_output_file
        filename, file_extension = temp_filename.split(".")
        output_filename = filename+'sn'+str(source_num)+"."+file_extension
        print(output_filename, flush = True)

        output = fire.File(output_filename, comm=comm.comm)
        comm.comm.barrier()

        X = fire.Function(self.function_space)
        if final_time == None:
            final_time = self.final_time
        if dt == None:
            dt = self.dt
        t = self.current_time
        nt = int( (final_time-t) / dt) +1# number of timesteps

        u_nm1 = self.u_nm1
        u_n = self.u_n
        u_np1 = fire.Function(self.function_space)

        rhs_forcing = fire.Function(self.function_space)
        usol = [fire.Function(self.function_space, name="pressure") for t in range(nt) if t % self.gradient_sampling_frequency == 0]
        usol_recv = []
        save_step = 0
        B = self.B
        rhs = self.rhs

        #assembly_callable = create_assembly_callable(rhs, tensor=B)

        for step in range(nt):
            rhs_forcing.assign(0.0)
            B = fire.assemble(rhs, tensor=B)
            f = excitations.apply_source(rhs_forcing, self.wavelet[step])
            B0 = B.sub(0)
            B0 += f
            self.solver.solve(X, B)

            u_np1.assign(X)

            usol_recv.append(self.receivers.interpolate(u_np1.dat.data_ro_with_halos[:]))

            if step % self.gradient_sampling_frequency == 0:
                usol[save_step].assign(u_np1)
                save_step += 1

            if (step-1) % self.output_frequency == 0:
                assert (
                    fire.norm(u_n) < 1
                ), "Numerical instability. Try reducing dt or building the mesh differently"
                if self.forward_output:
                    output.write(u_n, time=t, name="Pressure")
                if t > 0:
                    helpers.display_progress(self.comm, t)

            u_nm1.assign(u_n)
            u_n.assign(u_np1)

            t = step * float(dt)

        self.current_time = t
        helpers.display_progress(self.comm, t)

        usol_recv = helpers.fill(usol_recv, receivers.is_local, nt, receivers.num_receivers)
        usol_recv = utils.utils.communicate(usol_recv, comm)
        self.receivers_output = usol_recv

        return usol, usol_recv