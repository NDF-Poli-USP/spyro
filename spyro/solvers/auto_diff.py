from pyadjoint import minimize, MinimizationProblem, ROLSolver, TAOSolver
from firedrake import PETSc


class AutomatedGradientOptimisation:
    """Automated gradient-based optimisation methods.

    Parameters
    ----------
    reduced_functional : pyadjoint.ReducedFunctional
        Reduced functional to be minimised.
    """

    def __init__(self, reduced_functional):
        self.reduced_functional = reduced_functional

    def minimise_scipy(
            self, method="L-BFGS-B", max_iter=10, disp=True, bounds=None,
            riesz_representation='l2'
    ):
        """Minimise the reduced functional using scipy optimisation methods.

        Parameters
        ----------
        method : str
            Optimisation method.
        maxiter : int
            Maximum number of iterations.
        disp : bool
            Display the optimisation process.
        bounds : tuple
            Bounds for the optimisation.
        riesz_representation : str
            Riesz representation.
        """
        # c_optimised = minimize(
#     J_hat, method="L-BFGS-B", options={"disp": True, "maxiter": 10},
#     bounds=(lb, up), derivative_options={"riesz_representation": 'L2'}
# )
        return minimize(
            self.reduced_functional, method=method,
            options={"disp": disp, "maxiter": max_iter},
            bounds=bounds if bounds is not None else None,
            derivative_options={"riesz_representation": riesz_representation})

    def minimise_pyrol(
            self, maxiter=10, bounds=None, parameters=None):
        """Minimise the reduced functional using pyrol optimisation methods.

        Parameters
        ----------
        maxiter : int
            Maximum number of iterations.
        bounds : tuple
            Bounds for the optimisation.
        parameters : dict
            Parameters for the optimisation.
        """
        if parameters is None:
            parameters = {
                'General': {
                    'Secant': {'Type': 'Limited-Memory BFGS', 'Maximum Storage': 10}},
                'Step': {
                    'Type': 'Augmented Lagrangian',
                    'Line Search': {
                        'Descent Method': {
                            'Type': 'Quasi-Newton Step'
                        }
                    },
                    'Augmented Lagrangian': {
                        'Subproblem Step Type': 'Line Search',
                        'Subproblem Iteration Limit': 10
                    }
                },
                'Status Test': {
                    'Gradient Tolerance': 1e-7,
                    'Iteration Limit': maxiter
                }
            }
        solver = ROLSolver(
            self._minimization_problem(bounds), parameters, inner_product="L2")
        return solver.solve()

    def _minimization_problem(self, bounds):
        return MinimizationProblem(self.reduced_functional, bounds=bounds)

    def minimise_tao(
            self, maxiter=10, bounds=None, parameters=None, comm=None):
        """Minimise the reduced functional using tao optimisation methods.

        Parameters
        ----------
        maxiter : int
            Maximum number of iterations.
        bounds : tuple
            Bounds for the optimisation.
        parameters : dict
            Parameters for the optimisation.
        comm : MPI communicator
            MPI communicator.
        """
        problem = MinimizationProblem(self.reduced_functional, bounds=bounds)
        if parameters is None:
            parameters = {"tao_type": "blmvm", "tao_max_it": maxiter}
        solver = TAOSolver(
            problem, parameters, comm=my_ensemble.comm,
            convert_options={"riesz_representation": "L2"})
        solver.tao.setConvergenceTest(self._tao_convergence_tracker)
        return solver.solve()

    def _tao_convergence_tracker(self, tao, *, gatol=1.0e-7, max_its=15):
        its, _, res, _, _, _ = tao.getSolutionStatus()
        # outfile.write(J_hat.controls[0].control)
        if res < gatol or its >= max_its:
            tao.setConvergedReason(PETSc.TAO.ConvergedReason.CONVERGED_USER)
        else:
            tao.setConvergedReason(PETSc.TAO.ConvergedReason.CONTINUE_ITERATING)

