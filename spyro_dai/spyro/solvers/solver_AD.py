from firedrake import *

from ..domains import quadrature, space
from ..pml import damping
from ..io import ensemble_forward
from .. import utils
from ..sources import full_ricker_wavelet, delta_expr, delta_expr_adj
from . import helpers

import numpy as np
import copy

from mpi4py import MPI
class solver_AD():
    def __init__(self,fwi=False, Aut_Dif=True):
        self.Aut_Dif = Aut_Dif
        self.fwi=fwi
 
    def wave_propagation(self,
        model,
        mesh,
        comm,
        c,
        point_cloud,
        position_source, 
        freq_index=0,
        type="forward",
        output=False,
        **kwargs
    ):
        """Secord-order in time fully-explicit scheme
        with implementation of a Perfectly Matched Layer (PML) using
        CG FEM with or without higher order mass lumping (KMV type elements).

        Parameters
        ----------
        model: Python `dictionary`
            Contains model options and parameters
        comm: Firedrake.ensemble_communicator
            The MPI communicator for parallelism
        c: Firedrake.Function
            The velocity model interpolated onto the mesh.
        point_cloud: Firedrake.Function
            space function built with the receivers points
        position_source: array 
            Source position
        freq_index: `int`, optional
            The index in the list of low-pass cutoff values
        type: either "forwward" or "adjoint"
            Type o wave equation.
        """

        if "amplitude" in model["acquisition"]:
            amp = model["acquisition"]["amplitude"]
        else:
            amp = 1
        freq = model["acquisition"]["frequency"]

        if "inversion" in model:
            freq_bands = model["inversion"]["freq_bands"]
        method = model["opts"]["method"]
        degree = model["opts"]["degree"]
        dim    = model["opts"]["dimension"]
        dt     = model["timeaxis"]["dt"]
        tf     = model["timeaxis"]["tf"]
        delay  = model["acquisition"]["delay"]
        nspool = model["timeaxis"]["nspool"]
        fspool = model["timeaxis"]["fspool"]
      
        nt     = int(tf / dt)  # number of timesteps
        dstep  = int(delay / dt)  # number of timesteps with source

        if method == "KMV":
            params = {"ksp_type": "preonly", "pc_type": "jacobi"}
        elif (
            method == "CG"
            and mesh.ufl_cell() != quadrilateral
            and mesh.ufl_cell() != hexahedron
        ):
            params = {"ksp_type": "cg", "pc_type": "jacobi"}
        elif method == "CG" and (
            mesh.ufl_cell() == quadrilateral or mesh.ufl_cell() == hexahedron
        ):
            params = {"ksp_type": "preonly", "pc_type": "jacobi"}
        else:
            raise ValueError("method is not yet supported")

       
        element = space.FE_method(mesh, method, degree)
        V       = FunctionSpace(mesh, element)


        qr_x, qr_s, _ = quadrature.quadrature_rules(V)
        if dim == 2:
            z, x = SpatialCoordinate(mesh)
        elif dim == 3:
            z, x, y = SpatialCoordinate(mesh)

        u     = TrialFunction(V)
        v     = TestFunction(V)
        u_nm1 = Function(V)
        u_n   = Function(V)
        u_np1 = Function(V)

        cutoff   = freq_bands[freq_index] if "inversion" in model else None
        
        u_tt = ((u - 2.0 * u_n + u_nm1) / Constant(dt ** 2))
        m1   =  u_tt * v * dx(rule=qr_x)
        a    = c * c * dot(grad(u_n), grad(v)) * dx(rule=qr_x)  # explicit
        
        t = 0.0
        nf = 0
        if model["BCs"]["outer_bc"] == "non-reflective":
            nf = c * ((u_n - u_nm1) / dt) * v * ds(rule=qr_s)

        if type=="forward":
            RW = full_ricker_wavelet(dt, tf, freq, amp=amp, cutoff=cutoff)
            f, ricker = self.external_forcing(RW,mesh,position_source,V)
                 
        if type=="adjoint":
            f = Function(V)
            num_rec = len(position_source)
            excitation = []
            for i in range(num_rec):
                delta = Interpolator(delta_expr(position_source[i], z, x,sigma_x=2000), V)
                aux   = Function(delta.interpolate())
                excitation.append(aux.dat.data[:])
            
        FF   = m1 + a + nf - f * v * dx(rule=qr_x)  
        X    = Function(V)
        lhs_ = lhs(FF)
        rhs_ = rhs(FF)
        
        if not self.Aut_Dif:
            usol  = [Function(V, name="pressure") for t in range(nt) if t % fspool == 0]
        
        num_rec   = len(point_cloud.coordinates.dat.data_ro)
        usol_recv = []

        saveIT  = 0
        problem = LinearVariationalProblem(lhs_, rhs_, X)
        solver  = LinearVariationalSolver(problem, solver_parameters=params)
              
        if type=="adjoint":
            # Define gradient problem
            m_u = TrialFunction(V)
            m_v = TestFunction(V)
            mgrad = m_u * m_v * dx(rule=qr_x)

            uuadj = Function(V)  # auxiliarly function for the gradient compt.
            uufor = Function(V)  # auxiliarly function for the gradient compt.
            ffG = dot(grad(uuadj), grad(uufor)) * m_v * dx(rule=qr_x)

            G = mgrad - ffG
            lhsG, rhsG = lhs(G), rhs(G)

            gradi = Function(V)
            grad_prob = LinearVariationalProblem(lhsG, rhsG, gradi)
            if method == "KMV":
                grad_solver = LinearVariationalSolver(
                    grad_prob,
                    solver_parameters={
                        "ksp_type": "preonly",
                        "pc_type": "jacobi",
                        "mat_type": "matfree",
                    },
                )
            elif method == "CG":
                grad_solver = LinearVariationalSolver(
                grad_prob,
                solver_parameters={
                    "mat_type": "matfree",
                },
                )
            t = tf
            dJ = Function(V, name="gradient")

        if type=="forward":
            P              = FunctionSpace(point_cloud, "DG", 0)
            interpolator   = Interpolator(u_np1, P)
            usol_recv      = []
            t = 0
            if self.fwi:
                obj_func   = kwargs.get("obj_func")
                p_true_rec = kwargs.get("p_true_rec")
                misfit=[]   
        
        for IT in range(nt):
            if type=="forward":
                t += float(dt)
                if IT < dstep:
                    ricker.assign(RW[IT]*2000)
                elif IT == dstep:
                    ricker.assign(0.0)
            
            if type=="adjoint":
                misfit = kwargs.get("misfit")
                IT_adj = nt-1 - IT 
            
                rec_value = 0
                for i in range(num_rec):
                    rec_value+=misfit[IT_adj][i]*excitation[i]

                f.dat.data[:] = rec_value
      
            solver.solve()
            u_np1.assign(X)

            if not self.Aut_Dif:
                usol[IT].assign(u_np1)   
              
            if type=="forward" and num_rec>0:
                rec = Function(P)
                interpolator.interpolate(output=rec)
                usol_recv.append(rec.dat.data) 

                if self.fwi:
                    obj_func += self.objective_func(
                        rec,
                        p_true_rec[IT],
                        IT,
                        dt,
                        P,
                        misfit)
                 
            if IT % nspool == 0:
                assert (
                norm(u_n) < 1
                ), "Numerical instability. Try reducing dt or building the mesh differently"

                if t > 0:
                    helpers.display_progress(comm, t)

            u_nm1.assign(u_n)
            u_n  .assign(u_np1)

            if type=="adjoint":
                guess=kwargs.get("guess")
                # compute the gradient increment
                uuadj.assign(u_np1)
                uufor.assign(guess.pop())

                grad_solver.solve()
                dJ += gradi
         
        if comm.ensemble_comm.rank == 0 and comm.comm.rank == 0:
            print(
                "---------------------------------------------------------------",
                flush=True,
            )
        
        if type=="forward":
            if not self.Aut_Dif and self.fwi==True:
                return usol, misfit,obj_func
            elif self.Aut_Dif and self.fwi==True:
                return obj_func
            else:  
                return usol_recv
        
        if type=="adjoint":
            return dJ

    def objective_func(self, p_rec,p_true_rec, IT, dt,P,misfit):
        true_rec = Function(P)
        true_rec.dat.data[:] = p_true_rec
        J = 0.5 * assemble(inner(true_rec-p_rec, true_rec-p_rec) * dx)
        misfit.append(true_rec.dat.data[:]-p_rec.dat.data[:])
        return J

    # external forcing - older versions
    def external_forcing(self,RW, mesh,position_source,V):

        z, x       = SpatialCoordinate(mesh)
        source     = Constant(position_source)
        delta      = Interpolator(delta_expr(source, z, x,sigma_x=2000), V)
        excitation = Function(delta.interpolate())
        ricker     = Constant(0)
        f          = excitation * ricker
        ricker.assign(RW[0], annotate=True)

        return f, ricker


