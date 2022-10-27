from firedrake import set_log_level, parameters, dx
from firedrake import DirichletBC, Constant, Measure
from firedrake import File, CellDiameter, sqrt, inner, grad, TestFunction
from firedrake import TrialFunction, solve, lhs, rhs
from firedrake import FunctionSpace, Function
import firedrake as fire
import numpy as np
from ufl import SpatialCoordinate

# Work from Ruben Andres Salas,
# Andre Luis Ferreira da Silva,
# Luis Fernando Nogueira de Sá, Emilio Carlos Nelli Silva, Hybrid absorbing scheme based on hyperel-
# liptical layers with non-reflecting boundary conditions in scalar wave equations, Applied Mathematical
# Modelling (2022), doi: https://doi.org/10.1016/j.apm.2022.09.014

###############
# Execute this code before starting inversion process
##############

'''
# INFO PROGRESS DBG WARNING ERROR CRITICAL
# set_log_level(LogLevel.INFO)
CRITICAL  = 50, // errors that may lead to data corruption and suchlike
ERROR     = 40, // things that go boom
WARNING   = 30, // things that may go boom later
INFO      = 20, // information of general interest
PROGRESS  = 16, // what's happening (broadly)
TRACE     = 13, // what's happening (in detail)
DBG       = 10  // sundry
To turn of logging completely, use
set_log_active(False)
'''
set_log_level(20)
parameters['std_out_all_processes'] = True
parameters['form_compiler']['optimize'] = True
parameters['form_compiler']['cpp_optimize'] = True
# parameters['form_compiler']['cpp_optimize_flags'] = '-O2'
parameters['form_compiler'][
    'cpp_optimize_flags'] = '-O3 -ffast-math -march=native'
parameters['form_compiler']['representation'] = 'uflacs'
parameters['form_compiler']['quadrature_degree'] = 5
parameters['ghost_mode'] = 'shared_facet'


# def DefineBoundaries(possou, mesh, V, lmin):
#     '''
#     BCs for eikonal
#     possou: Positions of sources
#     V: Function space
#     lmin: Minimal dimension of finite element
#     '''
#     class Sigma(SubDomain):
#         def __init__(self, possou, tolx, toly, typsou='unique'):
#             super().__init__()
#             self.possou = possou
#             self.tolx = tolx
#             self.toly = toly
#             self.typsou = typsou

#         def inside(self, x, on_boundary):
#             cond = False
#             if self.typsou == 'unique':
#                 cond = cond or (self.possou[0] - self.tolx[0] <= x[0]
#                                 <= self.possou[0] + self.tolx[1]
#                                 and self.possou[1] - self.toly[0] <= x[1]
#                                 <= self.possou[1] + self.toly[1])
#             elif self.typsou == 'multiple':
#                 for p in self.possou:
#                     cond = cond or (p[0] - self.tolx[0] <= x[0]
#                                     <= p[0] + self.tolx[1]
#                                     and p[1] - self.toly[0]
#                                     <= x[1] <= p[1] + self.toly[1])

#             return cond

#     # Initialize mesh function for boundary domains
#     elreg_marker = MeshFunction('size_t', mesh, mesh.topology().dim())
#     elreg_marker.set_all(0)
#     facet_marker = MeshFunction('size_t', mesh, mesh.topology().dim() - 1)
#     facet_marker.set_all(0)
#     # Initialize sub-domain instances
#     if all(isinstance(x, list) for x in pH['possou']):
#         xl = 0.25*lmin
#         xu = 0.25*lmin
#         yl = 0.25*lmin
#         yu = 1.25*lmin
#         sigma = Sigma(possou, [xl, xu], [yl, yu], typsou='multiple')
#     else:
#         xl = 0.6*lmin
#         xu = 0.6*lmin
#         yl = 0.6*lmin
#         yu = 0.6*lmin
#         sigma = Sigma(possou, [xl, xu], [yl, yu])
#     sigma.mark(facet_marker, 1)
#     '''
#         OBS: Nodal approach may fail if there is 
#         not at least a node in the demarcated area
#         '''
#     # bcs = [DirichletBC(V, Constant(0.0), CompiledSubDomain(
#     #     '(near(x[0], xs) && near(x[1], ys))',
#     #     xs=possou[0], ys=possou[1]), 'pointwise')]

#     bcs = [DirichletBC(V, Constant(0.0), facet_marker, 1)]
#     # Define new measures associated with the bcs
#     # ds = Measure('ds')(subdomain_data=facet_marker)
#     dx = Measure('dx')(subdomain_data=elreg_marker)
#     souEik_file = File('/out/SouEik.pvd')
#     souEik_file << facet_marker

#     return dx, bcs

def SolveEikonal(c, Eik, mesh, sources, annotate=False):
    '''
    Solve for Eikonal
    '''
    sources.current_source = 0
    
    yp = Function(Eik)
    vy = TestFunction(Eik)
    u = TrialFunction(Eik)

    mask = Function(Eik)
    mask = sources.make_mask(mask)

    k = Constant(1e5)

    print('Solve Pre-Eikonal')
    f = Constant(1.0)
    F1 = inner(grad(u), grad(vy))*dx - f/c*vy*dx + mask * k * inner(u - 1e-12, vy) * dx
    A = fire.assemble(lhs(F1))
    
    B = fire.Function(Eik)
    B = fire.assemble(rhs(F1), tensor=B)

    output = File('linear.pvd')
    output.write(yp)


    B_data = B.dat.data[:]
    print(np.min(B_data))
    print(np.max(B_data))

    solve(A,yp,B)
    output = File('linear.pvd')
    output.write(yp)

    '''
    Eikonal with stabilizer term
    '''
    f = Constant(1.0)
    eps = CellDiameter(mesh)  # Stabilizer
    mask = Function(Eik)

    mask = sources.make_mask(mask)
    output = File('mask.pvd')
    output.write(mask)

    weak_bc = mask * k * inner(yp-1e-12, vy) * dx
    F = inner(sqrt(inner(grad(yp), grad(yp))),vy) *dx + eps*inner(grad(yp), grad(vy))*dx - f / c*vy*dx + weak_bc
    L = 0

    print('Solve Post-Eikonal')
    # Solver Parameters
    # https://petsc.org/release/docs/manualpages/KSP/KSPSetFromOptions.html
    # https://petsc.org/release/docs/manualpages/SNES/SNESSetFromOptions.html
    # Solver Parameters
    # PETScOptions.set("help")
    # newtonls newtontr test nrichardson ksponly vinewtonrsls vinewtonssls
    # ngmres qn shell ngs ncg fas ms nasm anderson aspin composite python

    # PETScOptions.set("snes_rtol", 1e-20)

    # # Newton LS
    # PETScOptions.set('snes_linesearch_type', 'l2')  # basic bt nleqerr cp l2
    # PETScOptions.set("snes_linesearch_damping", 1.00)  # for basic,l2,cp
    # PETScOptions.set("snes_linesearch_maxstep", 0.50)  # bt,l2,cp
    # PETScOptions.set('snes_linesearch_order', 2)  # for newtonls com bt

    # # damp 0.50 maxstep 1.00 64 it
    # # damp 0.50 maxstep 0.50 64 it
    # # damp 0.75 maxstep 1.00 32 it
    # # damp 0.75 maxstep 1.00 32 it
    # # damp 1.00 maxstep 1.00 08 it
    # # damp 1.00 maxstep 0.50 08 it

    # PETScOptions.set('ksp_type', 'gmres')
    # # jacobi pbjacobi bjacobi sor lu shell mg eisenstat ilu icc cholesky asm
    # # gasm ksp composite redundant nn mat fieldsplit galerkin exotic cp lsc
    # # redistribute svd gamg kaczmarz hypre pfmg syspfmg tfs bddc python
    # PETScOptions.set('pc_type', 'lu')

    solver_parameters = {'snes_type': 'vinewtonssls',
                        "snes_max_it": 1000,
                        "snes_atol": 5e-6,
                        "snes_rtol": 1e-20,
                        'snes_linesearch_type': 'l2',  # basic bt nleqerr cp l2
                        "snes_linesearch_damping": 1.00, # for basic,l2,cp
                        "snes_linesearch_maxstep": 0.50,  # bt,l2,cp
                        'snes_linesearch_order': 2,  # for newtonls com bt
                        'ksp_type': 'gmres',
                        'pc_type': 'lu',
                        'nonlinear_solver': 'snes',
                        }
    


    # solve(F == L, yp)
    solve(F == L, yp, solver_parameters=solver_parameters)#{"newton_solver": {"relative_tolerance": 1e-6}})

    return yp


def vel_bound(boundmesh, c_eik, Lx, Ly):
    '''
    Velocity profile at boundary
    boundmesh: Boundary Mesh
    c_eik: Velocity profile without layer
    Lx, Ly: Original domain dimensions
    '''
    print('Mapping Propagation Speed Boundary')
    name = 'VelBound (km/s)'

    # Create and define function space for boundary mesh
    B = FunctionSpace(boundmesh, 'CG', 1)
    cbound = Function(B, name=name)

    c_coords = cbound.function_space().tabulate_dof_coordinates()
    # Loop for identifying properties
    cb_array = cbound.vector().get_local()
    c_eik.set_allow_extrapolation(True)
    for dof, coord in enumerate(c_coords):
        xc = coord[0]
        yc = coord[1]
        if xc >= 0 and xc <= Lx and yc >= 0 and yc <= Ly:
            # print(xc, yc, Lx, Ly)
            cb_array[dof] = c_eik(xc, yc)

    cbound.vector().set_local(cb_array)
    cbound.vector().apply('insert')
    del cb_array

    velp_file = File('/out/VelpBound.pvd')
    velp_file << cbound

    return cbound


def bcMesh(mesh, c_eik, Lx, Ly, CamComp=False):
    '''
    Extract boundary from original domain
    mesh: Mesh without layer
    c_eik: Velocity profile without layer
    Lx, Ly: Original domain dimensions
    CamComp: False for top free surface
    '''
    boundmesh = BoundaryMesh(mesh, 'exterior')

    if not CamComp:
        dom_cut = CompiledSubDomain('(x[0] == x1 && x[1] >= y1 && \
            x[1] <= y2) || (x[0] == x2 && x[1] >= y1 && x[1] <= y2) || \
            (x[1] == y1 && x[0] >= x1 && x[0] <= x2)', x1=0, x2=Lx, y1=0, y2=Ly)
        boundmesh = SubMesh(boundmesh, dom_cut)
    # File('/out/boundmesh.pvd') << boundmesh
    # Coordinates of boundary mesh
    bcoord = boundmesh.coordinates()
    del boundmesh
    # Velocitiy profile in boundary mesh
    cbound = vel_bound(boundmesh, c_eik, Lx, Ly)
    cbound.set_allow_extrapolation(True)
    return cbound, bcoord


# def Eikonal(mesh, c_eik, possou, lmin, Lx, Ly):
def Eikonal(Wave):
    '''
    Solving eikonal for defining critical points from original domain
    mesh: Mesh without layer
    c_eik: Velocity profile without layer
    possou: Positions of sources
    Lx, Ly: Original domain dimensions
    '''
    mesh = Wave.mesh
    c_eik = Wave.c # somente o dominio 
    print(type(c_eik))
    # Create and define function space for current mesh
    # BCs for eikonal
    print('Defining Eikonal Boundaries')
    xs =[]
    ys = []
    for source in Wave.model_parameters.source_locations:
        x,y = source
        xs.append(x)
        ys.append(y)
    print(xs)
    print(ys)

    possou = [xs, ys]
    V = Wave.function_space
    # dx, bcs_eik = DefineBoundaries(possou, mesh, V, lmin)
    # Solving Eikonal
    yp = Function(V, name='Eikonal (Time [s])')

    yp.assign(SolveEikonal(c_eik, V, mesh, Wave.sources, annotate=False))
    eikonal_file = File('out/Eik.pvd')
    eikonal_file.write(yp)

    # Mesh coordinates
    # xmesh,ymesh = SpatialCoordinate(mesh)
    # ux = Function(V).interpolate(xmesh)
    # uy = Function(V).interpolate(ymesh)
    # ux_data = ux.dat.data[:]
    # uy_data = uy.dat.data[:]
    # mesh_coord = V.tabulate_dof_coordinates()
    # Velocity profile and coordinates at boundary
    cbound, bcoord = bcMesh(mesh, c_eik, Wave.length_z, Wave.length_x)

    # Defining critical points for vertical and horizontal boundaries
    yp.set_allow_extrapolation(True)
    coordCritEik = np.empty([2, 4])
    refx = refy = ref0 = ref1 = 0
    for i, coord in enumerate(bcoord):
        xc = coord[0]
        yc = coord[1]
        if xc <= 0.0 or xc >= Lx and yc > 0.0:
            ref0 = yp(xc, yc)
            if refx == 0 or ref0 < refx:
                refx = ref0
                coordCritEik[0, :] = [xc, yc, cbound(xc, yc), refx]
        else:
            ref1 = yp(xc, yc)
            if refy == 0 or ref1 < refy:
                refy = ref1
                coordCritEik[1, :] = [xc, yc, cbound(xc, yc), refy]
    del yp
    np.savetxt('/out/Eik.txt', coordCritEik, delimiter='\t')
    # Minimum eikonal at boundaries
    eik0cr = np.argmin(coordCritEik[:, -1])
    posCrit = np.array([coordCritEik[eik0cr, 0], coordCritEik[eik0cr, 1]])
    # Inverse of minimum Eikonal (For approximating c_bound/lref)
    Z = 1 / min(coordCritEik[:, -1])
    cref = coordCritEik[eik0cr, 2]

    return Z, posCrit, cref
