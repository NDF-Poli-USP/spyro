# ==========================================
# Main Function starts from Line 300
# ==========================================


# ===============================
# Define and Import Files
# ===============================

from firedrake import *
from firedrake.adjoint import *
import pyadjoint
import numpy as np
import matplotlib.pyplot as plt
import os
from firedrake import triplot
import math
import finat
import sys

from pyadjoint import Block
from pyadjoint.overloaded_function import overload_function
def float_print(var, name = ""):
    """
    Print function created and overloaded for AdjFloat.
    """

    if type(var).__name__ != 'AdjFloat':
        raise ValueError(" ❌️ ERROR: type(var).__name__ == '%s' != 'AdjFloat'" %(type(var).__name__))

    def float_print_(var):
        if name != "":
            if float(var) < 0.0:
               print(" 🌼️ %s = %1.7e ❗️❗️❗️❗️❗️❗️" %(name, float(var)))
            else:
               print(" 🌼️ %s = %1.7e" %(name, float(var)))
        else:
            if float(var) < 0.0:
               print(" 🌼️ %1.7e ❗️❗️❗️❗️❗️❗️" %(name, float(var)))
            else:
               print(" 🌼️ %1.7e" %(name, float(var)))
        return var

    backend_bound = float_print_
    class float_print_Block(Block):
        def __init__(self, var, **kwargs):
            super().__init__()
            self.kwargs = kwargs
            self.add_dependency(var) # Single input
        def __str__(self):
            return 'float_print_Block'
        def evaluate_adj_component(self, inputs, adj_inputs, block_variable, idx, prepared = None):
            return adj_inputs[0]
        def recompute_component(self, inputs, block_variable, idx, prepared):
            return backend_bound(inputs[0])
    float_print_ = overload_function(float_print_, float_print_Block)

    return float_print_(var)

# FunctionMixin._ad_convert_riesz
origRevert_FunctionMixin_ad_convert_riesz = FunctionMixin._ad_convert_riesz
def newRevert_FunctionMixin_ad_convert_riesz(self, value, options = None):
   options = {} if options is None else options
   options.setdefault("riesz_representation", "l2")
   return origRevert_FunctionMixin_ad_convert_riesz(self, value, options = options)
FunctionMixin._ad_convert_riesz = newRevert_FunctionMixin_ad_convert_riesz

# FunctionMixin._ad_convert_type
origRevert_FunctionMixin_ad_convert_type = FunctionMixin._ad_convert_type
def newRevert_FunctionMixin_ad_convert_type(self, value, options = None):
    options = {} if options is None else options
    options.setdefault("riesz_representation", "l2")
    return origRevert_FunctionMixin_ad_convert_type(self, value, options = options)
FunctionMixin._ad_convert_type = newRevert_FunctionMixin_ad_convert_type

# FunctionMixin._ad_dot
origRevert_FunctionMixin_ad_dot = FunctionMixin._ad_dot
def newRevert_FunctionMixin_ad_dot(self, other, options = None):
    options = {} if options is None else options
    options.setdefault("riesz_representation", "l2")
    return origRevert_FunctionMixin_ad_dot(self, other, options = options)
FunctionMixin._ad_dot = newRevert_FunctionMixin_ad_dot

# ===============================
#      Define Functions
# ===============================


def make_mesh(dim, cell_type, nx, ny, nz, Lx, Ly, Lz, FileName):
    """
    Create a Firedrake mesh for 2D or 3D with chosen cell type.

    Parameters
    ----------
    dim : int
        Dimension of the mesh (2 or 3)
    cell_type : str
        "tri" or "quad" for 2D, "tet" or "hex" for 3D
    nx, ny, nz : int
        Number of elements along each axis
    Lx, Ly, Lz : float
        Domain lengths
    """

    if dim == 2:
        if cell_type == "tri":
            mesh = RectangleMesh(nx, nz, Lx, Lz, diagonal='crossed')
        elif cell_type == "quad":
            mesh = RectangleMesh(nx, nz, Lx, Lz, quadrilateral=True)
        else:
            raise ValueError("For 2D, cell_type must be 'tri' or 'quad'.")

    elif dim == 3:
        if cell_type == "tet":
            mesh = BoxMesh(nx, ny, nz, Lx, Ly, Lz, diagonal='crossed')
        elif cell_type == "hex":
            mesh = BoxMesh(nx, ny, nz, Lx, Ly, Lz, quadrilateral=True)
        else:
            raise ValueError("For 3D, cell_type must be 'tet' or 'hex'.")

    else:
        raise ValueError("dim must be 2 or 3.")
    
    outfile = VTKFile(os.path.join(results_dir, FileName + ".pvd"))
    outfile.write(mesh)


    return mesh

def set_source(mesh, method, deg, x, xs, z, zs, rs, FileName):
    S1 = FunctionSpace(mesh, method, deg)
    S2 = Function(S1, name='Sx')
    sigma = Constant(100)
    # S2_expr = conditional(lt((x - xs - 1.0e-14)**2 + (z - zs + 1.0e-14)**2, rs**2), 1.0,0.0)
    S2_expr = exp(-sigma * ((x - xs)**2 + (z - zs)**2)) # exp(-sigma * ((x - xs)**2 + (z - zs)**2)) 
    S2.interpolate(S2_expr)
    mass = assemble(S2 * dx)
    S2.assign(S2/mass)
    final_mass = assemble(S2 * dx)
    assert abs(final_mass - 1.0) < 1e-10, \
        f"Normalization failed: {final_mass}"
    outfile = VTKFile(os.path.join(results_dir, FileName + ".pvd"))
    outfile.write(S2)

    return S2
    


def set_receivers(mesh, method, deg, x, xr1, xr2, z, zr1, zr2, FileName):
    R1 = FunctionSpace(mesh, method, deg) 
    R2 = Function(R1, name = 'Rx')   
    com1 = And(ge(x, xr1 - 1.0e-14), le(x, xr2 + 1.E-14) )
    com2 = And(ge(z, zr1 - 1.0e-14), le(z, zr2 + 1.E-14) )
    com = And(com1,com2)
    R2.interpolate(conditional(com,1.0,0.0))
    outfile = VTKFile(os.path.join(results_dir, FileName + ".pvd"))
    outfile.write(R2)
    return R2


"""
def set_receiver(x_start, x_end, z_start, z_end, n_receivers):

    # Generate positions linearly spaced between start and end
    x_coords = np.linspace(x_start, x_end, n_receivers)
    z_coords = np.linspace(z_start, z_end, n_receivers)

    # Combine into list of (x, z) tuples
    receiver_coords = np.column_stack((x_coords, z_coords))

    return receiver_coords
"""

def RickerWavelet(t, freq, amp):
    t_shifted = t - 1.0/freq
    ricker_time = amp * (1 - 2*(math.pi*freq*t_shifted)**2) * math.exp(-(math.pi*freq*t_shifted)**2)
    return ricker_time


def forcing_vec():
    return as_vector((Sx_spatial*ricker, 0.0))


def eps(w):
    return 0.5*(grad(w) + grad(w).T)


def camembert_coin(A, xc, zc, rc, x, z, VarName, FileName):
    rho_dv = Function(A, name = VarName)
    rho_dv.interpolate(conditional(lt((x - xc)**2 + (z - zc)**2, rc**2), 1.0,0.0))
    outfile = VTKFile(os.path.join(results_dir, FileName + ".pvd"))
    outfile.write(rho_dv)
    return rho_dv

def layer_model(A, dz, x, z, VarName, FileName):
    rho_dv = Function(A, name = VarName)
    rho_dv.interpolate(conditional(lt((z - dz), 0), 1.0, 0.0))
    outfile = VTKFile(os.path.join(results_dir, FileName + ".pvd"))
    outfile.write(rho_dv)
    return rho_dv


def apply_source(t, f0, v, F1, F_sou=1., integral=True):
    '''
    Ricker source for time t

    Parameters
    ----------
    t: `float`
        Current time
    f0: `float`
        Source central frequency
    v: `firedrake.TestFunction`
        Test function
    F1: `firedrake.Function`
        Source spatial distribution
    F_sou: `float`, optional
        Maximum source amplitude. Default is 1
    integral: `bool`, optional
        If True, returns the integral of the Ricker wavelet.

    Returns
    -------
    L: `firedrake.Form`
        Source term in weak form
    '''

    # Shifted time
    t_shifted = t - 1. / f0

    r = (np.pi * f0 * t_shifted)**2
    # Amplitude excitation
    if integral:
        amp = t_shifted * np.exp(-r) * F_sou
    else:
        amp = (1. - 2. * r) * np.exp(-r) * F_sou
    # print('Amp: {:3.2e}'.format(amp))

    # Traction force
    # F1 = fire.Constant((0., amp, 0.))
    # L = fire.inner(F1, v) * fire.ds(2)

    # Point load
    # F1.dat.data[source_vertex] = [0, amp, 0.]  # Point source
    # L = fire.inner(F1, v) * dx  # Source term

    # Explosion source
    L = -Constant(amp) * F1 * div(v) * dx  # Source term
    # print(fire.assemble(L))

    return L


def weak_form(V, u, v, rho, rho_dv, dt, nt, freq, amp, Sx_spatial, FileName):

    lmbda = lambda_1 + (lambda_2 - lambda_1)*rho_dv**p1
    mu    = mu_1 + (mu_2 - mu_1)*rho_dv**p2

    t = t_start

    u_n = Function(V)      # u^n ; value at current time step
    u_nm1 = Function(V)    # u^{n-1} ; value at previous time step
    u_new = Function(V, name = 'u')    # u^{n+1} ; value to be evaluated at next time step

    deg = V.ufl_element().sub_elements[0].degree()
    quad_rule = finat.quadrature.make_quadrature(V.finat_element.cell, deg, "KMV")
    dxlump = dx(scheme=quad_rule)

    F_m = (rho/Constant(dt*dt)) * inner(u - 2*u_n + u_nm1, v) * dxlump
    F_k = lmbda*div(u_n)*div(v)*dx + 2.0*mu*inner(eps(u_n), eps(v))* dx
    # F_k = lmbda*div(u)*div(v)*dx + 2.0*mu*inner(eps(u), eps(v))* dx


    # b_vec = forcing_vec()
    # F_s = inner(b_vec, v) * dx

    #F_s = apply_source(t=t, f0=freq, v=v, F1=Sx_spatial, F_sou=amp, integral=False)

    # t_left = Constant((1.0e3, 1.0e3))
    # t_right = Constant((6.0e3,3.0))
    # t_bottom = Constant((7.0e3,4.0))
    # t_top = Constant((8.0e3,1.0))

    # F_n = inner(t_left, v) * ds(1) # + inner(t_right, v) * ds(2) + inner(t_bottom, v) * ds(3) + inner(t_top, v) * ds(4) 

    #F = F_m + F_k - F_s # - F_n

    
    # bc_left   = DirichletBC(V, Constant((0.2, 0.2)), 1)
    # bc_right  = DirichletBC(V, Constant((0.0, 0.0)), 2)
    # bc_bottom = DirichletBC(V, Constant((0.0, 0.0)), 3)
    # bc_top    = DirichletBC(V, Constant((0.0, 0.0)), 4)
    bcs = []

    
    # Generate positions linearly spaced between start and end

    #xr1, xr2, zr1, zr2 = 1.0, 3.0, (4/80*20), (4/80*20)
    nr = 100 
    x_coords = np.linspace(xr1, xr2, nr)
    z_coords = np.linspace(zr1, zr2, nr)

    # Combine into list of (x, z) tuples
    Rx_1 = np.column_stack((x_coords, z_coords))


    time_values = np.zeros(nt)
    seismo_ux = [[] for _ in Rx_1]
    seismo_uz = [[] for _ in Rx_1]
    u_list = []

    outfile = VTKFile(os.path.join(results_dir, FileName + ".pvd"))

    # ==========================
    # Time Stepping Loop
    # ==========================

    for n in range(nt):
        #ricker.assign(RickerWavelet(t, freq, amp))
        #b = assemble(rhs(F))F = F_m + F_k + F_s


        # solve(lhs(F) == rhs(F), u_new, bcs=bcs, solver_parameters={"ksp_type": "preonly", "pc_type": "jacobi"})
        # #solver.solve(u_new, b)

        F_s = apply_source(t, freq, v, Sx_spatial, F_sou=1.0e6)

        F = F_m + F_k + F_s # + sign with F_s is because in apply_source function, the return with - sign


        solve(lhs(F) == rhs(F), u_new, bcs=bcs, solver_parameters={"ksp_type": "preonly", "pc_type": "jacobi"})
        time_values[n] = t

        # ==========================
        #    Record Seismogram
        # ==========================
        for i, (rx, rz) in enumerate(Rx_1):
            ux, uz = u_new.at((rx, rz))
            seismo_ux[i].append(ux)
            seismo_uz[i].append(uz)

        if n % 5 == 0:
            outfile.write(u_new, time=t)
            #print(f"Step {n:4d}/{nt}, t = {t:.3f}")
            print(f"{FileName:s} - Step {n:4d}/{nt}, t = {t:.3f}")

        u_latest = Function(V) # to be used in u_list
        u_nm1.assign(u_n)
        u_n.assign(u_new)
        u_latest.assign(u_new)
        u_list = u_list + [u_latest]
        t += float(dt)

    return u_list, time_values, seismo_ux, seismo_uz


def plot_seismogram(time, seismo, component, filename):
    data = np.array(seismo)
    data = data.T 

    plt.figure(figsize=(8, 6))
    plt.imshow(
        data,
        aspect='auto',
        interpolation='bilinear',
        cmap='seismic',
        origin='lower',
        extent=[0, data.shape[1], time.min(), time.max()]
    )
    plt.colorbar(label="Amplitude")
    plt.xlabel("Receiver index")
    plt.ylabel("Time")
    plt.title(f"Seismogram Image ({component})")

    plt.savefig(os.path.join("results", filename + ".png"), dpi=300, bbox_inches='tight')


def results_file(name):
    return VTKFile(os.path.join(results_dir, name + ".pvd"))


def cost_function(V, u, v, rho, rho_dv, dt, nt, freq, amp, Sx_spatial, u_list_ref, FileName):

    u_list_guess, time_values, seismo_ux, seismo_uz = \
        weak_form(V, u, v, rho, rho_dv, dt, nt, freq, amp, Sx_spatial, FileName)

    J_cost_list = []
    J_total = pyadjoint.AdjFloat(0.0)

    for n in range(nt):
        J_cost = assemble(inner(u_list_guess[n] - u_list_ref[n], u_list_guess[n] - u_list_ref[n]) * Rx * dx)
        # J_cost = assemble((u_list_guess[n] - u_list_ref[n])**2 * Rx * dx)
        # J_cost = float_print(J_cost, name = "[%s] J_cost_%d" %(FileName, n) if n!= 0 else "⭐️ [%s] J_cost_%d" %(FileName, n))
        J_cost_list = J_cost_list + [J_cost]

    print("J_cost_list = ", J_cost_list)

    for n in range(nt-1):
        J = ( J_cost_list[n+1] + J_cost_list[n] ) * float(dt) / 2
        # J = float_print(J, name = "[%s] J_(%d->%d)" %(FileName, n, n+1) if n!= 0 else "🕸️ [%s] J_(%d->%d)" %(FileName, n, n+1))
        J_total = J + J_total
        # J_total = float_print(J_total, name = "[%s] J_total_(%d->%d)" %(FileName, n, n+1) if n!= 0 else "🌀️ [%s] J_total_(%d->%d)" %(FileName, n, n+1))

    multiplier = 1.0 # 1.0e60


    # J_Reg = assemble( 0.5 * R_lmbda * inner(grad(lmbda), grad(lmbda)) * dx + 
    #                   0.5 * R_mu * inner(grad(mu), grad(mu)) * dx )
    
    J_Reg = assemble( 0.5 * R_lmbda * inner(grad(rho_dv), grad(rho_dv)) * dx)
    J_Reg = float_print(J_Reg, name = "🕸️ [%s] J_Reg" %(FileName))

    J_avg = ( J_total/t_end + J_Reg) * multiplier
    J_avg = float_print(J_avg, name = "🧭️ [%s] J_avg" %(FileName))


    # print("J_avg = ", J_avg)
    # print("J_Reg = ", J_Reg)
    # print("J_tot_type = ",type(J_total))
    # print("J_avg_type = ",type(J_avg))
    # print("J_Reg_type = ",type(J_Reg))    

    return J_avg

    
# =======================================
# =======================================
#      Main Function Starts Here
# =======================================
# =======================================

os.system('clear')

results_dir = "results"
os.makedirs(results_dir, exist_ok=True)

continue_annotation()

# =======================
#   Make Mesh 
# =======================

# Dimensions can either 2 or 3
# For 2D, cell type are tri and quad
# For 3D, cell type are tet and hex
# For 2D, need nx, nz, Lx and Lz ; give any arbitrary value for ny, and Ly
# For 3D, need nx, ny, nz, Lx, Ly and Lz ; make_amp =mesh(dim, cell_type, nx, ny, nz, Lx, Ly, Lz)


si_m = 1.0
dim = 2
cell_type = 'tri' # crossed
nx, ny, nz = 80, 0, 80
Lx, Ly, Lz = 4*si_m, 0, 4*si_m # meters
mesh = make_mesh(dim, cell_type, nx, ny, nz, Lx, Ly, Lz, "mesh")


x, z = SpatialCoordinate(mesh)

# ==========================
#   Set Point Source 
# ==========================

method_e = "CG" # method of element
deg_e = 1       # degree of element
A = FunctionSpace(mesh, method_e, deg_e)


# coords = mesh.coordinates.dat.data_ro

# dof = 3500
# x_dof, z_dof = coords[dof]

# print("DOF", dof, "is at x =", x_dof, ", z =", z_dof)

# coords = mesh.coordinates.dat.data_ro

# import sys
# sys.exit(0) 

# xs, zs: x and z coordinates of source ; rs: radius of the point source (very small)
# set_source returns source VTKFile and source expression to multiply with ricker time component

hz_s = 50

xs, zs, rs = 1.0*si_m, (Lz/nz)*hz_s*si_m, 10.0*si_m
Sx_spatial1 =  set_source(mesh, method_e, deg_e, x, xs, z, zs, rs, "Sx_1") # spatial point source

xs, zs, rs = 2.0*si_m, (Lz/nz)*hz_s*si_m, 10.0*si_m
Sx_spatial2 =  set_source(mesh, method_e, deg_e, x, xs, z, zs, rs, "Sx_2") # spatial point source

xs, zs, rs = 3.0*si_m, (Lz/nz)*hz_s*si_m, 10.0*si_m
Sx_spatial3 =  set_source(mesh, method_e, deg_e, x, xs, z, zs, rs, "Sx_3") # spatial point source

# ==========================
#   Set Receiver Line
# ==========================

# xr1, xr2: start and end points of receivers in x direction
# zr1, xr2: start and end pointricker = Constant(0.0)

hz_r = 30

xr1, xr2, zr1, zr2 = 0.5*si_m, 3.5*si_m, (Lz/nz*hz_r)*si_m, (Lz/nz*hz_r)*si_m
Rx = set_receivers(mesh, method_e, deg_e, x, xr1, xr2,  z, zr1, zr2, "Rx")

# print("-- Check: ", np.any(Rx.dat.data < 0))

# ======================================
#   Simulation and Material Parameters
# ======================================

ricker = Constant(0.0)

t_start = 0.0 # start time
t_end = 1.0 # end time
dt = Constant(0.002)  # time step
nt = int ( t_end/float(dt) + 1 )

freq = 6.0 # Hz
amp = 1.0 # 

rho = Constant(1.0) # kg/m^3 Constant(2.0e3)
lambda_1 = 2.0 # N/m^2 3.0e9
lambda_2 = 10.0 # N/m^2 6.0e9

mu_1 = 0.2 # N/m^2 0.2e9 
mu_2 = 1.0 # N/m^2 0.8e9

p1 = 3.0 
p2 = 3.0


# vp_1 = np.sqrt((lambda_1 + 2*mu_1)/float(rho))
# vp_2 = np.sqrt((lambda_2 + 2*mu_2)/float(rho))


# vs_1 = np.sqrt((mu_1)/float(rho))
# vs_2 = np.sqrt((mu_2)/float(rho))


# print("vp_1 = ", vp_1 , " vp_2 = ", vp_2 , " vs_1 = " ,  vs_1 , " vs_2 = " , vs_2)


# ======================================
#   Setup the design variable
# ======================================

# xc, zc, rc = 2.0*si_m , 3.0*si_m, 0.25*si_m
# rho_dv = camembert_coin(A, xc, zc, rc, x, z, "rho_dv_ref", "rho_dv_ref")

dz = Lz/2
rho_dv = layer_model(A, dz, x, z, "rho_dv_ref", "rho_dv_ref")

# ======================================
#   Setup the weak form and time step
# ======================================

method_v = "KMV" # method of vector space
deg_v = 2       # degree of vector space
V = VectorFunctionSpace(mesh, method_v, deg_v)
u = TrialFunction(V)   # trial function (u^{n+1})
v = TestFunction(V)    # test function

with stop_annotating():

    u_list_ref1, time_values1, seismo_ux1, seismo_uz1 = \
        weak_form(V, u, v, rho, rho_dv, dt, nt, freq, amp, Sx_spatial1, "wave_output_ref_1")
    
    u_list_ref2, time_values2, seismo_ux2, seismo_uz2 = \
        weak_form(V, u, v, rho, rho_dv, dt, nt, freq, amp, Sx_spatial2, "wave_output_ref_2")

    u_list_ref3, time_values3, seismo_ux3, seismo_uz3 = \
        weak_form(V, u, v, rho, rho_dv, dt, nt, freq, amp, Sx_spatial3, "wave_output_ref_3")

# ============================
#     Plot Seismograms
# ============================


# plot_seismogram(time_values1, seismo_ux1, component="ux", filename="ux_s1_ref")
# plot_seismogram(time_values1, seismo_uz1, component="uz", filename="uz_s1_ref")

# plot_seismogram(time_values2, seismo_ux2, component="ux", filename="ux_s2_ref")
# plot_seismogram(time_values2, seismo_uz2, component="uz", filename="uz_s2_ref")

# plot_seismogram(time_values3, seismo_ux3, component="ux", filename="ux_s3_ref")
# plot_seismogram(time_values3, seismo_uz3, component="uz", filename="uz_s3_ref")


# ============================
#     Forward Problem
# ============================

# rho_dv.assign(0.5)

rho_init_expr = conditional(le(z, dz), 0.9, 0.1)

rho_dv.interpolate(rho_init_expr)

# rho_init_expr = conditional(le(z, 1.0), 1.0, conditional(le(z, 3.0), 0.5, 0.0))
# rho_init_expr = conditional(le(z, 1.0), 1.0, 0.0)


# rho_dv.interpolate(rho_init_expr)

# with CheckpointFile("rho_dv_opt.h5", "r") as hdf5_file:
#     mesh = hdf5_file.load_mesh()

# # A = FunctionSpace(mesh, "KMV", 2)
# # rho_dv = Function(A)

# with CheckpointFile("rho_dv_opt.h5", "r") as hdf5_file:
#     hdf5_file.load_function(rho_dv, "rho_dv_opt")

VTKFile(os.path.join(results_dir, "rho_dv.pvd")).write(rho_dv)

# ==================================
#     Calculate Cost Function
# ==================================

R_lmbda = Constant(1.0e2)
R_mu    = Constant(1.0e-6)

J_avg_1 = cost_function(V, u, v, rho, rho_dv, dt, nt, freq, amp, Sx_spatial1, u_list_ref1, "wave_output_1")
J_avg_2 = cost_function(V, u, v, rho, rho_dv, dt, nt, freq, amp, Sx_spatial2, u_list_ref2, "wave_output_2")
J_avg_3 = cost_function(V, u, v, rho, rho_dv, dt, nt, freq, amp, Sx_spatial3, u_list_ref3, "wave_output_3")
J_avg = J_avg_1 + J_avg_2 + J_avg_3

# print("J_avg_1 = " , J_avg_1)
# print("J_avg_2 = " , J_avg_2)
# print("J_avg_3 = " , J_avg_3)
# print("J_avg = " , J_avg)

# sys.exit(0) 


# plot_seismogram(time_values1, seismo_ux1, component="ux", filename="ux_s1")
# plot_seismogram(time_values1, seismo_uz1, component="uz", filename="uz_s1")

# plot_seismogram(time_values2, seismo_ux2, component="ux", filename="ux_s2")
# plot_seismogram(time_values2, seismo_uz2, component="uz", filename="uz_s2")

# plot_seismogram(time_values3, seismo_ux3, component="ux", filename="ux_s3")
# plot_seismogram(time_values3, seismo_uz3, component="uz", filename="uz_s3")

# sys.exit(0) 
# accept_every_trial_step

#######

# epsilon = 1.0e-3
# dof = 3840

dJ_AD_function = compute_gradient(J_avg, Control(rho_dv), options = {"riesz_representation" : "l2"})
VTKFile(os.path.join(results_dir, "sensitivity.pvd")).write(dJ_AD_function)

# sys.exit(0) 


# Auto_Node = dJ_AD_function.dat.data[dof]

def tabulate_coordinates():
    A_vec = VectorFunctionSpace(mesh, method_e, deg_e)
    a_vec = Function(A_vec)
    a_vec.interpolate(mesh.coordinates)
    return a_vec.dat.data

def change_value(a, dof, delta_value):
    a.dat.data[dof] = a.dat.data[dof] + delta_value

# coords_value = tabulate_coordinates()
# my_value = change_value(rho_dv , dof , epsilon)

# J_avg_new = cost_function(V, u, v, rho, lmbda, dt, mu, nt, freq, amp, Sx_spatial, u_list_ref, "wave_output")
# dJ_FD = (J_avg_new - J_avg)/epsilon
# print("DoF =" , dof , " ;  FD =" , dJ_FD , "  ;  AD =", Auto_Node)


########

# =============================================
#     Calculate Adjoint and Sensitivity
# =============================================

control = Control(rho_dv)

rho_dv_viz = Function(A, name = 'rho_dv')
dj_viz = Function(A, name = 'dj')

rho_dv_pvd_file = VTKFile(os.path.join(results_dir, "rho_dv_optimization_iters.pvd"))
dj_pvd_file = VTKFile(os.path.join(results_dir, "dj_optimization_iters.pvd"))

global current_iteration
current_iteration = 0



import matplotlib
matplotlib.use("TkAgg")

import matplotlib.pyplot as plt
plt.ion()

J_history = []

fig, ax = plt.subplots()
line, = ax.plot([], [], marker='o')
ax.set_xlabel("Iteration")
ax.set_ylabel("J_avg")
ax.set_title("Live Objective Function Convergence")
ax.set_yscale("log")
ax.grid(True)


file_path = os.path.join(results_dir, "J_history.txt")
with open(file_path, "w") as f:
    f.write("Iteration\tJ\n")  # optional header


def derivative_cb_post(j, djs, current_rho_dvs):

    global current_iteration 

    # if current_iteration > 4:
    #     with pyadjoint.stop_annotating():
    #         R_mu.assign(1.0e-8)
    #         R_lmbda.assign(1.0e-8)

    if current_iteration > 0 and current_iteration % 20 == 0:
        with stop_annotating():
            R_lmbda.assign(R_lmbda * (1.0 / 3.0))

    if type(j).__name__ == 'NoneType':
        assert current_iteration in [0, 1]
        j = J_avg

    j_val = float(j)

    print("\n [Iteration: %d] J = %1.7e\n" %(current_iteration, j))

    with open(file_path, "a") as f:
        f.write(f"{current_iteration}\t{j_val}\n")


    J_history.append(j_val)
    line.set_data(range(len(J_history)), J_history)

    ax.relim()
    ax.autoscale_view()
    fig.canvas.draw_idle()
    plt.pause(0.01)

    current_iteration += 1

    # Update visualization variables
    rho_dv_viz.assign(current_rho_dvs[0])
    dj_viz.assign(djs[0])

    # Write to ParaView files
    rho_dv_pvd_file.write(rho_dv_viz)
    dj_pvd_file.write(dj_viz)

    return djs


###############
### Set bounds only in the middle part of domain from 1<z<3
##############

lb_expr = conditional(le(z, 1.0), 1.0, conditional(le(z, 3.0), 0.0, 0.0))
ub_expr = conditional(le(z, 1.0), 1.0, conditional(le(z, 3.0), 1.0, 0.0))

lb = Function(A, name="lower_bound").interpolate(lb_expr)
ub = Function(A, name="upper_bound").interpolate(ub_expr)

#############################3


J_avg_red = ReducedFunctional(J_avg, control, derivative_cb_post = derivative_cb_post)


# IPOPT

problem_min = MinimizationProblem(J_avg_red, bounds = (0.0, 1.0))
# problem_min = MinimizationProblem(J_avg_red, bounds = (lb, ub))

solver_opt = IPOPTSolver(problem_min, parameters = {'maximum_iterations': 200 , 'accept_every_trial_step':'yes'})
rho_dv_opt = solver_opt.solve()

filename = os.path.join(results_dir, "rho_dv_opt.h5")

# with CheckpointFile(filename, 'w') as hdf5_file:
#     hdf5_file.save_mesh(mesh)
#     hdf5_file.save_function(rho_dv_opt, name="rho_dv_opt")

# SciPy
# my_options = {'disp' : True, 'iprint' : 2, 'maxiter' : 40}
# rho_dv_opt = minimize(J_avg_red, method = "L-BFGS-B", 
#                       scale = 2.0, tol = 1e-10, bounds = (0.0, 1.0), options = my_options)
                      
plt.ioff()
# plt.show(block=False)

VTKFile(os.path.join(results_dir, "rho_dv_opt.pvd")).write(rho_dv_opt)


data = np.loadtxt(file_path, skiprows=1)

iterations = data[:, 0]
J_values = data[:, 1]

plt.figure(figsize=(6, 5))
plt.plot(iterations, J_values, marker='o')
plt.xlabel("Iteration")
plt.ylabel("J_avg")
plt.title("Objective Function Convergence")
plt.grid(True)
plt.yscale("log")   # remove this line if you don’t want log scale
plt.tight_layout()

plt.savefig(os.path.join(results_dir, "J_convergence.png"), dpi=300)
# plt.show()




rho_dv.assign(rho_dv_opt)

VTKFile(os.path.join(results_dir, "rho_dv_final.pvd")).write(rho_dv)


# ==================================
#     Calculate Cost Functionsys.exit(0) 

# ==================================

# cost_function(V,  , v, rho, rho_dv, dt, nt, freq, amp, Sx_spatial1, u_list_ref1, "wave_output_final_1")
# cost_function(V, 7u, v, rho, rho_dv, dt, nt, freq, amp, Sx_spatial2, u_list_ref2, "wave_output_final_2")
# cost_function(V, u, v, rho, rho_dv, dt, nt, freq, amp, Sx_spatial3, u_list_ref3, "wave_output_final_3")


sys.exit(0) 
















# 
rho_dv.assign(rho_dv_1) # Set the optimized design variable 

print("A1")

J_avg = cost_function(V, u, v, rho, rho_dv, dt, nt, freq, amp, Sx_spatial, u_list_ref, "wave_output_opt")

J_avg_red = ReducedFunctional(J_avg, control, derivative_cb_post = derivative_cb_post)

problem_min = MinimizationProblem(J_avg_red, bounds = (0.0, 1.0))
solver_opt = IPOPTSolver(problem_min, parameters = {'maximum_iterations': 1})
rho_dv_opt = solver_opt.solve()

VTKFile(os.path.join(results_dir, "rho_dv_opt.pvd")).write(rho_dv_opt)


print("A2")


#dj_opt = compute_gradient(J_avg, Control(rho_dv), options = {"riesz_representation" : "l2"})


print("J_avg(rho_dv_opt)", J_avg)
#VTKFile(os.path.join(results_dir, "dj_opt.pvd")).write(dj_opt)
