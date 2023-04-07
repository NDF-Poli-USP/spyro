import firedrake as fire
from firedrake import And, conditional, File
import numpy as np
import spyro
from SeismicMesh import write_velocity_model
import SeismicMesh
import meshio




def apply_box(mesh, c, x1, y1, x2, y2, value):
    y1 = 2.4 - y1
    y2 = 2.4 - y2
    x1 = 4.8 - x1
    x2 = 4.8 - x2
    y, x = fire.SpatialCoordinate(mesh)
    box = fire.conditional(And(And(x1>=x, x>=x2), And(y1>=-y, -y>=y2)), value, c)
    print("a")
    c.interpolate(box)
    return c

# def apply_triangle(mesh, c, x1, y1, x3, y3, value):
#     x, y = fire.SpatialCoordinate(mesh)
#     slope = (y3-y1)/(x3-x1)
#     print(slope)
#     # triangle = fire.conditional(And( (y-y1)/(x-x1) <= slope , And(y1<=y, x<=x3)), value, c)
#     triangle = fire.conditional(And( (y-y1)/(x-x1) <= slope, (y-y1)/(x-x1) >= 0.1) , value, c)
#     c.interpolate(triangle)
#     return c

def apply_slope(mesh, c, x1, y1, x3, y3, value):
    y, x = fire.SpatialCoordinate(mesh)
    y1 = 2.4 - y1
    y3 = 2.4 - y3
    x1 = 4.8 - x1
    x3 = 4.8 - x3
    slope = (y3-y1)/(x3-x1)
    print(slope)
    slope = fire.conditional(And( (-y-y1)/(x-x1) <= slope, x < x1 ), value, c)
    c.interpolate(slope)
    return c

def apply_vs_from_list(velmat, mesh, Lx, Ly, c):
    # (x1, y1, x2, y2, cm)
    for box in velmat:
        x1 = box[0]*Lx
        y1 = box[1]*Ly
        x2 = box[2]*Lx
        y2 = box[3]*Ly
        cm = box[4]
        c = apply_box(mesh, c, x1, y1, x2, y2, cm)
    
    return c


velmat = []
velmat.append([0.00, 0.00, 0.35, 0.10, 2.9*1000])
velmat.append([0.00, 0.10, 0.25, 0.30, 2.9*1000])
velmat.append([0.00, 0.30, 0.25, 0.70, 2.0*1000])
velmat.append([0.00, 0.70, 0.10, 1.00, 3.7*1000])
velmat.append([0.10, 0.70, 0.30, 0.90, 3.7*1000])
velmat.append([0.25, 0.10, 0.75, 0.30, 2.5*1000])
velmat.append([0.25, 0.30, 0.40, 0.70, 2.5*1000])
velmat.append([0.35, 0.00, 0.70, 0.10, 2.1*1000])
velmat.append([0.70, 0.00, 0.90, 0.10, 3.4*1000])
velmat.append([0.80, 0.10, 0.90, 0.35, 3.4*1000])
velmat.append([0.90, 0.00, 1.00, 0.20, 3.4*1000])
velmat.append([0.90, 0.20, 1.00, 0.65, 2.6*1000])
velmat.append([0.75, 0.10, 0.80, 0.50, 4.0*1000])
velmat.append([0.80, 0.35, 0.90, 0.80, 4.0*1000])
velmat.append([0.85, 0.80, 0.90, 0.95, 3.6*1000])
velmat.append([0.90, 0.65, 1.00, 1.00, 3.6*1000])
velmat.append([0.00, 0.00, 0.00, 0.00, 1.5*1000])  

Lx = 4.8
Ly = 2.4
mesh = fire.RectangleMesh(120,240,Ly,Lx, diagonal="crossed")
mesh.coordinates.dat.data[:, 0] *= -1.0

V = fire.FunctionSpace(mesh,"CG", 3)
c = fire.Function(V)

c.dat.data[:] = 1.5*1000

c = apply_slope(mesh, c, 0.4*Lx, 0.3*Ly, 0.75*Lx, 0.65*Ly, 3.3*1000)
c = apply_vs_from_list(velmat, mesh, Lx, Ly, c)

File("before_interpolation.pvd").write(c)

spyro.tools.saving_segy_from_function(c, V, 'first_segy.segy')
write_velocity_model('first_segy.segy', ofname = 'first_segy')

# # Creating seismicmeshmesh


Lz = Ly

bbox = (-Lz*1000, 0.0, 0.0, Lx*1000)
domain = SeismicMesh.Rectangle(bbox)

C = 2.5


print('Entering mesh generation')


fname = 'first_segy.segy'


# Desired minimum mesh size in domain

hmin = 1429.0/(C*5.0)

# Construct mesh sizing object from velocity model
print('a')
ef = SeismicMesh.get_sizing_function_from_segy(
    fname,
    bbox,
    hmin=hmin,
    wl=C,
    freq=5.0,
    grade=0.15,
)
print('b')

points, cells = SeismicMesh.generate_mesh(domain=domain, edge_length=ef, verbose = 2, mesh_improvement=False )

meshio.write_points_cells("meshes/test1.msh",
    points/1000,[("triangle", cells)],
    file_format="gmsh22", 
    binary = False
    )
meshio.write_points_cells("meshes/test1.vtk",
        points/1000,[("triangle", cells)],
        file_format="vtk"
        )


mesh = fire.Mesh(
    "meshes/test1.msh",
    distribution_parameters={
        "overlap_type": (fire.DistributedMeshOverlapType.NONE, 0)
    },
)

V = fire.FunctionSpace(mesh, 'KMV', 4)

spyro.tools.interpolate_to_pvd(
    'first_segy.hdf5',
    V,
    'after_interpolation_with_waveformed.pvd'
    )

# set_log_level(20)
# parameters['std_out_all_processes'] = True
# parameters['form_compiler']['optimize'] = True
# parameters['form_compiler']['cpp_optimize'] = True
# # parameters['form_compiler']['cpp_optimize_flags'] = '-O2'
# parameters['form_compiler'][
#     'cpp_optimize_flags'] = '-O3 -ffast-math -march=native'
# parameters['form_compiler']['representation'] = 'uflacs'
# parameters['form_compiler']['quadrature_degree'] = 5
# parameters['ghost_mode'] = 'shared_facet'


#     mp.my_print('Solve Post-Eikonal')
#     # Solver Parameters
#     # https://petsc.org/release/docs/manualpages/KSP/KSPSetFromOptions.html
#     # https://petsc.org/release/docs/manualpages/SNES/SNESSetFromOptions.html
#     # Solver Parameters
#     # PETScOptions.set("help")
#     # newtonls newtontr test nrichardson ksponly vinewtonrsls vinewtonssls
#     # ngmres qn shell ngs ncg fas ms nasm anderson aspin composite python
#     PETScOptions.set('snes_type', 'vinewtonssls')
#     PETScOptions.set("snes_max_it", 1000)
#     PETScOptions.set("snes_atol", 5e-6)
#     PETScOptions.set("snes_rtol", 1e-20)

#     # Newton LS
#     PETScOptions.set('snes_linesearch_type', 'l2')  # basic bt nleqerr cp l2
#     PETScOptions.set("snes_linesearch_damping", 1.00)  # for basic,l2,cp
#     PETScOptions.set("snes_linesearch_maxstep", 0.50)  # bt,l2,cp
#     PETScOptions.set('snes_linesearch_order', 2)  # for newtonls com bt

#     # damp 0.50 maxstep 1.00 64 it
#     # damp 0.50 maxstep 0.50 64 it
#     # damp 0.75 maxstep 1.00 32 it
#     # damp 0.75 maxstep 1.00 32 it
#     # damp 1.00 maxstep 1.00 08 it
#     # damp 1.00 maxstep 0.50 08 it

#     PETScOptions.set('ksp_type', 'gmres')
#     # jacobi pbjacobi bjacobi sor lu shell mg eisenstat ilu icc cholesky asm
#     # gasm ksp composite redundant nn mat fieldsplit galerkin exotic cp lsc
#     # redistribute svd gamg kaczmarz hypre pfmg syspfmg tfs bddc python
#     PETScOptions.set('pc_type', 'lu')