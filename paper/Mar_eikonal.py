# -*- coding: utf-8 -*-
"""
Created on Mon Jun  5 16:21:09 2017

@author: luis
"""

#from __future__ import print_function
from dolfin import *
from dolfin_adjoint import *

import numpy as np
import ufl

import sys

import time
import shutil, os

import loadDat as ld

start_time = time.time()

flag_debug = 0

#INFO PROGRESS DBG WARNING ERROR CRITICAL
set_log_level(LogLevel.INFO)

#adj_checkpointing(strategy='multistage', steps=1, snaps_on_disk=1, snaps_in_ram=1, verbose=True)
# turn off redundant output in parallel
parameters["std_out_all_processes"] = True
parameters["form_compiler"]["optimize"] = True
parameters["form_compiler"]["cpp_optimize"] = True
#parameters['form_compiler']['cpp_optimize_flags'] = '-O2'
parameters["form_compiler"]["cpp_optimize_flags"] = "-O3 -ffast-math -march=native"
parameters["form_compiler"]["representation"]='uflacs'
parameters["form_compiler"]["quadrature_degree"] = 5
parameters["ghost_mode"] = "shared_facet"

flag_mesh_from_cad = 0
recover = 0
flag_isola_mesh = 0
otimizar = 0
refinamento = 4
flag_turb = 0
geometria_init = 1
flag_filter = 0
flag_cut_rho = 0
flag_projection = 0
malha_quad = 0
otimizador = 'IPOPT' #ROL_QN - ROL_NK - IPOPT


#Geometria
l = 1
h = 1

global path_name

#Names
if otimizar==1:
#    path_name = "Teste_OPT_refino_" + str(refinamento) + "_" + otimizador + ""
#    path_name = "V2_2x500_Opt_"+str(ws)+"Jtop_"+str(wu)+"Ju_"+str(wa)+"Fa_k1e" + str(int(np.log10(k_0))) +"_q"+str(q_0)+"to"+str(q_1) + "_vol"+str(volfrac)+"_" + otimizador + ""
    path_name = "Apresentacao/V2_2x500_Opt_"+str(ws)+"Jtop_"+str(wu)+"Ju_k1e" + str(int(np.log10(k_0))) +"_q"+str(q_0)+"to"+str(q_1) + "_vol"+str(volfrac)+"_ref"+str(refinamento)+"_" + otimizador + ""
#    path_name = "Jtop_Juref_k1e" + str(int(np.log10(k_0))) +"_q"+str(q_0)+"to"+str(q_1) + "_vol"+str(volfrac)+"_" + otimizador + ""
else:
#    path_name = "Temp_mef"
    path_name = "Teste_mef0"#_refino_" + str(refinamento)


if flag_filter:
    path_name = path_name + "_Filtered"+str(filter_size)


#path_name = path_name + "/Turb"+str(flag_turb)+"_u"+str(int(u0))+"_rot"+str(int(rotacao))+"_kmax_1e"+str(int(np.log10(k_0)))+"_q"+str(q_0)+"to"+str(q_1)+"_Ref"+str(refinamento)+"_Init"+str(geometria_init)+"_Vol"+str(volfrac)
mesh_file_name = path_name + "/Mesh"

print( "\n"+path_name+"\n")

#Adjust name in case of recover
if recover:
    #otimizar = 0
    path_name_rcv = "TEste26_ref5/Turb1_kmax_99999/Turb1_u100_rot10000_q1.0to1.0_Ref3_Init2_mu0.1"

    path_name = path_name_rcv + "/Recovered"
    mesh_file_name = path_name + "/Mesh"
    rho_to_recover = 33

#==============================================================================MESH
#%%                Malha - Mesh
#==============================================================================
"""Definicao da Malha"""
from mshr import *

x_center = 0.0
y_center = 0.0

p0 = Point(0, 0)
p1 = Point(l, h)

rect1 = Rectangle(p0,p1)
domain = rect1

nel = 30*refinamento

pH = {}

if not recover:
    
    if True:
        # Source definition
        xs = (1/9200)*np.linspace(start=3000, stop=7175,
                                  num=int((7175-3000)/25)+1, endpoint=True)
        ys = (1 - 25/3000)*np.ones_like(xs)

        possou = [xs.round(8).tolist(), ys.round(8).tolist()] #% In realtive position between 0 and 1


        pH['possou'] = possou
        Lx = 9.2e6 #%[mm]
        Ly = 3.0e6 #%[mm]
        

      # Domain without layer
        if all(isinstance(x, list) for x in pH['possou']):
            # Sources definition
            possou = np.array(
                [[xp*Lx for xp in pH['possou'][0]],
                 [yp*Ly for yp in pH['possou'][1]]]).T


        lmin = 1e3 * min(9200 / 736, 3000 / 240) # %[mm]
        bdom = 9.2e6 #%[mm]
        hdom = 3.0e6 #%[mm]
        nx = 736
        ny = 240

        mesh = RectangleMesh(
                Point(0.0, 0.0), Point(bdom, hdom), nx, ny, 'left/right')
        velC = ld.loadFromCSV(mesh) #%(see loadDat.py)

        l=bdom
        h=hdom

    else:
            info("Malha nao estruturada\n")
            
            mesh = generate_mesh(domain, nel)
            mesh = Mesh(mesh)
    
    mesh_h5_filename = HDF5File(MPI.comm_world, mesh_file_name  +'.h5','w')
    mesh_h5_filename.write(mesh,"/Mesh")
    mesh_h5_filename.close()
    
        
else:
    mesh = Mesh()
    f = HDF5File(MPI.comm_world, path_name_rcv+"/Mesh.xdmf")
    f.read(mesh)
    f.close()
    

#==============================================================================ClassesControle
#%%                Classes Controle
#==============================================================================



Eik = FunctionSpace(mesh, "CG", 1)
A = FunctionSpace(mesh, "CG", 1)



if geometria_init == 1:     #CASO A
    class Rho_init_Geo1(UserExpression):
        def eval_cell(self, values, x, ufc_cell):
            values[0] = Constant(0.5)
            
            cond = ( x[0]>=l/2 )#+ 0.1 and x[1]>=l/2 +0.1 )
           
            if (cond):
                values[0] = Constant(0.5/2)
                
        def value_shape(self):
            return ()


    rho_init = Rho_init_Geo1(degree=2)

if geometria_init == 0:     #CASO A
    class Rho_init_Geo3(UserExpression):
        def eval_cell(self, values, x, ufc_cell):
            values[0] = Constant(1.0)
            d = 0.1

            if L <= x[1] <= L+l and ( x[0] < 0.05*h or x[0] > 0.45*h ):
                #values[0] = Constant(config_opt.vol_frac)
                values[0] = Constant(0)
                
            if (L+l+L)/2 -d <= x[1] <= (L+l+L)/2 +d and ( x[0] < 4*d or x[0] > h/2 -4*d ):
                values[0] = Constant(0)
                
            if (3*d <= x[0] <= 4*d and (L+l+L)/2 <= x[1] <= (L+l+L)/2 +4*d):
                values[0] = Constant(0) 
                
            if (h/2 -5*d <= x[0] <= h/2 -4*d and (L+l+L)/2 -4*d <= x[1] <= (L+l+L)/2 +d):
                values[0] = Constant(0)                
                
        def value_shape(self):
            return ()


    rho_init = Rho_init_Geo3(degree=2)
    
if geometria_init == 4 or flag_mesh_from_cad:
    rho_init = Constant(0.5)  
    
if geometria_init == 5:
    rho_init = velC
#    rho_init = Constant(1.0)     



def RecoverRho(obj,filename,rho_to_recover):
    A = obj.A
    rho_rcv = Function(A,name='rho_rcv')
    rho_h5_file = HDF5File(MPI.comm_world, filename+'/RhoH5/Rho_it' + str(rho_to_recover)+".h5", "r")
    rho_h5_file.read(rho_rcv, "/rho")
    
    return rho_rcv
    
def ThresholdRho(obj,rho,cut):
    A = obj.A
    r_array = rho.vector().get_local()
    
    r_array[np.where(r_array >= cut)] = 1.0
    r_array[np.where(r_array < cut)] = 0.0
    
    rho.vector()[:] = r_array
    
    return rho
    


#==============================================================================SubDomain
#%%                SubDomain
#==============================================================================
def DefineBoundaries(mesh):
    print("Creating Markers")

    class Sigma_s1(SubDomain):
        def inside(self, x, on_boundary):
            tolx = bdom/nx
            toly = hdom/ny
            
            cond = False
            for p in possou:
                cond = cond or (p[0] -tolx/2 <= x[0] <= p[0] +tolx/2 and p[1] -toly <= x[1] <= p[1] +toly)
#                cond = cond or (near(p[0],x[0]) and p[1] -toly*10 <= x[1] <= p[1] +toly)
            
            return  cond
        
    class Sigma_entrada1(SubDomain):
        def inside(self, x, on_boundary):
            return near(x[0],l) #and on_boundary and x[1] >0
        
    class Paredes(SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary

    
    # Initialize sub-domain instances
    sigma_s1 = Sigma_s1()
    sigma_entrada1 = Sigma_entrada1()
    sigma_p = Paredes()
    
    # Initialize mesh function for boundary domains
    facet_marker = MeshFunction("size_t", mesh, mesh.topology().dim() -1)
    facet_marker.set_all(0)
    sigma_p.mark(facet_marker,2)
    sigma_s1.mark(facet_marker,3)
#    sigma_entrada1.mark(facet_marker,1)
    
    # Define new measures associated with the exterior boundariess
    ds = Measure("ds")(subdomain_data=facet_marker)

    #Volume
#    class OmegaSub(SubDomain):
#        def inside(self, x, on_boundary):
#            return lc <= x[0] <= l + lc
#
#    osub = OmegaSub()
#    # create sumdomains
#    domains = MeshFunction("size_t", mesh, mesh.topology().dim())
#    domains.set_all(0)
#    osub.mark(domains, 1)
#
#    dx = Measure('dx', domain=mesh, subdomain_data=domains) #DX(1) é o sub domain de integrar!
    dx = Measure("dx")

    File("Temp_files/facet.pvd") << facet_marker
#    File("Temp_files/domain.pvd") << domains
        
    return dx,ds,facet_marker


dx,ds,facet_marker = DefineBoundaries(mesh)

#==============================================================================
#%%                Boundary Conditions
#==============================================================================
def DefineBcs(mesh):

    class InletProfile(UserExpression):
        def __init__(self,mesh,inlet_normal, inlet_center,inlet_radius,v_max, angulo,degree=1):
            super(InletProfile, self).__init__(self)
            self.mesh = mesh
            self.inlet_normal = inlet_normal
            self.inlet_center = inlet_center
            self.inlet_radius = inlet_radius
            self.v_max = v_max
            self.angulo= angulo
                
        def get_value(self, x, v_peak):
            # distance to center
            d = pow(pow(x[0]-self.inlet_center[0],2)+pow(x[1]-self.inlet_center[1],2),0.5)
            # parabolic profile
            v_value = v_peak*(1-pow(d/self.inlet_radius,2))
            return v_value
    
        def eval_cell(self, value, x, ufc_cell):
                cell = Cell(self.mesh, ufc_cell.index)
                n = cell.normal(ufc_cell.local_facet)
                                
                val = self.get_value(x,self.v_max)
                #v_drag_x = self.get_value(x,-self.omega[2]*x[1])               
                #v_drag_y = self.get_value(x,self.omega[2]*x[0])                
                
                #value[0] = val*self.inlet_normal[0] - v_drag_x
                #value[1] = val*self.inlet_normal[1] - v_drag_y
                #value[2] = val*self.inlet_normal[2]
                
#                uf = val
                uf = self.v_max
                theta = self.angulo * pi/180
    
                vx = cos(theta) * (-uf*n[0]) - sin(theta) * (-uf*n[1])
                vy = sin(theta) * (-uf*n[0]) + cos(theta) * (-uf*n[1])
    
                value[0] = vx
                value[1] = vy
    
        def value_shape(self):
            return (2,)
    
    
    Gamma2 = DirichletBC(Eik,(0.0),facet_marker,2)
    Gamma1 = DirichletBC(Eik,(0 ),facet_marker,3)
#    bcs = [Gamma1,Gamma2]
    bcs = [Gamma1]

    return bcs


bcs_eik = DefineBcs(mesh)

#==============================================================================
#%%                Forward
#==============================================================================
def AssembleEikonal(rho,yp,vy):
    f = 1.0
    eps = CellDiameter(mesh) #(obj.mesh.hmin()/1.*2.) #Estabilizador
    F = ( rho*sqrt(inner(grad(yp), grad(yp)))*vy*dx 
        - f*vy*dx 
        + rho*eps*inner(grad(yp), grad(vy))*dx 
        )
   
    return F

def SolveEikonal(rho, yp_final, F, annotate=False):
    Eik_fspace = Eik
    yp = yp_final

    vy = TestFunction(Eik_fspace)
    u = TrialFunction(Eik_fspace)
    f = (1.0)
    
    print("\n Solve Eikonal Pré\n")
    F1 = rho*inner(grad(u), grad(vy))*dx -f*vy*dx
    solve(lhs(F1) == rhs(F1), yp, bcs_eik, annotate=False)
    
    File('Temp_files/pre_eik.pvd') << yp
    
    F = AssembleEikonal(rho,yp,vy)
    
    print("\n Solve Eikonal Pós (Annotate: "+ str(annotate)+ ")\n")
    
    
    
    #% Solver Parameters
    #PETScOptions.set("help")
#    PETScOptions.set("snes_monitor_cancel")
    PETScOptions.set('snes_type', 'vinewtonssls') # newtonls newtontr test nrichardson ksponly vinewtonrsls vinewtonssls ngmres qn shell ngs ncg fas ms nasm anderson aspin composite python
    PETScOptions.set("snes_max_it", 1000)
    PETScOptions.set("snes_atol", 5e-6)
    PETScOptions.set("snes_rtol", 1e-20)
    
    #Newton LS
    PETScOptions.set('snes_linesearch_type','l2') # basic bt nleqerr cp l2
    PETScOptions.set("snes_linesearch_damping",0.5) #for basic,l2,cp
    PETScOptions.set("snes_linesearch_maxstep", 0.5) #bt,l2,cp
    PETScOptions.set('snes_linesearch_order', 2) #for newtonls com bt
    
    
    PETScOptions.set('ksp_type','gmres')
    PETScOptions.set('pc_type','lu') # jacobi pbjacobi bjacobi sor lu shell mg eisenstat ilu icc cholesky asm gasm ksp composite redundant nn mat fieldsplit galerkin exotic cp lsc redistribute svd gamg kaczmarz hypre pfmg syspfmg tfs bddc python
    
#    PETScOptions.set('ksp_type','pipegcr')
##    PETScOptions.set('pc_type','mg') # jacobi pbjacobi bjacobi sor lu shell mg eisenstat ilu icc cholesky asm gasm ksp composite redundant nn mat fieldsplit galerkin exotic cp lsc redistribute svd gamg kaczmarz hypre pfmg syspfmg tfs bddc python
##    PETScOptions.set('ksp_monitor_cancel')
#    #PETScOptions.set('pc_hypre_type','pilut')
#    
#    #PAULO
#    PETScOptions.set('pc_type','mg')
#    PETScOptions.set('pc_mg_levels',1)                  # Number of Levels <1>
#    PETScOptions.set('pc_mg_cycle_type', 'v')           #  Cylce type, one of: invalid v w <v>
#    PETScOptions.set('pc_mg_galerkin', 'TRUE')          # Use Galerkin process to compute coarser operators <FALSE>
#    PETScOptions.set('pc_mg_smoothup',2)                # Number of post-smoothing steps <2>
#    PETScOptions.set('pc_mg_smoothdown',2)              # Number of pre-smoothing steps <2>
#    PETScOptions.set('pc_mg_type','KASKADE')            # (choose one of) MULTIPLICATIVE ADDITIVE FULL KASKADE <MULTIPLICATIVE>
#    PETScOptions.set('mg_levels_0_ksp_type','richardson')
#    PETScOptions.set('mg_levels_0_esteig_ksp_type','gmres') # gmres
#    
#    PETScOptions.set('mg_levels_0_pc_type','lu') # sor eisenstat lu
#    PETScOptions.set('mg_levels_0_pc_factor_shift_type','POSITIVE_DEFINITE') # (choose one of) NONE NONZERO POSITIVE_DEFINITE INBLOCKS <NONE>
#    PETScOptions.set("mg_levels_0_pc_factor_mat_solver_package", "mumps") # petsc, superlu, superlu_dist, mumps, cusparse works only for LU and Cholesky
      
    solve(F == 0, yp_final, bcs_eik, solver_parameters={"nonlinear_solver": "snes"},annotate=annotate)

    return yp_final



#%% MAIN

#rho_init = Constant(0.)
rho = velC #Function(A,name='Medium velocity (m/s)')
#rho.interpolate(rho_init)

rho_file = File("Temp_files/Rho_init.pvd")
yp_file = File("Temp_files/YP.pvd")



yp = Function(Eik,name='eikonal (time [s])')
vy = TestFunction(Eik)
Feik = AssembleEikonal(rho,yp,vy)
yp.assign(SolveEikonal(rho,yp,Feik,annotate=False))

yp_file  << yp
rho_file << rho



