# Description of SEAM Elastic Model
#
# The SEAM model represents a deepwater subsalt region. The model dimensions are 35 km east-west by 40 km north-south by 15 km deep. Water depth varies between 800 m and 2000m. There are nine major sedimentary horizons in the model along with a major salt body. The base of the model contains a salt body that is the source of the salt body at shallower depth. The salt intrudes to the sea floor in some locations. The sedimentary layers are folded around the salt and are faulted by at least twelve faults. The faults cut across three of the sedimentary horizons. The top of the salt is rugose.
#
# Rock properties are defined using a petrophysical model and reservoirs, including overpressure portions of the reservoirs, are included in the model.
#
# Binary file size (Nx, Ny, Nz) = ( 1751  2001   1501) grid
# Spatial sampling a (Dx, Dy, Dz) = (20m, 20m, 10m) grid
# Origin (Ox,Oy,Oz) = (0, 0, 0)
#
# Horizontal and vertical refer to real axes relative to the Earth's surface; not local bedding plane.
#
# The model is provided in a binary gridded format with one file per model parameter. Grid interval is 20 m in both horizontal directions and 10 m in the vertical direction. Information about the binary files is included in a README file that is contained on the disk containing the model. Data are LITTLE endian. File dimensions are nx=1751, ny = 2001, nz = 1501. X varies fastest, then y and then z.
#
# Files and Ranges of parameters are as follows:
#
# Parameter       File Name                       Range of Values
# P-Velocity:     Vp.swab                         1490 to 4901 m/s
# S-Velocity:     Vs.Modify.Affine800.swab        0 to 3051 m/s
# Density:        Den.swab                        1.03 to 3.85 gm/cc
#
# Minimum non zero S velocity is greater than 600 m/s
#
# velocities in m/s
# density in gm/cc

import binascii
import struct
import segyio
import numpy as np
import sys
import matplotlib.pyplot as plt
import h5py

# FIXME test 3D

save_3D = 1     # 1 or 0. 1: save full 3D fields (do just once); 0: save a 2D slice
axis_slice = 2  # 1 or 2. 1: slice over the x axis (x fixed); 2: slice over the y axis (y fixed)
dist_slice = 20 # slice position (in km) between x (or y) min and x (or y) max (xmax=35 km, ymax=40 km)
#output_path = '/home/thiago.santos/SEAM/' 
output_path = '/home/thiago.santos/spyro/velocity_models/seam/' 

# input path and file names
path = '/home/public/SEAM/PHASE-I/SM1-006-mod-all/GL20130225-04899/SEAM_Elastic_Sim/'
vp_file = path+'Vp.swab'
vs_file = path+'Vs.Modify.Affine800.swab'
rho_file= path+'Den.swab'

# output file names
if save_3D:
	label = '_3D'
elif axis_slice==1: # slice at a given x
	label = '_at_x='+str(dist_slice)+'km'
elif axis_slice==2: # slice at a given y
	label = '_at_y='+str(dist_slice)+'km'
else:
	raise ValueError("Please specify axis slice as 1 (x) or 2 (y)")
	
output_vp_file = output_path + 'Vp' + label + '.hdf5'
output_vs_file = output_path + 'Vs' + label + '.hdf5'
output_rho_file = output_path + 'rho' + label + '.hdf5'

# 3D grid constants
km = 1000.
nx = 1751
ny = 2001
nz = 1501
dx = 20. #m 
dy = 20. #m
dz = 10. #m

# 3D grid definition
xmin = 0.
xmax = (nx-1)*dx
ymin = 0.
ymax = (ny-1)*dy
zmin = 0.
zmax = (nz-1)*dz

# loading files
print("Reading vp...")
vp = np.fromfile(vp_file, dtype=np.float32)
print("Reading vs...")
#vs = np.fromfile(vs_file, dtype=np.float32)
print("Reading rho...")
#rho= np.fromfile(rho_file, dtype=np.float32)

# save full 3D fields?
if save_3D: # FIXME not tested {{{
	print("Saving 3D vp...")	
	axes = [nz, nx, ny]
	axes_order = (0, 1, 2)
	axes_order_sort = "F"
	ix = np.argsort(axes_order)
	axes = [axes[o] for o in ix]
	
	#vs  = vs.reshape(*axes, order=axes_order_sort)
	#rho = rho.reshape(*axes, order=axes_order_sort)

	with h5py.File(output_vp_file, "w") as fp:
		vp  = vp.reshape(*axes, order=axes_order_sort)
		vpt = np.flipud(vp.transpose((*axes_order,)))
		fp.create_dataset("velocity_model", data=vpt, dtype="f")
		fp.attrs["shape"] = vpt.shape
		fp.attrs["units"] = "m/s"

	print("Saving 3D vs...")	
	#with h5py.File(output_vs_file, "w") as f:
	#	Ct = np.flipud(C.T)
	#	#Ct = C.T
	#	f.create_dataset("velocity_model", data=Ct, dtype="f") # FIXME check this in Spyro and SeismicMesh
	#	f.attrs["shape"] = Ct.shape
	#	f.attrs["units"] = "m/s"
	
	print("Saving 3D rho...")	
	#with h5py.File(output_rho_file, "w") as f:
	#	Ct = np.flipud(C.T)
		#Ct = C.T
	#	f.create_dataset("density_model", data=Ct, dtype="f") # FIXME check this in Spyro and SeismicMesh
	#	f.attrs["shape"] = Ct.shape
	#	f.attrs["units"] = "m/s"
	
	sys.exit("exit")
#}}}

# slice over x or y axes
print("Preparing the slices (2D fields)...")

dist = dist_slice * km # in m
if axis_slice==1: # slice at a given x distance
	ix = int(dist/dx)
	vp_slice = np.zeros((ny, nz))
	vs_slice = np.zeros((ny, nz))
	rho_slice = np.zeros((ny, nz))
	for iz in range(nz):		# loop over the z axis
		for iy in range(ny): # loop over the y axis
			vp_slice[iy, iz]  = vp[ix + iy*nx + iz*nx*ny]
			vs_slice[iy, iz]  = vs[ix + iy*nx + iz*nx*ny]
			rho_slice[iy, iz]= rho[ix + iy*nx + iz*nx*ny]
elif axis_slice==2: # slice at a given y distance
	iy = int(dist/dy)
	vp_slice = np.zeros((nx, nz))
	vs_slice = np.zeros((nx, nz))
	rho_slice = np.zeros((nx, nz))
	for iz in range(nz):		# loop over the z axis
		for ix in range(nx): # loop over the x axis
			vp_slice[ix, iz]  = vp[ix + iy*nx + iz*nx*ny]
			vs_slice[ix, iz]  = vs[ix + iy*nx + iz*nx*ny]
			rho_slice[ix, iz]= rho[ix + iy*nx + iz*nx*ny]
else:
	raise ValueError("Please specify axis slice as 1 (x) or 2 (y)")
	
print('Saving in hdf5 format')

with h5py.File(output_vp_file, "w") as fp:
	vpt = np.flipud(vp_slice.T)
	fp.create_dataset("velocity_model", data=vpt, dtype="f")
	fp.attrs["shape"] = vpt.shape
	fp.attrs["units"] = "m/s"

with h5py.File(output_vs_file, "w") as fs:
	vst = np.flipud(vs_slice.T)
	fs.create_dataset("velocity_model", data=vst, dtype="f")
	fs.attrs["shape"] = vst.shape
	fs.attrs["units"] = "m/s"

with h5py.File(output_rho_file, "w") as fr:
	rhot = np.flipud(rho_slice.T)
	fr.create_dataset("density_model", data=vst, dtype="f") # FIXME check it
	fr.attrs["shape"] = rhot.shape
	fr.attrs["units"] = "g/cc" # == Gt/km3 

print("Building data to plot...")
if axis_slice==1: # slice at a given x distance
	x = np.linspace(ymin, ymax, ny)
elif axis_slice==2: # slice at a given y distance
	x = np.linspace(xmin, xmax, nx)
else:
	raise ValueError("Please specify axis slice as 1 (x) or 2 (y)")

z = np.linspace(zmin, zmax, nz)*-1
X, Z = np.meshgrid(x, z)

print("Ploting vp...")
vmin=vp_slice.min()
vmax=vp_slice.max()
levels = np.linspace(vmin,vmax,20)
plt.contourf(X, Z, vp_slice.T, levels=levels, vmin=vmin, vmax=vmax)
plt.colorbar()
plt.axis('equal')
plt.show()
plt.savefig('vp' + label + '.png')
plt.close()

print("Ploting vs...")
vmin=vs_slice.min()
vmax=vs_slice.max()
levels = np.linspace(vmin,vmax,20)
plt.contourf(X, Z, vs_slice.T, levels=levels, vmin=vmin, vmax=vmax)
plt.colorbar()
plt.axis('equal')
plt.show()
plt.savefig('vs' + label + '.png')
plt.close()

print("Ploting rho...")
vmin=rho_slice.min()
vmax=rho_slice.max()
levels = np.linspace(vmin,vmax,20)
plt.contourf(X, Z, rho_slice.T, levels=levels, vmin=vmin, vmax=vmax)
plt.colorbar()
plt.axis('equal')
plt.show()
plt.savefig('rho' + label + '.png')
plt.close()

sys.exit("exit")

