import firedrake as fire
import numpy as np

def funVel(c, c_dist, dimensions):
    '''
    Velocity profile inside domain
    '''
    c_coords = c.function_space().tabulate_dof_coordinates()
    pmlx = dimensions[0]
    pmly = dimensions[1]
    Lx = dimensions[2]
    Ly = dimensions[3]
    # Loop for identifying properties
    valdef = c_dist[-1][-1]
    c_array = valdef*np.ones_like(c.vector().get_local())
    x1 = [pmlx + c_dist[i][0]*Lx for i in range(len(c_dist) - 1)]
    x2 = [pmlx + c_dist[i][2]*Lx for i in range(len(c_dist) - 1)]
    y1 = [pmly + c_dist[i][1]*Ly for i in range(len(c_dist) - 1)]
    y2 = [pmly + c_dist[i][3]*Ly for i in range(len(c_dist) - 1)]

    for dof, coord in enumerate(c_coords):
        xc = coord[0]
        yc = coord[1]
        valvel = [c_dist[i][4] for i in range(
            len(c_dist)-1) if xc >= x1[i] and xc <= x2[i]
            and yc >= y1[i] and yc <= y2[i]]
        if len(valvel) > 0:
            c_array[dof] = valvel[0]

    return c_array



velmat = []
velmat.append([0.00, 0.00, 0.35, 0.10, 2.9])
velmat.append([0.00, 0.10, 0.25, 0.30, 2.9])
velmat.append([0.00, 0.30, 0.25, 0.70, 2.0])
velmat.append([0.00, 0.70, 0.10, 1.00, 3.7])
velmat.append([0.10, 0.70, 0.30, 0.90, 3.7])
velmat.append([0.25, 0.10, 0.75, 0.30, 2.5])
velmat.append([0.25, 0.30, 0.40, 0.70, 2.5])
velmat.append([0.35, 0.00, 0.70, 0.10, 2.1])
velmat.append([0.70, 0.00, 0.90, 0.10, 3.4])
velmat.append([0.80, 0.10, 0.90, 0.35, 3.4])
velmat.append([0.90, 0.00, 1.00, 0.20, 3.4])
velmat.append([0.90, 0.20, 1.00, 0.65, 2.6])
velmat.append([0.75, 0.10, 0.80, 0.50, 4.0])
velmat.append([0.80, 0.35, 0.90, 0.80, 4.0])
#
step = 0.001  # 0.01
num = int((0.75 - 0.4) / step) + 1
x1 = np.linspace(start=0.4, stop=0.75, num=num, endpoint=False)
y1 = 0.3 * np.ones_like(x1)
x2 = step + x1
y2 = np.linspace(start=0.3 + step, stop=0.65, num=num, endpoint=True)
cm = 3.3 * np.ones_like(x1) # Propagation speed in [km/s]
d1 = np.stack((x1, y1, x2, y2, cm), axis=-1).round(8).tolist()
velmat += [d1[i] for i in range(len(d1))]
num = int((0.8 - 0.75) / step)
x1 = np.linspace(start=0.75, stop=0.8, num=num, endpoint=False)
y1 = 0.5 * np.ones_like(x1)
x2 = step + x1
y2 = np.linspace(start=0.65 + step, stop=0.7, num=num, endpoint=True)
cm = 3.3 * np.ones_like(x1) # Propagation speed in [km/s]
d2 = np.stack((x1, y1, x2, y2, cm), axis=-1).round(8).tolist()
velmat += [d2[i] for i in range(len(d2))]
#
velmat.append([0.85, 0.80, 0.90, 0.95, 3.6])
velmat.append([0.90, 0.65, 1.00, 1.00, 3.6])
velmat.append([0.00, 0.00, 0.00, 0.00, 1.5])  # Remaining domain
# Source definition in [m/ms^2] = [kN/g] 
possou = [0.35, 0.75] # Relative to domain dimensions
# Domain aspect ratio (Lx/Ly)
Lz = 4.8 # [km]
Lx = 2.4 # [km]
name = 'Velocity (km/s)'
mesh = fire
c = fire.Function(V, name=name)
print('Velocity Profile Inside Domain')
pmlx = pmly = 0
dimensions = np.array([pmlx, pmly, Lx, Ly])
# Velocity profile inside domain
c.dat.data_with_halos[:] = funVel(c, velmat, dimensions)