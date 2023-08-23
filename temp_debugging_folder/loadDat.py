from firedrake import FunctionSpace, Function, SpatialCoordinate, RectangleMesh
import os
import numpy as np


def vertex_to_dof_map(V):
    nx = 736
    ny = 240
    xb = np.linspace(0, 9.2, nx + 1)
    yh = np.linspace(0, 3.0, ny + 1)
    xp, yp = np.meshgrid(xb, yh, sparse=True)
    nox, noy = xp.size, yp.size
    # Maximum number of nodes
    nomax = int(nox * noy)
    # Array of node coordinates
    xnode = np.tile(xp, (1, noy)).T
    ynode = np.tile(yp, (1, nox)).reshape((nomax, 1))
    del xp, yp
    Cn = np.concatenate((xnode, ynode), axis=1)

     

    velc[]
 

    # for point in zip(xp.flatten(), yp.flatten()):
    for i in range((nx+1)*(ny+1)):
        x = i%(nx+1)*(9.2/nx)
        y = i//(nx+1)*(3.0/ny)
        # x,y = point

        for j in range(len(coord)):
            if np.isclose(coord[j][0],x) and np.isclose(coord[j][1],y):
                new_structured_dof_map.append(j)
                break
        
        if i%100 == 0:
            percent=i/((nx+1)*(ny+1))*100
            print(f'{percent}%')
    return new_structured_dof_map
        


def loadFromCSV():
    '''
    Loading a variable from an external file.
    https://sites.google.com/a/kaust.edu.sa/tariq/research/anismarmousi
    "Mar2csv": Marmousi2 9.2km x 3km
    '''
    # Source definition
    xs = (1 / 9200) * np.linspace(start=3000, stop=7175,
                                    num=int((7175 - 3000) / 25) + 1, endpoint=True)
    ys = (1 - 25 / 3000) * np.ones_like(xs)
    possou = [xs.round(8).tolist(), ys.round(8).tolist()]
        
    
    cadMod = '/Mar2VelP.csv'
    nx = 736
    ny = 240
    bdom = 9.2
    hdom = 3.0

    # mesh = RectangleMesh(Point(0.0, 0.0), Point(bdom, hdom), nx, ny, 'left/right')
    mesh = RectangleMesh(nx, ny, bdom, hdom, quadrilateral=True)

    pathDat = os.getcwd() + cadMod
    ########## Conversion Process ##########
    print('Processing External Data')
    # Domain shape depending on format
    shape = (nx+1, ny+1)
    # Get and fix velocity model format
    prop = np.loadtxt(pathDat) / 1e3
    prop = prop.reshape(np.flipud(shape))
    # Get field with indexes in the same order as vertices from function
    prop = np.flipud(prop).flat
    # Create and define function space
    V = FunctionSpace(mesh, 'CG', 1)
    v = Function(V)
    x, y = SpatialCoordinate(mesh)
    ux = Function(V).interpolate(x)
    uy = Function(V).interpolate(y)
    datax = ux.dat.data_ro_with_halos[:]
    datay = uy.dat.data_ro_with_halos[:]
    node_locations = np.zeros((len(datax), 2))
    node_locations[:, 0] = datax
    node_locations[:, 1] = datay
    # Dump values to function in km/s
    vertex_to_dof_map(V)
    # V.vector()[vertex_to_dof_map(M)] = prop
    prop = 0
    del prop

    # https://fenics2021.com/slides/yashchuk.pdf
    # https://github.com/firedrakeproject/firedrake/issues/1881
    # pc.setCoordinates(np.reshape(mesh.coordinates.dat.data_ro, (-1,mesh.geometric_dimension()))) 
    # conda list --revisions
    # conda install --revision xxx
    # pip3 install --no-binary=h5py h5py

    # import h5py
    # f1 = h5py.File(pathDat, 'w')
    # print(f1.keys())
    # from fenics import MPI, HDF5File, XDMFFile
    # file1 = HDF5File(MPI.comm_world, pathDat, 'r')
    # file1.read(self.cbound,'V')
    # file1.close()
    # file1 = XDMFFile(MPI.comm_world, pathDat)
    # file1.read_checkpoint(self.cbound, 'V', 0)
    # file1.close()

    return V


loadFromCSV()
print("END")