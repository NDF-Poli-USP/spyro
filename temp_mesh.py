import spyro


Lz = 1.0
Lx = 2.0
c = 1.5
freq = 5.0
lbda = c/freq
pad = 0.3

cpw = 3
Mesh_obj = spyro.meshing.AutomaticMesh(
    dimension=2,
    abc_pad=pad,
    mesh_type="SeismicMesh"
)

Mesh_obj.set_mesh_size(length_z=Lz, length_x=Lx)

Mesh_obj.set_seismicmesh_parameters(cpw=cpw, edge_length=lbda/cpw, output_file_name="test.msh")

Mesh_obj.create_mesh()
