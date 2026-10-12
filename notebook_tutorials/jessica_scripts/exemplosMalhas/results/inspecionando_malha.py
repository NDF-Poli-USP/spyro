import firedrake as fire

mesh = fire.Mesh("/workspaces/spyro/notebook_tutorials/jessica_scripts/exemplosMalhas/results/case_small_2d_without_bathymetry/mesh.msh")
print("dim:", mesh.topological_dimension, "células:", mesh.cell_set.size)
print("tags de células:", mesh.topology_dm.getLabelIdIS("Cell Sets").indices)
print("tags de facetas externas:", mesh.exterior_facets.unique_markers)
print("tags de facetas internas:", mesh.interior_facets.unique_markers)
c = mesh.coordinates.dat.data_ro
print("min:", c.min(axis=0), "max:", c.max(axis=0))


import numpy as np

z = c[:, 0]
print("vértices em z = -0.502:", np.sum(np.isclose(z, -0.502, atol=1e-6)))

for tag in (1, 2):
    sub = fire.Submesh(mesh, 2, tag)
    cs = sub.coordinates.dat.data_ro
    print(f"tag {tag}: z de {cs[:, 0].min():.4f} até {cs[:, 0].max():.4f}")