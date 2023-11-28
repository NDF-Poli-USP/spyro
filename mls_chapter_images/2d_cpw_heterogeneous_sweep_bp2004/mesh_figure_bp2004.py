import spyro

mesh_file = "automatic_mesh.msh"
title_str = "BP2004 mesh C = 2.41"

spyro.plots.plot_mesh_sizes(mesh_file, title_str=title_str, show=False, output_filename="adapted_mesh_marmousi.png")
