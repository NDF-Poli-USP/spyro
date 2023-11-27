import spyro

mesh_file = "automatic_mesh.msh"
title_str = "Marmousi mesh C = 2.29"

spyro.plots.plot_mesh_sizes(mesh_file, title_str=title_str, show=False, output_filename="adapted_mesh_marmousi.png")
