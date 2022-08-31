#!/bin/bash
# https://www.manpagez.com/info/gmsh/gmsh-2.8.4/gmsh_16.php
# "-algo front2d" generates equi triangles
gmsh -2 double_material_2_km_x_2_km.geo -clmax 0.150 -algo del2d -format msh2 -o double_material_2_km_x_2_km_h_150m.msh
