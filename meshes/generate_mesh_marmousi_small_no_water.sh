#!/bin/bash
# https://www.manpagez.com/info/gmsh/gmsh-2.8.4/gmsh_16.php
# "-algo front2d" generates equi triangles
gmsh -2 marmousi_small_no_water.geo -clmax 0.02 -algo del2d -format msh2 -o marmousi_small_no_water_h_20m.msh
