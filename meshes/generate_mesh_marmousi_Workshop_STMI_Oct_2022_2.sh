#!/bin/bash
# https://www.manpagez.com/info/gmsh/gmsh-2.8.4/gmsh_16.php
# "-algo front2d" generates equi triangles
# "-algo del2d" generates unstructured meshes
gmsh -2 marmousi_Workshop_STMI_Oct_2022.geo -clmax 0.07 -algo front2d -format msh2 -o marmousi_Workshop_STMI_Oct_2022_2.msh
