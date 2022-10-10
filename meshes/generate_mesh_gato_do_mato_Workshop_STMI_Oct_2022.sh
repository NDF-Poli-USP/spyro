#!/bin/bash
# https://www.manpagez.com/info/gmsh/gmsh-2.8.4/gmsh_16.php
# "-algo front2d" generates equi triangles
gmsh -2 gato_do_mato_Workshop_STMI_Oct_2022.geo -clmax 0.085 -algo del2d -format msh2 -o gato_do_mato_Workshop_STMI_Oct_2022.msh
