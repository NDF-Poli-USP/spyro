#!/bin/bash
gmsh -2 marmousi_mesh_for_moving_mesh_paper.geo -clmax 0.07 -algo front2d -format msh2 -o marmousi_mesh_for_moving_mesh_paper.msh
