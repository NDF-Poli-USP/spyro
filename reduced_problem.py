import firedrake as fire

Lx = 2
Ly = 4
user_mesh = fire.RectangleMesh(20*2, 20*4, Lx, Ly, diagonal="crossed")

