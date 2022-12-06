import firedrake as fire
import numpy as np

def distance(p1, p2, dim = 2):

    if dim ==2:
        x1, y1 = p1
        x2, y2 = p2
        dist = np.sqrt( (x1-x2)**2 + (y1-y2)**2 )

    if dim ==3:
        x1, y1, z1 = p1
        x2, y2, z2 = p2
        dist = np.sqrt( (x1-x2)**2 + (y1-y2)**2 +(z1-z2)**2 )

    return dist

def change_to_equilateral_triangle(p):
    a = (0.0, 0.0)
    b = (1.0, 0.0)
    c = (0.0, 1.0)
    (xa, ya) = a
    (xb, yb) = b
    (xc, yc) = c
    (px, py) = p
    xna = 0.0
    yna = 0.0
    xnb = 1.0
    ynb = 0.0
    xnc = 1.0*np.cos(np.pi/3)
    ync = 1.0*np.sin(np.pi/3)
    div = xa * yb - xb * ya - xa * yc + xc * ya + xb * yc - xc * yb
    a11 = -(xnb * ya - xnc * ya - xna * yb + xnc * yb + xna * yc - xnb * yc) / div
    a12 = (xa * xnb - xa * xnc - xb * xna + xb * xnc + xc * xna - xc * xnb) / div
    a13 = (
        xa * xnc * yb
        - xb * xnc * ya
        - xa * xnb * yc
        + xc * xnb * ya
        + xb * xna * yc
        - xc * xna * yb
    ) / div
    a21 = -(ya * ynb - ya * ync - yb * yna + yb * ync + yc * yna - yc * ynb) / div
    a22 = (xa * ynb - xa * ync - xb * yna + xb * ync + xc * yna - xc * ynb) / div
    a23 = (
        xa * yb * ync
        - xb * ya * ync
        - xa * yc * ynb
        + xc * ya * ynb
        + xb * yc * yna
        - xc * yb * yna
    ) / div
    pnx = px * a11 + py * a12 + a13
    pny = px * a21 + py * a22 + a23
    return (pnx, pny)

def min_equilateral_distance(mesh, V, rec_pos):
    x_mesh, y_mesh = fire.SpatialCoordinate(mesh)
    x_function = fire.Function(V).interpolate(x_mesh)
    y_function = fire.Function(V).interpolate(y_mesh)
    x = x_function.dat.data[:]
    y = y_function.dat.data[:]
    points_right = list(zip(x,y))
    point_close_rec = []

    for point in points_right:
        dist = distance(rec_pos[1], point)
        if dist<0.1:
            point_close_rec.append(point)
   
    points_equalateral = [change_to_equilateral_triangle(point) for point in point_close_rec]
    points = points_equalateral
    min_length = 1.0
    i = 0
    for point in points:
        for j in range(i+1,len(points)):
            dist = distance(point, points[j])
            if dist < min_length:
                min_length = dist
        i+=1

    return min_length

