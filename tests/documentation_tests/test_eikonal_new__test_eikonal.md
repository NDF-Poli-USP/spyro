# Complete documentation of `test_eikonal`

Source: [`tests/on_one_core/test_eikonal_new.py`](../on_one_core/test_eikonal_new.py) (function `test_eikonal`, line 255).

This is the complete, unabridged docstring. The test file keeps a reduced version that points here.

```text
Testing of eikonal for 2D and 3D case in Fig. 8 of Salas et al (2022).

See Salas et al (2022): Hybrid absorbing scheme based on hyperelliptical
layers with non-reflecting boundary conditions in scalar wave equations.
doi: https://doi.org/10.1016/j.apm.2022.09.014

Parameters
----------
element_geometry : `str`
    Geometry of the finite element. Options: "T" for triangles/tetrahedra or
    "Q" for quadrilaterals/hexahedra.
dimension : `int`
    Dimension of the problem. 2 for 2D and 3 for 3D.
element_type : `str`
    Finite element type. Options: "consistent" or "underintegrated".
source_bc : `str`
    Dirichlet condition at the source. Options: "nearest_node" (zero at
    the nearest node) or "local_time" (local travel time within one cell
    of the source).

Returns
-------
None

Notes
-----
Theoretical value. eik_min is the first-arrival time on the absorbing
boundaries: z = -1, x = 0, x = 1 and, in 3D, y = 0 and y = 1 (z = 0 is
the free surface). The source is at (z, x) = (-0.5, 0.25) km in 2D and
(z, x, y) = (-0.5, 0.25, 0.5) km in 3D. The velocity is 3.0 km/s for
x < 0.5 and 1.5 km/s for x >= 0.5. The first arrival on a boundary
follows the straight ray to its nearest point, since no other path is
shorter and none reaches a faster medium:

Boundary   Nearest point (z, x)   Path                          Time [ms]
x = 0      (-0.5, 0.0)            0.25 km at 3.0 km/s              83.333
z = -1     (-1.0, 0.25)           0.50 km at 3.0 km/s             166.667
y = 0, 1   3D: (-0.5, 0.25, 0|1)  0.50 km at 3.0 km/s             166.667
x = 1      (-0.5, 1.0)            0.25 km at 3.0 km/s
                                  + 0.50 km at 1.5 km/s           416.667

eik_min = 0.25 km / 3.0 km/s = 83.333 ms, on x = 0 at (-0.5, 0.0) km
(at (-0.5, 0.0, 0.5) km in 3D).

eik_min = 83.333 ms (Theoretical value)

Selection rule, applied separately for each source condition (source_bc).
Successful configurations: the nonlinear solver converges at f_est - 0.01,
f_est and f_est + 0.01 without retries, the three minimum Eikonal values
stay within 10 % of each other and increase with f_est ('nearest_node')
or change monotonically or by less than 0.1 % ('local_time', whose result
barely depends on f_est), and the relative error
abs(eik_min - 83.333) / 83.333 is below 3 %. For 'local_time' only meshes
with N >= 5 are considered: with N <= 4 the source region (radius
lmin = 1 / N >= 0.25 km) reaches the boundary x = 0, where the minimum
Eikonal would be imposed instead of computed. N is the number of
elements per direction (edge_length = 1 / N) and p the Eikonal degree.
The winner (*) is the coarsest mesh, then the smallest relative error.

*******************************
Source condition 'nearest_node'
*******************************

=========================
2D - Consistent triangles
=========================
 N  p  f_est  eik_min   error
 2  4  0.09    83.164   0.20 %*
 3  4  0.11    83.338   0.01 %
 4  3  0.02    83.120   0.26 %
 5  3  0.04    83.319   0.02 %
 5  4  0.08    83.304   0.03 %
 6  2  0.10    83.949   0.74 %
 6  3  0.16    83.544   0.25 %
 6  4  0.03    83.382   0.06 %
 7  3  0.14    83.056   0.33 %
 7  4  0.03    82.916   0.50 %
 8  1  0.10    83.206   0.15 %
 8  2  0.03    85.831   3.00 %
10  2  0.06    82.942   0.47 %
10  3  0.09    83.401   0.08 %
10  4  0.03    84.047   0.86 %
12  1  0.09    83.513   0.22 %
12  2  0.03    85.637   2.77 %
15  2  0.15    83.125   0.25 %
15  3  0.08    83.276   0.07 %
15  4  0.03    83.907   0.69 %
20  1  0.06    83.329   0.01 %
20  2  0.03    85.068   2.08 %
20  3  0.03    85.791   2.95 %

==============================
2D - Consistent quadrilaterals
==============================
 N  p  f_est  eik_min   error
 3  4  0.03    83.908   0.69 %*
 4  1  0.21    81.351   2.38 %
 5  4  0.09    83.335   0.00 %
 6  2  0.07    84.037   0.84 %
 6  3  0.15    83.352   0.02 %
 6  4  0.03    84.804   1.76 %
 7  2  0.81    81.744   1.91 %
 7  3  0.20    83.486   0.18 %
 7  4  0.02    85.409   2.49 %
 8  1  0.07    82.523   0.97 %
10  1  0.56    83.320   0.02 %
10  2  0.05    83.744   0.49 %
10  3  0.09    83.433   0.12 %
10  4  0.02    83.453   0.14 %
12  1  0.07    83.797   0.56 %
15  2  0.17    83.264   0.08 %
15  3  0.09    83.285   0.06 %
15  4  0.02    84.873   1.85 %
20  1  0.05    83.257   0.09 %
20  2  0.02    84.154   0.98 %
20  4  0.02    85.634   2.76 %

==============================
2D - Underintegrated triangles
==============================
 N  p  f_est  eik_min   error
 5  2  0.11    83.350   0.02 %*
 5  3  0.12    83.604   0.33 %
 5  4  0.12    83.207   0.15 %
 6  2  0.11    84.113   0.94 %
 6  3  0.13    83.516   0.22 %
 7  2  0.24    83.357   0.03 %
 7  3  0.05    84.738   1.69 %
10  2  0.08    84.272   1.13 %
10  3  0.08    83.405   0.09 %
15  2  0.11    83.381   0.06 %
15  3  0.05    85.108   2.13 %

===================================
2D - Underintegrated quadrilaterals
===================================
 N  p  f_est  eik_min   error
 3  1  0.01    83.789   0.55 %*
 5  4  0.11    83.289   0.05 %
 6  3  0.15    83.179   0.18 %
 6  4  0.04    85.582   2.70 %
 7  3  0.23    83.306   0.03 %
 8  1  0.19    83.243   0.11 %
10  3  0.09    83.402   0.08 %
10  4  0.04    85.593   2.71 %
12  1  0.14    83.258   0.09 %
15  2  0.20    83.315   0.02 %
15  3  0.10    83.166   0.20 %
20  1  0.13    85.045   2.05 %

==========================
3D - Consistent tetrahedra
==========================
 N  p  f_est  eik_min   error
 2  3  0.15    82.667   0.80 %*
 3  2  0.18    82.781   0.66 %
 3  3  0.19    83.896   0.68 %
 5  2  0.05    82.044   1.55 %
 6  2  0.04    84.603   1.52 %
 6  3  0.03    85.114   2.14 %
 7  2  0.05    82.259   1.29 %
 7  3  0.05    82.162   1.41 %
 8  1  0.05    82.516   0.98 %

=========================
3D - Consistent hexahedra
=========================
 N  p  f_est  eik_min   error
 2  3  0.10    85.386   2.46 %*
 3  3  0.16    82.970   0.44 %
 5  2  0.07    84.542   1.45 %
 5  3  0.05    83.563   0.28 %
 6  1  0.15    84.112   0.94 %
 7  2  0.08    84.377   1.25 %
 7  3  0.05    82.343   1.19 %

===============================
3D - Underintegrated tetrahedra
===============================
 N  p  f_est  eik_min   error
 4  1  0.19    82.222   1.33 %*
 5  2  0.08    84.861   1.83 %

==============================
3D - Underintegrated hexahedra
==============================
 N  p  f_est  eik_min   error
 4  1  0.21    84.007   0.81 %*
 5  3  0.06    85.008   2.01 %
 8  1  0.10    83.406   0.09 %

*****************************
Source condition 'local_time'
*****************************

=========================
2D - Consistent triangles
=========================
 N  p  f_est  eik_min   error
 5  4  0.05    80.904   2.92 %*
 6  3  0.05    82.474   1.03 %
 6  4  0.05    81.372   2.35 %
 7  1  0.07    81.712   1.94 %
 7  2  0.05    83.257   0.09 %
 7  3  0.05    82.614   0.86 %
 7  4  0.05    81.614   2.06 %
 8  2  0.02    84.070   0.88 %
 8  3  0.05    82.446   1.06 %
 8  4  0.05    82.801   0.64 %
10  1  0.06    83.613   0.34 %
10  2  0.03    83.477   0.17 %
10  3  0.06    83.333   0.00 %
10  4  0.05    83.169   0.20 %
12  1  0.09    83.513   0.22 %
12  2  0.02    83.504   0.21 %
12  3  0.03    83.686   0.42 %
12  4  0.02    83.525   0.23 %
15  1  0.05    84.293   1.15 %
15  2  0.04    83.869   0.64 %
15  3  0.04    84.098   0.92 %
15  4  0.03    83.647   0.38 %
20  1  0.04    83.599   0.32 %
20  2  0.02    84.027   0.83 %
20  3  0.03    83.881   0.66 %
20  4  0.03    83.838   0.61 %

==============================
2D - Consistent quadrilaterals
==============================
 N  p  f_est  eik_min   error
 5  4  0.05    81.368   2.36 %*
 6  1  0.40    83.155   0.21 %
 6  3  0.05    82.256   1.29 %
 6  4  0.05    81.169   2.60 %
 7  1  0.05    85.184   2.22 %
 7  2  0.05    83.959   0.75 %
 7  3  0.05    82.591   0.89 %
 7  4  0.05    81.286   2.46 %
 8  2  0.05    83.090   0.29 %
 8  3  0.05    82.519   0.98 %
 8  4  0.05    82.509   0.99 %
10  1  0.04    84.912   1.90 %
10  2  0.03    83.293   0.05 %
10  3  0.07    83.188   0.17 %
10  4  0.05    83.327   0.01 %
12  1  0.07    83.797   0.56 %
12  2  0.03    83.445   0.13 %
12  3  0.03    83.639   0.37 %
12  4  0.02    83.399   0.08 %
15  1  0.03    85.145   2.17 %
15  2  0.02    83.902   0.68 %
15  3  0.03    83.983   0.78 %
15  4  0.02    83.495   0.19 %
20  1  0.03    83.060   0.33 %
20  2  0.02    83.746   0.50 %
20  3  0.03    83.862   0.64 %
20  4  0.02    83.629   0.36 %

==============================
2D - Underintegrated triangles
==============================
 N  p  f_est  eik_min   error
 6  3  0.05    81.938   1.67 %*
 6  4  0.05    81.237   2.51 %
 7  2  0.08    82.155   1.41 %
 7  3  0.05    82.253   1.30 %
 7  4  0.05    81.769   1.88 %
 8  2  0.08    83.090   0.29 %
 8  3  0.08    81.788   1.85 %
 8  4  0.05    82.656   0.81 %
10  2  0.08    83.100   0.28 %
10  3  0.08    83.213   0.14 %
10  4  0.05    83.274   0.07 %
12  3  0.04    83.684   0.42 %
12  4  0.04    83.666   0.40 %
15  1  0.13    85.401   2.48 %
15  2  0.10    84.724   1.67 %
15  4  0.04    83.787   0.55 %
20  2  0.10    85.039   2.05 %
20  3  0.04    83.989   0.79 %
20  4  0.05    84.127   0.95 %

===================================
2D - Underintegrated quadrilaterals
===================================
 N  p  f_est  eik_min   error
 5  4  0.05    81.376   2.35 %*
 6  3  0.05    82.226   1.33 %
 6  4  0.05    81.168   2.60 %
 7  3  0.08    82.090   1.49 %
 7  4  0.06    81.018   2.78 %
 8  2  0.12    81.338   2.39 %
 8  3  0.08    82.061   1.53 %
 8  4  0.05    82.469   1.04 %
10  1  0.19    83.282   0.06 %
10  2  0.13    83.043   0.35 %
10  3  0.08    83.159   0.21 %
10  4  0.05    83.310   0.03 %
12  1  0.14    83.258   0.09 %
12  4  0.04    83.531   0.24 %
15  1  0.13    84.019   0.82 %
15  2  0.13    85.338   2.41 %
15  4  0.05    83.808   0.57 %
20  1  0.13    84.457   1.35 %
20  2  0.12    85.051   2.06 %
20  4  0.05    84.067   0.88 %

==========================
3D - Consistent tetrahedra
==========================
 N  p  f_est  eik_min   error
 5  1  0.10    83.563   0.28 %*
 6  1  0.20    82.911   0.51 %
 6  3  0.05    84.374   1.25 %
 7  3  0.04    85.171   2.21 %
 8  1  0.33    83.447   0.14 %
 8  3  0.03    84.417   1.30 %

=========================
3D - Consistent hexahedra
=========================
 N  p  f_est  eik_min   error
 6  2  0.14    83.361   0.03 %*
 6  3  0.05    83.778   0.53 %
 7  3  0.02    85.018   2.02 %
 8  1  0.09    84.002   0.80 %
 8  3  0.02    84.505   1.41 %

===============================
3D - Underintegrated tetrahedra
===============================
 N  p  f_est  eik_min   error
 6  1  0.20    83.084   0.30 %*
 7  3  0.06    84.486   1.38 %

==============================
3D - Underintegrated hexahedra
==============================
 N  p  f_est  eik_min   error
 6  1  0.22    83.733   0.48 %
 6  3  0.06    83.384   0.06 %*
 7  3  0.03    85.424   2.51 %
 8  1  0.34    83.346   0.02 %
 8  3  0.04    85.223   2.27 %

The tables were computed with Firedrake 2025.4.1. With Firedrake 2026.10.0
every selected configuration (*) converges at its f_est without retries
and gives the same minimum Eikonal within 0.01 ms, except (in ms,
2026.10.0 vs 2025.4.1):

Source condition   Case              2026.10.0   2025.4.1
'nearest_node'     T-3D consistent      82.101     82.667
'local_time'       T-3D consistent      84.164     83.563

'nearest_node' T-3D consistent passes with both versions (error 1.48 %
against 0.80 %, atol = 2.5e-2). 'local_time' T-3D consistent fails with
2026.10.0 (error 1.00 % against atol = 5e-3). Its mesh (N = 5) has 50 of
its 750 tetrahedra with the centroid exactly on the velocity interface
x = 0.5, where the cell velocity is decided by round-off in the condition
x < 0.5. With 2025.4.1 these cells are slow (83.563 ms); forcing them to
the fast side gives 84.014 ms and moves the minimum Eikonal on x = 1 from
511.8 to 480.7 ms, so the installations evaluate these cells differently.
```
