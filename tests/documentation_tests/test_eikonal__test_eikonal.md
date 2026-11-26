# Complete documentation of `test_eikonal`

Source: [`tests/on_one_core/test_eikonal.py`](../on_one_core/test_eikonal.py) (function `test_eikonal`, line 253).

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

Selection of f_est. The mesh size and the Eikonal degree are fixed for
each case, and f_est is calibrated separately for each source condition
(source_bc). The marked value (*) is the f_est with the smallest
relative error abs(eik_min - 83.333) / 83.333 among the values for which
the nonlinear solver converges with every supported Firedrake version
(2025.4.1 and 2026.10.0), and is the one used by the test.

*******************************
Source condition 'nearest_node'
*******************************
--/--: the nonlinear solver stalls at this f_est.

==============================
Eikonal for 2D model Δx = 100m
==============================
eik_min = 83.333 ms (Theoretical value)

f_est  T-cons  Q-cons f_est T-under Q-under
        p = 2   p = 2         p = 2   p = 4
 0.02  73.308  75.811  0.02   --/--   --/--
 0.03  77.178  79.845  0.03   --/--  84.245*
 0.04  79.680  82.101  0.04   --/--  85.593
 0.05  81.498  83.744* 0.05   --/--  86.887
 0.06  82.942* 85.118  0.06   --/--  88.145
 0.07  84.160  86.345  0.07  82.630* 89.376
 0.08  85.233  87.480  0.08  84.272  90.589

==============================
Eikonal for 3D model Δx = 150m
==============================
eik_min = 83.333 ms (Theoretical value)

f_est   T-cons  Q-cons f_est T-under Q-under
         p = 2   p = 2         p = 2   p = 3
 0.04   79.370  73.179  0.04   --/--  55.166
 0.05   82.259* 75.766  0.05   --/--   --/--
 0.06   85.344  78.548  0.06   --/--  83.476*
 0.07   88.562  81.431  0.07  83.719* 85.902
 0.08   91.877  84.377* 0.08  86.242  88.456
 0.09   95.272  87.376  0.09  88.867  91.121

*****************************
Source condition 'local_time'
*****************************
--/--: the nonlinear solver stalls at this f_est. #: converges with
Firedrake 2025.4.1 but not with 2026.10.0, where the solver retries with
the next f_est.

==============================
Eikonal for 2D model Δx = 100m
==============================
eik_min = 83.333 ms (Theoretical value)
f_est  T-cons  Q-cons f_est T-under Q-under
        p = 2   p = 2         p = 2   p = 4
 0.07  83.680  83.509  0.03   --/--  83.334*
 0.08  83.640  83.497  0.04   --/--  83.328
 0.09  83.585  83.476  0.05   --/--  83.310
 0.10  83.517  83.446  0.06   --/--  83.282
 0.11  83.437  83.406  0.07  83.088  83.245
 0.12  83.347* 83.357* 0.08  83.100* 83.199
 0.13  83.246  83.300  0.09  83.092  83.144

==============================
Eikonal for 3D model Δx = 150m
==============================
eik_min = 83.333 ms (Theoretical value)
f_est  T-cons  Q-cons f_est T-under Q-under
        p = 2   p = 2         p = 2   p = 3
 0.01   --/--   --/--  0.01   --/--   --/--
 0.02   --/--  85.710* 0.02   --/--  85.061#
 0.03  72.954  85.982  0.03   --/--  85.424*
 0.04  85.533* 86.353  0.04   --/--  85.832
 0.05  85.763  86.740  0.05   --/--  86.229
 0.06  85.872  87.114  0.06   --/--  86.588
 0.07  85.954  87.475  0.07  85.720* 86.900
 0.08  86.029  87.827  0.08  85.968  87.165

The tables were computed with Firedrake 2025.4.1. With Firedrake
2026.10.0 all cases pass. For 'nearest_node' the minimum Eikonal at the
marked f_est agrees within 0.01 ms except (in ms, 2026.10.0 vs 2025.4.1):

Case                    2026.10.0   2025.4.1
T-2D consistent            82.949     82.942
T-3D consistent            82.243     82.259
T-3D underintegrated       83.710     83.719
Q-3D underintegrated       83.302     83.476

For Q-3D underintegrated the critical point is (-0.532, 0, 0.532) km with
2026.10.0 and (-0.468, 0, 0.468) km with 2025.4.1: the source is snapped
to four equidistant nodes, and the installations break the tie between
the resulting boundary minima differently.

For 'local_time' the minimum Eikonal at the marked f_est agrees within
0.01 ms. The exception is Q-3D underintegrated: f_est = 0.02 (85.061 ms,
2.07 %) does not converge with 2026.10.0, so the test uses f_est = 0.03,
the value with the next smallest error, which converges with both
versions (85.424 ms, 2.51 %).
```
