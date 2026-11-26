# Complete documentation of `test_local_time_eikonal_3d_matches_theory_on_all_ranks`

Source: [`tests/parallel/test_eikonal_parallel.py`](../parallel/test_eikonal_parallel.py) (function `test_local_time_eikonal_3d_matches_theory_on_all_ranks`, line 637).

This is the complete, unabridged docstring. The test file keeps a reduced version that points here.

```text
Check the 3D Eikonal with the 'local_time' source condition in parallel.

Theoretical value. The velocity ``c = 3 - 1.5 x`` depends on x only, so
the ray from the source ``(-0.5, 0.25, 0.5)`` to the boundary ``x = 0``
keeps its z and y coordinates (the ray parameters along z and y are zero)
and its travel time is

    t = int_0^0.25 dx / (3 - 1.5 x) = ln(3 / 2.625) / 1.5 = 89.021 ms,

reached at ``(-0.5, 0, 0.5)``. Every other absorbing boundary is at least
0.5 km away through velocities of at most 3 km/s, i.e. at least 167 ms.

Calibration (degree 2, consistent, Firedrake 2025.4.1). Unlike in 2D, the
3D result depends on f_est, roughly in proportion to it (the artificial
diffusion acts on the point-source cone, whose Laplacian (d - 1) / r is
twice as large in 3D), so the stabilizing factor must be small. Minimum
Eikonal error for N = 1 / edge_length and f_est (r: the solver retried
with a larger f_est):

    T  N    f=0.01      f=0.02   f=0.03   f=0.04   f=0.05   f=0.06
       7    r (0.04)    r        r        +2.66 %  +3.00 %  +3.15 %
       8    r (0.02)    +2.70 %  +4.18 %  +5.20 %  +6.00 %  +6.71 %
       10   r (0.02)    +2.00 %  +3.64 %  +4.76 %  +5.68 %  +6.52 %
       12   r (0.02)    +0.70 %* +3.47 %  +5.11 %  +6.42 %  +7.61 %

    Q  N    f=0.05      f=0.07   f=0.08   f=0.09   f=0.10
       5    -7.05 %     -8.97 %  -9.72 %  -10.4 %  -10.9 %
       6    -1.17 %     -0.51 %  -0.29 %  -0.10 %* +0.07 %

T: N = 7 with f_est = 0.05 is the coarsest stable choice below 3 %, but
at +2.999 % it leaves no margin, so N = 12 with f_est = 0.02 (+0.70 %) is
used; below 0.02 the solver needs more stabilization and retries. Q: on
N = 5 the source region (radius lmin = 0.2 km) covers most of the
0.25 km to the boundary and the result falls as f_est grows, so N = 6
with f_est = 0.09 (-0.10 %, stable at 0.08 and 0.10) is used. One and
two ranks give the same values (T: to 4e-6 relative, Q: identical).
Firedrake 2026.10.0 on two ranks gives T 89.644 ms (+0.70 %) and
Q 88.936 ms (-0.10 %), within 3e-5 of the 2025.4.1 values. Both
configurations converge at their f_est without retries with both
versions.

Parameters
----------
geometry : `str`
    Element geometry, "T" or "Q".
edge_length : `float`
    Mesh size in km.
f_est : `float`
    Factor for the stabilizing term in the Eikonal equation.

Returns
-------
None
```
