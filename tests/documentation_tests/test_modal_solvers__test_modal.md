# Complete documentation of `test_modal`

Source: [`tests/on_one_core/test_modal_solvers.py`](../on_one_core/test_modal_solvers.py) (function `test_modal`, line 361).

This is the complete, unabridged docstring. The test file keeps a reduced version that points here.

```text
Testing modal solvers for 2D and 3D case in Fig. 8 of Salas et al (2022).

See Salas et al (2022): Hybrid absorbing scheme based on hyperelliptical
layers with non-reflecting boundary conditions in scalar wave equations.
doi: https://doi.org/10.1016/j.apm.2022.09.014

Parameters
----------
wave_instance : acoustic_wave.AcousticWave
    An instance of the :class:`~spyro.solvers.acoustic_wave.AcousticWave`.
element_geometry : `str`
    Geometry of the finite element. Options: "T" for triangles/tetrahedra or
    "Q" for quadrilaterals/hexahedra.
dimension : `int`
    Dimension of the problem. 2 for 2D and 3 for 3D.
layer_shape : `str`
    Shape of the absorbing layer. Options: "rectangular" or "hypershape".
    The hypershape degree is set in MODAL_CONFIG. Q elements are only
    tested with rectangular layers.
homogeneous : `bool`
    If `True`, the velocity model is homogeneous. If `False`, it is heterogeneous.

Returns
-------
None

Notes
-----
Theoretical value. eik_min is the first-arrival time on the absorbing
boundaries: z = -1, x = 0, x = 1 and, in 3D, y = 0 and y = 1 (z = 0 is
the free surface). The source is at (z, x) = (-0.5, 0.25) km in 2D and
(z, x, y) = (-0.5, 0.25, 0.5) km in 3D. The velocity is 1.5 km/s in the
homogeneous model. In the heterogeneous one it is 3.0 km/s for x < 0.5
and 1.5 km/s for x >= 0.5. The first arrival on a boundary follows the
straight ray to its nearest point:

Boundary   Nearest point (z, x)   Homogeneous [ms]   Heterogeneous [ms]
x = 0      (-0.5, 0.0)                     166.667               83.333
z = -1     (-1.0, 0.25)                    333.333              166.667
y = 0, 1   3D: (-0.5, 0.25, 0|1)           333.333              166.667
x = 1      (-0.5, 1.0)                     500.000              416.667

eik_min is on x = 0 at (-0.5, 0.0) km (at (-0.5, 0.0, 0.5) km in 3D):
0.25 km / 1.5 km/s = 166.667 ms in the homogeneous model and
0.25 km / 3.0 km/s = 83.333 ms in the heterogeneous one.

Selection of the parameters. The Eikonal is solved as in test_eikonal.py
for the source condition 'nearest_node' and consistent elements with
p = 2. The meshes, edge_length = 1/10 km in 2D and 1/7 km in 3D, have
the same 10 and 7 cells per side as the 100 m and 150 m meshes of
test_eikonal.py. N = 1 / edge_length is the coarsest number of cells for
which the Eikonal of every case is within 3 % of its theoretical value,
the layered hypershape mesh is conforming, and the analytical
approximation stays within the test tolerance for the homogeneous model
with fitting_c = (0, 0, 0, 0), its default. The marked
f_est (*) is the
value with the smallest relative error abs(eik_min - eik_thr) / eik_thr,
where eik_thr is the theoretical value, and is the one used by the test.
In every table eik_min increases with f_est, so the smallest error is at
one of the two listed values around eik_thr. For the heterogeneous model
the marked f_est is that of test_eikonal.py. Every marked f_est
converges, without retries, with Firedrake 2025.4.1 and 2026.10.0. The
expected values are the fnum of the FEM
solvers; for the homogeneous rectangular layers they equal
c / (2 (length_x + 2 pad)).

Every case runs the nine solvers of MODAL_CONFIG. The 3D cases are
marked slow, so the CI runs them with the slow tests. In the
heterogeneous rectangular case, T-3D has 192964 dofs (KMV degree 3, six
tetrahedra per cube) against 71944 for Q-3D (spectral degree 3) on the
same grid of cubes, and the T-3D case takes about 150 s here.

texe and pmem are the runtime and the peak memory traced by Python for
each solver, measured serially on one core with Firedrake 2025.4.1 and
the compilation caches already filled. PETSc allocates its memory outside
Python, so pmem leaves out the MUMPS factorization of ARNOLDI and
LANCZOS, the GAMG preconditioner of LOBPCG and the SLEPc solves of the
KRYLOVSCH methods. With empty compilation caches, as in the CI, the
Q-3D heterogeneous case took 109 s here (setup included), against
about 27 s with filled caches: compiling the forms costs more than the
solvers.

Calibration of fitting_c. The ANALYTICAL solver needs the equivalent
velocity c_eq of a static problem with Neumann conditions on the whole
boundary, loaded at the source. `Modal_Analytical_Solver.c_equivalent`
removes the mean of the load, since such a problem only has a solution
for a zero-mean load, and solves it with linear elements on the mesh of
the modal problem, which makes the ANALYTICAL solver much cheaper than
the FEM solvers (its Notes explain both). A constant velocity needs no
static solve, so the homogeneous model keeps c_eq = c and passes
fitting_c = (0, 0, 0, 0), the default of the analytical solver.

c_eq depends on the static problem, so the heterogeneous fitting_c is
calibrated for it, separately for 2D and 3D. The values fitted to the
zero-mean solve in the space of the modal problem, (-0.9, 0.3, -2.6, -1.4)
in 2D and (-0.8, 0.3, -2.4, -1.7) in 3D, give errors of about -9 % in 2D
and -11 % in 3D with linear elements. fitting_c only enters the
regression of the equivalent velocity, after the static solves, so a
grid with step 0.1 (fc1 and fc2 from -1 to 1, fp1 and fp2 from -5 to 3)
was searched for the smallest maximum relative error
abs(fana - fnum) / fnum over the three heterogeneous cases of each
dimension. fc2 has no effect on these cases: in
c_ref = max(c_eq * f3**fc1, c_eqref * f3**fc2), where f3 is the ratio
of the rectangular to the elliptical frequency factor, the first term is
the larger one, so fc2 keeps its previous value of 0.3.

Dim   fitting_c                 Max error   Max error, one parameter +-0.1
2D    (-0.5, 0.3, -1.5, -0.9)      0.64 %         3.32 %
3D    (-0.9, 0.3, -2.1, -1.5)      0.73 %         3.80 %

Both approaches for the heterogeneous models: the calibrated fitting_c
of this test, used without optimization, and the fitting_c optimized
for each model separately by test_modal_analytical_solver.py, with SLSQP
from (0, 0, 0, 0). fana in Hz and its error against fnum:

Case       Approach    fitting_c                           fana    Error
T-2D N2.0  calibrated  (-0.500,  0.300, -1.500, -0.900)  0.50526  +0.17 %
           optimized   ( 0.277,  0.006,  0.000, -0.551)  0.50473  +0.07 %
T-2D REC   calibrated  (-0.500,  0.300, -1.500, -0.900)  0.45256  -0.62 %
           optimized   ( 0.569,  0.174, -0.640,  0.000)  0.45572  +0.07 %
Q-2D REC   calibrated  (-0.500,  0.300, -1.500, -0.900)  0.45832  +0.64 %
           optimized   ( 0.056,  0.974, -0.887,  0.000)  0.45625  +0.18 %
T-3D N2.4  calibrated  (-0.900,  0.300, -2.100, -1.500)  0.51669  +0.26 %
           optimized   ( 0.156,  0.655, -0.168, -0.988)  0.51555  +0.04 %
T-3D REC   calibrated  (-0.900,  0.300, -2.100, -1.500)  0.42541  -0.05 %
           optimized   ( 0.169,  0.860, -1.032, -0.001)  0.42584  +0.05 %
Q-3D REC   calibrated  (-0.900,  0.300, -2.100, -1.500)  0.41427  +0.73 %
           optimized   ( 0.184,  0.857, -1.044, -0.001)  0.41160  +0.08 %

The homogeneous models pass (0, 0, 0, 0). The optimization keeps
(0, 0, 0, 0) for them, except for T-2D N2.5, where it finds
(0.010, 0.008, -0.022, -0.113) and moves fana from 0.52430 Hz (+2.71 %)
to 0.51057 Hz (+0.02 %). The others stop at the start, where the
gradient of the objective is below the tolerance of SLSQP, and T-3D N6.0
stays at -1.37 % (see test_modal_analytical_solver.py). With Firedrake
2026.10.0 the optimized values of the 2D heterogeneous models are
identical, and those of the 3D ones differ slightly, since the velocity
of the cells with a centroid on x = 0.5 depends on round-off (see
test_modal_analytical_solver.py).

Theoretical Eikonal values. eik_min is the first-arrival time on the
absorbing boundaries (z = -1, x = 0, x = 1 and, in 3D, y = 0 and y = 1).
The source is at (z, x) = (-0.5, 0.25) km ((-0.5, 0.25, 0.5) km in 3D),
so the nearest absorbing boundary is x = 0, 0.25 km away, and the
straight ray to it stays in the medium around the source. Hence
eik_min = 0.25 / 1.5 = 166.667 ms in the homogeneous model (1.5 km/s) and
eik_min = 0.25 / 3.0 = 83.333 ms in the heterogeneous one (3.0 km/s for
x < 0.5).
Every other absorbing boundary is at least 0.5 km away in the same
medium, so its first arrival is at least twice as late.

=======================================================
Natural Frequency for 2D model Δx = 1000/10 m - Ele = T
=======================================================

*EIKONAL HOMOGENEOUS
eik_min = 166.667 ms (Theoretical value)
f_est  eik[ms]
 0.05  163.373
 0.06  166.193*
 0.07  168.555

*RESULTS HOMOGENEOUS
Frequency[Hz]    N2.5      (texe/pmem)     REC      (texe/pmem)
ANALYTICAL    0.52430 (0.217s/1.301MB) 0.46875 (0.147s/0.452MB)
ARNOLDI       0.51046 (0.106s/5.514MB) 0.46875 (0.096s/5.977MB)
LANCZOS       0.51046 (0.041s/5.189MB) 0.46875 (0.048s/5.629MB)
LOBPCG        0.51046 (0.211s/6.655MB) 0.46875 (0.219s/7.207MB)
KRYLOVSCH_CH  0.51046 (0.051s/0.139MB) 0.46875 (0.059s/0.111MB)
KRYLOVSCH_CG  0.51046 (0.061s/0.122MB) 0.46875 (0.059s/0.120MB)
KRYLOVSCH_GH  0.51046 (0.049s/0.138MB) 0.46875 (0.059s/0.097MB)
KRYLOVSCH_GG  0.51046 (0.047s/0.125MB) 0.46875 (0.058s/0.133MB)
RAYLEIGH      0.51084 (0.244s/1.151MB) 0.46875 (0.218s/1.245MB)

*ANALYTICAL
   Case      N2.5*     REC*
fnum[Hz]   0.51046  0.46875
fana[Hz]   0.52430  0.46875
fray[Hz]   0.51084  0.46875

*EIKONAL HETEROGENEOUS
eik_min = 83.333 ms (Theoretical value)
f_est  eik[ms]
 0.05  81.498
 0.06  82.942*
 0.07  84.160

*RESULTS HETEROGENEOUS - fitting_c = (-0.5, 0.3, -1.5, -0.9)
Frequency[Hz]    N2.0       (texe/pmem)     REC       (texe/pmem)
ANALYTICAL    0.50526 (0.546s/ 1.853MB) 0.45256 (0.399s/ 1.662MB)
ARNOLDI       0.50440 (0.088s/ 7.637MB) 0.45539 (0.138s/ 8.438MB)
LANCZOS       0.50440 (0.050s/ 7.585MB) 0.45539 (0.080s/ 8.071MB)
LOBPCG        0.50440 (0.292s/ 9.731MB) 0.45539 (0.229s/10.373MB)
KRYLOVSCH_CH  0.50440 (0.094s/ 0.141MB) 0.45539 (0.073s/ 0.122MB)
KRYLOVSCH_CG  0.50440 (0.085s/ 0.098MB) 0.45539 (0.069s/ 0.132MB)
KRYLOVSCH_GH  0.50440 (0.082s/ 0.131MB) 0.45539 (0.071s/ 0.123MB)
KRYLOVSCH_GG  0.50440 (0.078s/ 0.138MB) 0.45539 (0.080s/ 0.110MB)
RAYLEIGH      0.52768 (0.205s/ 1.123MB) 0.47634 (0.220s/ 1.527MB)

*ANALYTICAL
   Case      N2.0*     REC*
fnum[Hz]   0.50440  0.45539
fana[Hz]   0.50526  0.45256
fray[Hz]   0.52768  0.47634

*RAYLEIGH N2.0
n_eigfunc        2      *4       6       8
freq[Hz]   0.65545 0.52768 0.51695 0.51347
texe[s]      0.123   0.165   0.337   0.613
mem[MB]      0.742   1.044   2.186   3.863

=======================================================
Natural Frequency for 2D model Δx = 1000/10 m - Ele = Q
=======================================================

*EIKONAL HOMOGENEOUS
eik_min = 166.667 ms (Theoretical value)
f_est  eik[ms]
 0.04  164.457
 0.05  167.703*
 0.06  170.399

*RESULTS HOMOGENEOUS
Frequency[Hz]     REC      (texe/pmem)
ANALYTICAL    0.46875 (0.189s/0.490MB)
ARNOLDI       0.46875 (0.127s/6.100MB)
LANCZOS       0.46875 (0.056s/5.643MB)
LOBPCG        0.46875 (0.097s/5.836MB)
KRYLOVSCH_CH  0.46875 (0.071s/0.166MB)
KRYLOVSCH_CG  0.46875 (0.074s/0.128MB)
KRYLOVSCH_GH  0.46875 (0.068s/0.121MB)
KRYLOVSCH_GG  0.46875 (0.054s/0.125MB)
RAYLEIGH      0.46875 (0.258s/1.360MB)

*ANALYTICAL
   Case       REC*
fnum[Hz]   0.46875
fana[Hz]   0.46875
fray[Hz]   0.46875

*EIKONAL HETEROGENEOUS
eik_min = 83.333 ms (Theoretical value)
f_est  eik[ms]
 0.04  82.101
 0.05  83.744*
 0.06  85.118

*RESULTS HETEROGENEOUS - fitting_c = (-0.5, 0.3, -1.5, -0.9)
Frequency[Hz]     REC      (texe/pmem)
ANALYTICAL    0.45832 (0.580s/2.665MB)
ARNOLDI       0.45542 (0.118s/8.571MB)
LANCZOS       0.45542 (0.053s/8.086MB)
LOBPCG        0.45542 (0.092s/8.377MB)
KRYLOVSCH_CH  0.45542 (0.067s/0.143MB)
KRYLOVSCH_CG  0.45542 (0.073s/0.118MB)
KRYLOVSCH_GH  0.45542 (0.068s/0.143MB)
KRYLOVSCH_GG  0.45542 (0.075s/0.129MB)
RAYLEIGH      0.47619 (0.255s/1.495MB)

*ANALYTICAL
   Case       REC*
fnum[Hz]   0.45542
fana[Hz]   0.45832
fray[Hz]   0.47619

======================================================
Natural Frequency for 3D model Δx = 1000/7 m - Ele = T
======================================================

*EIKONAL HOMOGENEOUS
eik_min = 166.667 ms (Theoretical value)
f_est  eik[ms]
 0.04  159.016
 0.05  164.698*
 0.06  170.789

*RESULTS HOMOGENEOUS
Frequency[Hz]    N6.0        (texe/pmem)     REC        (texe/pmem)
ANALYTICAL    0.51628 (0.749s/  4.654MB) 0.47741 (0.667s/  2.424MB)
ARNOLDI       0.52345 (4.440s/193.113MB) 0.47727 (5.777s/232.052MB)
LANCZOS       0.52345 (3.152s/191.810MB) 0.47727 (4.672s/230.583MB)
LOBPCG        0.52345 (4.863s/242.746MB) 0.47727 (5.942s/291.832MB)
KRYLOVSCH_CH  0.52345 (2.960s/  0.110MB) 0.47727 (4.907s/  0.098MB)
KRYLOVSCH_CG  0.52345 (3.189s/  0.119MB) 0.47727 (4.598s/  0.123MB)
KRYLOVSCH_GH  0.52345 (3.212s/  0.135MB) 0.47727 (4.387s/  0.109MB)
KRYLOVSCH_GG  0.52345 (3.626s/  0.128MB) 0.47727 (4.449s/  0.104MB)
RAYLEIGH      0.52678 (2.466s/ 36.157MB) 0.47727 (2.343s/ 43.253MB)

*ANALYTICAL
   Case      N6.0*     REC*
fnum[Hz]   0.52345  0.47727
fana[Hz]   0.51628  0.47741
fray[Hz]   0.52678  0.47727

*EIKONAL HETEROGENEOUS
eik_min = 83.333 ms (Theoretical value)
f_est  eik[ms]
 0.04  79.370
 0.05  82.259*
 0.06  85.344

*RESULTS HETEROGENEOUS - fitting_c = (-0.9, 0.3, -2.1, -1.5)
Frequency[Hz]    N2.4         (texe/pmem)     REC         (texe/pmem)
ANALYTICAL    0.51669 ( 1.199s/  3.656MB) 0.42541 ( 1.854s/  4.636MB)
ARNOLDI       0.51535 ( 7.437s/331.678MB) 0.42562 (24.468s/524.757MB)
LANCZOS       0.51535 ( 5.233s/330.273MB) 0.42562 (14.862s/522.047MB)
LOBPCG        0.51535 (14.008s/417.952MB) 0.42562 (40.075s/660.653MB)
KRYLOVSCH_CH  0.51535 ( 7.231s/  0.119MB) 0.42562 (19.994s/  0.127MB)
KRYLOVSCH_CG  0.51535 ( 7.134s/  0.121MB) 0.42562 (16.964s/  0.122MB)
KRYLOVSCH_GH  0.51535 ( 6.796s/  0.141MB) 0.42562 (12.236s/  0.105MB)
KRYLOVSCH_GG  0.51535 ( 6.687s/  0.123MB) 0.42562 (11.812s/  0.134MB)
RAYLEIGH      0.54073 ( 3.219s/ 60.774MB) 0.44257 ( 4.738s/ 95.369MB)

*ANALYTICAL
   Case      N2.4*     REC*
fnum[Hz]   0.51535  0.42562
fana[Hz]   0.51669  0.42541
fray[Hz]   0.54073  0.44257

*RAYLEIGH N2.4
n_eigfunc        2      *4       6
freq[Hz]   0.61849 0.54073 0.52941
texe[s]      2.279   3.422  13.142
mem[MB]      9.812  60.665 207.066

======================================================
Natural Frequency for 3D model Δx = 1000/7 m - Ele = Q
======================================================

*EIKONAL HOMOGENEOUS
eik_min = 166.667 ms (Theoretical value)
f_est  eik[ms]
 0.07  162.852
 0.08  168.726*
 0.09  174.708

*RESULTS HOMOGENEOUS
Frequency[Hz]     REC        (texe/pmem)
ANALYTICAL    0.47741 (0.693s/  3.335MB)
ARNOLDI       0.47727 (2.868s/174.051MB)
LANCZOS       0.47727 (1.954s/172.121MB)
LOBPCG        0.47727 (0.737s/174.172MB)
KRYLOVSCH_CH  0.47727 (1.727s/  0.131MB)
KRYLOVSCH_CG  0.47727 (1.892s/  0.132MB)
KRYLOVSCH_GH  0.47727 (1.703s/  0.144MB)
KRYLOVSCH_GG  0.47727 (1.820s/  0.119MB)
RAYLEIGH      0.47727 (2.027s/ 19.146MB)

*ANALYTICAL
   Case       REC*
fnum[Hz]   0.47727
fana[Hz]   0.47741
fray[Hz]   0.47727

*EIKONAL HETEROGENEOUS
eik_min = 83.333 ms (Theoretical value)
f_est  eik[ms]
 0.07  81.431
 0.08  84.377*
 0.09  87.376

*RESULTS HETEROGENEOUS - fitting_c = (-0.9, 0.3, -2.1, -1.5)
Frequency[Hz]     REC        (texe/pmem)
ANALYTICAL    0.41427 (1.922s/  8.021MB)
ARNOLDI       0.41127 (7.781s/391.781MB)
LANCZOS       0.41127 (4.209s/389.293MB)
LOBPCG        0.41127 (1.499s/393.876MB)
KRYLOVSCH_CH  0.41127 (5.201s/  0.144MB)
KRYLOVSCH_CG  0.41127 (5.315s/  0.096MB)
KRYLOVSCH_GH  0.41127 (5.159s/  0.144MB)
KRYLOVSCH_GG  0.41127 (5.609s/  0.122MB)
RAYLEIGH      0.42935 (3.331s/ 38.496MB)

*ANALYTICAL
   Case       REC*
fnum[Hz]   0.41127
fana[Hz]   0.41427
fray[Hz]   0.42935

*RAYLEIGH REC
n_eigfunc        2      *4       6
freq[Hz]   0.46713 0.42935 0.42125
texe[s]      2.073   3.150  11.544
mem[MB]      7.528  36.147 123.922

The tables were computed with Firedrake 2025.4.1. With Firedrake
2026.10.0 and the final code, all cases pass (12 passed, 4 skipped, in
141 s). eik_min agrees within 0.02 ms (in ms, 2026.10.0 vs 2025.4.1):

Case     Homogeneous         Heterogeneous
T-2D     166.187  166.193    82.949  82.942
Q-2D     167.703  167.703    83.744  83.744
T-3D     164.686  164.698    82.243  82.259
Q-3D     168.744  168.726    84.375  84.377

In the homogeneous model every solver gives the same frequency with both
versions. In the heterogeneous model the frequencies differ as follows
(in Hz, 2026.10.0 vs 2025.4.1; FEM is the value of all the FEM solvers,
which agree with each other):

Case          FEM              RAYLEIGH
T-2D N2.0     0.50440 0.50440  0.52768 0.52768
T-2D REC      0.45539 0.45539  0.47634 0.47634
Q-2D REC      0.45542 0.45542  0.47619 0.47619
T-3D N2.4     0.52941 0.51535  0.55614 0.54073
T-3D REC      0.43868 0.42562  0.45670 0.44257
Q-3D REC      0.42883 0.41127  0.44576 0.42935

The velocity has one value per cell, evaluated at the cell centroid,
and on the 1/7 km meshes some centroids lie exactly on the interface
x = 0.5. Whether such a cell takes the fast or the slow velocity depends
on round-off, which differs between the versions. In 2D (1/10 km) no
centroid lies on x = 0.5, and the values are identical.

Deviation from the expected value with 2026.10.0 (2025.4.1) and the
tolerance of the test:

Case          FEM (4 % T, 5 % Q)  RAYLEIGH (9 %)
T-2D N2.0     +0.00 % (+0.00 %)   +4.62 % (+4.62 %)
T-2D REC      +0.00 % (+0.00 %)   +4.60 % (+4.60 %)
Q-2D REC      +0.00 % (+0.00 %)   +4.56 % (+4.56 %)
T-3D N2.4     +2.73 % (+0.00 %)   +7.92 % (+4.92 %)
T-3D REC      +3.07 % (+0.00 %)   +7.30 % (+3.98 %)
Q-3D REC      +4.27 % (+0.00 %)   +8.39 % (+4.40 %)

The tolerances cover both versions. The smallest margins are Q-3D REC
with 2026.10.0: 0.73 % for FEM and 0.61 % for RAYLEIGH.

The ANALYTICAL frequencies of the homogeneous models and of the 2D
heterogeneous models are identical with both versions. In 3D the
heterogeneous c_eq changes with the velocity of the cells on x = 0.5, as
FEM does (in Hz, 2026.10.0 vs 2025.4.1, and deviation from the expected
value, tolerance 9 %):

Case          ANALYTICAL       Deviation
T-3D N2.4     0.53261 0.51669  +3.35 % (+0.26 %)
T-3D REC      0.43814 0.42541  +2.94 % (-0.05 %)
Q-3D REC      0.42305 0.41427  +2.86 % (+0.73 %)

Against the FEM frequency of its own version, ANALYTICAL is within
1.35 % with 2026.10.0 and 0.73 % with 2025.4.1. Before the zero-mean
load, the 2D ANALYTICAL frequencies differed by up to 5.8 % between the
versions.

With 2026.10.0, when the 3D cases still ran fewer solvers, those of Q-3D
REC heterogeneous took 24.3 s in total (ARNOLDI 6.0 s, LANCZOS 3.8 s,
LOBPCG 1.2 s), and those of T-3D REC heterogeneous 27.9 s (KRYLOVSCH and
RAYLEIGH). LOBPCG reached its tolerance in every case.
```
