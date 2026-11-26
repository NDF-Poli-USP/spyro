# Complete documentation of `run_modal`

Source: [`tests/on_one_core/test_modal_analytical_solver.py`](../on_one_core/test_modal_analytical_solver.py) (function `run_modal`, line 246).

This is the complete, unabridged docstring. The test file keeps a reduced version that points here.

```text
Solve the eigenvalue problem for models 2D and 3D.

Parameters
----------
wave : acoustic_wave.AcousticWave
    An instance of the :class:`~spyro.solvers.acoustic_wave.AcousticWave`.
fitting_c : `tuple`
    Parameters for fitting equivalent velocity regression, only reported.
    (0, 0, 0, 0) for the homogeneous model.
    Structure: (fc1, fc2, fp1, fp2):
    - fc1: Magnitude order
    - fc2: Monotonicity
    - fp1: Rectangle frequency
    - fp2: Ellipse frequency
exp_value : `float`
    Expected value for the fundamental frequency
n_root : `int`, optional
    n-th Root selected as the size of the absorbing layer. Default is 1.

Returns
-------
None

Notes
-----
The optimization always starts from fitting_c = (0, 0, 0, 0), so the
test checks that a fit is found without a prior guess. The 'fitting_c'
argument is only reported, and is not the initial guess.

The optimizer evaluates the ANALYTICAL solver once per iteration, for a
new fitting_c. The equivalent velocities c_eq and c_eqref do not depend
on fitting_c, only on the model, and computing them takes two static
solves. They are therefore computed at the first evaluation and reused
for the rest of the optimization of this model; every evaluation still
runs the regression and the frequency of the ANALYTICAL solver.

Choice of SLSQP. The scipy methods were compared on the 12 cases with the
options below (at most 15 iterations or evaluations), serially on one
core with Firedrake 2025.4.1: relative error abs(fana - fexp) / fexp at
the returned fitting_c, evaluations of the ANALYTICAL solver and time of
the optimization, both summed over the cases, and failed cases.

Method       Max error  Mean error  Evaluations    Time   Failed
SLSQP           1.37 %      0.16 %          250  10.8 s        0
L-BFGS-B        0.81 %      0.23 %          205   9.6 s        0
COBYQA          0.90 %      0.30 %          180   9.7 s        0
COBYLA          1.54 %      0.44 %          180   9.1 s        0
TNC             5.57 %      0.70 %          595  14.1 s        1
Nelder-Mead    62.85 %     18.99 %          180   9.3 s        6
Powell         47.19 %     14.76 %          180   9.4 s       10

SLSQP is the most accurate: every heterogeneous case ends within
0.18 %, and its largest error is T-3D N6.0 homogeneous, which stops at
the start (see the Notes of test_modal). L-BFGS-B, COBYQA and COBYLA stop
at their evaluation limit in most cases, with heterogeneous errors up to
0.8 % to 1.5 %. TNC needs the most evaluations and fails Q-3D
heterogeneous, and Nelder-Mead and Powell barely move from (0, 0, 0, 0)
in 15 evaluations. The test checks the frequency of the last evaluation,
which for these methods can differ from the returned fitting_c, so they
fail more cases than their errors alone suggest. The cost differs by
less than 2 s, in a file that takes about 52 s, most of it in building
the meshes.
```
