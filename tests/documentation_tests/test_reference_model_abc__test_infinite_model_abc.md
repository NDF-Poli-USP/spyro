# Complete documentation of `test_infinite_model_abc`

Source: [`tests/on_one_core/test_reference_model_abc.py`](../on_one_core/test_reference_model_abc.py) (function `test_infinite_model_abc`, line 220).

This is the complete, unabridged docstring. The test file keeps a reduced version that points here.

```text
Testing modal solvers for 2D and 3D case in Fig. 8 of Salas et al (2022).

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
calc_eik : `bool`
    If `True`, eikonal analysis is performed; otherwise, it is skipped.

Returns
-------
None

==============================
Eikonal for 2D model Δx = 250m
==============================
eik_min = 83.333 ms

f_est  T-ele   Q-ele
 0.03 84.586*  --/--
 0.04 87.945   --/--
 0.05  --/--  93.368*
 0.06  --/--  97.456

==============================
Eikonal for 3D model Δx = 500m
==============================
eik_min = 83.333 ms

f_est   T-ele    Q-ele
 0.06   3.618   63.319
 0.07   4.782   78.381*
 0.08   6.509   93.750
 0.09   8.398  109.148
 0.10  10.336  124.374
 0.20  28.403    --/--
 0.30  41.259    --/--
 0.40  48.893    --/--
 0.50  52.831    --/--
 0.60  54.439    --/--
 0.65  54.672    --/--
 0.66  54.685    --/--
 0.67  54.687*   --/--
 0.68  54.680    --/--
 0.69  54.664    --/--
 0.70  54.639    --/--
```
