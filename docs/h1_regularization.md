# Physical H1 regularization

The implementation extends `dolci/proximal_opt`. `spyro.functionals` owns
receiver misfits, spatial penalties and their composition. Coordinate
wrappers live in `spyro.functionals.reduced`; the old
`spyro.reduced_functionals` imports remain compatible.

```python
from spyro import ElasticMaterialParameter as Parameter
from spyro.functionals import H1Regularization, InversionObjective
from spyro.utils.typing import AdjointType

objective = InversionObjective(
    regularization=H1Regularization(
        weights={
            Parameter.P_WAVE_VELOCITY: 1e-6,
            Parameter.S_WAVE_VELOCITY: 1e-6,
        },
    ),
)
result = fwi.run_fwi(
    adjoint_type=AdjointType.AUTOMATED_ADJOINT,
    objective=objective,
    maxiter=20,
)
```

The example weights illustrate the API, not recommended universal values.
For acoustic inversion, use the acoustic material-parameter enum instead.
Existing bounds and latent options keep their existing meanings.
The updated base removed proximal subproblems; this branch does not restore them.

## Mathematical meaning

The objective is `J_data(m) + R(m)`, with

`R(m) = sum_p alpha_p/2 * integral(|grad((m_p - reference_p)/scale_p)|^2 dx)`.

References default to zero, scales to one. Reference Functions are copied
without annotation at construction and stay fixed. This is the squared H1 **seminorm**, not the full
H1 norm. It is evaluated on physical fields, even for latent optimization.
It is not the proximal term or its moving anchor. The regularization uses
ordinary finite-element stiffness integration, not a lumped mass penalty.

The data misfit integrates squared scalar or vector receiver residuals over
time by the trapezoidal rule. VOM assembly reduces over the spatial
communicator. Each ensemble member records `R / ensemble_size`, so summing
the local reduced objectives includes the regularizer exactly once.

## Scope and validation

The initial implementation supports continuous scalar independent physical
controls and the automated adjoint. Discontinuous controls (including DG0)
are rejected rather than silently producing a zero cellwise penalty.
The hand-implemented adjoint does not yet include this regularization and
is explicitly rejected when an objective is requested through `run_fwi`.
The integration currently requires one propagation per ensemble member;
sequential shot accumulation is explicitly rejected. A propagation can
contain simultaneous sources using the existing acquisition settings.

The penalty is assembled while the forward tape is still recording; no
separate wave solve or per-timestep penalty evaluation is needed. Complete
and partial reduced functionals use the same recorded objective. Existing
checkpoint schedules remain responsible for wavefield storage.

Tests cover analytic values, parameter scales and references, temporal
integration, zero weights, coordinate-wrapper Taylor tests, and backward
compatible imports. Integration and MPI tests cover the composed objective.

## Contribution disclosure

Implementation and tests were assisted by OpenAI Codex. Human review and
submission of any upstream pull request are required by this repository.
