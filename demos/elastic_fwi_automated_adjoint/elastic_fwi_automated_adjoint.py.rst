Full-waveform inversion of an isotropic elastic medium with the automated adjoint
==================================================================================

This demo runs a synthetic full-waveform inversion (FWI) of a two-dimensional
isotropic elastic medium with spyro. The gradient of the misfit is computed
via algorithmic differentiation of the forward solver, through
``firedrake.adjoint``. Along the way the demo presents the elastic wave
equation and its material parameters, the true and starting models, the acquisition geometry,
the gradient verification, and the results of the inversion.

The demo is meant to be run with one MPI process per shot, three in this case::

    mpiexec -n 3 python elastic_fwi_automated_adjoint.py

The Python file is extracted from this document with `pylit
<https://pypi.org/project/pylit/>`__, exactly as the Firedrake demos are::

    pylit --code-block-marker ".. code-block:: python" elastic_fwi_automated_adjoint.py.rst

What full-waveform inversion does
---------------------------------

FWI is a local optimisation that looks for the material parameters
:math:`m` of the subsurface [Tarantola1984]_, [Virieux2009]_.
To this end, the FWI process consists of minimising the misfit between an observed
and numerically predicted seismogram data. The misfit is quantified by a functional,
which in general is a summation of the cost functions for multiple wave sources:

.. math::

    J(m) = \sum_{s=1}^{N_s} J_s(m),
    \qquad
    J_s(m) = \frac{1}{2} \sum_{r=1}^{N_r} \int_0^T
    \left| \mathbf{u}_s(m, \mathbf{x}_r, t) - \mathbf{d}_s(\mathbf{x}_r, t) \right|^2 \, dt,
    \quad \quad (1)

where :math:`\mathbf{u}_s` is the numerically simulated displacement for shot
:math:`s = 1, \dots, N_s`, :math:`\mathbf{d}_s` the observed one, and
:math:`\mathbf{x}_r`, :math:`r = 1, \dots, N_r`, the receivers. In an elastic
medium, both :math:`\mathbf{u}_s` and :math:`\mathbf{d}_s` are vectors
with one component per space dimension, and the norm in (1) sums over the
components. spyro integrates (1) in time with the trapezoidal rule, on the
same time grid the wave equation is solved on.

The inversion needs, besides the observed data, a *starting model*: FWI is a
local method, and it converges to the model nearest to the starting point.
In a synthetic experiment like this, we emulate an observed data by considering
a said true model. The observed data is then obtained by running the forward solver
on the true model.

The isotropic elastic wave equation
-----------------------------------

The displacement :math:`\mathbf{u}(\mathbf{x}, t)` of a linear elastic solid
of density :math:`\rho` satisfies the second-order wave equation

.. math::

    \rho \frac{\partial^2 \mathbf{u}}{\partial t^2}
    - \nabla \cdot \boldsymbol{\sigma}(\mathbf{u}) = \mathbf{f}(\mathbf{x}, t),
    \quad \quad (2)

where :math:`\mathbf{f}` is a body force and, for an isotropic medium, the
stress is given by Hooke's law in terms of the two Lamé parameters
:math:`\lambda` and :math:`\mu`,

.. math::

    \boldsymbol{\sigma}(\mathbf{u}) = \lambda \, (\nabla \cdot \mathbf{u}) \, \mathbf{I}
    + 2 \mu \, \boldsymbol{\varepsilon}(\mathbf{u}),
    \qquad
    \boldsymbol{\varepsilon}(\mathbf{u}) = \tfrac{1}{2}
    \left( \nabla \mathbf{u} + \nabla \mathbf{u}^{T} \right).
    \quad \quad (3)

Such a medium carries two kinds of waves, a pressure (P) wave and a
shear (S) wave, whose speeds are

.. math::

    c_p = \sqrt{\frac{\lambda + 2\mu}{\rho}},
    \qquad
    c_s = \sqrt{\frac{\mu}{\rho}},
    \qquad \text{or, inversely,} \qquad
    \mu = \rho \, c_s^2, \quad \lambda = \rho \left( c_p^2 - 2 c_s^2 \right).
    \quad \quad (4)

The medium is therefore described by three fields, the density and the two
wave speeds, and the inversion of this demo is for the two velocity models,
:math:`c_p` and :math:`c_s`.

The source is a point force, :math:`\mathbf{f}(\mathbf{x}, t) = A \, r(t) \,
\delta(\mathbf{x} - \mathbf{x}_s) \, \mathbf{e}`, where :math:`\mathbf{e}` is
a unit vector giving its direction, :math:`A` its amplitude and :math:`r(t)` a
Ricker wavelet [Ricker1953]_ of peak frequency :math:`f`,

.. math::

    r(t) = \left( 1 - 2 \pi^2 f^2 (t - t_0)^2 \right)
    \exp \left( - \pi^2 f^2 (t - t_0)^2 \right),

delayed by :math:`t_0` so that it starts from rest. The medium is at rest at
:math:`t = 0`, :math:`\mathbf{u} = \partial_t \mathbf{u} = \mathbf{0}`.

Discretisation
--------------

spyro solves (2) first writing in the weak form. Multiplying by a test
function :math:`\mathbf{v}` and integrating by parts,

.. math::

    \int_\Omega \rho \, \frac{\partial^2 \mathbf{u}}{\partial t^2} \cdot \mathbf{v} \, dx
    + \int_\Omega \left( \lambda \, (\nabla \cdot \mathbf{u})(\nabla \cdot \mathbf{v})
    + 2 \mu \, \boldsymbol{\varepsilon}(\mathbf{u}) : \boldsymbol{\varepsilon}(\mathbf{v}) \right) dx
    - \int_{\partial \Omega} \left( \boldsymbol{\sigma}(\mathbf{u}) \, \mathbf{n} \right) \cdot \mathbf{v} \, ds
    = \int_\Omega \mathbf{f} \cdot \mathbf{v} \, dx.
    \quad \quad (5)

The boundary term is where the boundary conditions enter. The domain of a
seismic simulation is a window cut out of a much larger medium, and waves
that reach its edges should leave rather than reflect. The demo uses the
local absorbing condition of Stacey [Stacey1988]_, an improvement of the
paraxial condition of Clayton and Engquist [Clayton1977]_: it prescribes on
:math:`\partial \Omega` a traction that damps the normal motion at the speed
of the P wave and the tangential motion at the speed of the S wave,

.. math::

    \boldsymbol{\sigma}(\mathbf{u}) \, \mathbf{n} =
    - \rho \, c_p \left( \frac{\partial \mathbf{u}}{\partial t} \cdot \mathbf{n} \right) \mathbf{n}
    - \rho \, c_s \left( \frac{\partial \mathbf{u}}{\partial t}
    - \left( \frac{\partial \mathbf{u}}{\partial t} \cdot \mathbf{n} \right) \mathbf{n} \right)
    + \ldots,

plus terms in the tangential derivatives of :math:`\mathbf{u}` that improve
the absorption at oblique incidence; the complete expressions are in
``spyro/solvers/elastic_wave/local_abc.py``. The condition is applied on all
four sides, so the domain behaves as a window into an unbounded medium. A
survey carried out at the surface would instead leave the top free
(traction-free, which is the natural condition in (5)), at the price of
surface waves and reflections that make the inversion harder; this demo keeps
to a transmission geometry.

In space, this tutorial uses the spectral element method [Komatitsch1998]_:
quadrilateral elements carrying Lagrange polynomials on the
Gauss–Lobatto–Legendre (GLL) points. In time, the second derivative in
(5) is replaced by the
central difference

.. math::

    \frac{\partial^2 \mathbf{u}}{\partial t^2} \approx
    \frac{\mathbf{u}^{n+1} - 2 \mathbf{u}^n + \mathbf{u}^{n-1}}{\Delta t^2},

with the stiffness and boundary terms evaluated at :math:`\mathbf{u}^n`.

The automated adjoint
---------------------

The gradient of (1) with respect to the material (also called control) parameters
is computed with ``firedrake.adjoint``. The forward solve is recorded on a
*tape* as it runs, and differentiating :math:`J` is a reverse traversal of
that tape, which gives the derivative of the discrete functional with respect
to the discrete parameters for any parameter that enters the variational
forms. The tape is wrapped in a *reduced functional*, :math:`\hat{J}(m)`,
which the optimiser re-evaluates
at new values of :math:`m` to obtain functional values and gradients. The
tape holds the forward states, so its memory grows with the number of time
steps. For largest FWI, e.g., trhee dimensional, long-time, high-frequency problems,
the tape is too large, which means high memory usage. To keep the memory usage under control,
firedrake.adjoint offers checkpointing, which trades memory for computation by recomputing
the forward states between a number of *snapshots* that are kept in memory.
Here, de demo runs with the Single Memory schedule, which keeps all the states used for adjoint
computation in memory and recomputes nothing. For larger problems, the user can choose a
a number of snapshots to keep in memory, and the forward solve is recomputed between them.
firedrake.adjoint uses de python library checkpoint\_schedules [Dolci2024]_,
fell free to check the documentation for more information on the available schedules.


The shots are distributed with Firedrake's *ensemble parallelism*. With
``"parallelism": {"type": "automatic"}`` the MPI processes are split into as
many *ensemble members* as there are shots; each member propagates its own
shot and records its own tape, and the reduced functional -- an
``EnsembleReducedFunctional`` -- sums the per-shot functionals and gradients
across members. Running with ``mpiexec -n 3`` gives one process per shot;
``-n 6`` would additionally split each shot's mesh over two processes. The
number of processes has to be a multiple of the number of shots. For additional details on
the ensemble parallelism used in this demo, please refer to the `Firedrake FWI documentation
<https://www.firedrakeproject.org/demos/full_waveform_inversion.py.html>`__:.

Setting up the problem
----------------------

We begin with the imports. ``ElasticMaterialParameter`` is the enumeration
spyro keys elastic material parameters by, and ``AdjointType`` is how the
inversion driver is told which adjoint to use.

.. code-block:: python

    import numpy as np
    import firedrake as fire
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    import spyro
    from spyro.domains.space import create_function_space
    from spyro.io.basicio import parallel_print
    from spyro.utils.typing import AdjointType

    Parameter = spyro.ElasticMaterialParameter

The material values. spyro works in km, s and km/s, so the density is given
in g/cm³ (:math:`10^3` kg/m³) to match: :math:`\rho c^2` is then in GPa, and
so are the Lamé parameters (4) computes from these values, 6.25 and 3.125 GPa
in the background and 9.0 and 4.5 GPa in the circle.

.. code-block:: python

    rho = 2.0              # density, g/cm^3 (the same everywhere)
    cp_background = 2.5    # P-wave velocity, km/s
    cs_background = 1.25   # S-wave velocity, km/s
    cp_circle = 3.0        # km/s
    cs_circle = 1.5        # km/s

    center_z, center_x = -0.5, 0.5   # centre of the circle, km
    radius = 0.125                   # km

The domain is a 1 km × 1 km square. spyro's first coordinate is the depth
:math:`z`, which is zero at the top and *negative* below it, and the second
is the horizontal position :math:`x`. The mesh is a uniform grid of
quadrilaterals of 100 m (``edge_length``), and the Ricker wavelet has a peak
frequency of 5 Hz.

.. code-block:: python

    length_z = 1.0   # depth of the domain, km
    length_x = 1.0   # width of the domain, km

    edge_length = 0.1     # element size, km
    dt = 0.0016           # time step, s
    maxiter = 20          # optimiser iterations

    final_time = 1.0      # length of the records, s
    frequency = 5.0       # peak frequency of the Ricker wavelet, Hz
    number_of_shots = 3


spyro is configured through a dictionary. The ``options`` choose the
discretisation: quadrilaterals (``"Q"``) with the mass-lumped variant, which
on quadrilaterals is the spectral element method described above, of degree
4, in two dimensions. ``parallelism`` chooses the ensemble parallelism just
described.

.. code-block:: python

    dictionary = {}
    dictionary["options"] = {
        "cell_type": "Q",
        "variant": "lumped",
        "degree": 4,
        "dimension": 2,
    }
    dictionary["parallelism"] = {
        "type": "automatic",
    }
    dictionary["mesh"] = {
        "length_z": length_z,
        "length_x": length_x,
        "length_y": 0.0,
        "mesh_file": None,
        "mesh_type": "firedrake_mesh",
    }

The three sources are at :math:`z = -0.15` km, at :math:`x = 0.2`, 0.5
and 0.8 km; the 41 receivers are on the line :math:`z = -0.85` km, from
:math:`x = 0.1` to 0.9 km.

.. code-block:: python

    dictionary["acquisition"] = {
        "source_type": "ricker",
        "source_locations": spyro.create_transect(
            (-0.15, 0.2), (-0.15, 0.8), number_of_shots,
        ),
        "frequency": frequency,
        "amplitude": np.array([1.0, 0.0]),   # vertical unit force (z, x)
        "receiver_locations": spyro.create_transect(
            (-0.85, 0.1), (-0.85, 0.9), 41,
        ),
    }

The material parameters of the solver are read from ``synthetic_data``: the
density and the two velocities, from which :math:`\lambda` and :math:`\mu`
are computed by (4). The values written here are the *starting model* of the
inversion; the true model is set separately, further down.

.. code-block:: python

    dictionary["synthetic_data"] = {
        "type": "object",
        "density": rho,
        "p_wave_velocity": cp_background,
        "s_wave_velocity": cs_background,
        "real_velocity_file": None,
    }

The absorbing boundary condition, the time axis and the outputs. The
``output_frequency`` only controls how often progress is printed, since the
wavefield output is switched off.

.. code-block:: python

    dictionary["absorving_boundary_conditions"] = {
        "status": True,
        "abc_type": "nrbc",
        "nrbc": {"type": "Stacey", "dt_scheme": "backward"},
    }
    dictionary["time_axis"] = {
        "initial_time": 0.0,
        "final_time": final_time,
        "dt": dt,
        "output_frequency": 100,
        "gradient_sampling_frequency": 1,
    }
    dictionary["visualization"] = {
        "forward_output": False,
        "gradient_output": False,
        "adjoint_output": False,
        "debug_output": False,
    }

The inversion is driven by ``FullWaveformInversion``. By default it builds an
acoustic solver; passing ``wave_class=spyro.IsotropicWave`` makes it build
the isotropic elastic one instead, which the driver keeps on ``fwi.wave``.
Constructing it also creates the ensemble communicator, ``fwi.comm``:
``ensemble_comm`` connects the members (one per shot here) and ``comm`` is
the spatial communicator inside each member.

.. code-block:: python

    fwi = spyro.FullWaveformInversion(
        dictionary=dictionary, wave_class=spyro.IsotropicWave,
    )
    comm = fwi.comm
    parallel_print(
        f"{comm.ensemble_comm.size} ensemble member(s), "
        f"{comm.comm.size} process(es) each", comm,
    )

The true model and the observed data
------------------------------------

The true model is defined on the mesh the observed data are generated on,
which ``set_real_mesh`` builds. The circle is written as a UFL expression in
the mesh coordinates ``fwi.wave.mesh_z`` and ``fwi.wave.mesh_x``, with the
function of the Firedrake demo: :math:`\tanh(200 \, (0.125 - r))`, where
:math:`r` is the distance to the centre in km.

.. code-block:: python

    fwi.set_real_mesh(input_mesh_parameters={"edge_length": edge_length})

    z, x = fwi.wave.mesh_z, fwi.wave.mesh_x
    distance = fire.sqrt((z - center_z) ** 2 + (x - center_x) ** 2)
    inside = 0.5 * (1.0 + fire.tanh(200.0 * (radius - distance)))
    cp_true = cp_background + (cp_circle - cp_background) * inside
    cs_true = cs_background + (cs_circle - cs_background) * inside

``set_real_model`` takes the true value of each parameter, keyed by the
parameter it belongs to.

.. code-block:: python

    fwi.set_real_model({
        Parameter.DENSITY: rho,
        Parameter.P_WAVE_VELOCITY: cp_true,
        Parameter.S_WAVE_VELOCITY: cs_true,
    })

``generate_real_shot_record`` builds a solver of its own for the true model,
interpolates the values above into its material fields, and propagates every
shot. The records are kept on ``fwi.real_shot_record``, each ensemble member
holding the shots it owns, and handed to the inversion's solver, which needs
them to evaluate (1).

.. code-block:: python

    fwi.generate_real_shot_record(save_shot_record=False)

Before moving on, it is worth looking at what has been built, with spyro's
plotting helpers.

.. code-block:: python

    material_space = create_function_space(
        fwi.wave.mesh, fwi.wave.method, fwi.wave.degree, dim=1,
    )
    cp_true_field = fire.Function(material_space).interpolate(cp_true)
    cs_true_field = fire.Function(material_space).interpolate(cs_true)

    sources = dictionary["acquisition"]["source_locations"]
    receivers = dictionary["acquisition"]["receiver_locations"]
    first_member = comm.ensemble_comm.rank == 0

    if first_member:
        spyro.plots.plot_scalar_field(
            [cp_true_field, cs_true_field], "elastic_fwi_true_model.png",
            titles=["true $c_p$", "true $c_s$"],
            vmin=[cp_background, cs_background], vmax=[cp_circle, cs_circle],
            colorbar_label="km/s", cmap="jet",
            sources=sources, receivers=receivers,
        )

``plot_shots`` draws the record of a shot, the displacement at the receivers
as a function of time, one component at a time for an elastic solver. It
reads the record from the solver's ``forward_solution_receivers``, where a
forward solve leaves its own, so the observed record is copied there first.

.. code-block:: python

    fwi.wave.forward_solution_receivers = fwi.real_shot_record
    for component, name in enumerate(("uz", "ux")):
        spyro.plots.plot_shots(
            fwi.wave, filename=f"elastic_fwi_observed_{name}",
            out_index=component, file_format="png",
        )

.. image:: elastic_fwi_true_model.png
    :width: 90 %
    :alt: true P- and S-wave velocity models with sources and receivers
    :align: center

.. image:: elastic_fwi_observed_uz[0].png
    :width: 45 %
    :alt: observed vertical displacement record of the first shot

.. image:: elastic_fwi_observed_ux[0].png
    :width: 45 %
    :alt: observed horizontal displacement record of the first shot

The starting model and the controls
-----------------------------------

The inversion runs on the *guess* mesh, built here with the same element
size as the mesh the data were generated on.

.. code-block:: python

    fwi.set_guess_mesh(input_mesh_parameters={"edge_length": edge_length})

The starting model is a Gaussian anomaly in the place of the circle. It is told
to the driver with ``set_guess_control``, keyed by parameter in the same way as
``set_real_model``; the density is the true one.

.. code-block:: python

    z, x = fwi.wave.mesh_z, fwi.wave.mesh_x
    distance = fire.sqrt((z - center_z) ** 2 + (x - center_x) ** 2)
    gaussian = fire.exp(-(distance / radius) ** 2)
    cp_start = cp_background + 0.5 * (cp_circle - cp_background) * gaussian
    cs_start = cs_background + 0.5 * (cs_circle - cs_background) * gaussian
    fwi.set_guess_control({
        Parameter.P_WAVE_VELOCITY: cp_start,
        Parameter.S_WAVE_VELOCITY: cs_start,
    })
    at_center = fire.PointEvaluator(fwi.wave.mesh, [(center_z, center_x)])
    cp_start_center = float(at_center.evaluate(fwi.wave.c)[0])
    cs_start_center = float(at_center.evaluate(fwi.wave.c_s)[0])

    if first_member:
        spyro.plots.plot_scalar_field(
            [fwi.wave.c, fwi.wave.c_s], "elastic_fwi_starting_model.png",
            titles=["starting $c_p$", "starting $c_s$"],
            vmin=[cp_background, cs_background], vmax=[cp_circle, cs_circle],
            colorbar_label="km/s", cmap="jet",
            sources=sources, receivers=receivers,
        )

.. image:: elastic_fwi_starting_model.png
    :width: 90 %
    :alt: starting P- and S-wave velocity models
    :align: center

The *controls*, the fields the inversion moves, are the two velocities; the
density is kept fixed at its true value. The selection is made when the
automated adjoint is enabled on the solver, and names the parameters the
tape is differentiated with respect to; with no selection, every material
parameter would be taken. The adjoint is enabled with checkpointing, in
its single memory schedule, and with a garbage collection every 50 time
steps, which keeps the memory of the tape from creeping up with reference
cycles.

.. code-block:: python

    control_parameters = {
        Parameter.P_WAVE_VELOCITY,
        Parameter.S_WAVE_VELOCITY,
    }
    fwi.wave.enable_automated_adjoint(
        control_parameters=control_parameters,
        checkpointing=True, gc_timestep_frequency=50,
    )

Gradient verification
---------------------

Before handing the problem to the optimiser, it is good practice to compute
the gradient at the starting model and check it with a Taylor test. The
gradient is differentiated from a recorded forward solve, so one is made
first; ``gradient_solve`` then returns the gradient of :math:`J` with
respect to each control, keyed by the parameter it belongs to:

.. code-block:: python

    fwi.wave.forward_solve()
    gradient = fwi.wave.gradient_solve()
    parallel_print(
        f"Gradient computed for {[parameter.value for parameter in gradient]}",
        comm,
    )

The gradient can be largest at the sources and, less so, at the receivers,
where the wavefields are strongest. What it carries there is the imprint
of the acquisition rather than information on the medium, and it is usual
practice in FWI to correct the gradient around the sources and receivers
[Modrak2016]_; the optimiser below is given a mask that zeroes it there.

The Taylor test checks the gradient against the misfit itself: for a
perturbation :math:`\delta m` of the controls, the residual

.. math::

    \left| J(m + h \, \delta m) - J(m) - h \, \nabla J \cdot \delta m \right|

falls as :math:`h^2` if :math:`\nabla J` is the gradient of :math:`J`, and
only as :math:`h` if it is not. ``verify_gradient`` runs pyadjoint's test
and returns the rate at which the residual falls, which should be close to 2.
The perturbation is random, with a fixed seed so that every ensemble member draws
the same one, and at most 5 % of each velocity.

.. code-block:: python

    controls = fwi.wave.automated_adjoint.controls
    rng = np.random.default_rng(42)
    directions = [
        fire.Function(
            control.function_space(),
            val=0.05 * control.dat.data_ro * rng.random(control.dat.data_ro.shape),
        )
        for control in controls
    ]
    rate = fwi.wave.automated_adjoint.verify_gradient(
        controls, direction=directions, dJdm=gradient,
    )
    parallel_print(
        f"Taylor test: the residual falls at rate {rate:.3f} (2 expected)",
        comm,
    )

Running the inversion
---------------------

Each control gets bounds of its own, since a P-wave velocity and an S-wave
velocity are not bounded by the same numbers. The bounds keep the optimiser
in the physical region: with the pairs allowed below, :math:`c_p / c_s \geq
2.3 / 1.6 > \sqrt{2}`, so :math:`\lambda = \rho (c_p^2 - 2 c_s^2)` in (4)
stays positive whatever the optimiser does. The explicit time stepping is
only conditionally stable, and the speed that matters for it is the largest
one the optimiser may produce, not that of the starting model: the time step
was checked to be stable at :math:`c_p = 3.5` km/s. The bounds are
given in the order the controls are held in: :math:`c_p` before
:math:`c_s`.

.. code-block:: python

    cp_bounds = (2.3, 3.5)   # km/s
    cs_bounds = (1.0, 1.6)   # km/s

The gradient mask is a field equal to one where the model may change and
zero where it may not: here, within 150 m of the line of sources and 100 m
of the line of receivers. The optimiser multiplies every gradient it is
given by it, so the model is never updated there.

.. code-block:: python

    gradient_mask = fire.conditional(fire.And(z < -0.3, z > -0.75), 1.0, 0.0)

``run_fwi`` with ``adjoint_type=AdjointType.AUTOMATED_ADJOINT`` runs the
forward solve of the starting model, evaluates :math:`J` there, and hands
the problem to ``spyro.tools.optimization.LumpedTAOSolver``, spyro's driver
for PETSc/TAO's bound-constrained limited-memory quasi-Newton method, BLMVM
[Benson2001]_. TAO then asks for :math:`J` and its gradient at each trial
model, applies the mask, projects its steps onto the box defined by the
bounds, and stops after ``maxiter`` iterations. Its convergence tolerances
are left at their defaults, so the run ends on the iteration budget, which
is how FWI is normally run; the driver warns about that and returns the
last iterate. (Had no gradient been wanted beforehand, ``run_fwi`` could
also have enabled the adjoint itself, through
``adjoint_options={"control_parameters": control_parameters}``.)

.. code-block:: python

    controls = fwi.run_fwi(
        adjoint_type=AdjointType.AUTOMATED_ADJOINT,
        vmin=[cp_bounds[0], cs_bounds[0]],
        vmax=[cp_bounds[1], cs_bounds[1]],
        maxiter=maxiter,
        gradient_mask=gradient_mask,
    )

Each accepted iterate is written to ``control_<iteration>.pvd`` as it is
produced, the final one to ``control_end.pvd``, and the functional values to
``functional_values.txt``; ParaView opens the ``.pvd`` files. The result is
returned as one ``Function`` per control, in the order of the bounds, and the
functional history is kept on the driver, starting from the value at the
starting model.

.. code-block:: python

    cp_result, cs_result = controls
    parallel_print(
        f"Controls: {[control.name() for control in controls]}", comm,
    )
    parallel_print(f"Functional history: {fwi.functional_history}", comm)

Results
-------

The recovered velocities at the centre of the circle, where the starting
model is farthest from the truth, measure how far the inversion got:

.. code-block:: python

    cp_center = float(at_center.evaluate(cp_result)[0])
    cs_center = float(at_center.evaluate(cs_result)[0])
    parallel_print(
        f"At the centre of the circle: c_p = {cp_center:.3f} km/s "
        f"(started at {cp_start_center:.3f}, true {cp_circle}), "
        f"c_s = {cs_center:.3f} km/s "
        f"(started at {cs_start_center:.3f}, true {cs_circle})", comm,
    )

The inverted models are drawn next to the true and the starting ones, one
row per velocity with a common colour scale, and the decrease of the
functional with a plain matplotlib line, by the first process.

.. code-block:: python

    if first_member:
        space = cp_result.function_space()
        cp_start_field = fire.Function(space).interpolate(cp_start)
        cs_start_field = fire.Function(space).interpolate(cs_start)
        spyro.plots.plot_scalar_field(
            [cp_true_field, cp_start_field, cp_result,
             cs_true_field, cs_start_field, cs_result],
            "elastic_fwi_models.png", columns=3,
            titles=["true $c_p$", "starting $c_p$", "inverted $c_p$",
                    "true $c_s$", "starting $c_s$", "inverted $c_s$"],
            vmin=[cp_background] * 3 + [cs_background] * 3,
            vmax=[cp_circle] * 3 + [cs_circle] * 3,
            colorbar_label="km/s", cmap="jet",
        )

    if first_member and comm.comm.rank == 0:
        figure, axes = plt.subplots(figsize=(5.5, 4))
        axes.semilogy(fwi.functional_history, "o-")
        axes.set_xlabel("iteration")
        axes.set_ylabel("$J$ (km$^2$ s)")
        axes.grid(True, which="both", alpha=0.3)
        figure.tight_layout()
        figure.savefig("elastic_fwi_functional_history.png", dpi=120)

.. image:: elastic_fwi_models.png
    :width: 100 %
    :alt: true, starting and inverted P- and S-wave velocity models
    :align: center

.. image:: elastic_fwi_functional_history.png
    :width: 50 %
    :alt: misfit functional against iteration
    :align: center


.. admonition:: Exercise

    Run the inversion in two stages with the same budget of 20 iterations:
    ten moving :math:`c_s` alone, then ten moving :math:`c_p` alone with
    :math:`c_s` held where the first stage left it. Compare the functional
    history and the recovered :math:`c_p` with the joint inversion above.

    *Hint.* ``run_fwi`` takes ``stages`` in place of ``maxiter``, a list of
    ``(parameters, iterations)`` pairs::

        stages=[
            (Parameter.S_WAVE_VELOCITY, 10),
            (Parameter.P_WAVE_VELOCITY, 10),
        ],

    The tape is recorded once, with both controls; each stage is a TAO run
    that holds the other control by zeroing its gradient, as the mask holds
    the model around the sources and receivers, and starts from where the
    previous stage stopped. The iteration count and the functional history
    run through the stages. Repeat the pairs, ``[(S, 5), (P, 5)] * 2``, to
    alternate. Expect the second stage to bring :math:`J` below what the
    joint inversion reaches in 20 iterations and :math:`c_p` at the centre
    of the circle to about 2.9 km/s, with :math:`c_s` as the first stage
    left it.

.. note::

    This tutorial is a simplified, toy example: it leaves out the
    complexities of a realistic FWI, such as field data with noise, a
    free surface, source and receiver imprints to be masked, regularisation,
    multi-scale strategies and the size of real models.

.. rubric:: References

.. [Benson2001] Benson, S. J., & Moré, J. J. (2001). A limited memory
    variable metric method in subspaces and bound constrained optimization
    problems. Technical Report ANL/MCS-P909-0901, Argonne National
    Laboratory.

.. [Clayton1977] Clayton, R., & Engquist, B. (1977). Absorbing boundary
    conditions for acoustic and elastic wave equations. Bulletin of the
    Seismological Society of America, 67(6), 1529–1540.

.. [Dolci2024] Dolci, D. I., Maddison, J. R., Ham, D. A., Pallez, G., &
    Herrmann, J. (2024). checkpoint_schedules: schedules for incremental
    checkpointing of adjoint simulations. Journal of Open Source Software,
    9(95), 6148.

.. [Komatitsch1998] Komatitsch, D., & Vilotte, J.-P. (1998). The spectral
    element method: an efficient tool to simulate the seismic response of 2D
    and 3D geological structures. Bulletin of the Seismological Society of
    America, 88(2), 368–392.

.. [Modrak2016] Modrak, R., & Tromp, J. (2016). Seismic waveform inversion
    best practices: regional, global and exploration test cases.
    Geophysical Journal International, 206(3), 1864–1889.

.. [Munson2012] Munson, T., Sarich, J., Wild, S., Benson, S., & McInnes, L.
    C. (2012). TAO 2.0 users manual. Technical Memorandum ANL/MCS-TM-322,
    Argonne National Laboratory.

.. [Ricker1953] Ricker, N. (1953). The form and laws of propagation of
    seismic wavelets. Geophysics, 18(1), 10–40.

.. [Shearer2009] Shearer, P. M. (2009). Introduction to Seismology (2nd
    ed.). Cambridge University Press.

.. [Stacey1988] Stacey, R. (1988). Improved transparent boundary
    formulations for the elastic-wave equation. Bulletin of the Seismological
    Society of America, 78(6), 2089–2097.

.. [Tarantola1984] Tarantola, A. (1984). Inversion of seismic reflection
    data in the acoustic approximation. Geophysics, 49(8), 1259–1266.

.. [Tarantola1986] Tarantola, A. (1986). A strategy for nonlinear elastic
    inversion of seismic reflection data. Geophysics, 51(10), 1893–1903.

.. [Virieux2009] Virieux, J., & Operto, S. (2009). An overview of
    full-waveform inversion in exploration geophysics. Geophysics, 74(6),
    WCC1–WCC26.
