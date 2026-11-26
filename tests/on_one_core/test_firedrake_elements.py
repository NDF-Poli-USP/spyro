"""Reference elements of Firedrake used by spyro.

The degree 3 KMV tetrahedron of the T-3D cases of test_reference_model_abc and
the spectral elements of its Q cases are checked on their reference cells, so
that a change of an element between Firedrake versions is reported by the
element itself and not only by the time step or the signals of a simulation.
"""

import itertools

import FIAT
import firedrake as fire
import numpy as np
import pytest
from finat.quadrature import make_quadrature
from numpy import sum
from spyro.domains.quadrature import quadrature_rules
from spyro.domains.space import create_function_space
from spyro.mpi.spyro_mpi import SpyroEnsemble
from spyro.version_control import is_firedrake_new

# Expected failure of the T-3D cases with newer Firedrake: the degree 3 KMV
# tetrahedron of firedrake-fiat 2026.10.0 has another basis than that of
# 2025.4.0 (see the Notes of test_infinite_model_abc in
# test_reference_model_abc, which also applies this mark). The 'ANALYTICAL'
# case becomes unstable and the 'ARNOLDI' case selects another time step, both
# raising AssertionError. strict=True reports the cases as soon as they pass.
KMV3_TETRAHEDRON_XFAIL = pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="the degree 3 KMV tetrahedron of newer Firedrake has another, "
    "non-symmetric basis, which makes the stable time step of T-3D about 11 "
    "times smaller",
)

# Stiffness diagonal of the degree 3 KMV tetrahedron on the UFC reference
# element, computed with its "KMV" quadrature, in the order of
# element.dual_basis(): 4 vertex, 12 edge, 12 face and 4 interior nodes (see
# test_kmv3_tetrahedron_reference_stiffness). The 2026.10.0 values were printed
# with four decimals.
KMV3_STIFFNESS_DIAGONAL = {
    "2025.4.1": (
        "0.08853607706 0.03741682612 0.03741682612 0.03741682612 "
        "0.3758670933 0.3758670933 0.3758670933 0.3758670933 0.3758670933 "
        "0.3758670933 0.4487776041 0.4396933483 0.4396933483 0.4487776041 "
        "0.4396933483 0.4487776041 2.660955652 2.660955652 2.660955652 "
        "1.689657416 1.689657416 1.667340012 1.689657416 1.689657416 "
        "1.667340012 1.689657416 1.689657416 1.667340012 5.980986394 "
        "5.980986394 5.980986394 6.202625231"
    ),
    "2026.10.0": (
        "1.9126 1.4815 2.0302 3.9271 14.4662 34.3064 34.3064 14.4662 13.3298 "
        "26.3461 54.6138 22.9601 15.1043 27.0778 8.9991 10.4341 14.3867 47.6151 "
        "14.3867 28.7872 103.6042 27.427 28.7872 103.6042 27.427 28.7872 "
        "103.6042 27.427 5.981 5.981 5.981 6.2026"
    ),
}
# Absolute tolerance of each recorded diagonal
KMV3_STIFFNESS_ATOL = {"2025.4.1": 1e-8, "2026.10.0": 1e-4}

# Stiffness diagonal of the spectral (Gauss-Lobatto-Legendre) elements of the Q
# cases on one unit cell, which is their reference cell: the degree 4
# quadrilateral (Q-2D) and the degree 3 quadrilateral x interval element of the
# extruded meshes (Q-3D), with spyro's quadrature. The dofs are grouped by their
# sorted distances to the faces of the cell, which a symmetry of the cell
# preserves (see test_spectral_element_reference_stiffness). Firedrake 2025.4.1.
SPECTRAL_STIFFNESS_DIAGONAL = {
    2: {
        (0.0, 0.0): 0.7,
        (0.0, 0.172673): 2.54074074074,
        (0.0, 0.5): 2.96296296296,
        (0.172673, 0.172673): 6.91646090535,
        (0.172673, 0.5): 7.09794238683,
        (0.5, 0.5): 6.74238683128,
    },
    3: {
        (0.0, 0.0, 0.0): 0.0902777777778,
        (0.0, 0.0, 0.276393): 0.358796296296,
        (0.0, 0.276393, 0.276393): 1.33101851852,
        (0.276393, 0.276393, 0.276393): 4.34027777778,
    },
}


def test_kmv3_tetrahedron_reference_stiffness(request: pytest.FixtureRequest) -> None:
    """Check the basis of the degree 3 KMV tetrahedron on the reference element.

    The T-3D cases of test_infinite_model_abc use this element. With Firedrake
    2026.10.0 their stable time step is about 11 times smaller than with
    2025.4.1 (see the Notes of test_infinite_model_abc in
    test_reference_model_abc). This test isolates the cause with FIAT and
    FInAT only: it builds the element and its "KMV" quadrature on the UFC
    reference tetrahedron, and the mass and stiffness matrices with that
    quadrature. Run it with ``pytest -s`` to print the stiffness diagonal node
    by node, next to the values recorded with 2025.4.1 and 2026.10.0
    (KMV3_STIFFNESS_DIAGONAL), so that each version shows how it differs from
    the other.

    Parameters
    ----------
    request : `pytest.FixtureRequest`
        Fixture that marks the test as an expected failure with newer Firedrake
        (KMV3_TETRAHEDRON_XFAIL).

    Returns
    -------
    None

    Notes
    -----
    Both versions agree on the 32 nodes (4 at the vertices, 2 on each edge, 3
    on each face, 4 interior), on the quadrature, whose points are the nodes,
    on the diagonal mass matrix, whose entries are the quadrature weights, and
    on the nodal basis (identity at the nodes). The test checks these first.
    The stiffness diagonal differs: it sums to 52.39 with 2025.4.1 and to
    865.75 with 2026.10.0, and only the four interior nodes keep their values.
    A permutation of the axes maps the reference tetrahedron onto itself and
    permutes vertices 1, 2 and 3, so their stiffness entries must be equal:
    0.0374 each with 2025.4.1, but 1.4815, 2.0302 and 3.9271 with 2026.10.0.
    The test also tabulates the basis at 35 points of a lattice inside the
    cell, between the nodes, and checks two properties that hold for any
    correct degree 3 KMV tetrahedron, without recorded values: the basis
    interpolates every polynomial of degree 3 or less exactly (both versions
    pass), and a swap P of two axes, which maps the cell and the nodes onto
    themselves, satisfies phi_i(P x) = phi_P(i)(x) (2026.10.0 fails, see
    below). Then it requires the symmetry of the stiffness diagonal of
    vertices 1, 2 and 3 and the 2025.4.1 values, so it fails with 2026.10.0,
    first at the symmetry of the basis, and with ``strict=True`` it is
    reported as soon as it passes there.

    Results. With 2025.4.1 the test passes (6 s); the current diagonal matches
    the 2025.4.1 values to 3.3e-10 and differs from the 2026.10.0 values at the
    28 vertex, edge and face nodes, by up to 101.9. With 2026.10.0 it is an
    expected failure (xfailed, 5.6 s); the current diagonal matches the
    2026.10.0 values to 4.8e-05, within their four-decimal rounding, and
    differs from the 2025.4.1 values at the same 28 nodes, by up to 101.9. The
    checks shared by both versions pass with both: the cubics are reproduced
    to 1.1e-15 (2025.4.1) and 1.0e-15 (2026.10.0). The symmetry of the basis
    holds to 1.0e-15 with 2025.4.1; with 2026.10.0 it fails by 2.48 for each
    swap (vertex 0.31 to 0.50, edge 1.34 to 1.44, face 2.48, interior 5.6e-16),
    where the test stops as an expected failure (8.7 s for both element tests
    here, 18.1 s there).

    Degree. The element is of degree 3: it contains every polynomial of degree
    3 or less. To have positive quadrature weights at its nodes (mass
    lumping), it adds bubble functions up to degree 5, which vanish on the
    boundary of the cell, so FIAT reports ``degree() == 5``. Its 32 nodes are
    the 20 of the cubic Lagrange element (4 vertex, 12 edge, 4 face) and 12
    more for the bubbles (8 face and 4 interior).

    Comparison of the two versions (diag_kmv3 files of each version, with the
    full matrices and the basis at 35 points of a lattice inside the cell).
    The nodes, the quadrature and the mass matrix are identical. Both
    stiffness matrices are symmetric with zero row sums, but the 2026.10.0 one
    is much larger (largest entry 103.6 against 6.2), and every block except
    the interior-interior one changes. In both versions the basis sums to one
    and interpolates every polynomial of degree 3 or less exactly (to 1e-15),
    so both spaces contain the cubics. Between the nodes, the vertex, edge and
    face basis functions differ by up to 0.33, 1.17 and 2.36, and the interior
    ones are identical. A nodal basis is unique for a given space and given
    nodes, so the 2026.10.0 space itself differs, in its bubble enrichment.
    It is also not symmetric: a swap P of two axes maps the cell and the
    nodes onto themselves, so phi_i(P x) = phi_P(i)(x) must hold; it holds to
    1e-15 with 2025.4.1, and fails by 2.48 with 2026.10.0 (vertex, edge and
    face functions; the interior ones keep it).
    """
    if is_firedrake_new():
        request.applymarker(KMV3_TETRAHEDRON_XFAIL)

    cell = FIAT.ufc_simplex(3)
    element = FIAT.KongMulderVeldhuizen(cell, 3)
    rule = make_quadrature(cell, 3, "KMV")
    points = np.asarray(rule.point_set.points)
    weights = np.asarray(rule.weights)
    nodes = np.array([list(node.get_point_dict())[0] for node in element.dual_basis()])

    tab = element.tabulate(1, points)
    phi = tab[(0, 0, 0)]
    grads = np.array([tab[key] for key in ((1, 0, 0), (0, 1, 0), (0, 0, 1))])
    mass = np.einsum("ip,p,jp->ij", phi, weights, phi)
    stiffness = np.einsum("dip,p,djp->ij", grads, weights, grads)
    diagonal = np.diag(stiffness)

    # Current stiffness diagonal against the recorded versions
    recorded = {
        version: np.array(values.split(), dtype=float)
        for version, values in KMV3_STIFFNESS_DIAGONAL.items()
    }
    entity = [""] * element.space_dimension()
    names = {0: "vertex", 1: "edge", 2: "face", 3: "interior"}
    for dim, dofs in element.entity_dofs().items():
        for ids in dofs.values():
            for i in ids:
                entity[i] = names[dim]
    SpyroEnsemble.print(
        "\nStiffness diagonal of the degree 3 KMV tetrahedron\n"
        + f"{'Node':>4} {'Entity':>8} {'Current':>10}"
        + "".join(f" {version:>10}" for version in recorded)
    )
    for i, value in enumerate(diagonal):
        SpyroEnsemble.print(
            f"{i:>4} {entity[i]:>8} {value:>10.4f}"
            + "".join(f" {values[i]:>10.4f}" for values in recorded.values())
        )
    matches = []
    for version, values in recorded.items():
        diff = np.abs(diagonal - values)
        SpyroEnsemble.print(
            f"Max |current - {version}|: {diff.max():.3e}, at nodes "
            f"{np.flatnonzero(diff > KMV3_STIFFNESS_ATOL[version]).tolist()}"
        )
        if np.allclose(diagonal, values, rtol=0.0, atol=KMV3_STIFFNESS_ATOL[version]):
            matches.append(version)
    SpyroEnsemble.print(f"Current stiffness diagonal matches: {matches or 'none'}")

    # Basis at 35 points of a lattice inside the cell, between the nodes
    lattice = np.asarray(cell.make_points(3, 0, 8))
    phi_lattice = element.tabulate(0, lattice)[(0, 0, 0)]

    # Interpolation of every polynomial of degree 3 or less from its nodal values
    exponents = [e for e in itertools.product(range(4), repeat=3) if sum(e) <= 3]
    cubic_error = max(
        np.abs(
            np.prod(nodes ** np.array(e), axis=1) @ phi_lattice
            - np.prod(lattice ** np.array(e), axis=1)
        ).max()
        for e in exponents
    )
    SpyroEnsemble.print(
        f"Interpolation error of the {len(exponents)} monomials of degree 3 or less "
        f"at the lattice: {cubic_error:.2e}"
    )

    # Symmetry of the basis: a swap P of two axes maps the cell and the nodes
    # onto themselves, so phi_i(P x) = phi_P(i)(x)
    entity = np.array(entity)
    symmetry_error = 0.0
    for swap in ((1, 0, 2), (0, 2, 1), (2, 1, 0)):
        perm = [
            np.flatnonzero(np.all(np.isclose(nodes, node[list(swap)]), axis=1))[0]
            for node in nodes
        ]
        phi_swapped = element.tabulate(0, lattice[:, list(swap)])[(0, 0, 0)]
        diff = np.abs(phi_swapped - phi_lattice[perm]).max(axis=1)
        symmetry_error = max(symmetry_error, diff.max())
        SpyroEnsemble.print(
            f"Swap {swap}: max |phi_i(P x) - phi_P(i)(x)| {diff.max():.2e}, by node: "
            + ", ".join(
                f"{name} {diff[entity == name].max():.2e}" for name in names.values()
            )
        )

    # Same in both versions: nodes, quadrature, lumped mass and nodal basis
    assert element.space_dimension() == len(nodes) == len(weights) == 32
    dist = np.linalg.norm(nodes[:, None, :] - points[None, :, :], axis=2)
    assert np.all(dist.min(axis=1) < 1e-12), "✗ Quadrature points are not the nodes"
    assert np.allclose(mass, np.diag(np.diag(mass)), atol=1e-14), "✗ Mass not lumped"
    assert np.allclose(np.sort(np.diag(mass)), np.sort(weights)), "✗ Mass weights"
    assert np.allclose(element.tabulate(0, nodes)[(0, 0, 0)], np.eye(len(nodes)))
    assert (
        cubic_error < 1e-12
    ), f"✗ The basis does not reproduce the cubics: error {cubic_error:.2e}"

    # Differs with 2026.10.0: symmetry of the basis, and symmetry and values of
    # the stiffness diagonal
    assert symmetry_error < 1e-12, (
        "✗ The basis is not symmetric under the swaps of two axes, which map the "
        f"cell and the nodes onto themselves: error {symmetry_error:.2e}"
    )
    assert np.allclose(diagonal[1:4], diagonal[1], rtol=1e-10), (
        "✗ Stiffness entries of vertices 1, 2 and 3 differ, although a "
        f"permutation of the axes exchanges them: {diagonal[1:4]}"
    )
    reference = recorded["2025.4.1"]
    assert np.allclose(diagonal, reference, rtol=0.0, atol=1e-8), (
        "✗ Stiffness diagonal differs from Firedrake 2025.4.1 by up to "
        f"{np.abs(diagonal - reference).max():.3e} (sum {diagonal.sum():.4f} "
        f"against {reference.sum():.4f}); it matches: {matches or 'none'}"
    )


@pytest.mark.parametrize("dimension", [2, 3])
def test_spectral_element_reference_stiffness(dimension: int) -> None:
    """Check the spectral elements of the Q cases on the reference cell.

    This is the counterpart of test_kmv3_tetrahedron_reference_stiffness for
    the elements of the Q cases of test_infinite_model_abc, whose time steps
    are the same with Firedrake 2025.4.1 and 2026.10.0: the degree 4
    quadrilateral (Q-2D) and the degree 3 quadrilateral x interval element of
    the extruded meshes (Q-3D), both spectral (Gauss-Lobatto-Legendre). They
    are built with spyro's ``create_function_space`` and ``quadrature_rules``
    on one unit cell, which is their reference cell, and the mass and
    stiffness matrices are assembled with that quadrature. Run it with
    ``pytest -s`` to print the stiffness diagonal by group of dofs next to the
    values recorded with 2025.4.1 (SPECTRAL_STIFFNESS_DIAGONAL).

    Parameters
    ----------
    dimension : `int`
        Dimension of the problem. 2 for 2D and 3 for 3D.

    Returns
    -------
    None

    Notes
    -----
    The dofs are grouped by their distances to the faces of the cell, sorted,
    so the groups do not depend on the dof numbering: in 3D, the 8 vertex
    dofs, the 24 dofs on the edges, the 24 on the faces and the 8 interior
    ones. A permutation of the axes or a reflection maps the cell onto itself
    and keeps these distances, so the stiffness diagonal must be the same in
    each group; the degree 3 KMV tetrahedron of Firedrake 2026.10.0 breaks the
    corresponding symmetry. The whole matrices must also be symmetric in that
    sense: a swap of two axes or a reflection x -> 1 - x maps the cell onto
    itself and permutes the dofs, which must leave the mass and stiffness
    matrices unchanged. The space must reproduce every monomial of degree 4 (2D)
    or 3 (3D) or less in each variable: their nodal interpolants, all in one
    vector function, must have zero L2 error over the cell. The test also
    requires a diagonal mass matrix and the recorded values, which are rational
    numbers (0.7, 80/27, ... in 2D; 13/144, 155/432, 575/432 and 625/144 in
    3D).

    Results. Both cases pass with Firedrake 2025.4.1 (4.9 s) and 2026.10.0
    (4.6 s), with the same value in every group and the same values in both
    versions, so these elements did not change, unlike the degree 3 KMV
    tetrahedron. With both versions the matrices change by at most 5.3e-15
    under the symmetries of the cell, and the 25 (2D) and 64 (3D) monomials
    are reproduced to 9.3e-16 and 1.1e-15.
    """
    degree = 4 if dimension == 2 else 3  # as in wave_dict
    base = fire.UnitSquareMesh(1, 1, quadrilateral=True)
    mesh = base if dimension == 2 else fire.ExtrudedMesh(base, 1, layer_height=1.0)
    V = create_function_space(mesh, "spectral_quadrilateral", degree)
    quad_rule = quadrature_rules(V)[0]
    u, v = fire.TrialFunction(V), fire.TestFunction(V)
    mass = fire.assemble(u * v * fire.dx(**quad_rule)).petscmat
    stiffness = fire.assemble(
        fire.inner(fire.grad(u), fire.grad(v)) * fire.dx(**quad_rule)
    ).petscmat
    coords = fire.Function(fire.VectorFunctionSpace(mesh, V.ufl_element()))
    coords.interpolate(fire.SpatialCoordinate(mesh))

    # Stiffness diagonal by group of dofs with the same distances to the faces
    groups = {}
    for x, value in zip(coords.dat.data_ro, stiffness.getDiagonal().getArray()):
        key = tuple(sorted(float(round(min(c, 1.0 - c), 6)) for c in x))
        groups.setdefault(key, []).append(value)
    recorded = SPECTRAL_STIFFNESS_DIAGONAL[dimension]
    SpyroEnsemble.print(
        f"\nStiffness diagonal of {V.ufl_element()}\n"
        + f"{'Distances to the faces':>26} {'Dofs':>5} {'Min':>12} {'Max':>12} "
        + f"{'2025.4.1':>12}"
    )
    for key in sorted(groups):
        values = np.array(groups[key])
        SpyroEnsemble.print(
            f"{str(key):>26} {len(values):>5} {values.min():>12.8f} "
            f"{values.max():>12.8f} {recorded.get(key, np.nan):>12.8f}"
        )

    # Dense copies; the converted matrices must stay alive while they are read
    dense = {}
    for name, matrix in (("mass", mass), ("stiffness", stiffness)):
        converted = matrix.convert("dense")
        dense[name] = converted.getDenseArray().copy()

    # Symmetry of the matrices: a swap of two axes or a reflection x -> 1 - x
    # maps the cell onto itself and permutes the dofs, which must leave the
    # mass and stiffness matrices unchanged
    x_dofs = coords.dat.data_ro
    index = {tuple(np.round(x, 10)): i for i, x in enumerate(x_dofs)}
    transforms = [
        ("swap", axes) for axes in itertools.combinations(range(dimension), 2)
    ]
    transforms += [("reflection", (axis,)) for axis in range(dimension)]
    symmetry_error = 0.0
    for kind, axes in transforms:
        y_dofs = x_dofs.copy()
        if kind == "swap":
            y_dofs[:, list(axes)] = y_dofs[:, list(axes[::-1])]
        else:
            y_dofs[:, axes[0]] = 1.0 - y_dofs[:, axes[0]]
        perm = [index[tuple(np.round(y, 10))] for y in y_dofs]
        errors = {
            name: np.abs(matrix[np.ix_(perm, perm)] - matrix).max()
            for name, matrix in dense.items()
        }
        symmetry_error = max(symmetry_error, *errors.values())
        SpyroEnsemble.print(
            f"{kind} {axes}: change of the mass {errors['mass']:.2e}, "
            f"of the stiffness {errors['stiffness']:.2e}"
        )

    # Nodal interpolation of every monomial up to the degree in each variable,
    # all in one vector function, and its L2 error over the cell
    exponents = list(itertools.product(range(degree + 1), repeat=dimension))
    xs = fire.SpatialCoordinate(mesh)
    exact = fire.as_vector(
        [np.prod([xs[k] ** e[k] for k in range(dimension)]) for e in exponents]
    )
    W = fire.VectorFunctionSpace(mesh, V.ufl_element(), dim=len(exponents))
    interpolant = fire.Function(W)
    interpolant.dat.data[:] = np.array(
        [np.prod(x_dofs ** np.array(e), axis=1) for e in exponents]
    ).T
    poly_error = np.sqrt(
        fire.assemble(fire.inner(interpolant - exact, interpolant - exact) * fire.dx)
    )
    SpyroEnsemble.print(
        f"L2 error of the nodal interpolants of the {len(exponents)} monomials of "
        f"degree {degree} or less in each variable: {poly_error:.2e}"
    )

    # Lumped mass, symmetric matrices, polynomials and the same groups as recorded
    off_diagonal = dense["mass"] - np.diag(np.diag(dense["mass"]))
    assert np.abs(off_diagonal).max() < 1e-12, "✗ Mass matrix not diagonal"
    assert (
        symmetry_error < 1e-12
    ), f"✗ Mass or stiffness changes under a symmetry of the cell: {symmetry_error:.2e}"
    assert (
        poly_error < 1e-12
    ), f"✗ The space does not reproduce the monomials: error {poly_error:.2e}"
    assert sorted(groups) == sorted(recorded), "✗ Groups of dofs differ"

    # Symmetry within each group and the recorded values
    for key, values in groups.items():
        assert np.allclose(values, values[0], rtol=1e-10), (
            f"✗ Stiffness entries of the dofs {key} differ, although symmetries "
            f"of the cell exchange them: {values}"
        )
        assert np.isclose(np.mean(values), recorded[key], rtol=1e-10), (
            f"✗ Stiffness entry of the dofs {key} is {np.mean(values)}, "
            f"recorded {recorded[key]} with Firedrake 2025.4.1"
        )
