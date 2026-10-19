import numpy as np
import pytest

gmsh = pytest.importorskip("gmsh")

from spyro.meshing.meshing_gmsh2d import _classify_rectangular_curves  # noqa: E402
from spyro.meshing.meshing_gmsh3d import (  # noqa: E402
    configure_gmsh_3d_meshing_mode,
    padding_grading_factor,
)


@pytest.fixture
def gmsh_session():
    """Yield an initialized Gmsh module with an empty model.

    Yields
    ------
    module
        The initialized Gmsh Python module, finalized after the test.
    """
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("helpers")
    try:
        yield gmsh
    finally:
        gmsh.finalize()


def test_classify_rectangular_curves(gmsh_session):
    """Check that rectangle boundary curves are sorted by side and others dropped.

    Parameters
    ----------
    gmsh_session : module
        Initialized Gmsh module with an empty model.
    """
    length_x, depth_z = 5.0, -3.0
    rectangle = gmsh_session.model.occ.addRectangle(0, 0, 0, length_x, depth_z)
    # A curve away from the box sides must be dropped.
    p0 = gmsh_session.model.occ.addPoint(1.0, -1.0, 0.0)
    p1 = gmsh_session.model.occ.addPoint(2.0, -2.0, 0.0)
    interior = gmsh_session.model.occ.addLine(p0, p1)
    gmsh_session.model.occ.synchronize()

    curves = [
        tag
        for _, tag in gmsh_session.model.getBoundary([(2, rectangle)], oriented=False)
    ]
    top, left, right, bottom = _classify_rectangular_curves(
        gmsh_session, curves + [interior], length_x, depth_z
    )

    def bbox(tags):
        """Return the bounding box of a single classified curve.

        Parameters
        ----------
        tags : list of int
            Classified curve tags; exactly one is expected.

        Returns
        -------
        tuple of float
            ``(xmin, ymin, xmax, ymax)`` of that curve.
        """
        assert len(tags) == 1
        xmin, ymin, _, xmax, ymax, _ = gmsh_session.model.getBoundingBox(1, tags[0])
        return xmin, ymin, xmax, ymax

    # Gmsh pads bounding boxes by about 1e-7.
    tol = 1.0e-6
    assert bbox(top)[1] == pytest.approx(0.0, abs=tol)
    assert bbox(top)[3] == pytest.approx(0.0, abs=tol)
    assert bbox(left)[0] == pytest.approx(0.0, abs=tol)
    assert bbox(left)[2] == pytest.approx(0.0, abs=tol)
    assert bbox(right)[0] == pytest.approx(length_x, abs=tol)
    assert bbox(bottom)[1] == pytest.approx(depth_z, abs=tol)
    assert interior not in top + left + right + bottom


@pytest.mark.parametrize("structured_mesh", [False, True])
def test_configure_gmsh_3d_meshing_mode(gmsh_session, structured_mesh):
    """Check the Gmsh options set for structured and unstructured 3-D meshing.

    Parameters
    ----------
    gmsh_session : module
        Initialized Gmsh module with an empty model.
    structured_mesh : bool
        Whether the structured meshing mode is requested.
    """
    gmsh_session.model.occ.addBox(0, 0, 0, 1, 1, 1)
    gmsh_session.model.occ.synchronize()

    configure_gmsh_3d_meshing_mode(
        gmsh=gmsh_session, structured_mesh=structured_mesh, min_element_size=0.25
    )

    assert gmsh_session.option.getNumber("Mesh.Algorithm3D") == 10
    assert gmsh_session.option.getNumber("Mesh.Optimize") == 1
    assert gmsh_session.option.getNumber("Mesh.OptimizeNetgen") == 1
    assert gmsh_session.option.getNumber("Mesh.OptimizeThreshold") == 0.5
    if structured_mesh:
        assert gmsh_session.option.getNumber("Mesh.MeshSizeMin") == 0.25
        assert gmsh_session.option.getNumber("Mesh.MeshSizeMax") == 0.25
    else:
        assert gmsh_session.option.getNumber("Mesh.MeshSizeMax") != 0.25


@pytest.mark.parametrize(
    "padding_type, expected",
    [
        ("rectangular", 0.5),
        (None, 0.5),
        ("hyperelliptical", (0.5**2 + 0.25**2 + 0.0**2) ** 0.5),
    ],
)
def test_padding_grading_factor_scalar(padding_type, expected):
    """Check the scalar padding grading factor for each padding type.

    Parameters
    ----------
    padding_type : str or None
        Padding geometry passed to `padding_grading_factor`.
    expected : float
        Expected grading factor.
    """
    t = padding_grading_factor(50.0, 50.0, 0.0, 100.0, 200.0, 300.0, padding_type, 2.0)

    assert float(t) == pytest.approx(expected)


def test_padding_grading_factor_array_clamps_and_ignores_zero_padding():
    """Check that array grading factors are clamped to one and zero padding ignored."""
    distances = np.array([0.0, 50.0, 100.0, 400.0])

    t = padding_grading_factor(
        distances, distances, distances, 100.0, 0.0, 0.0, "rectangular", 3.0
    )

    np.testing.assert_array_equal(t, [0.0, 0.5, 1.0, 1.0])
