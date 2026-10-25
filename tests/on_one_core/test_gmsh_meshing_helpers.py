import pytest

gmsh = pytest.importorskip("gmsh")

from spyro.meshing.meshing_gmsh2d import _classify_rectangular_curves  # noqa: E402


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
            `(xmin, ymin, xmax, ymax)` of that curve.
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
