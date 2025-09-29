from spyro.utils import Mask
from spyro.utils import Gradient_mask_for_pml
from spyro.examples.rectangle import Rectangle_acoustic
import firedrake as fire
from random import uniform as rand
import numpy as np


class Interval(object):
    def __init__(self, lower, upper):
        self.lower = lower
        self.upper = upper

    def __contains__(self, item):
        return self.lower <= item <= self.upper


def interval(lower, upper):
    return Interval(lower, upper)


def test_mask():
    dictionary = {}
    dictionary["options"] = {
        "cell_type": "T",
    }
    dictionary["absorving_boundary_conditions"] = {
        "status": False,
        "pad_length": 0.,
        "damping_type": None,
    }
    dictionary["mesh"] = {
        "length_z": 1.0,
        "length_x": 1.0,
        "h": 0.03
    }
    Wave_obj = Rectangle_acoustic(dictionary=dictionary)
    boundaries = {
        "z_min": -0.9,
        "z_max": -0.1,
        "x_min": 0.2,
        "x_max": 0.8,
    }

    # Points we are going to check
    tol = Wave_obj.input_dictionary["mesh"]["h"]
    points_not_masked = [
        # Interior
        (-0.15, 0.25),
        (-0.5, 0.5),
        (-0.85, 0.4),
        (-0.3, 0.6),
        (-0.4, 0.7),
        # Vertices plus tol
        (-0.1-tol, 0.2+tol),
        (-0.9+tol, 0.2+tol),
        (-0.1-tol, 0.8-tol),
        (-0.9+tol, 0.8-tol),
    ]
    points_on_boundary = [
        # Vertices
        (-0.1, 0.2),
        (-0.9, 0.2),
        (-0.1, 0.8),
        (-0.9, 0.8),
        # Edges
        (-0.1, 0.3),
        (-0.9, 0.6),
        (-0.5, 0.2),
        (-0.7, 0.8),
    ]
    points_masked = [
        # Inside pml
        (-0.05, 0.25),
        (-0.95, 0.5),
        (-0.85, 0.1),
        (-0.3, 0.9),
        (-0.02, 0.7),
        # Vertices plus tol
        (-0.1+tol, 0.2-tol),
        (-0.9-tol, 0.2-tol),
        (-0.1+tol, 0.8+tol),
        (-0.9-tol, 0.8+tol),
    ]
    points_in_tolerance = [
        # Vertices
        (-0.1+tol*rand(-1, 1), 0.2+tol*rand(-1, 1)),
        (-0.9+tol*rand(-1, 1), 0.2+tol*rand(-1, 1)),
        (-0.1+tol*rand(-1, 1), 0.8+tol*rand(-1, 1)),
        (-0.9+tol*rand(-1, 1), 0.8+tol*rand(-1, 1)),
        # Edges
        (-0.1+tol*rand(-1, 1), rand(0.2, 0.8)),
        (-0.9+tol*rand(-1, 1), rand(0.2, 0.8)),
        (rand(-0.1, -0.9), 0.2+tol*rand(-1, 1)),
        (rand(-0.1, -0.9), 0.8+tol*rand(-1, 1)),
    ]

    # Testing mask that applies zeros to a function in the objects space
    Mask_not_dg = Mask(boundaries, Wave_obj)
    V = Wave_obj.function_space
    u = fire.Function(V)
    u.interpolate(fire.Constant(10))
    u = Mask_not_dg.apply_mask(u)

    unmasked_results = u.at(points_not_masked)
    boundary_results = u.at(points_on_boundary)
    close_to_boundary_results = u.at(points_in_tolerance)
    masked_results = u.at(points_masked)

    # Checking results close to or in the boundaries
    for result in boundary_results:
        assert result in interval(-2.1, 12.5), f"Value of point failling in boundary: {result}"
    for result in close_to_boundary_results:
        assert result in interval(-2.1, 12.5), f"Value of point failing close to boundary: {result}"

    # Checking results in mask
    for result in masked_results:
        assert np.isclose(result, 0.0), f"Mask not zero: {result}"

    # Checking interior points
    for result in unmasked_results:
        assert np.isclose(result, 10.0), f"Interior is masked: {result}"

    # Testing DG mask for 1 in mask and 0 outside
    Mask_dg = Mask(boundaries, Wave_obj, dg=True)
    dg_func = Mask_dg.dg_mask

    unmasked_results = dg_func.at(points_not_masked)
    boundary_results = dg_func.at(points_on_boundary)
    close_to_boundary_results = dg_func.at(points_in_tolerance)
    masked_results = dg_func.at(points_masked)

    # Checking results close to or in the boundaries
    for result in boundary_results:
        assert result in interval(0 - 1e-5, 1.0 + 1e-5), f"Value of DG point failling in boundary: {result}"
    for result in close_to_boundary_results:
        assert result in interval(0 - 1e-5, 1.0 + 1e-5), f"Value of DG point failling close to boundary: {result}"

    # Checking results in mask
    for result in masked_results:
        assert np.isclose(result, 1.0), f"Value of DG point in mask should be 1 not: {result}"
    # Checking interior points
    for result in unmasked_results:
        assert np.isclose(result, 0.0), f"Value of DG point unmask should be zero not: {result}"

    # Testing DG inverse mask for 0 in mask and 1 outside
    Mask_dg = Mask(boundaries, Wave_obj, dg=True, inverse_mask=True)
    dg_func_inverted = Mask_dg.dg_mask

    unmasked_results = dg_func_inverted.at(points_not_masked)
    boundary_results = dg_func_inverted.at(points_on_boundary)
    close_to_boundary_results = dg_func_inverted.at(points_in_tolerance)
    masked_results = dg_func_inverted.at(points_masked)

    # Checking results close to or in the boundaries
    for result in boundary_results:
        assert result in interval(0, 10), f"Value of inv DG point failling in boundary: {result}"
    for result in close_to_boundary_results:
        assert result in interval(0, 10), f"Value of inv DG point failling close to boundary: {result}"

    # Checking results in mask
    for result in masked_results:
        assert np.isclose(result, 0.0), f"Inverted DG mask not zero, but {result}"

    # Checking interior points
    for result in unmasked_results:
        assert np.isclose(result, 1.0), f"Inverted DG mask interior not 1, but {result}"

    assert True


def test_gradient_mask():
    dictionary = {}
    dictionary["options"] = {
        "cell_type": "T",
    }
    dictionary["absorving_boundary_conditions"] = {
        "status": True,
        "pad_length": 0.2,
    }
    dictionary["mesh"] = {
        "length_z": 1.0,
        "length_x": 1.0,
        "h": 0.03
    }
    Wave_obj = Rectangle_acoustic(dictionary=dictionary)
    # Points we are going to check
    tol = Wave_obj.input_dictionary["mesh"]["h"]
    points_not_masked = [
        # Interior
        (-0.15, 0.25),
        (-0.5, 0.5),
        (-0.85, 0.4),
        (-0.3, 0.6),
        (-0.4, 0.7),
        # Vertices plus tol
        (-0.-tol, 0.+tol),
        (-1.0+tol, 0.+tol),
        (-0.-tol, 1.0-tol),
        (-1.0+tol, 1.0-tol),
    ]
    points_on_boundary = [
        # Vertices
        (-0., 0.),
        (-1., 0.),
        (-0., 1.),
        (-1., 1.),
        # Edges
        (-0., 0.3),
        (-1., 0.6),
        (-0.5, 0.),
        (-0.7, 1.),
    ]
    points_masked = [
        # Inside pml
        (-0.05, -0.1),
        (-1.10, 0.5),
        (-0.95, -0.15),
        (-0.3, 1.12),
        (-1.13, 1.15),
        # Vertices plus tol
        (-0.-tol, 0.-tol),
        (-1.-tol, 0.-tol),
        (-0.-tol, 1.+tol),
        (-1.-tol, 1.+tol),
    ]
    points_in_tolerance = [
        # Vertices
        (-0., 0.+tol*rand(-1, 1)),
        (-1.+tol*rand(-1, 1), 0.+tol*rand(-1, 1)),
        (-0., 1.+tol*rand(-1, 1)),
        (-1.+tol*rand(-1, 1), 1.+tol*rand(-1, 1)),
        # Edges
        (-0., rand(0.2, 0.8)),
        (-1.+tol*rand(-1, 1), rand(0.2, 0.8)),
        (rand(-0.1, -0.9), 0.+tol*rand(-1, 1)),
        (rand(-0.1, -0.9), 1.+tol*rand(-1, 1)),
    ]

    # Testing mask that applies zeros to a function in the objects space
    test1 = True

    Mask_not_dg = Gradient_mask_for_pml(Wave_obj)
    V = Wave_obj.function_space
    u = fire.Function(V)
    u.interpolate(fire.Constant(10))
    u = Mask_not_dg.apply_mask(u)

    unmasked_results = u.at(points_not_masked)
    boundary_results = u.at(points_on_boundary)
    close_to_boundary_results = u.at(points_in_tolerance)
    masked_results = u.at(points_masked)

    # Checking results close to or in the boundaries
    for result in boundary_results:
        if result not in interval(-2.1, 13):
            test1 = False
    for result in close_to_boundary_results:
        if result not in interval(-2.1, 13):
            test1 = False
    if test1 is False:
        print("Boundary going crazy")
        assert False
    # Checking results in mask
    for result in masked_results:
        if np.isclose(result, 0.0) is False:
            test1 = False
    if test1 is False:
        print("Mask not zero")
        assert False
    # Checking interior points
    for result in unmasked_results:
        if np.isclose(result, 10.0) is False:
            test1 = False
    if test1 is False:
        print("Interior is masked")
        assert False

    assert test1


if __name__ == "__main__":
    test_mask()
    test_gradient_mask()
