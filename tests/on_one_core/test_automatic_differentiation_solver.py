from collections import OrderedDict

from spyro.solvers import automatic_differentiation_solver as ad_solver


def test_create_reduced_functional_normalizes_mapping_controls(monkeypatch):
    first = object()
    second = object()
    controls = OrderedDict((("first", first), ("second", second)))
    captured = {}

    monkeypatch.setattr(
        ad_solver.fire_ad,
        "Control",
        lambda control: ("wrapped", control),
    )

    def reduced_functional(functional, wrapped_controls, ensemble, **kwargs):
        captured["functional"] = functional
        captured["controls"] = wrapped_controls
        captured["ensemble"] = ensemble
        captured["kwargs"] = kwargs
        return "reduced-functional"

    monkeypatch.setattr(
        ad_solver.fire_ad,
        "EnsembleReducedFunctional",
        reduced_functional,
    )

    automated_adjoint = ad_solver.AutomatedAdjoint(
        ensemble="ensemble",
        controls=controls,
    )

    result = automated_adjoint.create_reduced_functional("functional")

    assert result == "reduced-functional"
    assert captured["controls"] == [
        ("wrapped", first),
        ("wrapped", second),
    ]
    assert captured["ensemble"] == "ensemble"
    assert captured["kwargs"]["scatter_control"] is True


def test_create_reduced_functional_keeps_single_control_scalar(monkeypatch):
    control = object()
    captured = {}

    monkeypatch.setattr(
        ad_solver.fire_ad,
        "Control",
        lambda value: ("wrapped", value),
    )

    def reduced_functional(functional, wrapped_control, ensemble, **kwargs):
        captured["control"] = wrapped_control
        return "reduced-functional"

    monkeypatch.setattr(
        ad_solver.fire_ad,
        "EnsembleReducedFunctional",
        reduced_functional,
    )

    automated_adjoint = ad_solver.AutomatedAdjoint(
        ensemble="ensemble",
        controls=control,
    )
    automated_adjoint.create_reduced_functional("functional")

    assert captured["control"] == ("wrapped", control)
