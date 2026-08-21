"""HABC-related xCR plotting routines."""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

from .plot_helpers import _finalize_figure


def plot_xCR_regression(
    wave,
    regression_data,
    output_folder: str | Path = "output/",
    show: bool = False,
):
    """Plot the regression curves used to determine the optimal xCR.

    Creates a plot showing quadratic regressions of the integral and peak
    errors as functions of the heuristic factor xCR. The optimal xCR is
    highlighted together with the corresponding errors. The plot also
    displays the regression equations and their coefficients of
    determination (R²).

    Parameters
    ----------
        - xCR_values : `array-like`
            Values of xCR used for the regression. The last value is the
            optimal xCR.
        - integral_errors : `array-like`
            Maximum integral error associated with each xCR value. The last
            value corresponds to the optimal xCR.
        - peak_errors : `array-like`
            Maximum peak error associated with each xCR value. The last value
            corresponds to the optimal xCR.
        - optimization_criterion : `str`
            Criterion used to determine the optimal xCR. Supported values are:

            - ``"err_difference"`` : Minimizes the difference between the
              integral and peak errors.
            - ``"err_integral"`` : Minimizes the integral error.
            - ``"err_sum"`` : Minimizes the sum of the integral and peak
              errors.

    output_folder : `str` or `pathlib.Path`, optional
        Directory where the generated figures are saved. The directory is
        created if it does not exist. Default is ``"output/"``.

    show : `bool`, optional
        Whether to display the plot interactively. Default is ``False``.

    Returns
    -------
    None

    Raises
    ------
    TypeError
        If ``output_folder`` is not a string or ``pathlib.Path``, or if
        ``show`` is not a boolean.
    ValueError
        If ``regression_data`` does not contain four elements, if the
        regression arrays have incompatible sizes, if fewer than three
        regression points are provided, or if an unsupported optimization
        criterion is specified.
    """
    if not isinstance(output_folder, (str, Path)):
        raise TypeError("output_folder must be a str or pathlib.Path.")

    if not isinstance(show, bool):
        raise TypeError("show must be a bool.")

    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    curve_spacing = 0.1
    polynomial_degree = 2

    try:
        (
            xCR_values,
            integral_errors,
            peak_errors,
            optimization_criterion,
        ) = regression_data
    except (TypeError, ValueError) as error:
        raise ValueError(
            "regression_data must contain four elements: "
            "(xCR_values, integral_errors, peak_errors, "
            "optimization_criterion)."
        ) from error

    xCR_values = np.asarray(xCR_values, dtype=float)
    integral_errors = np.asarray(integral_errors, dtype=float)
    peak_errors = np.asarray(peak_errors, dtype=float)

    if not (xCR_values.ndim == integral_errors.ndim == peak_errors.ndim == 1):
        raise ValueError("Regression arrays must be one-dimensional.")

    if not (len(xCR_values) == len(integral_errors) == len(peak_errors)):
        raise ValueError(
            "xCR_values, integral_errors, and peak_errors must have " "the same length."
        )

    if len(xCR_values) < polynomial_degree + 2:
        raise ValueError(
            "At least three regression points and one optimal point " "are required."
        )

    valid_criteria = {
        "err_difference",
        "err_integral",
        "err_sum",
    }

    if optimization_criterion not in valid_criteria:
        raise ValueError(
            f"Unsupported optimization criterion "
            f"{optimization_criterion!r}. Expected one of "
            f"{sorted(valid_criteria)}."
        )

    # The final entry contains the selected optimal xCR and its errors.
    regression_xCR_values = xCR_values[:-1]
    regression_integral_errors = integral_errors[:-1]
    regression_peak_errors = peak_errors[:-1]

    xCR_opt = xCR_values[-1]
    integral_error_opt = integral_errors[-1]
    peak_error_opt = peak_errors[-1]

    # Fit quadratic regression models to the sampled data.
    integral_regression_coefficients = np.polyfit(
        regression_xCR_values,
        regression_integral_errors,
        polynomial_degree,
    )
    peak_regression_coefficients = np.polyfit(
        regression_xCR_values,
        regression_peak_errors,
        polynomial_degree,
    )

    # Evaluate the regression models at the sampled xCR values.
    predicted_integral_errors = np.polyval(
        integral_regression_coefficients,
        regression_xCR_values,
    )
    predicted_peak_errors = np.polyval(
        peak_regression_coefficients,
        regression_xCR_values,
    )

    # Compute coefficients of determination.
    integral_r2 = coeff_of_determination(
        regression_integral_errors,
        predicted_integral_errors,
        polynomial_degree,
    )
    peak_r2 = coeff_of_determination(
        regression_peak_errors,
        predicted_peak_errors,
        polynomial_degree,
    )

    # Format the regression equations for the legend.
    equation_template = r"${:.3e} x^{{2}} + {:.3e} x + {:.3e}, " r"R^{{2}} = {:.3f}$"

    integral_equation = (
        r"$e_I = $"
        + equation_template.format(
            *integral_regression_coefficients,
            integral_r2,
        )
    ).replace("+ -", "- ")

    peak_equation = (
        r"$e_P = $"
        + equation_template.format(
            *peak_regression_coefficients,
            peak_r2,
        )
    ).replace("+ -", "- ")

    # Generate smooth regression curves within the configured xCR bounds.
    xCR_lower_bound, xCR_upper_bound = wave.xCR_lim

    curve_point_count = max(
        2,
        int((xCR_upper_bound - xCR_lower_bound) / curve_spacing) + 1,
    )

    curve_xCR_values = np.linspace(
        xCR_lower_bound,
        xCR_upper_bound,
        curve_point_count,
    )

    curve_integral_errors = np.polyval(
        integral_regression_coefficients,
        curve_xCR_values,
    )
    curve_peak_errors = np.polyval(
        peak_regression_coefficients,
        curve_xCR_values,
    )

    # Evaluate the regression models at the selected optimal xCR.
    predicted_integral_error_opt = np.polyval(
        integral_regression_coefficients,
        xCR_opt,
    )
    predicted_peak_error_opt = np.polyval(
        peak_regression_coefficients,
        xCR_opt,
    )

    fig, ax = plt.subplots()

    # Plot the regression data points.
    ax.plot(
        regression_xCR_values,
        100 * regression_integral_errors,
        "o",
        label=f"Integral Error: {integral_equation}",
    )
    ax.plot(
        regression_xCR_values,
        100 * regression_peak_errors,
        "o",
        label=f"Peak Error: {peak_equation}",
    )

    # Plot the regression curves.
    ax.plot(
        curve_xCR_values,
        100 * curve_integral_errors,
        linestyle="--",
    )
    ax.plot(
        curve_xCR_values,
        100 * curve_peak_errors,
        linestyle="--",
    )

    # Highlight the optimal xCR.
    ax.plot(
        [xCR_opt, xCR_opt],
        [0.0, 100 * integral_error_opt],
        linestyle="-",
    )

    if np.isclose(
        predicted_integral_error_opt,
        predicted_peak_error_opt,
    ):
        optimal_label = (
            r"Optimized Heuristic Factor: "
            r"$X^{*}_{C_{R}} = {:.3f}$ | "
            r"$e_{{I}} = e_{{P}} = {:.2f}\%$"
        ).format(
            xCR_opt,
            100 * integral_error_opt,
        )
    else:
        optimal_label = (
            r"Optimized Heuristic Factor: "
            r"$X^{*}_{C_{R}} = {:.3f}$ | "
            r"$e_{{I}} = {:.2f}\%$ | "
            r"$e_{{P}} = {:.2f}\%$"
        ).format(
            xCR_opt,
            100 * integral_error_opt,
            100 * peak_error_opt,
        )

    ax.plot(
        xCR_opt,
        100 * integral_error_opt,
        marker="*",
        markersize=10,
        label=optimal_label,
    )

    criterion_labels = {
        "err_difference": r" (Criterion: Min $(e_I - e_P)$)",
        "err_integral": r" (Criterion: Min $e_I$)",
        "err_sum": r" (Criterion: Min $(e_I + e_P)$)",
    }

    ax.set_xlabel(r"$X_{C_{R}}$" + criterion_labels[optimization_criterion])
    ax.set_ylabel(r"$e_I \; | \; e_P \; (\%)$")

    ax.legend(loc="best", fontsize=8.5)

    maximum_error = max(
        np.max(regression_integral_errors),
        np.max(regression_peak_errors),
    )

    ax.set_xlim(
        0,
        round(xCR_upper_bound, 1) + 0.1,
    )
    ax.set_ylim(
        0,
        round(100 * maximum_error, 1) + 0.1,
    )

    fig.tight_layout(pad=2)

    _finalize_figure(
        fig,
        filename=output_folder / "xCR",
        formats=("png", "pdf"),
        show=show,
        bbox_inches="tight",
    )
