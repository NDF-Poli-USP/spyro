import numpy as np
from scipy.optimize import curve_fit, minimize
from scipy.special import beta, betainc, gamma
from spyro.utils.stats_tools import coeff_of_determination
from scipy.stats import norm
from sys import float_info


def reg_geometry_hyp(n_fix, cut_plane_percent=1.):

    cut_plane_percent = np.clip(cut_plane_percent, 0., 1.)

    n = np.arange(2., 100., 0.1)
    A_max = 2. * (cut_plane_percent + 1.)

    def area_function(n, cut_plane_percent):

        A = 2. * gamma(1 + 1 / n)**2 / gamma(1 + 2 / n)
        if cut_plane_percent == 1.:
            A *= 2.
        else:
            w = np.maximum(cut_plane_percent ** n, float_info.min)  # w <= 1
            p = 1 / n
            q = 1 + 1 / n
            B_w = beta(p, q) * betainc(p, q, w)  # Non-regularized Beta
            A += (2. / n) * B_w

        return A

    # Pre-compute constants
    fn2 = area_function(2., cut_plane_percent)
    cn2 = A_max - fn2

    # Function values
    f_data = area_function(n, cut_plane_percent)

    # Constraint points
    constraint_points = [(n_cns, area_function(
        n_cns, cut_plane_percent)) for n_cns in [n_fix, 10.]]

    # Define the fit function
    def fit_function(x, a, b):
        '''
        Fit function: A_max - (A_max - fn2) * (1 / (b * x + 1 - 2 * b)) ** a
        '''
        return A_max - cn2 * (1 / (b * x + 1 - 2 * b)) ** a

    def objective_function(params, x, y):
        '''
        Objective function to minimize (sum of squared residuals)
        '''
        a, b = params
        y_pred = fit_function(x, a, b)
        return np.sum((y_pred - y) ** 2)

    def create_constraint_functions(params, constraint_points):
        '''
        Create constraint functions for each constraint point
        '''

        constraints = []
        for x_const, y_const in constraint_points:
            def constraint_func(params, x_const=x_const, y_const=y_const):
                '''
                Constraint function to ensure the fit
                passes through the constraint points
                '''
                a, b = params
                return fit_function(x_const, a, b) - y_const

            constraints.append({'type': 'eq', 'fun': constraint_func})

        return constraints

    # Initial guess
    init_guess = np.array([1/3, 1/3])

    # Create constraints dynamically based on constraint_points
    # constraints = create_constraint_functions(init_guess, constraint_points)

    # Parameter bounds
    # bounds = [(0, None), (0, None)]  # a >= 0, b >= 0
    lower_bnds = [0, 0]
    upper_bnds = [np.inf, np.inf]

    # options = {'gtol': 1e-12,  # gtol: || ∇F(x) || ≤ gtol
    #            'xtol': 1e-14,  # xtol: || delta x || < xtol*|| x ||
    #            'barrier_tol': 1e-14,
    #            'maxiter': 10000,
    #            'initial_tr_radius': 1.0,  # Default is 1.0
    #            'initial_constr_penalty': 1.0,  # Default is 1.0
    #            'initial_barrier_parameter': 0.1,  # Default is 0.1
    #            'initial_barrier_tolerance': 0.1,  # Default is 0.1
    #            'disp': True}

    try:
        # Perform constrained optimization
        # result = minimize(objective_function, init_guess, args=(n, f_data),
        #                   bounds=bounds, constraints=constraints,
        #                   method='trust-constr', options=options)
        # a_fit, b_fit = result.x

        popt, pcov = curve_fit(fit_function, n, f_data, p0=init_guess,
                               bounds=(lower_bnds, upper_bnds), maxfev=10000)
        a_fit, b_fit = popt

        # Calculate goodness of fit metrics
        f_pred = fit_function(n, a_fit, b_fit)
        residuals = f_data - f_pred

        r_squared = coeff_of_determination(f_data, f_pred, 1)  # 1 predictor
        rmse = np.sqrt(np.mean(residuals**2))  # root-mean-square error, RMSE

        print(f"Optimization successful!")
        print(f"Fitted parameters: a = {a_fit:.8f}, b = {b_fit:.8f}")
        print(f"R-squared: {r_squared:.8f}")
        print(f"RMSE: {rmse:.8f}")
        # print(f"Function evaluations: {result.nfev}")
        # print(f"Iterations: {result.nit}")

        # Calculate confidence intervals
        perr = np.sqrt(np.diag(pcov))
        # z_95 = 1.959963984540054  # valor más preciso que 1.96
        # print(f"Fitted parameters: a = {a_fit:.8f} ± {z_95 * perr[0]:.8f}, "
        #       f"b = {b_fit:.8f} ± {z_95 * perr[1]:.8f}")
        ci_a = norm.interval(0.95, loc=a_fit, scale=perr[0])
        ci_b = norm.interval(0.95, loc=b_fit, scale=perr[1])
        delta_a = a_fit - ci_a[0]
        delta_b = b_fit - ci_b[0]
        print(f"Curve fit successful!")
        print(f"Fitted parameters: a = {a_fit:.8f} ± {delta_a:.8f}, "
              f"b = {b_fit:.8f} ± {delta_b:.8f}")

        # Verify constraint satisfaction
        print("Constraint verification:")
        for x_const, y_target in constraint_points:
            y_fitted = fit_function(x_const, a_fit, b_fit)
            error = abs(y_fitted - y_target)
            print(f"x={x_const}: Target={y_target:.8f}, "
                  f"Fitted={y_fitted:.8f}, Error={error:.2e}")
    except Exception as e:
        print(f"Optimization failed: {e}")

    f_req = 0.5 * np.pi * (2. * (1. / 0.5)**2.)**0.5
    f_ell = 2.40482556 / 0.5
    cn2 = f_req - f_ell
    f_hyp = f_req - cn2 * (1 / (b_fit * n_fix + 1 - 2 * b_fit)) ** a_fit

    print(f"\n-+Frequency: Rec: {f_req:.6f}, Ell: {f_ell:.6f}, Hyp: {f_hyp:.6f}")
    print("-" * 80)


# Example usage with different constraint points
if __name__ == "__main__":
    # Example: Different constraint points for a different problem
    # reg_geometry_hyp(3., cut_plane_percent=0.5/1.5)
    for i in range(30, 100):
        reg_geometry_hyp(round(i/10, 1), cut_plane_percent=0.5/1.5)
