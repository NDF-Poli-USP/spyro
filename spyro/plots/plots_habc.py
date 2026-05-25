# This file contains methods for plotting results from the HABC scheme
from matplotlib.pyplot import (close, figure, gca, grid, legend, plot, rcParams, savefig,
                               scatter, setp, subplots, tight_layout
                               xlabel, xlim, xticks, ylabel, ylim)
from matplotlib.pyplot import show as plt_show
from numpy import arange, asarray, ceil, clip, linspace, inf, polyfit, polyval, zeros
from os import makedirs, path
from spyro.habc.lay_len import f_layer, loop_roots
from spyro.utils.stats_tools import coeff_of_determination
rcParams.update({"font.family": "serif"})
rcParams['text.latex.preamble'] = r'\usepackage{bm} \usepackage{amsmath}'


def create_folder(folder):
    """Verify if a folder exists, if not, it creates the folder.

    Parameters
    ----------
    folder: `str`
        Path to the folder to be created

    Returns
    -------
    None
    """

    # Create the folder if it does not exist
    if not path.isdir(folder):
        makedirs(folder)


def plot_function_layer_size(lay_par, freq_par, geom_par, FLpos,
                             output_folder="output/", show=False):
    """Plot the function of the layer size criterion for the HABC scheme.

    Parameters
    ----------
    lay_par: `list`
        Parameters of the layer:
        - a : `float`
            Adimensional propagation speed parameter (a = z / f, z = c / l)
        - z_par : `float`
            Inverse of min. Eikonal (1 / phi_min, equivalent to c_bound/lref)
    freq_par: `list`
        Parameters of the frequency:
        - reference_frequency : `float`
            Reference frequency of the wave
        - source_frequency : `float`
            Source frequency
    geom_par: `list`
        Parameters of the domain geometry:
        - lmin : `float`
            Minimal dimension of finite element in mesh
        - lref : `float`
            Reference length for the size of the absorbing layer
    FLpos: `list`
        List of size parameters for the reference frequency
    output_folder: `str`, optional
        Folder to save the output plots. Default is "output/".
    show: `bool`, optional
        Whether to show the plot. Default is False.

    Returns
    -------
    None
    """

    # Create the output folder if it does not exist
    create_folder(output_folder)

    # Unpack the parameters
    a, z = lay_par
    reference_frequency, source_frequency = freq_par
    lmin, lref = geom_par

    # Prepare the data for the plot
    a_lst = [a]
    F_lst = [FLpos]
    l_lst = ["{:.2f}".format(reference_frequency)]
    c_lst = ['C0']

    if source_frequency == reference_frequency:

        # Layer size based on source frequency
        FLsou = []
        w_lst = ['f_{{sou}}']

    else:

        # Calculate the size parameter for the source frequency
        a_sou = z / source_frequency   # Adimensional parameter
        FLsou = loop_roots(a_sou, lmin, lref, len(FLpos), show_ig=False)
        a_lst.append(a_sou)
        F_lst.append(FLsou)
        l_lst.append("{:.2f}".format(source_frequency))
        c_lst.append('C1')
        w_lst = ['f_{{bnd}}', 'f_{{sou}}']

    # Calculate the maximum layer size for the plot
    FL_max = max(FLpos + FLsou) + 0.4
    FL_lim = ceil(FL_max * 10) / 10
    F_L = linspace(0.001, FL_lim, int(FL_lim * 1e3))

    # Plot the size criterion
    figure(figsize=(12, 6))  # Set figure size
    ax = gca()
    lim_crit = inf
    for a_pr, FL_rt, lab, col, w_str in zip(a_lst, F_lst, l_lst, c_lst, w_lst):
        crit = f_layer(F_L, a_pr)
        lim_crit = min(lim_crit, crit.min())
        plot(F_L, crit, color=col, zorder=2,
             label=r'$\Psi_{{F_L}}({}={}\text{{Hz}})$'.format(w_str, lab))
        scatter(FL_rt, zeros(len(FL_rt)), color=col, zorder=3)

    # Identify the roots of the criterion function
    delta_x = FL_lim / 40.
    delta_y = abs(lim_crit) / 2.
    off_x = 0.5 * delta_x
    off_y = 0.85 * delta_y
    for lay, (FL_rt, col) in enumerate(zip(F_lst, c_lst)):
        base_y = -1.3 * delta_y if lay == 0 else 0.8 * delta_y
        used_positions = []

        for rt, FL_par in enumerate(FL_rt):
            xFL = FL_par + delta_x if rt % 2 == 0 else FL_par - delta_x
            y_FL = base_y

            # Check for overlap and adjust if needed
            for prev_x, prev_y in used_positions:
                if abs(xFL - prev_x) < 2.6 * delta_x and abs(y_FL - prev_y) < 0.9 * off_y:
                    xFL += -off_x if rt % 2 == 0 else off_x
                    y_FL += -off_y if lay == 0 else off_y
            used_positions.append((xFL, y_FL))

            ax.annotate(
                f"{FL_par:.4f}",  # Text
                xy=(FL_par, 0),  # Point to connect to
                xytext=(xFL, y_FL),  # Text position
                ha='center', va='bottom', zorder=4, bbox=dict(facecolor=col, alpha=0.9),
                arrowprops=dict(arrowstyle='-', color='black', linewidth=0.8,
                                alpha=0.9, connectionstyle="arc3,rad=0."))

    # Formatting the plot
    FL_str = r'$F_L \; (L_{{\xi}} \; = \; L_{{ref}} \, F_L \;$'
    lref_str = r'$\therefore \; L_{{ref}} \; = \; {:.4f}\text{{km}})$'
    xlabel((FL_str + lref_str).format(lref))
    ylabel(r'$\Psi_{{F_L}} \; = \; |C_{Rmin}| \; - \; R$')
    xticks(arange(0, FL_lim + 0.01, 0.5 if FL_lim > 1 else 0.2))
    xlim((0, FL_lim))
    ylim((lim_crit - 0.01, 1.01))
    grid(zorder=1)
    legend()

    # Saving the plot
    layer_str = output_folder + "layer_opts"
    savefig(layer_str + ".png", bbox_inches='tight')
    savefig(layer_str + ".pdf", bbox_inches='tight')
    plt_show() if show else None
    close()


def plot_hist_receivers(wave_object, show=False):
    """Plot time-domain receiver response comparison.

    Creates a multi-panel plot comparing the time-domain response at each
    receiver between the computed solution and a reference solution. Each
    receiver is plotted in its own subplot with the computed solution in
    green and the reference solution in red dashed line.

    Parameters
    ----------
    Wave_object : `wave`
        The Wave object containing the simulation results. Must have the
        following attributes:
        - forward_solution_receivers: Computed receiver data
        - receivers_reference: Reference receiver data
        - dt: Time step
        - final_time: Final simulation time
        - number_of_receivers: Number of receivers
        - path_save: Directory path for saving plots
        - case_abc: Case name for file naming
    show : `bool`, optional
        Whether to display the plot interactively. Default is False.

    Returns
    -------
    None

    Notes
    -----
    The function saves two files:
    - {path_save}/{case_abc}/time.png
    - {path_save}/{case_abc}/time.pdf

    The green solid line represents the computed transient solution,
    while the red dashed line represents the reference transient solution.
    """

    print("\nPlotting Time Comparison", flush=True)

    # Time data
    dt = wave_object.dt
    tf = wave_object.final_time
    nt = int(round(tf / dt)) + 1  # number of timesteps
    t_rec = linspace(0., tf, nt)

    # Setting fonts
    rcParams['font.size'] = 7

    # Setting subplots
    num_recvs = wave_object.number_of_receivers
    rcParams['axes.grid'] = True
    fig, axes = subplots(nrows=num_recvs, ncols=1)
    fig.subplots_adjust(hspace=0.6)

    # Setting colormap
    cl_rc = (0., 1., 0., 1.)  # RGB-alpha (Green)
    cl_rf = (1., 0., 0., 1.)  # RGB-alpha (Red)

    for rec in range(num_recvs):

        # Plot the receiver data
        rc_dat = wave_object.forward_solution_receivers[:, rec]
        rf_dat = wave_object.receivers_reference[:, rec]
        axes[rec].plot(t_rec, rc_dat, color=cl_rc, linestyle='-', linewidth=2)
        axes[rec].plot(t_rec, rf_dat, color=cl_rf, linestyle='--', linewidth=2)

        # Adding the receiver number label
        axes[rec].text(0.995, 0.9, "R" + str(rec + 1), fontsize=8.5,
                       transform=axes[rec].transAxes, fontweight='bold',
                       verticalalignment='top', horizontalalignment='right',
                       bbox=dict(facecolor='none', edgecolor='none'))

        # Centered title
        if rec == num_recvs // 2:
            axes[rec].set_ylabel(r'$sol \; recs$')

        # Hide all the xticks for receiver different of the last one
        hide_xticks = False if rec < num_recvs - 1 else True
        setp(axes[rec].get_xticklabels(), visible=hide_xticks)

        # Axis format
        axes[rec].set_xlim(0, tf)
        axes[rec].ticklabel_format(
            axis='y', style='scientific', scilimits=(-2, 2))
        if rec == num_recvs - 1:
            axes[rec].set_xlabel(r'$t \; (s)$')

    # Saving the plot
    time_str = wave_object.path_case_abc + "time"
    savefig(time_str + ".png", bbox_inches='tight')
    savefig(time_str + ".pdf", bbox_inches='tight')
    plt_show() if show else None
    close()


def plot_rfft_receivers(wave_object, factor_xlim=4., show=False):
    """Plot frequency-domain receiver response comparison.

    Creates a multi-panel plot comparing the normalized frequency-domain
    (FFT) response at each receiver between the computed solution and a
    reference solution. Vertical lines indicate the source and reference
    frequencies.

    Parameters
    ----------
    Wave_object : `wave`
        Wave object containing the simulation results. Must have the
        following attributes:
        - receivers_out_fft: FFT of computed receiver data
        - receivers_ref_fft: FFT of reference receiver data
        - freq_Nyq: Nyquist frequency
        - frequency: Source frequency
        - freq_ref: Reference frequency
        - number_of_receivers: Number of receivers
        - path_save: Directory path for saving plots
        - case_abc: Case name for file naming
    factor_xlim : `float`, optional
        Factor to set the x-axis limits relative to the source frequency.
        The plot will show frequencies up to factor_xlim * source_frequency,
        capped at the Nyquist frequency. Minimum value is 2.
        Default is 4.

    show : `bool`, optional
        Whether to display the plot interactively. Default is False.

    Returns
    -------
    None

    Notes
    -----
    The function saves two files:
    - {path_save}/{case_abc}/freq.png
    - {path_save}/{case_abc}/freq.pdf

    The green solid line represents the FFT of the computed solution,
    while the red dashed line represents the FFT of the reference solution.
    Black vertical lines mark the source and reference frequencies.
    """

    print("\nPlotting Frequency Comparison", flush=True)

    # Frequency data
    freq_Nyq = wave_object.freq_Nyq
    freq_sou = wave_object.frequency
    samples_fft = wave_object.receivers_out_fft.shape[0] - 1
    df = freq_Nyq / samples_fft
    limf = round(clip(factor_xlim * freq_sou, 2 * freq_sou, freq_Nyq), 1)
    idx_lim = int(limf / df) + 1
    f_rec = linspace(0, df * idx_lim, idx_lim)

    # Setting fonts
    rcParams['font.size'] = 7

    # Setting subplots
    num_recvs = wave_object.number_of_receivers
    rcParams['axes.grid'] = True
    fig, axes = subplots(nrows=num_recvs, ncols=1)
    fig.subplots_adjust(hspace=0.6)

    # Setting colormap
    cl_rc = (0., 1., 0., 1.)  # RGB-alpha (Green)
    cl_rf = (1., 0., 0., 1.)  # RGB-alpha (Red)

    for rec in range(num_recvs):

        # Plot the receiver data
        rc_dat = wave_object.receivers_out_fft[:idx_lim, rec]
        rf_dat = wave_object.receivers_ref_fft[:idx_lim, rec]
        axes[rec].plot(f_rec, rc_dat, color=cl_rc, linestyle='-', linewidth=2)
        axes[rec].plot(f_rec, rf_dat, color=cl_rf, linestyle='--', linewidth=2)

        # Add a vertical line at f_ref and freq_sou
        if freq_sou == wave_object.freq_ref:
            f_ref = freq_sou
            f_str = r'$f_{ref} = f_{sou}$'
        else:
            f_ref = wave_object.freq_ref
            f_str = r'$f_{ref}$'
            axes[rec].axvline(x=freq_sou, color='black', linestyle='-', linewidth=1.25)

        axes[rec].axvline(x=f_ref, color='black', linestyle='-', linewidth=1.25)

        # Adding the receiver number label
        axes[rec].text(0.995, 0.9, "R" + str(rec + 1), fontsize=8.5,
                       transform=axes[rec].transAxes, fontweight='bold',
                       verticalalignment='top', horizontalalignment='right',
                       bbox=dict(facecolor='none', edgecolor='none'))

        # Centered title
        if rec == num_recvs // 2:
            axes[rec].set_ylabel(r'$FFT \; recs_{norm}$')

        # Hide all the xticks for receiver different of the last one
        hide_xticks = False if rec < num_recvs - 1 else True
        setp(axes[rec].get_xticklabels(), visible=hide_xticks)

        # Axis format
        axes[rec].set_xlim(0, limf)
        axes[rec].ticklabel_format(axis='y', style='scientific', scilimits=(-2, 2))
        if rec == num_recvs - 1:
            axes[rec].set_xlabel(r'$f \; (Hz)$')

            # Adding the frequency labels
            axes[rec].text(
                f_ref - limf / 500., axes[rec].get_ylim()[0] * 1.05, f_str,
                color='black', fontsize=8, fontweight='bold', ha='right', va='bottom')
            axes[rec].text(
                freq_sou + limf / 500., axes[rec].get_ylim()[0] * 1.05, r'$f_{sou}$',
                color='black', fontsize=8, fontweight='bold', ha='left',
                va='bottom') if freq_sou != wave_object.freq_ref else None

    # Saving the plot
    time_str = wave_object.path_case_abc + "freq"
    savefig(time_str + ".png", bbox_inches='tight')
    savefig(time_str + ".pdf", bbox_inches='tight')
    plt_show() if show else None
    close()


def plot_xCR_opt(wave_object, data_regr_xCR, show=False):
    """Plot the regression curve for the optimal xCR parameter.

    Creates a plot showing the quadratic regression of integral and peak
    errors as a function of the heuristic factor xCR, highlighting the
    optimal value based on a specified criterion. The plot includes
    regression equations, R² values, and the optimal xCR marker.

    Parameters
    ----------
    wave_object: `wave`
        The Wave object containing the simulation results and configuration.
        Must have attributes:
        - xCR_bounds: Bounds for the xCR parameter
        - path_save: Directory path for saving plots
        - case_abc: Case name for file naming
    data_regr_xCR: `list`
        Data for the regression of the parameter xCR with structure:
        [xCR, max_errIt, max_errPk, crit_opt] where:
        - xCR : array-like
            Values of xCR used in the regression. The last value is the optimal xCR.
        - max_errIt : array-like
            Values of the maximum integral error at each xCR.
            The last value corresponds to the optimal xCR.
        - max_errPk : array-like
            Values of the maximum peak error at each xCR.
            The last value corresponds to the optimal xCR.
        - crit_opt : `str`
            Criterion used to determine the optimal xCR.
            Options:
            - 'err_difference' : Minimizes difference between integral and peak errors
            - 'err_integral' : Minimizes integral error
            - 'err_sum' : Minimizes the sum of integral and peak errors
    show : bool, optional
        Whether to display the plot interactively. Default is False.

    Returns
    -------
    None
    """

    # Data for regression
    xCR, max_errIt, max_errPk, crit_opt = data_regr_xCR
    xCR_opt = xCR[-1]
    err_opt = max_errIt[-1]
    eq_eI = polyfit(xCR[:-1], max_errIt[:-1], 2)
    eq_eP = polyfit(xCR[:-1], max_errPk[:-1], 2)

    # Compute R^2 values
    y_eI_true = max_errIt[:-1]
    y_eI_pred = polyval(eq_eI, xCR[:-1])
    y_eP_true = max_errPk[:-1]
    y_eP_pred = polyval(eq_eP, xCR[:-1])
    p = 2  # Quadratic model (Predictors: x and x^2)
    r2_eI = coeff_of_determination(y_eI_true, y_eI_pred, p)
    r2_eP = coeff_of_determination(y_eP_true, y_eP_pred, p)

    # Format equations
    qua_reg = r'${:.3e} x^{{2}} + {:.3e} x + {:.3e}, R^{{2}} = {:.3f}$'
    eq_str_eI = (r'$e_I = $' + qua_reg).format(*eq_eI, r2_eI).replace("+ -", "- ")
    eq_str_eP = (r'$e_P = $' + qua_reg).format(*eq_eP, r2_eP).replace("+ -", "- ")

    # Regression points
    plot(xCR[:-1], 100 * asarray(max_errIt[:-1]), 'ro',
         label=r'Integral Error: ' + eq_str_eI)
    plot(xCR[:-1], 100 * asarray(max_errPk[:-1]), 'bo',
         label=r'Peak Error: ' + eq_str_eP)

    # xCR limits
    xCR_inf, xCR_sup = wave_object.xCR_lim

    # Regression curves
    xgraf = linspace(xCR_inf, xCR_sup, int((xCR_sup - xCR_inf) / 0.1))
    y_eI = polyval(eq_eI, xgraf)
    y_eP = polyval(eq_eP, xgraf)
    plot(xgraf, 100 * y_eI, color='r', linestyle='--')
    plot(xgraf, 100 * y_eP, color='b', linestyle='--')

    # Locating the optimal value
    plot([xCR_opt, xCR_opt], [0., 100 * err_opt], 'k-')
    xopt_str = r'Optimized Heuristic Factor: $X^{{*}}_{{C_{{R}}}} = {:.3f}$'
    if round(100 * polyval(eq_eI, xCR_opt), 2) == round(100 * polyval(eq_eP, xCR_opt), 2):
        xopt_str += r' | $e_{{I}} = e_{{P}} = {:.2f}\%$'
        label = xopt_str.format(xCR_opt, 100 * err_opt)
    else:
        xopt_str += r' | $e_{{I}} = {:.2f}\%$ | $e_{{P}} = {:.2f}\%$'
        label = xopt_str.format(xCR_opt, 100 * err_opt, 100 * max_errPk[-1])
    plot(xCR_opt, 100 * err_opt, marker=r'$\ast$', color='k', markersize=10, label=label)
    legend(loc="best", fontsize=8.5)

    # Formatting the plot
    max_err = max(max(max_errIt[:-1]), max(max_errPk[:-1]))
    xlim(0, round(xCR_sup, 1) + 0.1)
    ylim(0, round(100 * max_err, 1) + 0.1)
    if crit_opt == 'err_difference':
        str_crt = r' (Criterion: Min $(e_I - e_P)$)'
    elif crit_opt == 'err_integral':
        str_crt = r' (Criterion: Min $e_I$)'
    elif crit_opt == 'err_sum':
        str_crt = r' (Criterion: Min $(e_I + e_P)$)'

    xlabel(r'$X_{C_{R}}$' + str_crt)
    tight_layout(pad=2)
    ylabel(r'$e_I \; | \; e_P \; (\%)$')

    # Saving the plot
    xcr_str = wave_object.path_case_abc + "xCR"
    savefig(xcr_str + '.png', bbox_inches='tight')
    savefig(xcr_str + '.pdf', bbox_inches='tight')
    plt_show() if show else None
    close()
