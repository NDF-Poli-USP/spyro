import matplotlib.pyplot as plt
import numpy as np
import spyro.habc.lay_len as lay_len
from spyro.utils.stats_tools import coeff_of_determination
plt.rcParams.update({"font.family": "serif"})
plt.rcParams['text.latex.preamble'] = r'\usepackage{bm} \usepackage{amsmath}'


def plot_function_layer_size(a, fref, fsou, z, lmin, lref, FLpos, show=False):

    a_lst = [a]
    F_lst = [FLpos]
    l_lst = ["{:.2f}".format(fref)]
    c_lst = ['C0']

    if fsou == fref:
        FLsou = []
        w_lst = ['f_{{sou}}']

    else:

        # Calculate the size parameter for the source frequency
        a_sou = a * fref / fsou
        a_lst.append(a_sou)
        FLsou = lay_len.calc_size_lay(
            fsou, z, lmin, lref, nz=len(FLpos))[-1]
        F_lst.append(FLsou)
        l_lst.append("{:.2f}".format(fsou))
        c_lst.append('C1')
        w_lst = ['f_{{bnd}}', 'f_{{sou}}']

    FL_max = max(FLpos + FLsou) + 0.4
    FL_lim = np.ceil(FL_max * 10) / 10
    F_L = np.linspace(0.001, FL_lim, int(FL_lim * 1e3))
    delta_x = FL_lim / 40

    # Plot the size criterion
    plt.figure(figsize=(12, 6))
    lim_crit = np.inf
    for a_pr, FL_rt, lab, col, w_str in zip(a_lst, F_lst, l_lst, c_lst, w_lst):
        crit = lay_len.f_layer(F_L, a_pr)
        lim_crit = min(lim_crit, crit.min())
        plt.plot(F_L, crit, color=col, zorder=2,
                 label=r'$\Psi_{{F_L}}({}={}\text{{Hz}})$'.format(w_str, lab))
        plt.scatter(FL_rt, np.zeros(5), color=col, zorder=3)

    delta_y = abs(lim_crit) / 2
    for lay, (FL_rt, col) in enumerate(zip(F_lst, c_lst)):
        y_FL = -0.9 * delta_y if lay == 0 else 0.5 * delta_y
        for rt, FL_par in enumerate(FL_rt):
            xFL = FL_par + delta_x if rt % 2 == 0 else FL_par - delta_x
            plt.text(x=xFL, y=y_FL, s="{:.4f}".format(FL_par),
                     horizontalalignment='center', verticalalignment='bottom',
                     bbox=dict(facecolor=col, alpha=0.9), zorder=4)

    plt.xlabel(r'$F_L$')
    plt.ylabel(r'$\Psi_{{F_L}} \; = \; |C_{Rmin}| \; - \; R$')

    plt.xticks(np.arange(0, FL_lim + 0.01, 0.5 if FL_lim > 1 else 0.2))

    plt.xlim((0, FL_lim))
    plt.ylim((lim_crit - 0.01, 1.01))
    plt.grid(zorder=1)
    plt.legend()
    layer_str = "layer_opts"
    plt.savefig(layer_str + ".png", bbox_inches='tight')
    plt.savefig(layer_str + ".pdf", bbox_inches='tight')
    plt.show() if show else None
    plt.close()


def plot_hist_receivers(Wave_object, show=False):
    '''
    Plot the comparison of the time-domain response at the
    receivers between the reference model and the HABC scheme.
    The plots are saved in PDF and PNG formats.

    Parameters
    ----------
    Wave_object: `wave`
        The Wave object containing the simulation results.
    show: `bool`, optional
        Whether to show the plot. Default is False.

    Returns
    -------
    None
    '''

    print("\nPlotting Time Comparison")

    # Time data
    dt = Wave_object.dt
    tf = Wave_object.final_time
    nt = int(tf / dt) + 1  # number of timesteps
    t_rec = np.linspace(0.0, tf, nt)

    # Setting fonts
    plt.rcParams['font.size'] = 7

    # Setting subplots
    num_recvs = Wave_object.number_of_receivers
    plt.rcParams['axes.grid'] = True
    fig, axes = plt.subplots(nrows=num_recvs, ncols=1)
    fig.subplots_adjust(hspace=0.6)

    # Setting colormap
    cl_rc = (0., 1., 0., 1.)  # RGB-alpha (Green)
    cl_rf = (1., 0., 0., 1.)  # RGB-alpha (Red)

    for rec in range(num_recvs):

        # Plot the receiver data
        rc_dat = Wave_object.receivers_output[:, rec]
        rf_dat = Wave_object.receivers_reference[:, rec]
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
        plt.setp(axes[rec].get_xticklabels(), visible=hide_xticks)

        # Axis format
        axes[rec].set_xlim(0, tf)
        axes[rec].ticklabel_format(
            axis='y', style='scientific', scilimits=(-2, 2))
        if rec == num_recvs - 1:
            axes[rec].set_xlabel(r'$t \; (s)$')

    # Saving the plot
    time_str = Wave_object.path_save + Wave_object.case_habc + "/time"
    plt.savefig(time_str + ".png", bbox_inches='tight')
    plt.savefig(time_str + ".pdf", bbox_inches='tight')
    plt.show() if show else None
    plt.close()


def plot_rfft_receivers(Wave_object, fxlim=4., show=False):
    '''
    Plot the comparison of the frequency-domain response at the
    receivers between the reference model and the HABC scheme.
    The plots are saved in PDF and PNG formats.

    Parameters
    ----------
    Wave_object: `wave`
        Wave object containing the simulation results.
    fxlim: `float`, optional
        Factor to set the x-axis limits in the plots realtive to
        the source frequency. Default is 4 and the minimum is 2.
    show: `bool`, optional
        Whether to show the plot. Default is False.

    Returns
    -------
    None
    '''

    print("\nPlotting Frequency Comparison")

    # Frequency data
    f_Nyq = Wave_object.f_Nyq
    f_sou = Wave_object.frequency
    pfft = Wave_object.receivers_out_fft.shape[0] - 1
    df = f_Nyq / pfft
    limf = round(min(max(fxlim, 2.) * f_sou, f_Nyq), 1)
    idx_lim = int(limf / df) + 1
    f_rec = np.linspace(0, df * idx_lim, idx_lim)

    # Setting fonts
    plt.rcParams['font.size'] = 7

    # Setting subplots
    num_recvs = Wave_object.number_of_receivers
    plt.rcParams['axes.grid'] = True
    fig, axes = plt.subplots(nrows=num_recvs, ncols=1)
    fig.subplots_adjust(hspace=0.6)

    # Setting colormap
    cl_rc = (0., 1., 0., 1.)  # RGB-alpha (Green)
    cl_rf = (1., 0., 0., 1.)  # RGB-alpha (Red)

    for rec in range(num_recvs):

        # Plot the receiver data
        rc_dat = Wave_object.receivers_out_fft[:idx_lim, rec]
        rf_dat = Wave_object.receivers_ref_fft[:idx_lim, rec]
        axes[rec].plot(f_rec, rc_dat, color=cl_rc, linestyle='-', linewidth=2)
        axes[rec].plot(f_rec, rf_dat, color=cl_rf, linestyle='--', linewidth=2)

        # Add a vertical line at f_ref and f_sou
        if f_sou == Wave_object.freq_ref:
            f_ref = f_sou
            f_str = r'$f_{ref} = f_{sou}$'
        else:
            f_ref = Wave_object.freq_ref
            f_str = r'$f_{ref}$'
            axes[rec].axvline(
                x=f_sou, color='black', linestyle='-', linewidth=1.25)

        axes[rec].axvline(
            x=f_ref, color='black', linestyle='-', linewidth=1.25)

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
        plt.setp(axes[rec].get_xticklabels(), visible=hide_xticks)

        # Axis format
        axes[rec].set_xlim(0, limf)
        axes[rec].ticklabel_format(
            axis='y', style='scientific', scilimits=(-2, 2))
        if rec == num_recvs - 1:
            axes[rec].set_xlabel(r'$f \; (Hz)$')

            # Adding the frequency labels
            axes[rec].text(f_ref - limf / 500., axes[rec].get_ylim()[0] * 1.05,
                           f_str, color='black', fontsize=8, fontweight='bold',
                           ha='right', va='bottom')
            axes[rec].text(f_sou + limf / 500., axes[rec].get_ylim()[0] * 1.05,
                           r'$f_{sou}$', color='black', fontsize=8,
                           fontweight='bold', ha='left', va='bottom') \
                if f_sou != Wave_object.freq_ref else None

    # Saving the plot
    time_str = Wave_object.path_save + Wave_object.case_habc + "/freq"
    plt.savefig(time_str + ".png", bbox_inches='tight')
    plt.savefig(time_str + ".pdf", bbox_inches='tight')
    plt.show() if show else None
    plt.close()


def plot_xCR_opt(Wave_object, data_regr_xCR, show=False):
    '''
    Plot the regression curve for the optimal xCR value.

    Parameters
    ----------
    Wave_object: `wave`
        The Wave object containing the simulation results
    data_regr_xCR: `list`
        Data for the regression of the parameter xCR.
        Structure: [xCR, max_errIt, max_errPK, crit_opt]
        - xCR: Values of xCR used in the regression.
          The last value IS the optimal xCR
        - max_errIt: Values of the maximum integral error.
          The last value corresponds to the optimal xCR
        - max_errPK: Values of the maximum peak error.
          The last value corresponds to the optimal xCR
        - crit_opt : Criterion for the optimal heuristic factor.
          * 'error_difference' : Difference between integral and peak errors
          * 'error_integral' : Minimum integral error
    show: `bool`, optional
        Whether to show the plot. Default is False.

    Returns
    -------
    None
    '''

    # Data for regression
    xCR, max_errIt, max_errPk, crit_opt = data_regr_xCR
    xCR_opt = xCR[-1]
    err_opt = max_errIt[-1]
    eq_eI = np.polyfit(xCR[:-1], max_errIt[:-1], 2)
    eq_eP = np.polyfit(xCR[:-1], max_errPk[:-1], 2)

    # Compute R^2 values
    y_eI_true = max_errIt[:-1]
    y_eI_pred = np.polyval(eq_eI, xCR[:-1])
    y_eP_true = max_errPk[:-1]
    y_eP_pred = np.polyval(eq_eP, xCR[:-1])
    p = 2  # Quadratic model (Predictors: x and x^2)
    r2_eI = coeff_of_determination(y_eI_true, y_eI_pred, p)
    r2_eP = coeff_of_determination(y_eP_true, y_eP_pred, p)

    # Format equations
    qua_reg = r'${:.3e} x^{{2}} + {:.3e} x + {:.3e}, R^{{2}} = {:.3f}$'
    eq_str_eI = (
        r'$e_I = $' + qua_reg).format(*eq_eI, r2_eI).replace("+ -", "- ")
    eq_str_eP = (
        r'$e_P = $' + qua_reg).format(*eq_eP, r2_eP).replace("+ -", "- ")

    # Regression points
    plt.plot(xCR[:-1], 100 * np.asarray(max_errIt[:-1]), 'ro',
             label=r'Integral Error: ' + eq_str_eI)
    plt.plot(xCR[:-1], 100 * np.asarray(max_errPk[:-1]), 'bo',
             label=r'Peak Error: ' + eq_str_eP)

    # xCR limits
    xCR_inf, xCR_sup = Wave_object.xCR_bounds[0]

    # Regression curves
    xgraf = np.linspace(xCR_inf, xCR_sup, int((xCR_sup - xCR_inf) / 0.1))
    y_eI = np.polyval(eq_eI, xgraf)
    y_eP = np.polyval(eq_eP, xgraf)
    plt.plot(xgraf, 100 * y_eI, color='r', linestyle='--')
    plt.plot(xgraf, 100 * y_eP, color='b', linestyle='--')

    # Locating the optimal value
    plt.plot([xCR_opt, xCR_opt], [0., 100 * err_opt], 'k-')
    xopt_str = r'Optimized Heuristic Factor: $X^{{*}}_{{C_{{R}}}} = {:.3f}$'
    if round(100 * np.polyval(eq_eI, xCR_opt), 2) == round(
            100 * np.polyval(eq_eP, xCR_opt), 2):
        xopt_str += r' | $e_{{I}} = e_{{P}} = {:.2f}\%$'
        label = xopt_str.format(xCR_opt, 100 * err_opt)
    else:
        xopt_str += r' | $e_{{I}} = {:.2f}\%$ | $e_{{P}} = {:.2f}\%$'
        label = xopt_str.format(xCR_opt, 100 * err_opt, 100 * max_errPk[-1])
    plt.plot(xCR_opt, 100 * err_opt, marker=r'$\ast$', color='k',
             markersize=10, label=label)
    plt.legend(loc="best", fontsize=8.5)

    # Formatting the plot
    max_err = max(max(max_errIt[:-1]), max(max_errPk[:-1]))
    plt.xlim(0, round(xCR_sup, 1) + 0.1)
    plt.ylim(0, round(100 * max_err, 1) + 0.1)
    if crit_opt == 'error_difference':
        str_crt = r' (Criterion: Min $(e_I - e_P)$)'
    elif crit_opt == 'error_integral':
        str_crt = r' (Criterion: Min $e_I$)'

    plt.xlabel(r'$X_{C_{R}}$' + str_crt)
    plt.tight_layout(pad=2)
    plt.ylabel(r'$e_I \; | \; e_P \; (\%)$')

    # Saving the plot
    xcr_str = Wave_object.path_save + Wave_object.case_habc + "/xCR"
    plt.savefig(xcr_str + '.png', bbox_inches='tight')
    plt.savefig(xcr_str + '.pdf', bbox_inches='tight')
    plt.show() if show else None
    plt.close()
