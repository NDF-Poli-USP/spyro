import spyro
import firedrake as fire
import numpy as np
import matplotlib.pyplot as plt

final_time = 1.0


def get_error(dt):

    dts = [
        0.0005,
        0.0003,
        0.0001,
        8e-05,
        5e-05,
    ]

    errors = [
        1.6396549870165494e-06,
        5.896549156143283e-07,
        6.547845892529218e-08,
        4.189559379765933e-08,
        1.6385623108142117e-08,
    ]

    if dt not in dts:
        raise ValueError("dt not in dts")

    return errors[dts.index(dt)]


if __name__ == "__main__":
    dts = [
        5e-4,
        3e-4,
        1e-4,
        8e-5,
        5e-5,
    ]

    errors = []
    for dt in dts:
        errors.append(get_error(dt))

    for dt in dts:
        print(f"dt = {dt}, error = {errors[dts.index(dt)]}")

    plt.loglog(dts, errors, '-o', label='numerical error')

    theory = [t**2 for t in dts]
    theory = [errors[0]*th/theory[0] for th in theory]

    plt.loglog(dts, theory, '--^', label='theoretical 2nd order in time')

    plt.legend()
    plt.title("MMS convergence for triangles")
    plt.xlabel("dt [s]")
    plt.ylabel("error")
    plt.show()
