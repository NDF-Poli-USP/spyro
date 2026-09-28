import matplotlib.pyplot as plt
from matplotlib import use
use("agg")
import numpy as np
from spyro.tools.error_measure import MeasureError

cpws = [
    1.0,
    1.1,
    1.2,
    1.4,
    1.5,
    1.6,
    1.8,
    1.9,
    2.0,
    2.1,
    2.2,
    2.4,
    2.5,
    2.6,
    2.8,
    3.0,
    3.1,
    3.2,
    3.3,
    3.4,
    3.5,
]
is_quadrilateral = True
if is_quadrilateral:
    cell_text = "quad"
    image_text = "SEM"
    # c_reference = 3.5
else:
    cell_text = "ml"
    image_text = "ML"

#establish reference
degree = 4
c_reference = 3.4
# c_reference = 3.5
reference_result_filename = f"2dhetsol_ml4_{c_reference:.1f}".replace(".", "") + ".npy"
reference = np.load(reference_result_filename)

#time variables
time_vector = np.linspace(0.0, 2.0, round(2.0/1e-4)+1)
dt = 1e-4

#initializing error vector
errors = []

actual_cpws = []
#get comparison result
for c in cpws:
# c = cpws[0]
    result_filename = f"2dhetsol_{cell_text}{degree}_{c:.1f}".replace(".", "") + ".npy"
    try:
        result = np.load(result_filename)
        #error calc
        rec_error = MeasureError.calculate_receiver_error(result, reference, dt)*100
        errors.append(rec_error)
        actual_cpws.append(c)
        print(f"{c}, {rec_error:.7f},")
    except:
        print("\n")


# if we need to convert frequency to 2.5:
# cpws_array = np.array(cpws) * (2.5/2.0)
# cpws = cpws_array.tolist()

plt.plot(
    actual_cpws,
    errors,
    "o-",
    label=f"{cell_text.upper()}{degree}",
)
# Error threshold
plt.axhline(
    5e-2,
    linestyle="--",
    linewidth=1.5,
    label=r"Error = $5\times10^{-2}$",
)
plt.xlabel("Cells per wavelength (CPW)")
plt.ylabel("Normalized $L^2$ error in receivers")

plt.yscale("log")

plt.grid(
    True,
    which="both",
    linestyle=":",
    alpha=0.5,
)

plt.legend()

plt.tight_layout()
plt.savefig(f"2d_het_cpw_{image_text}{degree}.png")

#looking at specific receiver
rec_id = 10
plt.close()
plt.plot(time_vector, reference[:, rec_id, 1], label="reference")
plt.plot(time_vector, result[:, rec_id, 1], "--", label="result")
plt.legend()
plt.savefig("debug.png")

print("END")