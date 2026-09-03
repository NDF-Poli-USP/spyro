import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import correlate

from spyro.plots.receiver_plots import plot_receiver_response, plot_displacement_components
from spyro.tools.error_measure import MeasureError

NPZ_PATH = "results/spyro_receiver_data.npz"
GAR6MORE_DIR_FLUID = "/workspaces/spyro/notebook_tutorials/results_fluid"
GAR6MORE_DIR_SOLID = "/workspaces/spyro/notebook_tutorials/results_solid"
FLIP_UZ = True
OUTPUT_DIR = "results"

os.makedirs(OUTPUT_DIR, exist_ok=True)

data = np.load(NPZ_PATH)
dt = float(data["dt"])
final_time = float(data["final_time"])
p_spyro = data["p_spyro"]
u_solid = data["u_solid"]

uz_spyro = u_solid[:, 0]
ux_spyro = u_solid[:, 1]
t_spyro = np.linspace(0.0, final_time, len(p_spyro))

print(f"[OK] Dados carregados de: {os.path.abspath(NPZ_PATH)}")

plot_receiver_response(
    p_spyro,
    final_time=final_time,
    filename=f"{OUTPUT_DIR}/receiver_fluid.png",
    receiver_id_for_title=0,
)

plot_displacement_components(
    time_vector=t_spyro,
    receiver_results=u_solid,
    source_type="Ricker",
    filename=f"{OUTPUT_DIR}/receiver_solid.png",
)


def load_gar6more_file(path):
    values = np.loadtxt(path)
    if values.ndim == 1:
        raise ValueError(f"{path}: só 1 coluna encontrada — esperado (tempo, valor).")
    return values[:, 0], values[:, 1]


def normalize(x):
    peak = np.abs(x).max()
    return x / peak if peak > 0 else x


def estimate_time_shift(time_vector, numerical, reference):
    numerical = numerical - np.mean(numerical)
    reference = reference - np.mean(reference)
    correlation = correlate(numerical, reference, mode="full")
    lag_samples = np.argmax(correlation) - (len(reference) - 1)
    dt_ = time_vector[1] - time_vector[0]
    return lag_samples * dt_


t_gar6_p,  p_gar6      = load_gar6more_file(f"{GAR6MORE_DIR_FLUID}/P.dat")
t_gar6_ux, ux_gar6_raw = load_gar6more_file(f"{GAR6MORE_DIR_SOLID}/Ux.dat")
t_gar6_uy, uy_gar6_raw = load_gar6more_file(f"{GAR6MORE_DIR_SOLID}/Uy.dat")

p_gar6_i  = np.interp(t_spyro, t_gar6_p,  p_gar6,  left=0.0, right=0.0)
uz_sign = -1.0 if FLIP_UZ else 1.0
uz_gar6_i = uz_sign * np.interp(t_spyro, t_gar6_ux, ux_gar6_raw, left=0.0, right=0.0)
ux_gar6_i = -np.interp(t_spyro, t_gar6_uy, uy_gar6_raw, left=0.0, right=0.0)

p_spyro_n,  p_gar6_n  = normalize(p_spyro),  normalize(p_gar6_i)
uz_spyro_n, uz_gar6_n = normalize(uz_spyro), normalize(uz_gar6_i)
ux_spyro_n, ux_gar6_n = normalize(ux_spyro), normalize(ux_gar6_i)

shift = estimate_time_shift(t_spyro, p_spyro_n, p_gar6_n)
print(f"Deslocamento temporal estimado: {shift:.6f}s  (dt={dt}s)")

displacement_results = np.stack([uz_spyro_n, ux_spyro_n], axis=1)
fig = plot_displacement_components(
    time_vector=t_spyro,
    receiver_results=displacement_results,
    source_type="Ricker",
    hold=True,
)
axes = fig.get_axes()
axes[0].plot(t_spyro, uz_gar6_n, "--", label="Gar6more2D")
axes[0].legend()
axes[1].plot(t_spyro, ux_gar6_n, "--", label="Gar6more2D")
axes[1].legend()
displacement_path = f"{OUTPUT_DIR}/compare_displacement.png"
fig.savefig(displacement_path)
plt.close(fig)
print(f"[OK] Salvo em: {os.path.abspath(displacement_path)}")

fig_p, ax_p = plt.subplots(figsize=(10, 4))
ax_p.plot(t_spyro, p_spyro_n, label="Spyro")
ax_p.plot(t_spyro, p_gar6_n, "--", label="Gar6more2D")
ax_p.set_xlabel("Time (s)"); ax_p.set_ylabel("P (normalized)")
ax_p.legend(); ax_p.grid(True, alpha=0.3)
p_path = f"{OUTPUT_DIR}/compare_P.png"
fig_p.savefig(p_path)
plt.close(fig_p)
print(f"[OK] Salvo em: {os.path.abspath(p_path)}")

error_p  = MeasureError.calculate_normalized_L2_error(p_spyro_n,  p_gar6_n)
error_uz = MeasureError.calculate_normalized_L2_error(uz_spyro_n, uz_gar6_n)
error_ux = MeasureError.calculate_normalized_L2_error(ux_spyro_n, ux_gar6_n)
print(f"Erro L2 normalizado — P:   {error_p:.4f}")
print(f"Erro L2 normalizado — u_z: {error_uz:.4f}")
print(f"Erro L2 normalizado — u_x: {error_ux:.4f}")