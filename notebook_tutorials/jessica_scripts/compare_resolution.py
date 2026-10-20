import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

cases = ["case_2d_res_h50", "case_2d_res_h25", "case_2d_res_h12"]
labels = ["h = 50 m", "h = 25 m", "h = 12.5 m"]
fig, axs = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
for case, label in zip(cases, labels):
    d = np.load(f"results/spyro_receiver_data_{case}_sequential_2d.npz")
    p = np.ravel(d["p_spyro"])
    u = np.asarray(d["u_solid"])
    t = np.arange(len(p)) * float(d["dt"])
    tu = np.arange(len(u)) * float(d["dt"])
    axs[0].plot(t, p / np.abs(p).max(), label=label)
    axs[1].plot(tu, u[:, 0] / np.abs(u[:, 0]).max(), label=label)
    axs[2].plot(tu, u[:, 1] / np.abs(u[:, 1]).max(), label=label)
for ax, title in zip(axs, ["P", "u_z", "u_x"]):
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
axs[-1].set_xlabel("Time (s)")
plt.tight_layout()
plt.savefig("results/compare_resolution.png", dpi=150)