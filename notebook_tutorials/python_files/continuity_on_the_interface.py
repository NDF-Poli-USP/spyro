import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

RESULTS_MIXED = "results_mixed/li_et_al_monolithic"
fem_data = np.load(os.path.join(RESULTS_MIXED, "receiver_data.npz"))
t_fem = fem_data["time"]
ux_fem = fem_data["ux_recv"]
uf_x_recv = fem_data["uf_x_recv"]

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(t_fem, uf_x_recv, 'b-', lw=1.5, label='u_x lado fluido')
ax.plot(t_fem, ux_fem, 'r--', lw=1.2, label='u_x lado sólido')
ax.set_xlabel("Time (s)"); ax.set_ylabel("Displacement x (normal)")
ax.set_title("Interface normal displacement continuity")
ax.legend(); ax.grid(True, ls=':')
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_MIXED, "interface_displacement_check.png"), dpi=200)
print("Salvo")