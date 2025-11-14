import numpy as np
import matplotlib.pyplot as plt
import sys

# tot_it = int(sys.argv[1])
# filename = f"functional_values{tot_it}.txt"
# output_filename = f"analyze_functional_output{tot_it}.txt"
# fig_name = f"analyse_functional{tot_it}.png"

filename = "functional_values50_queijao.txt"
output_filename = "analyze_functional_values50_queijao_5hz_m5.txt"
fig_name = "analyse_functional_values_50_queijao_5hz_m5.png"

M = 5   # window size for running average

iters, J = [], []
with open(filename, "r") as f:
    for line in f:
        if "Iteration" in line and "Functional" in line:
            parts = line.strip().split(",")
            it = int(parts[0].split(":")[1])
            val = float(parts[1].split(":")[1])
            iters.append(it)
            J.append(val)

J = np.array(J)
iters = np.array(iters)

# === Option 1: relative misfit reduction per iteration ===
# ΔJ_rel[k] = (J[k-1] - J[k]) / J[k-1]
dJ_rel = np.zeros_like(J)
dJ_rel[1:] = (J[:-1] - J[1:]) / J[:-1]
dJ_rel[0] = np.nan  # undefined for first iteration

# === Option 2: running (smoothed) average over the last M iterations ===
def running_mean(x, M):
    """Centered running mean of length M (NaN-padded edges)."""
    if len(x) < M:
        return np.full_like(x, np.nan)
    kernel = np.ones(M) / M
    return np.convolve(x, kernel, mode="same")

dJ_rel_smooth = running_mean(dJ_rel, M)

# Save output to text file

with open(output_filename, "w") as f:
    f.write(f"{'Iter':>5} | {'J':>14} | {'ΔJ_rel':>12} | {f'AvgΔJ_rel(M={M})':>16}\n")
    f.write("-"*55 + "\n")
    for i in range(len(J)):
        f.write(f"{iters[i]:5d} | {J[i]:14.8e} | {dJ_rel[i]:12.4e} | {dJ_rel_smooth[i]:16.4e}\n")

print(f"Output saved to {output_filename}")

print(f"{'Iter':>5} | {'J':>14} | {'ΔJ_rel':>12} | {'AvgΔJ_rel(M=3)':>16}")
print("-"*55)
for i in range(len(J)):
    print(f"{iters[i]:5d} | {J[i]:14.8e} | {dJ_rel[i]:12.4e} | {dJ_rel_smooth[i]:16.4e}")

# === Plot ===
fig, ax1 = plt.subplots(figsize=(7,4))
ax1.plot(iters, J, 'o-', color='tab:blue', label='Functional J')
ax1.set_yscale('log')
ax1.set_xlabel("Iteration")
ax1.set_ylabel("Functional J", color='tab:blue')

ax2 = ax1.twinx()
ax2.plot(iters, dJ_rel, 'r.--', alpha=0.5, label='ΔJ_rel')
ax2.plot(iters, dJ_rel_smooth, 'k-', lw=2, label=f'AvgΔJ_rel (M={M})')
ax2.axhline(y=0.1, color='gray', linestyle='--', alpha=0.7, label='0.1 threshold')
ax2.set_ylabel("Relative reduction per iter")
ax2.legend(loc="upper right")

plt.title("Functional decay and relative misfit reduction")
plt.tight_layout()
plt.savefig(fig_name)

