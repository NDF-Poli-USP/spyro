import segyio
import numpy as np
import matplotlib.pyplot as plt

segy_file = "final_vp_it14.segy"
output_file = "final_vp_it14.png"

# Open the SEG-Y file
with segyio.open(segy_file, "r", ignore_geometry=True) as f:
    # Read all trace data
    data = segyio.tools.collect(f.trace[:])

# Plot and save as PNG
plt.figure(figsize=(10, 10))
plt.imshow(data.T, aspect='equal', cmap='seismic', interpolation='nearest')
plt.colorbar(label='Amplitude')
plt.title('SEG-Y Visualization')
plt.xlabel('Trace')
plt.ylabel('Sample')
plt.tight_layout()
plt.savefig(output_file)
plt.close()