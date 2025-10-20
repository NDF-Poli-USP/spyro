import matplotlib.pyplot as plt
import numpy as np

# Read and parse the functional values
def read_functional_values(filename):
    iterations = []
    functionals = []
    
    with open(filename, 'r') as file:
        for line in file:
            if line.strip():  # Skip empty lines
                # Parse line format: "Iteration: X, Functional: Y"
                parts = line.strip().split(', ')
                iteration = int(parts[0].split(': ')[1])
                functional = float(parts[1].split(': ')[1])
                iterations.append(iteration)
                functionals.append(functional)
    
    return np.array(iterations), np.array(functionals)

# Read the data
iterations, functionals = read_functional_values('/home/olender/spyro_development/fwi_tutorial/queijo_minas_big_results/functional_values.txt')

# # Create the plot
# plt.figure(figsize=(12, 8))

# Plot with both linear and log scale options
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

# Linear scale plot
ax1.plot(iterations, functionals, 'b-o', linewidth=2, markersize=4)
ax1.set_xlabel('Iteration')
ax1.set_ylabel('Functional Value')
ax1.set_title('Functional Value vs Iteration (Linear Scale)')
ax1.grid(True, alpha=0.3)
ax1.set_xlim(0, max(iterations))

# Log scale plot
ax2.semilogy(iterations, functionals, 'r-o', linewidth=2, markersize=4)
ax2.set_xlabel('Iteration')
ax2.set_ylabel('Functional Value (log scale)')
ax2.set_title('Functional Value vs Iteration (Log Scale)')
ax2.grid(True, alpha=0.3)
ax2.set_xlim(0, max(iterations))

plt.tight_layout()
plt.show()

# Print some statistics
print(f"Initial functional value: {functionals[0]:.6e}")
print(f"Final functional value: {functionals[-1]:.6e}")
print(f"Reduction factor: {functionals[0]/functionals[-1]:.2f}")
print(f"Total iterations: {len(iterations)}")