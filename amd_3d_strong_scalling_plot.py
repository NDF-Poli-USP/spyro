# import matplotlib.pyplot as plt

# # Define the data
# cores = [40, 30, 20, 15, 10, 5]
# runtimes = [1182.7118127346039, 1233.0374219417572, 1761.6108815670013, 2240.1770391464233, 3279.3396253585815, 6266.367906570435]

# # Plot the data
# fig, ax = plt.subplots()
# ax.plot(cores, runtimes, 'o-')

# # Set the labels and title
# ax.set_xlabel('Number of cores')
# ax.set_ylabel('Computational runtime (s)')
# ax.set_title('Strong Scaling Plot')

# # Add a grid and adjust the axis limits
# ax.grid()
# ax.set_xlim([min(cores)-5, max(cores)+5])
# ax.set_ylim([min(runtimes)/2, max(runtimes)*2])

# # Display the plot
# plt.show()

import matplotlib.pyplot as plt

# Define the data
cores = [40, 30, 20, 15, 10, 5]
runtimes = [1182.7118127346039, 1233.0374219417572, 1761.6108815670013, 2240.1770391464233, 3279.3396253585815, 6266.367906570435]

# Calculate the ideal runtimes assuming perfect linear scaling
fraction = runtimes[-1] / cores[-1]
ideal_runtimes = [fraction * c for c in cores]

# Plot the data and ideal runtimes
fig, ax = plt.subplots()
ax.plot(cores, runtimes, 'o-', label='Actual Runtimes')
ax.plot(cores, ideal_runtimes, 'k--', label='Ideal Runtimes')

# Set the labels and title
ax.set_xlabel('Number of cores')
ax.set_ylabel('Computational runtime (s)')
ax.set_title('Strong Scaling Plot')

# Add a legend, grid, and adjust the axis limits
ax.legend()
ax.grid()
ax.set_xlim([min(cores)-5, max(cores)+5])
ax.set_ylim([min(runtimes)/2, max(runtimes)*2])

# Display the plot
plt.show()

