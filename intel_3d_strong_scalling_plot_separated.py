import matplotlib.pyplot as plt

# Define the data
# Cores,Minimum Runtime pre forward,Minimum Runtime forward,Minimum Runtime total
# 40,36.55581760406494,661.1032140254974,703.0348110198975
# 30,36.272324323654175,810.6684374809265,846.3902587890625
# 20,37.0566132068634,1164.3955035209656,1201.1293776035309
# 15,38.510701417922974,1517.3705337047577,1555.5230677127838
# 10,38.62554669380188,2089.359314441681,2127.857751607895
# 5,44.504101514816284,4186.835275888443,4231.718212127686


runtimes_pre_forward = [36.55581760406494, 36.272324323654175, 37.0566132068634,38.510701417922974, 38.62554669380188, 44.504101514816284]

runtimes_forward = [661.1032140254974,810.6684374809265,1164.3955035209656,1517.3705337047577,2089.359314441681,4186.835275888443]

cores = [40, 30, 20, 15, 10, 5]
# runtimes = [683.6608626842499, 869.5781161785126, 1238.8793861865997, 1595.3227922916412, 2203.073546409607, 4295.813792228699]
runtimes = runtimes_forward

# Plot the data
fig, ax = plt.subplots()
ax.plot(cores, runtimes, 'o-')

# Set the labels and title
ax.set_xlabel('Number of cores')
ax.set_ylabel('Computational runtime (s)')
ax.set_title('Strong Scaling Plot')

# Add a grid and adjust the axis limits
ax.grid()
ax.set_xlim([min(cores)-5, max(cores)+5])
ax.set_ylim([min(runtimes)/2, max(runtimes)*2])
ax.set_yscale('log')

# Display the plot
plt.show()

# import matplotlib.pyplot as plt

# # Define the data
# cores = [40, 30, 20, 15, 10, 5]
# runtimes = [1182.7118127346039, 1233.0374219417572, 1761.6108815670013, 2240.1770391464233, 3279.3396253585815, 6266.367906570435]

# # Calculate the ideal runtimes assuming perfect linear scaling
# fraction = runtimes[-1] / cores[-1]
# ideal_runtimes = [fraction * c for c in cores]

# # Plot the data and ideal runtimes
# fig, ax = plt.subplots()
# ax.plot(cores, runtimes, 'o-', label='Actual Runtimes')
# ax.plot(cores, ideal_runtimes, 'k--', label='Ideal Runtimes')

# # Set the labels and title
# ax.set_xlabel('Number of cores')
# ax.set_ylabel('Computational runtime (s)')
# ax.set_title('Strong Scaling Plot')

# # Add a legend, grid, and adjust the axis limits
# ax.legend()
# ax.grid()
# ax.set_xlim([min(cores)-5, max(cores)+5])
# ax.set_ylim([min(runtimes)/2, max(runtimes)*2])

# # Display the plot
# plt.show()

