# Open the file for reading
with open('p3_cpw_results.txt', 'r') as file:
    # Initialize empty lists for each column
    C_values = []
    dt_values = []
    error_values = []
    runtime_values = []

    # Read each line in the file
    for line in file:
        # Split the line into columns based on whitespace
        columns = line.split()

        # Convert the columns to the appropriate data types
        C = float(columns[0])
        dt = float(columns[1])
        error = float(columns[2])
        runtime = float(columns[3])

        # Append the values to their respective lists
        C_values.append(C)
        dt_values.append(dt)
        error_values.append(error)
        runtime_values.append(runtime)

    print(f"c = {C_values}")
    print(f"dt = {dt_values}")
    print(f"error = {error_values}")
    print(f"runtime = {runtime_values}")
    print("END")

# Now you have the data in separate lists: C_values, dt_values, error_values, and runtime_values
