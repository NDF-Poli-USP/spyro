import sys
import re
import csv

# Open the input and output files
if len(sys.argv) < 2:
    print("Please provide input file name as command line argument")
    sys.exit()

# Get input file name from command line argument
input_file = sys.argv[1]

# input_file = "amd_strong_scalling/test1_overthurst.23184.out"

core_count = []
runtimes = []
# Open input file for reading
with open(input_file, 'r') as f:
    # Initialize variables for number of cores and biggest number
    num_cores = None
    num_runtimes = 0
    
    # Loop through each line in the file
    for line in f:
        # Check if the line contains the number of cores
        if "Each shot is using" in line:
            num_cores = int(line.split()[-2])
            core_count.append(num_cores)
            max_runtime = 0
        # Check if the line contains the biggest number

        if "Spatial parallelism, reducing to comm 0" in line:
            getting_runtime = True
            num_runtimes = 0   

        match = re.match(r'^\d+\.\d+', line)
        if match and getting_runtime:
            num_runtimes += 1
            runtime = float(match.group())
            if runtime > max_runtime:
                max_runtime = runtime
            
            if num_runtimes == num_cores:
                runtimes.append(max_runtime)
                getting_runtime = False


# Save data to file with same name as input file plus "data"
output_file = input_file + "data.csv"
with open(output_file , 'w', newline='') as output_file:
    # Create a CSV writer object
    csv_writer = csv.writer(output_file)

    # Write the header row to the output file
    csv_writer.writerow(['Cores Used', 'Runtime'])

    # Write the shot information to the output file
    for i in range(len(core_count)):
        core = core_count[i]
        r = runtimes[i]
        csv_writer.writerow([core_count[i], runtimes[i]])

