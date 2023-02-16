import sys
import re
import csv

# # Open the input and output files
# if len(sys.argv) < 2:
#     print("Please provide input file name as command line argument")
#     sys.exit()

# # Get input file name from command line argument
# input_file = sys.argv[1]

input_file = "amd_strong_scalling/intel_test1_overthurst.23210.out"

core_count = []
runtimes_total = []
runtimes_no_forward = []
runtimes_forward = []
# Open input file for reading
with open(input_file, 'r') as f:
    # Initialize variables for number of cores and biggest number
    num_cores = None
    num_runtimes = None
    
    # Loop through each line in the file
    first_loop = True
    for line in f:
        # Check if the line contains the number of cores
        if "Each shot is using" in line:
            num_cores = int(line.split()[-2])
            core_count.append(num_cores)
            if not first_loop:
                runtimes_forward.append(max_runtime_forward)
                runtimes_no_forward.append(max_runtime_no_forward)
                runtimes_total.append(max_runtime_total)
            max_runtime_total = 0
            max_runtime_no_forward = 0
            max_runtime_forward = 0
        
        # Check if the line contains the biggest number
        if 'Time with only forward problem:' in line:
            match = re.search(r'\d+\.\d+', line)
            runtime_forward = float(match.group())
            if runtime_forward > max_runtime_forward:
                max_runtime_forward= runtime_forward
        if 'Time without forward problem:' in line:
            match = re.search(r'\d+\.\d+', line)
            runtime_no_forward = float(match.group())
            if runtime_no_forward > max_runtime_no_forward:
                max_runtime_no_forward= runtime_no_forward
        if 'Total time problem:' in line:
            match = re.search(r'\d+\.\d+', line)
            runtime_total = float(match.group())
            if runtime_total > max_runtime_total:
                max_runtime_total = runtime_total
    
    # Getting last loop data
    runtimes_forward.append(max_runtime_forward)
    runtimes_no_forward.append(max_runtime_no_forward)
    runtimes_total.append(max_runtime_total)


# Save data to file with same name as input file plus "data"
output_file = input_file + "data.csv"
with open(output_file , 'w', newline='') as output_file:
    # Create a CSV writer object
    csv_writer = csv.writer(output_file)

    # Write the header row to the output file
    csv_writer.writerow(['Cores Used', 'Runtime pre forward', 'Runtime forward', 'Runtime total'])

    # Write the shot information to the output file
    for i in range(len(core_count)):
        core = core_count[i]
        r_forward = runtimes_forward[i]
        r_no_forward = runtimes_no_forward[i]
        r_total = runtimes_total[i]
        csv_writer.writerow([core, r_no_forward, r_forward, r_total])

