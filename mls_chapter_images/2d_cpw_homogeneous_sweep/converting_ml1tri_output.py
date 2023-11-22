data = []
current_row = []

with open('intel_ml1tri_cpw.24877.out', 'r') as file:
    for line in file:
        if line.startswith('Trying cells-per-wavelength '):
            current_row.append(float(line.split()[-1]))
        elif line.startswith('Maximum dt is '):
            current_row.append(float(line.split()[-1]))
        elif line.startswith('Error is '):
            current_row.append(float(line.split()[-1]))
            current_row.append(0)
            data.append(current_row)
            current_row = []

with open('p1_cpw_results_all.txt', 'w') as file:
    for row in data:
        file.write(' '.join(map(str, row)) + '\n')
