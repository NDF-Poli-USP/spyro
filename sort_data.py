import csv

# initialize a dictionary to store the minimum values for each Cores value
min_values = {}

# open the text file and read its contents as CSV data
with open('strong_scalling_2_nodes.txt', 'r') as f:
    csv_reader = csv.DictReader(f)

    # iterate over each row in the CSV data
    cores = 0
    skip_count = 0
    for row in csv_reader:
        if cores == 5:
            skip_count +=1

        if cores != 5 or skip_count >= 2:
            skip_count = 0
            cores = int(row['Cores Used'])
            pre_forward = float(row['Runtime pre forward'])
            forward = float(row['Runtime forward'])
            total = float(row['Runtime total'])

            # if this is the first row for a given Cores value, store its values as the minimum
            if cores not in min_values:
                min_values[cores] = {
                    'Runtime pre forward': pre_forward,
                    'Runtime forward': forward,
                    'Runtime total': total
                }
            # otherwise, update the minimum values if the current row's values are smaller
            else:
                if pre_forward < min_values[cores]['Runtime pre forward']:
                    min_values[cores]['Runtime pre forward'] = pre_forward
                if forward < min_values[cores]['Runtime forward']:
                    min_values[cores]['Runtime forward'] = forward
                if total < min_values[cores]['Runtime total']:
                    min_values[cores]['Runtime total'] = total

# print the minimum values for each Cores value
for cores, values in min_values.items():
    print(f'Cores: {cores}')
    print(f'Minimum Runtime pre forward: {values["Runtime pre forward"]}')
    print(f'Minimum Runtime forward: {values["Runtime forward"]}')
    print(f'Minimum Runtime total: {values["Runtime total"]}')

# save the minimum values for each Cores value to a CSV file
with open('strong_scalling_2_nodes.csv', 'w', newline='') as f:
    fieldnames = ['Cores', 'Minimum Runtime pre forward', 'Minimum Runtime forward', 'Minimum Runtime total']
    writer = csv.DictWriter(f, fieldnames=fieldnames)

    writer.writeheader()
    for cores, values in min_values.items():
        writer.writerow({
            'Cores': cores,
            'Minimum Runtime pre forward': values['Runtime pre forward'],
            'Minimum Runtime forward': values['Runtime forward'],
            'Minimum Runtime total': values['Runtime total']
        })