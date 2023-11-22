import os
import pandas as pd
import pickle


# Define the directory path where the files are located
directory = '/media/olender/T7 Shield/Development/mls_chapter_ALL_images/spyro-1'

# Initialize an empty dictionary to store the file data
file_data = {}

# Specify the column names
column_names = ['c', 'dt', 'error', 'runtime']

# Iterate over all files in the directory
for filename in os.listdir(directory):
    if filename.endswith('.txt') and filename.startswith('p'):
        # Read the contents of the file into a DataFrame
        df = pd.read_csv(os.path.join(directory, filename), sep='\s+', names=column_names)

        # Form the new key
        new_key = 'ml' + filename[1] + 't'

        # Check if the key already exists in the dictionary
        if new_key in file_data:
            # If it does, append the new DataFrame to the existing one
            file_data[new_key] = pd.concat([file_data[new_key], df])
        else:
            # If it doesn't, store the DataFrame in the dictionary using the filename as the key
            file_data[new_key] = df

# Print the dictionary
for key, value in file_data.items():
    print(f"{key}:\n{value}\n")

# Save the dictionary to a file
with open(new_key+'.pkl', 'wb') as f:
    pickle.dump(file_data, f)

print("END")
