import os
import pandas as pd
import pickle


# Define the directory path where the files are located
directory = '/media/alexandre/T7 Shield/Development/mls_chapter_ALL_images/spyro-1'

# Initialize an empty dictionary to store the file data
file_data = {}

# Specify the column names
column_names = ['c', 'dt', 'error', 'runtime']

# Iterate over all files in the directory
for filename in os.listdir(directory):
    if filename.endswith('.txt') and filename.startswith('p'):
        # Read the contents of the file into a DataFrame
        df = pd.read_csv(os.path.join(directory, filename), sep='\s+', names=column_names)

        # Round the 'C' column to 2 decimal places
        df['c'] = df['c'].round(2)

        # Form the new key
        new_key = 'ml' + filename[1] + 't'

        # Check if the key already exists in the dictionary
        if new_key in file_data:
            # If it does, append the new DataFrame to the existing one
            file_data[new_key] = pd.concat([file_data[new_key], df])
        else:
            # If it doesn't, store the DataFrame in the dictionary using the filename as the key
            file_data[new_key] = df

        # eliminating repeated Cs and using the ones with lowest runtimes
        filtered_df = file_data[new_key]
        # Sort the DataFrame by 'c' and 'runtime' columns
        filtered_df = filtered_df.sort_values(['c', 'runtime'])

        # Drop duplicate rows based on the 'c' column and keep the first occurrence
        filtered_df = filtered_df.drop_duplicates(subset='c', keep='first') # Sort the DataFrame by 'c' and 'runtime' columns
        filtered_df = filtered_df.sort_values(['c', 'runtime'])

        # Drop duplicate rows based on the 'c' column and keep the first occurrence
        filtered_df = filtered_df.drop_duplicates(subset='c', keep='first')

        file_data[new_key] = filtered_df

# Print the dictionary
for key, value in file_data.items():
    print(f"{key}:\n{value}\n")

# Save the dictionary to a file
with open('ml_results.pkl', 'wb') as f:
    pickle.dump(file_data, f)

print("END")
