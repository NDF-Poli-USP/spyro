import pickle
import numpy as np

old_dt = 0.0001
final_time = 0.9
new_dt = 0.001

shot_ids = [[0], [1], [2], [3], [4], [5], [6], [7], [8], [9], [10], [11]]

for shot_id in shot_ids:
    file_name = "shots/shot_record_"
    array = np.zeros(())
    file_name = file_name + str(shot_id) + ".dat"

    with open(file_name, "rb") as f:
        array = np.asarray(pickle.load(f), dtype=float)

    # Slice the array to get every fourth value along the first dimension
    sliced_array = array[::10, :]

    print("PAUSE")

    with open(file_name, "wb") as f:
        pickle.dump(sliced_array, f)
