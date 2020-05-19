import pickle
import os


def save_object(checkpoint_path, obj_arr, folder_arr, filename_arr):

    """ Function to save objects
    """

    print(f"[INFO] Saving {' '.join(filename_arr)}")

    for i, v in enumerate(obj_arr):
        with open(
            os.path.join(checkpoint_path, folder_arr[i], f"{filename_arr[i]}.pkl"),
            "wb",
        ) as f:
            pickle.dump(obj_arr[i], f)

    return True
