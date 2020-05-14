import glob
import os


def get_latest_file(dir_path, file_type="*"):
    list_of_files = glob.glob(os.path.join(dir_path, file_type))
    latest_file = max(list_of_files, key=os.path.getctime)
    return latest_file
