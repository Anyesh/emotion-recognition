import glob
import os


def get_latest_file(dir_path, file_type="*"):
    """ Function to get latest file from dir

    Arguments:
        dir_path {[string]} -- [directory path to find file on]

    Keyword Arguments:
        file_type {str} -- [type of file] (default: {"*"})

    Returns:
        [latest_file] -- [path of the latest file]
    """
    list_of_files = glob.glob(os.path.join(dir_path, file_type))
    latest_file = max(list_of_files, key=os.path.getctime)
    return latest_file
