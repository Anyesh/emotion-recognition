from tqdm import tqdm
import requests
import os
from src.config import config


def download_dataset(url):

    """ Download the dataset file from url

    This function will download the dataset file from the given url parameter and store
    in the [data] path at root directory.


    Parameters
    -----------
    url: String
        Valid URL of a file to download.

    Returns
    --------
    Boolean
        False- if file already exists in the directory
        True- otherwise

    """

    response = requests.get(url, stream=True)

    file_path = os.path.join(config.DATA_PATH, config.DATASET_NAME)
    if os.path.exists(file_path):
        print("[ALERT] File already exists!!")
        return False

    with open(file_path, "wb") as handle:
        for data in tqdm(response.iter_content()):
            handle.write(data)
    return True


if __name__ == "__main__":
    download_dataset(config.DATASET_URL)
