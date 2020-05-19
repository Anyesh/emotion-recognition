import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DATA_PATH = os.path.join(BASE_DIR, "data", "raw")

DATASET_NAME = "ISEAR_dataset.csv"

DATASET_URL = "https://www.floydhub.com/api/v1/resources/qM4BHN3pNjkfkvYjMMtjU4/ISEAR.csv?content=true&rename=isearcsv"

MODEL_PATH = os.path.join(BASE_DIR, "models")

CHECKPOINT_PATH = os.path.join(BASE_DIR, "checkpoints")
