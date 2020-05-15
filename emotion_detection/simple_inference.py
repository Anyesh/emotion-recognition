from .config import config
from .models.test_model import TestModel
import numpy as np
import pickle
import os
from .utils.file_utils import get_latest_file


test_req = [
    "its very unpleasent.",
    "happy me",
    "Very good",
    "this is so disgusting, why it has to be like this.",
]

with open(
    os.path.join(config.CHECKPOINT_PATH, "sklearn_models", "naive_bayes-model.pkl"),
    "rb",
) as f:
    model = pickle.load(f)


clf = TestModel.from_path(
    model=model,
    use_processor=True,
    processor_path=get_latest_file(
        os.path.join(config.CHECKPOINT_PATH, "data_processor")
    ),
)

results = clf.predict(test_req)


label_encoder = np.load(
    get_latest_file(os.path.join(config.CHECKPOINT_PATH, "label_encoder")),
    allow_pickle=True,
)


for idx, val in enumerate(results):
    print(f"{test_req[idx]} - predicted {label_encoder.classes_[val]}.")
