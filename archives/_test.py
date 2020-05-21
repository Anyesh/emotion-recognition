from emotion_detection.models.test_model import TestModel
from emotion_detection.config import config
from emotion_detection.utils.file_utils import get_latest_file
import pickle
import pandas as pd
import os
from sklearn.metrics import classification_report

processor_path = get_latest_file(
    os.path.join(config.CHECKPOINT_PATH, "data_processor"), "*.pkl"
)

model_name = "naive_bayes"

data = pd.read_csv(os.path.join(config.DATA_PATH, "test_dataset.csv"))
X = data["texts"].values
y = data["emotions"].map(
    {
        "anger": 0,
        "disgust": 1,
        "fear": 2,
        "guilt": 3,
        "joy": 4,
        "sadness": 5,
        "shame": 6,
    }
)

with open(
    os.path.join(config.CHECKPOINT_PATH, "sklearn_models", f"{model_name}-model.pkl"),
    "rb",
) as f:
    model = pickle.load(f)

clf = TestModel.from_path(
    model=model, processor_path=processor_path, use_processor=True
)

results, _ = clf.predict(X)

print(classification_report(results, y))
