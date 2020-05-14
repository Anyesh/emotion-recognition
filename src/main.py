from .features.feature_generator import process_data
from .models.train_model import TrainModel

# from .models.test_model import TestModel
from .config import config
from sklearn.metrics import accuracy_score, f1_score
from .utils.serializer import save_object
import numpy as np
import os
import mlflow
import mlflow.sklearn


VOCAB_SIZE = 2000
MODEL_NAME = "xgboost"
processor_engine = "SKLEARN"


if __name__ == "__main__":
    model_params = {"model_name": MODEL_NAME, "vocab_size": VOCAB_SIZE, "gpu_id": 0}

    mlflow.log_params(model_params)

    (
        train_data,
        train_label,
        test_data,
        test_label,
        processor,
        label_encoder,
    ) = process_data(
        config.DATA_PATH,
        config.DATASET_NAME,
        VOCAB_SIZE,
        processor_engine=processor_engine,
        train_size=0.7,
    )

    save_status = save_object(
        config.CHECKPOINT_PATH,
        [processor, label_encoder],
        ["data_processor", "label_encoder"],
        ["sklearn_data_processor", "label_encoder"],
    )

    trainer = TrainModel.sklearn(**model_params)
    clf, history = trainer.train(train_data, train_label)
    print(clf)
    # test_clf = TestModel.from_path(clf)
    predictions = clf.predict(test_data)

    accuracy_score = accuracy_score(predictions, test_label)
    f1_score = f1_score(predictions, test_label, average="weighted")

    print("Accuracy score: ", accuracy_score)
    print("F1 score: ", f1_score)

    mlflow.log_metric("Accuracy score", accuracy_score)
    mlflow.log_metric("F1 score", f1_score)
    mlflow.sklearn.log_model(clf, MODEL_NAME)

    np.save(
        os.path.join(
            config.CHECKPOINT_PATH,
            "frozen_data",
            f"{processor_engine}-data-{VOCAB_SIZE}",
        ),
        [train_data, train_label, test_data, test_label],
    )
