from .features.feature_generator import process_data
from .models.train_model import TrainModel
from .models.test_model import TestModel
from .config import config, model_params
from sklearn.metrics import accuracy_score, f1_score
from .utils.serializer import save_object
from .utils.file_utils import get_latest_file
from .utils.colors import bcolors

import numpy as np
import os
import mlflow
import mlflow.sklearn
import pickle


def train_pipeline(model_name, vocab_size, train_size=0.7):

    """ Train pipeline function
    """

    processor_engine = "SKLEARN"
    input_shape = vocab_size

    try:
        _model_params = model_params.parameters[model_name]
    except Exception as e:
        print("Invalid model name passed", e)
        return False

    _essential_params = {
        "model_name": model_name,
        "vocab_size": vocab_size,
        **_model_params,
    }

    ## Keras trainer
    # trainer = TrainModel.keras(**_model_params)

    ## Sklearn trainer
    trainer = TrainModel.sklearn(**_essential_params)

    mlflow.log_params(_essential_params)

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
        vocab_size,
        train_size,
        processor_engine=processor_engine,
    )

    save_status = save_object(
        config.CHECKPOINT_PATH,
        [processor, label_encoder],
        ["data_processor", "label_encoder"],
        ["sklearn_data_processor", "label_encoder"],
    )

    kwargs = {"train_data": train_data, "train_label": train_label}

    clf, history = trainer.train(**kwargs)
    print(f"{bcolors.OKBLUE}{clf} {bcolors.ENDC}")
    # test_clf = TestModel.from_path(clf)
    predictions = clf.predict(test_data)

    score = accuracy_score(predictions, test_label)
    f1 = f1_score(predictions, test_label, average="weighted")

    print(f"{bcolors.OKGREEN }Test accuracy score: {bcolors.ENDC}", score)
    print(f"{bcolors.OKGREEN }Test f1 score: {bcolors.ENDC}", f1)

    mlflow.log_metric("Accuracy score", score)
    mlflow.log_metric("F1 score", f1)
    mlflow.sklearn.log_model(clf, model_name)

    print(f"{bcolors.WARNING }[INFO] Freezing data.. {bcolors.ENDC}")

    np.save(
        os.path.join(
            config.CHECKPOINT_PATH,
            "frozen_data",
            f"{processor_engine}-data-{vocab_size}",
        ),
        [train_data, train_label, test_data, test_label],
    )


def test_pipeline(model_name, input_text):
    """ Test pipeline function
    """

    model = None
    processor_path = get_latest_file(
        os.path.join(config.CHECKPOINT_PATH, "data_processor"), "*.pkl"
    )

    with open(
        os.path.join(
            config.CHECKPOINT_PATH, "sklearn_models", f"{model_name}-model.pkl"
        ),
        "rb",
    ) as f:
        model = pickle.load(f)

    clf = TestModel.from_path(
        model=model, processor_path=processor_path, use_processor=True
    )

    results, probabilities = clf.predict([input_text])

    label_encoder = np.load(
        get_latest_file(os.path.join(config.CHECKPOINT_PATH, "label_encoder")),
        allow_pickle=True,
    )
    probabilities = dict(
        zip(
            label_encoder.inverse_transform(list(probabilities.keys())),
            list(probabilities.values()),
        )
    )
    ## Lime explainer added

    try:
        from lime.lime_text import LimeTextExplainer

        explainer = LimeTextExplainer(
            class_names=["anger", "disgust", "fear", "guilt", "joy", "sadness", "shame"]
        )
        exp = explainer.explain_instance(
            input_text, clf.predict_proba, num_features=7, top_labels=2
        )
    except:
        print("Error on explainer!")
        exp = None
    ## End
    return label_encoder.classes_[results], probabilities, exp
