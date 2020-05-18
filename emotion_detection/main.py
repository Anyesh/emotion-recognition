from .features.feature_generator import process_data
from .models.train_model import TrainModel

# from .models.test_model import TestModel
from .config import config, model_params
from sklearn.metrics import accuracy_score, f1_score
from .utils.serializer import save_object
import numpy as np
import os
import mlflow
import mlflow.sklearn


# train_data_loader, test_data_loader = process_data(
#     config.DATA_PATH,
#     config.DATASET_NAME,
#     config.MAX_LEN,
#     processor_engine=processor_engine,
#     train_size=0.8,
#     train_batch_size=config.TRAIN_BATCH_SIZE,
#     test_batch_size=config.VALID_BATCH_SIZE,
# )

# trainer = TrainModel.bert(
#     model_name="bert_classifier",
#     vocab_size=config.MAX_LEN,
#     batch_size=config.TRAIN_BATCH_SIZE,
#     epochs=config.EPOCHS,
# )

# clf, _ = trainer.train(
#     train_data_loader=train_data_loader, test_data_loader=test_data_loader
# )
# print(clf)


def main(processor_engine, model_name, input_shape, output_shape):

    VOCAB_SIZE = input_shape
    _model_params = model_params["model_name"]

    _essential_params = {
        "input_shape": input_shape,
        "output_shape": output_shape,
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
        input_shape,
        processor_engine=processor_engine,
        train_size=0.7,
    )

    save_status = save_object(
        config.CHECKPOINT_PATH,
        [processor, label_encoder],
        ["data_processor", "label_encoder"],
        ["sklearn_data_processor", "label_encoder"],
    )

    kwargs = {"train_data": train_data, "train_label": train_label}

    clf, history = trainer.train(**kwargs)
    print(clf)
    # test_clf = TestModel.from_path(clf)
    predictions = clf.predict(test_data)
    print(predictions)

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


if __name__ == "__main__":
    main(processor_engine, model_name, input_shape, output_shape)
