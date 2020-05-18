from emotion_detection.dispatcher import dispatcher
from emotion_detection.features import feature_generator
from emotion_detection.config import config
from emotion_detection.utils.serializer import save_object
from sklearn.metrics import accuracy_score, f1_score, classification_report
import numpy as np
import os
import pickle


class TrainModel:
    """

    """

    def __init__(
        self,
        model_name,
        engine,
        epochs=50,
        batch_size=128,
        validation_split=0.1,
        **kwargs,
    ):
        self._engine = engine
        self._model_name = model_name
        self._model = dispatcher.MODELS[model_name](**kwargs)

        self._epochs = epochs
        self._batch_size = batch_size
        self._validation_split = validation_split

    # def _keras(self, **kwargs):
    #     """

    #     """
    #     train_data = kwargs["train_data"]
    #     train_label = kwargs["train_label"]
    #     clf = self._model._build()
    #     print("[INFO] Training DL model..")
    #     history = clf.fit(
    #         train_data,
    #         train_label,
    #         epochs=self._epochs,
    #         batch_size=self._batch_size,
    #         validation_split=self._validation_split,
    #     )
    #     # clf.evaluate(test_data, test_label, batch_size=16)

    #     print("[INFO] Saving model..")
    #     clf.save(os.path.join(config.CHECKPOINT_PATH, "keras_models", "saved_model.h5"))

    #     return clf, history

    def _sklearn(self, **kwargs):
        """

        """

        train_data = kwargs["train_data"]
        train_label = kwargs["train_label"]

        print(f"[INFO] Training on {self._model_name} model..")
        clf = self._model.fit(train_data, train_label)

        save_object(
            config.CHECKPOINT_PATH,
            [self._model],
            ["sklearn_models"],
            [f"{self._model_name}-model"],
        )

        return clf, clf

    def _bert(self, **kwargs):
        """

        """
        import torch
        from transformers import AdamW, get_linear_schedule_with_warmup
        from .bert_engine import engine

        device = torch.device("cuda")
        model = self._model
        model.to(device)
        train_data_loader = kwargs["train_data_loader"]
        test_data_loader = kwargs["test_data_loader"]

        optimizers_parameters = list(model.named_parameters())
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        optimizers_parameters = [
            {
                "params": [
                    p
                    for n, p in optimizers_parameters
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_deacy": 0.001,
            },
            {
                "params": [
                    p
                    for n, p in optimizers_parameters
                    if any(nd in n for nd in no_decay)
                ],
                "weight_deacy": 0.00,
            },
        ]

        num_train_steps = len(train_data_loader) / self._batch_size * self._epochs

        optimizers = AdamW(optimizers_parameters, lr=3e-5)
        scheduler = get_linear_schedule_with_warmup(
            optimizers, num_warmup_steps=0, num_training_steps=num_train_steps
        )

        best_accuracy = 0
        for epoch in range(self._epochs):
            engine.train(train_data_loader, model, optimizers, device, scheduler)
            outputs, targets = engine.eval(test_data_loader, model, device)

            # outputs = np.array(outputs) >= 0.5
            accuracy = accuracy_score(
                np.argmax(targets, axis=1), np.argmax(outputs, axis=1)
            )
            print("Validation accuracy: ", accuracy)
            if accuracy > best_accuracy:
                torch.save(model.state_dict(), config.MODEL_PATH)
                best_accuracy = accuracy

        return model, None

    def train(self, **kwargs):
        """

        """

        history = None
        clf = None

        clf, history = getattr(
            self, "_" + str(self._engine).lower(), lambda **kwargs: "Invalid!"
        )(**kwargs)

        return clf, history

    @classmethod
    def sklearn(cls, model_name, vocab_size, **kwargs):

        return cls(model_name, vocab_size, engine="SKLEARN", **kwargs)

    # @classmethod
    # def keras(cls, model_name, vocab_size, **kwargs):

    #     return cls(model_name, vocab_size, engine="KERAS", **kwargs)

    @classmethod
    def bert(cls, model_name, vocab_size, **kwargs):

        return cls(model_name, vocab_size, engine="BERT", **kwargs)


if __name__ == "__main__":
    TrainModel.sklearn(model_name="naive_bayes", vocab_size=1000)
