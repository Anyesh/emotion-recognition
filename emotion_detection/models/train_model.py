from emotion_detection.dispatcher import dispatcher
from emotion_detection.features import feature_generator
from emotion_detection.config import config
import numpy as np
import os
import pickle
from emotion_detection.utils.serializer import save_object


class TrainModel:
    def __init__(
        self,
        model_name,
        vocab_size,
        engine,
        epochs=5,
        batch_size=16,
        validation_split=0.1,
        **kwargs,
    ):
        self._engine = engine
        self._model_name = model_name
        self._model = dispatcher.MODELS[model_name](**kwargs)
        self._vocab_size = vocab_size
        self._epochs = epochs
        self._batch_size = batch_size
        self._validation_split = validation_split

    def train(self, train_data, train_label):

        history = None
        clf = None
        self._train_data = train_data
        self._train_label = train_label

        if self._engine == "SKLEARN":
            print(f"[INFO] Training on {self._model_name} model..")
            clf = self._model.fit(self._train_data, self._train_label)

            save_object(
                config.CHECKPOINT_PATH,
                [self._model],
                ["sklearn_models"],
                [f"{self._model_name}-model"],
            )

        elif self._engine == "KERAS":

            clf = self._model._build()
            print("[INFO] Training DL model..")
            history = clf.fit(
                self._train_data,
                self._train_label,
                epochs=self._epochs,
                batch_size=self._batch_size,
                validation_split=self._validation_split,
            )
            # clf.evaluate(test_data, test_label, batch_size=16)

            print("[INFO] Saving model..")
            clf.save(
                os.path.join(config.CHECKPOINT_PATH, "keras_models", "saved_model.h5")
            )

        print(f"[INFO] Model saved in {config.CHECKPOINT_PATH}")
        return clf, history

    @classmethod
    def sklearn(cls, model_name, vocab_size, **kwargs):

        return cls(model_name, vocab_size, engine="SKLEARN", **kwargs)

    @classmethod
    def keras(cls, model_name, vocab_size, **kwargs):

        return cls(model_name, vocab_size, engine="KERAS", **kwargs)


if __name__ == "__main__":
    TrainModel.sklearn(model_name="naive_bayes", vocab_size=1000)
