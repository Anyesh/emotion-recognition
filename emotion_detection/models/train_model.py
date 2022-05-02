from emotion_detection.dispatcher import dispatcher
from emotion_detection.features import feature_generator
from emotion_detection.config import config
import numpy as np
import os
import pickle
from emotion_detection.utils.serializer import save_object


# epochs=5,
# batch_size=16,
# validation_split=0.1,


class TrainModel:
    def __init__(
        self, model_name, vocab_size, engine, **kwargs,
    ):
        self._engine = engine
        self._model_name = model_name
        self._model = dispatcher.MODELS[model_name](**kwargs)
        self._vocab_size = vocab_size
        # self._epochs = epochs
        # self._batch_size = batch_size
        # self._validation_split = validation_split

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

    def train(self, **kwargs):
        """
        """

        history = None
        clf = None

        clf, history = getattr(
            self, f"_{str(self._engine).lower()}", lambda **kwargs: "Invalid!"
        )(**kwargs)


        return clf, history

    @classmethod
    def sklearn(cls, model_name, vocab_size, **kwargs):

        return cls(model_name, vocab_size, engine="SKLEARN", **kwargs)

    # @classmethod
    # def keras(cls, model_name, vocab_size, **kwargs):

    #     return cls(model_name, vocab_size, engine="KERAS", **kwargs)


if __name__ == "__main__":
    TrainModel.sklearn(model_name="naive_bayes", vocab_size=1000)
