import pickle
import os
import numpy as np


class TestModel(object):
    def __init__(self, model, processor, use_processor):
        self._model = model
        self._processor = processor
        self._use_processor = use_processor

    def predict(self, instances):
        processed_data = instances

        if self._use_processor:
            processed_data = self._processor.transform_text(instances)

        try:
            predictions = self._model.predict(processed_data)

            try:
                proba = self._model.predict_proba(processed_data)
            except:
                proba = np.zeros((2, self._model.classes_.shape[0]))

            classes = self._model.classes_
            probabilities = dict(zip(classes, proba[0]))

            return predictions, probabilities
        except Exception as e:
            print("Invalid model !!", e)

    def predict_proba(self, instances):

        processed_data = self._processor.transform_text(instances)

        proba = self._model.predict_proba(processed_data)

        classes = self._model.classes_
        probabilities = dict(zip(classes, proba[0]))

        return proba

    @classmethod
    def from_path(cls, model, processor_path=None, use_processor=False):

        processor = None

        if use_processor:
            with open(processor_path, "rb",) as f:
                processor = pickle.load(f)

        return cls(model, processor, use_processor)

    # @classmethod
    # def keras(cls, model_path):
    #     import tensorflow.keras as keras

    #     model = keras.models.load_model(model_path)
    #     with open(
    #         os.path.join(config.CHECKPOINT_PATH, "keras-data_processor_state.pkl"), "rb"
    #     ) as f:
    #         processor = pickle.load(f)
    #     return cls(model, processor)
