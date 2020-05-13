import pickle
import os


class TestModel(object):
    def __init__(self, model, processor):
        self._model = model
        self._processor = processor

    def predict(self, instances, **kwargs):
        processed_data = self._processor.transform_text(instances)
        predictions = self._model.predict(processed_data)
        return predictions.tolist()

    @classmethod
    def from_path(cls, mode_dir):
        import tensorflow.keras as keras

        model = keras.models.load_model(os.path.join("checkpoints", "classifier.h5"))
        with open(os.path.join("checkpoints", "processor_state.pkl"), "rb") as f:
            processor = pickle.load(f)
        return cls(model, processor)
