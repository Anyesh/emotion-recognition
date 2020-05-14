import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy


class EmotionClassifier:
    def __init__(self, input_shape, output_shape):
        self.input_shape = input_shape
        self.optimizer = Adam(
            learning_rate=0.001,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-07,
            amsgrad=False,
            name="Adam",
        )
        self.criterion = BinaryCrossentropy(
            from_logits=False, label_smoothing=0, name="binary_crossentropy"
        )
        self.output_shape = output_shape

    def _build(self):
        model = tf.keras.models.Sequential()
        model.add(
            tf.keras.layers.Dense(
                50, input_shape=(self.input_shape,), activation="relu"
            )
        )
        model.add(tf.keras.layers.Dense(self.output_shape, activation="sigmoid"))
        model.compile(
            loss=self.criterion, optimizer=self.optimizer, metrics=["accuracy"]
        )
        return model
