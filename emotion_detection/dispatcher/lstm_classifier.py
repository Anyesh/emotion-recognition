
# import tensorflow as tf
# from tensorflow.keras import Sequential
# from tensorflow.keras.callbacks import ModelCheckpoint
# from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout


# class LSTMClassifier:

#     def __init__(self, input_shape, output_shape, optimizer='adam', criterion='binary_crossentropy'):
#         self.input_shape = input_shape
#         self.optimizer = optimizer
#         self.criterion = criterion
#         self.output_shape = output_shape

#     def _build(self):

#         model = Sequential()
#         model.add(Dense(100, input_shape=(
#             self.input_shape, ), activation='relu'))
#         model.add(Dropout(0.5))
#         model.add(Dense(200, activation='relu'))
#         model.add(Dropout(0.5))
#         model.add(Dense(
#             self.output_shape, activation='softmax'))
#         model.compile(loss=self.criterion, optimizer=self.optimizer,
#                         metrics=['accuracy'])
#         return model


