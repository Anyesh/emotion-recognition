# from tensorflow.keras.preprocessing import text
# from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import pickle
import os


# class KerasTextPreprocessor:
#     def __init__(self, vocab_size):
#         self._vocab_size = vocab_size
#         self._tokenizer = None

#     def create_tokenizer(self, corpus):

#         tokenizer = text.Tokenizer(num_words=self._vocab_size)
#         tokenizer.fit_on_texts(corpus)
#         self._tokenizer = tokenizer

#     def transform_text(self, text_list):
#         text_sequence = self._tokenizer.texts_to_sequences(text_list)
#         padded_sequence = pad_sequences(
#             text_sequence, maxlen=self._vocab_size, padding="post"
#         )

#         return padded_sequence


class TFIDFProcessor:
    def __init__(
        self, feature_size=None, n_grams=None,
    ):

        self._feature_size = feature_size
        self._vectorizer = None
        self._n_grams = n_grams

    def create_vocab(self, corpus):
        _vectorizer = TfidfVectorizer(max_features=self._feature_size)

        self._vectorizer = _vectorizer.fit(corpus)
        # self._vocabulary = self._vectorizer.vocabulary_

    def transform_text(self, text_list):

        return self._vectorizer.transform(text_list).toarray()
