# from tensorflow.keras.preprocessing import text
# from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.feature_extraction.text import TfidfVectorizer
from emotion_detection.config import config
import torch

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
#         text_sequence = self._tokenizer.texts_to_matrix(text_list)
#         # padded_sequence = pad_sequences(
#         #     text_sequence, maxlen=self._vocab_size, padding="post"
#         # )

#         return text_sequence


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

        text_matrix = self._vectorizer.transform(text_list).toarray()
        return text_matrix


class BERTProcessor:
    def __init__(self, features, target, vocab_size):
        self.features = features
        self.target = target
        self.tokenizer = config.TOKENIZER
        self.max_len = vocab_size

    def __len__(self):
        return len(self.features)

    def __getitem__(self, item):
        features = str(self.features)
        features = " ".join(features.split())

        input = self.tokenizer.encode_plus(
            features, None, add_special_token=True, max_length=self.max_len
        )

        ids = input["input_ids"]
        mask = input["attention_mask"]
        token_type_ids = input["token_type_ids"]
        padding_length = self.max_len - len(ids)
        ids = ids + ([0] * padding_length)
        mask = mask + ([0] * padding_length)
        token_type_ids = token_type_ids + ([0] * padding_length)

        return {
            "ids": torch.tensor(ids, dtype=torch.long),
            "mask": torch.tensor(mask, dtype=torch.long),
            "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
            "target": torch.tensor(self.target[item], dtype=torch.long),
        }
