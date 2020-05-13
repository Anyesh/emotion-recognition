from tensorflow.keras.preprocessing import text
from sklearn.feature_extraction.text import TfidfVectorizer
from src.utils.tokenizer import nltk_tokenizer, nltk_tokenizer_keras
import numpy as np

import pickle
import os


class KerasTextPreprocessor:
    def __init__(self, vocab_size):
        self._vocab_size = vocab_size
        self._tokenizer = None

    def create_tokenizer(self, df_corpus):

        ## Removing stopwords using nltk
        clean_corpus = df_corpus.apply(nltk_tokenizer_keras)
        tokenizer = text.Tokenizer(num_words=self._vocab_size)
        tokenizer.fit_on_texts(clean_corpus.values)
        self._tokenizer = tokenizer

    def transform_text(self, text_list):
        text_matrix = self._tokenizer.texts_to_matrix(text_list)
        return text_matrix


class TFIDFProcessor:
    def __init__(
        self, feature_size=None, use_n_grams=False, n_grams=None,
    ):

        self._feature_size = feature_size
        self._vocabulary = None
        self._n_grams = n_grams
        self._use_n_grams = use_n_grams

        self._tokenizer = nltk_tokenizer

    def create_vocab(self, corpus):
        _vectorizer = TfidfVectorizer(
            tokenizer=self._tokenizer, max_features=self._feature_size,
        )

        self._vectorizer = _vectorizer.fit(corpus)
        self._vocabulary = self._vectorizer.vocabulary_

    def transform_text(self, text_list):

        text_matrix = self._vectorizer.transform(text_list).toarray()
        return text_matrix
