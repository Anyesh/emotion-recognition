from sklearn import ensemble
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from xgboost import XGBClassifier
from .neuralnet import EmotionClassifier


MODELS = {
    "randomforest": ensemble.RandomForestClassifier,
    "naive_bayes": MultinomialNB,
    "xgboost": XGBClassifier,
    "emotion_classifier": EmotionClassifier,
    "logistic": LogisticRegression,
    "sgd": SGDClassifier,
}
