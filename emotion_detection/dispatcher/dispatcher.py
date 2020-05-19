from sklearn import ensemble
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from xgboost import XGBClassifier
from sklearn.svm import SVC


MODELS = {
    "randomforest": ensemble.RandomForestClassifier,
    "naive_bayes": MultinomialNB,
    "xgboost": XGBClassifier,
    "logistic": LogisticRegression,
    "sgd_classifier": SGDClassifier,
    "svm_svc": SVC,
}
