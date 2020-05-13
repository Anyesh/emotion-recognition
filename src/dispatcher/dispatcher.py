from sklearn import ensemble
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import MultinomialNB


MODELS = {
    "randomforest": ensemble.RandomForestClassifier(
        n_estimators=200, n_jobs=-1, verbose=2
    ),
    "extra_trees": ensemble.ExtraTreesClassifier(
        n_estimators=200, n_jobs=-1, verbose=2
    ),
    "naive_bayes": OneVsRestClassifier(MultinomialNB()),
}
