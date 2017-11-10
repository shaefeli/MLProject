import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array, check_is_fitted
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from scipy import stats

class LDAwithYHandling(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.lda = LinearDiscriminantAnalysis();

    def fit(self, X, y, sample_weight=None):
        #transform y by just taking the max label
        y_new = np.empty((y.shape[0],1))
        for i in range(0,y.shape[0]):
            maxYIndex = np.argmax(y[i])
            y_new[i] = maxYIndex
        self.lda.fit(X, y_new)
        return self

    def score(self, X, y, sample_weight=None):
        return stats.spearmanr(y)

    def predict_proba(self, X):
        return self.lda.predict_proba(X)

class MeanPredictor(BaseEstimator, TransformerMixin):
    """docstring for MeanPredictor"""
    def fit(self, X, y):
        self.mean = y.mean(axis=0)
        return self

    def predict_proba(self, X):
        check_array(X)
        check_is_fitted(self, ["mean"])
        n_samples, _ = X.shape
        return np.tile(self.mean, (n_samples, 1))




