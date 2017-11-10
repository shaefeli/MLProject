import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array, check_is_fitted
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from scipy import stats

class LDAwithYHandling(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.lda = LinearDiscriminantAnalysis();

    def maxIndexWithSampling(y):
        chosenIndices = np.empty((y.shape[0],1))
        for i in range(0,y.shape[0]):
            rand = random.random()
            rowY = y[i]
            cumSum = 0;
            for j in range(0,rowY.shape[0]):
                cumSum += rowY[j]
                if rand<cumSum:
                    chosenIndices[i] = j
                    break
        return chosenIndices

    def maxIndex(y):
        y_n = np.empty((y.shape[0],1))
        for i in range(0,y.shape[0]):
            maxYIndex = np.argmax(y[i])
            y_n[i] = maxYIndex
        return y_n

    def fit(self, X, y, sample_weight=None):
        chosenIndices = np.empty((y.shape[0],1))
        for i in range(0,y.shape[0]):
            rand = random.random()
            rowY = y[i]
            cumSum = 0;
            for j in range(0,rowY.shape[0]):
                cumSum += rowY[j]
                if rand<cumSum:
                    chosenIndices[i] = j
                    break
        self.lda.fit(X, chosenIndices)
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




