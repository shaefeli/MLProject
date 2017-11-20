import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from scipy import stats
import random


class LDAwithYHandling(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.lda = LinearDiscriminantAnalysis()

    def maxIndexWithSampling(y):
        chosenIndices = np.empty((y.shape[0], 1))
        for i in range(0, y.shape[0]):
            rand = random.random()
            rowY = y[i]
            cumSum = 0
            for j in range(0, rowY.shape[0]):
                cumSum += rowY[j]
                if rand < cumSum:
                    chosenIndices[i] = j
                    break
        return chosenIndices

    def maxIndex(y):
        y_n = np.empty((y.shape[0], 1))
        for i in range(0, y.shape[0]):
            maxYIndex = np.argmax(y[i])
            y_n[i] = maxYIndex
        return y_n

    def fit(self, X, y, sample_weight=None):
        chosenIndices = np.argmax(y, axis=1)
        self.lda.fit(X, chosenIndices)
        return self

    def score(self, X, y, sample_weight=None):
        y_p = self.predict_proba(X)
        n_samples = X.shape[0]
        correl = 0
        for i in range(0, n_samples):
            correla, _ = stats.spearmanr(y[i], y_p[i])
            correl = correl+correla
        return correl/n_samples

    def predict_proba(self, X):
        return self.lda.predict_proba(X)
