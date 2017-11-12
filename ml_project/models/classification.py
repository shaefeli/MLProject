#import numpy as np
#from sklearn.base import BaseEstimator, TransformerMixin
#from sklearn.utils.validation import check_array, check_is_fitted
#from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
#from scipy import stats

#class LDAwithYHandling(BaseEstimator, TransformerMixin):
#    def __init__(self):
#        self.lda = LinearDiscriminantAnalysis();

#    def maxIndexWithSampling(y):
#        chosenIndices = np.empty((y.shape[0],1))
#        for i in range(0,y.shape[0]):
#            rand = random.random()
#            rowY = y[i]
#            cumSum = 0;
#            for j in range(0,rowY.shape[0]):
#                cumSum += rowY[j]
#                if rand<cumSum:
#                    chosenIndices[i] = j
#                    break
#        return chosenIndices
#        def maxIndex(y):
#        y_new = np.empty((y.shape[0],1))
#        for i in range(0,y.shape[0]):
#            maxYIndex = np.argmax(y[i])
#            y_new[i] = maxYIndex
#        return y_new

#    def fit(self, X, y, sample_weight=None):
#        #transform y by just taking the max label
#        y_new = maxIndex(y)
#        self.lda.fit(X, y_new)
#        return self

#    def score(self, X, y, sample_weight=None):
#        return stats.spearmanr(y)

#    def predict_proba(self, X):
#        return self.lda.predict_proba(X)
#    


#class MeanPredictor(BaseEstimator, TransformerMixin):
#    """docstring for MeanPredictor"""
#    def fit(self, X, y):
#        self.mean = y.mean(axis=0)
#        return self

#    def predict_proba(self, X):
#        check_array(X)
#        check_is_fitted(self, ["mean"])
#        n_samples, _ = X.shape
#        return np.tile(self.mean, (n_samples, 1))



import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array, check_is_fitted
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from scipy import stats
import random
import sys

class LDAwithYHandling(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.lda=MLPClassifier()

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
        print("START FITTING")
        sys.stdout.flush();
        chosenIndices = np.argmax(y,axis=1)
        self.lda.fit(X, chosenIndices)
        return self

    def score(self, X, y, sample_weight=None):
        y_p = self.predict_proba(X)
        n_samples=X.shape[0]
        correl=0
        for i in range(0,n_samples):
            correla,_=stats.spearmanr(y[i],y_p[i])
            correl = correl+correla
        return correl/n_samples

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






