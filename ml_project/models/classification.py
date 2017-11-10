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
from scipy import stats
import random
import sys

class LDAwithYHandling(BaseEstimator, TransformerMixin):
    def __init__(self,nrClassifiers=100):
        self.nrClassifiers=nrClassifiers
        classifs = np.empty(nrClassifiers,dtype = LinearDiscriminantAnalysis);
        for i in range(0,nrClassifiers):
            print(i);
            sys.stdout.flush();
            classifs[i]=LinearDiscriminantAnalysis();
        self.classifiers = classifs
        print(endInit)
        sys.stdout.flush();

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
        for e in range(0,self.nrClassifiers):
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
            np.ravel(chosenIndices)
            print(e)
            sys.stdout.flush();
            ldaToFit = self.classifiers[e]
            ldaToFit.fit(X, chosenIndices)
        return self

    def score(self, X, y, sample_weight=None):
        return stats.spearmanr(y)

    def predict_proba(self, X):
        y=np.empty((X.shape[0],4));
        for e in range(0,self.nrClassifiers):
            y = y + self.classifiers[e].predict_proba(X)
        return y/self.nrClassifiers;
    


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






