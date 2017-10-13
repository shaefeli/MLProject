from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_array, check_is_fitted
from sklearn.utils.random import sample_without_replacement
import numpy as np


class RandomSelection(BaseEstimator, TransformerMixin):
    """Random Selection of features"""
    def __init__(self, n_components=1000):
        self.n_components = n_components

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        #check_is_fitted(self, ["components"])
        X = check_array(X)
        n_samples, n_features = X.shape
	
        maxValue=4420
        X_new =[]
        for i in range (0,n_samples):
                brain = X[i,:]
                counts = np.histogram(brain, maxValue, (1,maxValue))
                X_new.append(counts[0])

        #X_new = X[:, self.components]

        return X_new
