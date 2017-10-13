from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_array, check_is_fitted
from sklearn.utils.random import sample_without_replacement
import numpy as np


class HistogramSlices(BaseEstimator, TransformerMixin):
    """Create histogram out of all image"""
    def __init__(self, sliceWidth=20, random_state=None):
        self.sliceWidth = n_components
        self.random_state = sliceWidth

    def fit(self, X, y=None):
        X = check_array(X)
        n_samples, n_features = X.shape
        random_state = check_random_state(self.random_state)
        return self

    def transform(self, X, y=None):
        check_is_fitted(self, ["components"])
        X = check_array(X)
        n_samples, n_features = X.shape
	
	
        maxValue=4420
	nrBins = maxValue/sliceWidth;
        X_new =[]
        for i in range (0,n_samples):
                brain = X[i,:]
                counts = np.histogram(brain, nrBins)
                X_new.append(counts[0])


        return X_new
