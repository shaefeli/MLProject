from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_array, check_is_fitted
from sklearn.utils.random import sample_without_replacement
import numpy as np


class HistogramSlice(BaseEstimator, TransformerMixin):
    """Histogram of slices"""
    def __init__(self, sliceWidth=10, random_state=None):
        self.sliceWidth = sliceWidth
        self.random_state = random_state
        #self.components = None

    def fit(self, X, y=None):
        X = check_array(X)
        n_samples, n_features = X.shape

        random_state = check_random_state(self.random_state)
#        self.components = sample_without_replacement(
#                            n_features,
#                            self.n_components,
#                            random_state=random_state)

        return self

    def transform(self, X, y=None):
        #check_is_fitted(self, ["components"])
        X = check_array(X);
        n_samples, n_features = X.shape;
        
        images = np.reshape(X, (-1,176,208,176));
        dimensions = [176,208,176];
        sliceWidth = self.sliceWidth;
        nrBins = X.max()-X.min();
        X_new =[];
        startingPoint = -sliceWidth;
        endingPoint = 0;
        for j in range(3):
            dimensionLength = dimensions(j);
            startingPoint += sliceWidth;
            if startingPoint>=dimensionLength:
                break;
            endingPoint += sliceWidth;
            if endingPoint>dimensionLength:
                endingPoint = dimensionLength;
            toSlice = range(startingPoint,endingPoint);
            for i in range (0,n_samples):
                if i=0:
                    counts = np.histogram(images[i,toSlice,:,:], nrBins);
                    X_new.append(counts[0]);
                else if i=1:
                    counts = np.histpgram(images[i,:,toSlice,:], nrBins);
                    X_new.append(counts[0]);
                else if i=2:
                    counts = np.histpgram(images[i,:,:,toSlice], nrBins);
                    X_new.append(counts[0]);
                             
       
        return X_new;
