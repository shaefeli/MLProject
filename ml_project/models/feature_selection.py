from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_array
from sklearn.utils.random import sample_without_replacement
import numpy as np


class SliceHistogram(BaseEstimator, TransformerMixin):
    """Histogram of slices"""
    def __init__(self, slice_width=10, random_state=None):
        self.sliceWidth = slice_width
        self.random_state = random_state

    def fit(self, X, y=None):
        X = check_array(X)
        n_samples, n_features = X.shape

        random_state = check_random_state(self.random_state)

        return self

    def transform(self, X, y=None):
        X = check_array(X);

        n_samples, n_features = X.shape;
        images = np.reshape(X, (-1,176,208,176));
        dimensions = [176,208,176];
        sliceW = 100
        nrBins = X.max()-X.min();
        print(nrBins)
        X_new = []
        for i in range(0,n_samples):
            for j in range (3):
                dimensionLength = dimensions[j];
                startingPoint = 0
                endingPoint = sliceW
                while(startingPoint<dimensionLength-1):
                    toSlice = range(startingPoint,endingPoint);
                    if j==0:
                        counts = np.histogram(images[i,toSlice,:,:], nrBins);
                        X_new = np.hstack([X_new, counts[0]])
                    elif j==1:
                        counts = np.histogram(images[i,:,toSlice,:], nrBins);
                        X_new = np.hstack([X_new, counts[0]])
                    elif j==2:
                        counts = np.histogram(images[i,:,:,toSlice], nrBins);
                        X_new = np.hstack([X_new, counts[0]])
                    endingPoint += sliceW;
                    if endingPoint>dimensionLength:
                        endingPoint = dimensionLength;
                    startingPoint += sliceW
                
        return np.reshape(X_new,(n_samples,-1))
