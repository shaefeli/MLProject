from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_array, check_is_fitted
from sklearn.utils.random import sample_without_replacement
import numpy as np


class RandomSelection(BaseEstimator, TransformerMixin):
    """Histogram of slices"""
    def __init__(self, n_components=10, random_state=None):
        self.sliceWidth = n_components
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
        sliceW = 20
        nrBins = X.max()-X.min();
        X_new =np.zeros(nrBins);
        for j in range(3):
            dimensionLength = dimensions[j];
            startingPoint = -sliceW;
            endingPoint = 0;
            for i in range (0,n_samples):
                startingPoint += sliceW;
                if startingPoint>=dimensionLength:
                    break;
                endingPoint += sliceW;
                if endingPoint>dimensionLength:
                    endingPoint = dimensionLength;
                toSlice = range(startingPoint,endingPoint);
                if j==0:
                    counts = np.histogram(images[i,toSlice,:,:], nrBins);
                    X_new = np.vstack([X_new, counts[0]]);
                elif j==1:
                    counts = np.histogram(images[i,:,toSlice,:], nrBins);
                    X_new = np.vstack([X_new, counts[0]]);
                elif j==2:
                    counts = np.histogram(images[i,:,:,toSlice], nrBins);
                    X_new = np.vstack([X_new, counts[0]]);
           
        print(X_new.shape)
        print(X_new.reshape(X_new.shape[0], -1).shape)
        return X_new.reshape(X_new.shape[0], -1)
