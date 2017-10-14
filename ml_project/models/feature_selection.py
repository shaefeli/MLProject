from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_array
from sklearn.utils.random import sample_without_replacement
import numpy as np
import sys

class CubeHistogram(BaseEstimator, TransformerMixin):
    """Histogram of slices"""
    def __init__(self, cut = 9, nrBins = 45,  random_state=None):
        self.cut = cut
        self.random_state = random_state
        self.nrBins = nrBins
        

    def fit(self, X, y=None):
        X = check_array(X)
        n_samples, n_features = X.shape
        print("fit")
        sys.stdout.flush()
        random_state = check_random_state(self.random_state)
        return self

    def transform(self, X, y=None):
        X = check_array(X);
        n_samples, n_features = X.shape;
        images = np.reshape(X, (-1,176,208,176));
        dimensions = [176,208,176];
        cut = self.cut
        cubeX = int(dimensions[0]/cut);
        cubeY = int(dimensions[1]/cut);
        cubeZ = int(dimensions[2]/cut);
        nrBins = self.nrBins
        X_new = np.empty((n_samples,cut*cut*cut*nrBins))
        for i in range(0,n_samples):
            image = images[i,:,:,:]
            image = np.extract(image>50 and image<1900,image)
            for e in range(0,cut):
                for f in range(0,cut):
                    for g in range(0,cut):
                        img = image[e*cubeX:(e+1)*cubeX,f*cubeY:(f+1)*cubeY,g*cubeZ:(g+1)*cubeZ]
                        counts = np.histogram(img, nrBins);
                        X_new[i,(g+f*cut+e*cut*cut)*nrBins:(g+f*cut+e*cut*cut+1)*nrBins]=counts[0]     
        return X_new


class SliceHistogram(BaseEstimator, TransformerMixin):
    """Histogram of slices"""
    def __init__(self, slice_width=10, nrBins = 4000,  random_state=None):
        self.sliceWidth = slice_width
        self.random_state = random_state
        self.nrBins = nrBins

    def fit(self, X, y=None):
        X = check_array(X)
        n_samples, n_features = X.shape
        print("fit")
        sys.stdout.flush
        random_state = check_random_state(self.random_state)
        sys.stdout.flush()
        return self

    def transform(self, X, y=None):
        X = check_array(X);
        sys.stdout.flush()
        n_samples, n_features = X.shape;
        images = np.reshape(X, (-1,176,208,176));
        dimensions = [176,208,176];
        sliceW = 100
        nrBins = self.nrBins
        X_new = []
        for i in range(0,n_samples):
            sys.stdout.flush()
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
