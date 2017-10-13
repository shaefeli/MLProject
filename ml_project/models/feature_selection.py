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
        sys.stdout.flush
        random_state = check_random_state(self.random_state)
        sys.stdout.flush()
        return self

    def transform(self, X, y=None):
        X = check_array(X);
        n_samples, n_features = X.shape;
        images = np.reshape(X, (-1,176,208,176));
        dimensions = [176,208,176];
        cut = 9
        cubeX = int(dimensions[0]/cut);
        print(cubeX)
        #restX = dimensions[0]-cubeX*cut
        cubeY = int(dimensions[1]/cut);
        print(cubeY)
        #restY = dimensions[1]-cubeY*cut
        cubeZ = int(dimensions[2]/cut);
        print(cubeZ)
        #restZ = dimensions[2]-cubeZ*cut
        nrBins = 45
        X_new = np.empty((n_samples,cubeX*cubeY*cubeZ*nrBins))
        print(X_new.shape)
        for i in range(0,n_samples):
            print(i)
            image = images[i,:,:,:]
            for e in range(0,cubeX):
                region1 = image[e*cubeX:(e+1)*cubeX,:,:]
                sys.stdout.flush()
                for f in range(0,cubeY):
                    region2 = region1[:, f*cubeY:(f+1)*cubeY,:]
                    for g in range(0,cubeZ):
                        counts = np.histogram(region2[:,:,g*cubeZ:(g+1)*cubeZ], nrBins);
                        X_new[i,(g+f*cubeZ+e*cubeY)*nrBins:(g+f*cubeZ+e*cubeY+1)*nrBins]=counts[0]     

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
