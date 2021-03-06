from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array
import numpy as np


class CubeHistogram(BaseEstimator, TransformerMixin):
    """Histogram of cubes"""
    def __init__(self, cut=9, nrBins=45):
        self.cut = cut
        self.nrBins = nrBins

    def fit(self, X, y=None):
        X = check_array(X)
        n_samples, n_features = X.shape
        return self

    def transform(self, X, y=None):
        X = check_array(X)
        n_samples, n_features = X.shape
        images = np.reshape(X, (-1, 176, 208, 176))
        dimensions = [176, 208, 176]
        cut = self.cut
        cubeX = int(dimensions[0]/cut)
        cubeY = int(dimensions[1]/cut)
        cubeZ = int(dimensions[2]/cut)
        nrBins = self.nrBins
        X_new = np.empty((n_samples, cut*cut*cut*nrBins))
        for i in range(0, n_samples):
            image = images[i, :, :, :]
            for e in range(0, cut):
                for f in range(0, cut):
                    for g in range(0, cut):
                        img = image[e*cubeX:(e+1)*cubeX, f*cubeY:(f+1)*cubeY,
                                    g*cubeZ:(g+1)*cubeZ]
                        counts = np.histogram(img, nrBins)
                        X_new[i, (g+f*cut+e*cut*cut)*nrBins:
                              (g+f*cut+e*cut*cut+1)*nrBins] = counts[0]
        return X_new
