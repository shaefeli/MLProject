
from sklearn.utils import check_random_state
from sklearn.utils.random import sample_without_replacement
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array, check_is_fitted
import numpy as np


class SlidingWindowSelection(BaseEstimator, TransformerMixin):
    def __init__(self, nrOfBin=10):
        self.MAXIMUM_TRAIN = 4418
        self.MINIMUM_TRAIN = 13
        self.MAXIMUM_TEST = 4347
        self.MINIMUM_TEST = 12
        self.pixelThreshold = 3000
        self.nrOfBin = nrOfBin
        self.features = []

    def findCoordinates(self,image):

        rowCoordiantes = []
        columnCoordinates = []
        coordinates = []
        flag = True

        #for finding rows

        for row in range(len(image)):
            if any(image[row][i] != 0  for i in range(len(image[row]))):

                if(flag == True):

                    rowCoordiantes.append(row)
                    flag = False

            if all( image[row][j] == 0  for j in range(len(image[row]))):

                if flag == False:
                    rowCoordiantes.append(row)
                    flag = True


        for column in range(len(image[0])):

            if any(image[r][column] != 0 for r in range(len(image))):

                if flag == True:
                    columnCoordinates.append(column)
                    flag = False
            if all(image[r][column] == 0 for r in range(len(image))):

                if flag == False:
                    columnCoordinates.append(column-1)
                    flag = True
        coordinates.append(rowCoordiantes)
        coordinates.append(columnCoordinates)
        return coordinates

    def histogramFactory(self,image, subRows,subCols,train):

        if(train == True):

            maximum = self.MAXIMUM_TRAIN
            minimum = self.MINIMUM_TRAIN
        else:
            maximum = self.MAXIMUM_TEST
            minimum = self.MINIMUM_TEST


        subMatrix = image[subRows][:,subCols]

        sM = np.array(subMatrix)

        final = sM.flatten()

        hist = np.histogram(final, self.nrOfBin,(minimum,maximum))

        return hist

    def modules(self,coordinates,offset):

        outRows    = (coordinates[0][1] - coordinates[0][0]+1)%offset
        outColumns = (coordinates[1][1] - coordinates[1][0]+1)%offset

        return outRows,outColumns

    def evaluateSlice(self,slice):
        counter = 0
        for row in range(len(slice)):
            elem = slice[row]
            counter += sum(1 for element in elem if element != 0)

        if counter < self.pixelThreshold:
            return False
        else:
            return True

    def slidingWindow(self,image,offset, coordinates,train):


        histogramList = []
        for rowPointer in range(coordinates[0][0], coordinates[0][1],offset):

            if(rowPointer + offset < coordinates[0][1]):
                subRows = range(rowPointer,rowPointer+offset)
            for columnPointer in range(coordinates[1][0], coordinates[1][1],offset):
                if(columnPointer + offset < coordinates[1][1]):
                    subCols = range(columnPointer,columnPointer+offset)
                    hist = self.histogramFactory(image, subRows,subCols,train)
                    histogramList.extend(hist)


        rowSelector, columnSelector = self.modules(coordinates,offset)

        if(rowSelector == 0 and columnSelector != 0):

            for pointer in range(coordinates[0][0], coordinates[0][1], offset):
                subRows = range(pointer, pointer + offset)
                subCols = range(columnPointer, coordinates[1][1])
                hist = self.histogramFactory(image, subRows, subCols,train)
                histogramList.extend(hist)

        if(rowSelector != 0 and columnSelector == 0):

            for pointer in range(coordinates[1][0], coordinates[1][1],offset):

                for columnPointer in range(coordinates[1][0], coordinates[1][1], offset):
                    subCols = range(pointer, pointer + offset)
                    subRows= range(rowPointer, coordinates[0][1])
                    hist = self.histogramFactory(image, subRows, subCols,train)
                    histogramList.extend(hist)

        if(rowSelector != 0 and columnSelector != 0):

            for pointer in range(coordinates[0][0], coordinates[0][1]-offset, offset):
                subRows = range(pointer, pointer + offset)
                subCols = range(columnPointer, coordinates[1][1])
                hist = self.histogramFactory(image, subRows, subCols,train)
                histogramList.extend(hist)

            for pointer in range(coordinates[1][0], coordinates[1][1]-offset, offset):
                subCols = range(pointer, pointer + offset)
                subRows = range(rowPointer + offset , coordinates[0][1])
                hist = self.histogramFactory(image,subRows,subCols,train)
                histogramList.extend(hist)

            subRows = range(columnPointer,coordinates[0][1])
            subCols = range(columnPointer,coordinates[1][1])
            hist = self.histogramFactory(image, subRows, subCols,train)
            histogramList.extend(hist)

        #must flatten this histogram 
        return histogramList

    def brainFeature(self,brainCollection,brainNumber,train):

        brainHist = []

        for xSlice in range(25,140,20):

            brain = brainCollection[brainNumber][xSlice][:][:]
            value = self.evaluateSlice(brain)
            while not value:

                if (xSlice < (brainCollection.shape[1])/2):

                    xSlice += 5
                    brain = brainCollection[brainNumber][xSlice][:][:]
                    value = self.evaluateSlice(brain)

                else:
                    xSlice -= 5
                    brain = brainCollection[brainNumber][xSlice][:][:]
                    value = self.evaluateSlice(brain)


            coordinates = self.findCoordinates(brain)
            if len(coordinates[0]) == 2 and len(coordinates[1]) == 2:
                hist = self.slidingWindow(brain,9, coordinates,train)
                brainHist.append(hist)

        for ySlice in range(40,161,30):

            brain = brainCollection[brainNumber][:][ySlice][:]
            coordinates = self.findCoordinates(brain)
            value = self.evaluateSlice(brain)
            while not value:

                if (ySlice < (brainCollection.shape[1])/2):
                    ySlice += 5
                    brain = brainCollection[brainNumber][:][ySlice][:]
                    value = self.evaluateSlice(brain)

                else:
                    ySlice -= 5
                    brain = brainCollection[brainNumber][:][ySlice][:]
                    value = self.evaluateSlice(brain)

            if len(coordinates[0]) == 2 and len(coordinates[1]) == 2:

                hist = self.slidingWindow(brain,9,coordinates,train)
                brainHist.append(hist)

        for zSlice in range(30,140,20):

            brain = brainCollection[brainNumber][:][:][zSlice]
            coordinates = self.findCoordinates(brain)
            value = self.evaluateSlice(brain)

            while not value:

                if (zSlice < (brainCollection.shape[1])/2):
                    zSlice += 5
                    brain = brainCollection[brainNumber][:][:][zSlice]
                    value = self.evaluateSlice(brain)

                else:
                    zSlice -= 5
                    brain = brainCollection[brainNumber][:][:][zSlice]
                    value = self.evaluateSlice(brain)


            if len(coordinates[0]) == 2 and len(coordinates[1]) == 2:

                hist = self.slidingWindow(brain,9,coordinates,train)
                brainHist.append(hist)

        flatBrain = [value for x in brainHist for y in x for value in y]

        return flatBrain

    def totalBrainsFeature(self,brainCollection,train):

        features = []

        for brain in range(brainCollection.shape[0]):

            print ("Sono il cervello %d" %brain, flush=True)
            feature = self.brainFeature(brainCollection,brain,train)
            features.append(feature)

        return features

    def foo(self,stringa1):
        stringa = stringa1
        return stringa

    def fit(self, X, y=None):
        X = check_array(X)
        stringa = self.foo('ciao fit')
        print(stringa)
        brainCollection = np.reshape(X, (-1, 176, 208, 176))
        self.features = self.totalBrainsFeature(brainCollection,True)
        print("Features in fit:")
        print(self.features)
        print ("fine fit")
        return self

    def transform(self, X, y = None):
        X = check_array(X)
        n_samples, n_features = X.shape
        stringa = self.foo('ciao transform')
        print(stringa,flush=True)
        X_new = self.features
        print("Features in transform:")
        print(X_new, flush = True)
        return X_new

class NonZeroSelection(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        X = check_array(X)
        print('ciao fit 1')
        self.non_zero = X.sum(axis=0) > 0
        return self

    def transform(self, X, y=None):
        check_is_fitted(self, ["non_zero"])
        print('ciao transform 1')
        X = check_array(X)
        return X[:, self.non_zero]

class RandomSelection(BaseEstimator, TransformerMixin):

    """Random Selection of features"""
    def __init__(self, n_components=1000, random_state=None):
        self.n_components = n_components
        self.random_state = random_state
        self.components = None

    def foo(self):
        stringa = 'ciao sono fit 2'
        return stringa

    def fit(self, X, y=None):
        X = check_array(X)
        n_samples, n_features = X.shape
        stringa = self.foo()
        print(stringa)

        random_state = check_random_state(self.random_state)
        self.components = sample_without_replacement(
            n_features,
            self.n_components,
            random_state=random_state)
        return self

    def transform(self, X, y=None):
        check_is_fitted(self, ["components"])
        stringa = self.foo()
        print(stringa)
        X = check_array(X)
        n_samples, n_features = X.shape
        X_new = X[:, self.components]

        return X_new