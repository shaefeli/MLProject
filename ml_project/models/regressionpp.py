
import sklearn as skl
import numpy as np
import pandas as pd
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from matplotlib import pyplot as plt

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import RidgeCV




class KernelEstimator(skl.base.BaseEstimator, skl.base.TransformerMixin):
    """docstring"""
    def __init__(self, save_path=None):
        super(KernelEstimator, self).__init__()
        self.save_path = save_path

    def fit(self, X, y):
        print(len(X),flush=True)
        print(len(X[0]),flush=True)
        print(y)
        X, y = check_X_y(X, y)
        self.y_mean = np.mean(y)
        y -= self.y_mean
        Xt = np.transpose(X)
        cov = np.dot(X, Xt)
        alpha, _, _, _ = np.linalg.lstsq(cov, y)
        self.coef_ = np.dot(Xt, alpha)

        if self.save_path is not None:
            plt.figure()
            plt.hist(self.coef_[np.where(self.coef_ != 0)], bins=50,
                     normed=True)
            plt.savefig(self.save_path + "KernelEstimatorCoef.png")
            plt.close()

        return self

    def predict(self, X):
        check_is_fitted(self, ["coef_", "y_mean"])
        X = check_array(X)

        prediction = np.dot(X, self.coef_) + self.y_mean

        if self.save_path is not None:
            plt.figure()
            plt.plot(prediction, "o")
            plt.savefig(self.save_path + "KernelEstimatorPrediction.png")
            plt.close()

        return prediction

    def score(self, X, y, sample_weight=None):
        scores = (self.predict(X) - y)**2 / len(y)
        score = np.sum(scores)

        if self.save_path is not None:
            plt.figure()
            plt.plot(scores, "o")
            plt.savefig(self.save_path + "KernelEstimatorScore.png")
            plt.close()

            df = pd.DataFrame({"score": scores})
            df.to_csv(self.save_path + "KernelEstimatorScore.csv")

        return score

    def set_save_path(self, save_path):
        self.save_path = save_path


class RidgeWithPost(BaseException, TransformerMixin):
    def __init__(self, weight=1.0):
        self.ridge = RidgeCV(weight)
        print(weight, flush=True)



    def fit(self, X, y, sample_weight = None):
        print('fit regression',flush=True)
        print(len(X),flush=True)
        print(len(X[0]))
        print(len(X[140]))

        print(len(y),flush=True)
        self.ridge.fit(X,y)
        print('fine regression',flush=True)
        return self

    def predict(self, X):
        print('predict regression',flush=True)
        y = self.ridge.predict(X)
        ranged = np.empty(len(y))
        for i in range(len(y)):
            if y[i]<18:
                ranged[i]=18
            else:
                ranged[i]=y[i]
        return ranged

    def score(self,X, y, sample_weight=None):
        return self.ridge.score(X,y)