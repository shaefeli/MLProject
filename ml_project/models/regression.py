from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import RidgeCV
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
import numpy as np

class RidgeWithPost(BaseEstimator,TransformerMixin):
    def __init__(self, weight=1.0):
        self.ridge = RidgeCV(weight)
    def fit(self,X,y):
        self.ridge.fit(X,y)
        return self;
    def predict(self,X):
        y = self.ridge.predict(X)
        ranged = np.empty(len(y))
        for i in range(0,len(y)):
            if y[i]<18:
                ranged[i] = 18;
            else:
                ranged[i] = y[i];
        print(ranged)
        return ranged
    def score(self,X,y):
        return self.ridge.score(X,y)
	

#class KernelEstimator(skl.base.BaseEstimator, skl.base.TransformerMixin):
#    """docstring"""
#    def __init__(self, save_path=None):
#        super(KernelEstimator, self).__init__()
#        self.save_path = save_path

#    def fit(self, X, y):
#        X, y = check_X_y(X, y)
#        self.y_mean = np.mean(y)
#        y -= self.y_mean
#        Xt = np.transpose(X)
#        cov = np.dot(X, Xt)
#        alpha, _, _, _ = np.linalg.lstsq(cov, y)
#        self.coef_ = np.dot(Xt, alpha)

#        if self.save_path is not None:
#            plt.figure()
#            plt.hist(self.coef_[np.where(self.coef_ != 0)], bins=50,
#                     normed=True)
#            plt.savefig(self.save_path + "KernelEstimatorCoef.png")
#            plt.close()

#        return self

#    def predict(self, X):
#        check_is_fitted(self, ["coef_", "y_mean"])
#        X = check_array(X)

#        prediction = np.dot(X, self.coef_) + self.y_mean

#        if self.save_path is not None:
#            plt.figure()
#            plt.plot(prediction, "o")
#            plt.savefig(self.save_path + "KernelEstimatorPrediction.png")
#            plt.close()

#        return prediction

#    def score(self, X, y, sample_weight=None):
#        scores = (self.predict(X) - y)**2 / len(y)
#        score = np.sum(scores)

#        if self.save_path is not None:
#            plt.figure()
#            plt.plot(scores, "o")
#            plt.savefig(self.save_path + "KernelEstimatorScore.png")
#            plt.close()

#            df = pd.DataFrame({"score": scores})
#            df.to_csv(self.save_path + "KernelEstimatorScore.csv")

#        return score

#    def set_save_path(self, save_path):
#        self.save_path = save_path
