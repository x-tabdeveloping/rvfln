from typing import Iterable

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelBinarizer


def ReLU(a: np.ndarray) -> np.ndarray:
    """
    Simple numpy implementation of Rectified Linear Unit
    """
    return np.maximum(a, 0)


def LeakyReLU(a: np.ndarray) -> np.ndarray:
    """
    Simple numpy implementation of LeakyReLU
    """
    return np.where(a > 0, a, a * 0.01)


class RVFLN(BaseEstimator):
    """
    Base class Random Vector Functional Link Network estimator using ridge regression as its fitting procedure.
    The model uses Sklearns API for ease of use.
    ----------
    Attributes:
        n_enhancement: int - Number of enhancement nodes to use for training
        activation: Callable = LeakyReLU - Activation function for the enhancement nodes
        alpha: float = 0.5 - Penalty parameter for the ridge regression

    Training data:
        X: ndarray - Matrix containing the input of the network
        Y: ndarray - Matrix containing the output of the network

    Parameters:
        Wh: ndarray - Randomly generated enhancement weights of the network (sampled from standard normal distribution)
        bh: ndarray - Randomly generated enhancement biases of the network (sampled from standard normal distribution)
        W: ndarray - Fitted weights of the model
    """

    def __init__(
        self,
        n_enhancement: int,
        activation=LeakyReLU,
        alpha=0.5,
    ):
        self.n_enhancement = n_enhancement
        self.activation = activation
        self.alpha = alpha

    def fit(self, X: np.ndarray, Y: np.ndarray):
        if X.ndim != 2:
            raise ValueError("X is not a matrix, please provide a matrix")
        if Y.ndim != 2:
            raise ValueError("Y is not a matrix, please provide a matrix")
        self.input_shape = X.shape[1]
        self.output_shape = Y.shape[1]
        self.Wh = np.random.standard_normal(size=(self.n_enhancement, X.shape[1]))
        self.bh = np.random.standard_normal(size=self.n_enhancement)
        E = X @ self.Wh.T + self.bh
        E = self.activation(E)
        D = np.concatenate((X, E), axis=1)
        m = Ridge(alpha=self.alpha, fit_intercept=False).fit(D, Y)
        self.W = m.coef_.T
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.W is None:
            raise ValueError(
                "The model has not been fitted yet, please fit the model before trying to use it for prediction."
            )
        if self.input_shape != X.shape[1]:
            raise ValueError(
                "The input shape of the model is different from that of the supplied matrix. Please make sure that the training data and the data fro prediction has vectors of the same dimensionality."
            )
        E = self.activation(X @ self.Wh.T + self.bh)
        D = np.concatenate((X, E), axis=1)
        return D @ self.W

    def mse(self, X: np.ndarray, Y: np.ndarray) -> float:
        if self.output_shape != Y.shape[1]:
            raise ValueError(
                "The shape of the output doesn't match the ouput shape the model was fitted to."
            )
        Yhat = self.predict(X)
        return mean_squared_error(Y, Yhat)


class RVFLNRegressor(BaseEstimator, RegressorMixin):
    """
    RVFLN Regressor using the Sklearn Regressor API.
    ----
    Attributes:
        n_enhancement: int - Number of enhancement nodes to use for training
        activation: Callable - Activation function for the enhancement nodes
        alpha: float - Penalty parameter for the ridge regression

    Training data:
        X: ndarray - Matrix containing the input of the network
        y: ndarray - Vector containing the output of the network

    Parameters:
        model: Base class model, that is wrapped in the class.
    """

    def __init__(self, *args, **kwargs):
        self.model = RVFLN(*args, **kwargs)

    def fit(self, X: np.ndarray, y: np.ndarray):
        Y = np.atleast_2d(y).T
        return self.model.fit(X, Y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X).flatten()

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        return r2_score(self.predict(X), y)


class RVFLNClassifier(BaseEstimator, ClassifierMixin):
    """
    RVFLN Classifier using the Sklearn Classifier API.
    ----
    Attributes:
        n_enhancement: int - Number of enhancement nodes to use for training
        activation: Callable - Activation function for the enhancement nodes
        alpha: float - Penalty parameter for the ridge regression

    Training data:
        X: ndarray - Matrix containing the input of the network
        y: Iterable - List of desired output classes

    Parameters:
        model: Base class model, that is wrapped in the class.
    """

    def __init__(self, *args, **kwargs):
        self.model = RVFLN(*args, **kwargs)

    def fit(self, X: np.ndarray, y: Iterable):
        self.encoder = LabelBinarizer()
        Y = self.encoder.fit_transform(y)
        return self.model.fit(X, Y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        Y = self.model.predict(X)
        Y = np.around(Y)
        y = self.encoder.inverse_transform(Y)
        return np.array(y)

    def score(self, X: np.ndarray, y: Iterable) -> float:
        y_hat = self.predict(X)
        y = np.array(y)
        return sum(y == y_hat) / y.size
