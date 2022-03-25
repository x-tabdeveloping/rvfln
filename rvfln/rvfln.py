from typing import Callable, Iterable

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.preprocessing import LabelBinarizer

from rvfln.activation import ActivationFunction, LeakyReLU


def matricise(a: np.ndarray) -> np.ndarray:
    if a.ndim == 1:
        return np.atleast_2d(a).T
    if a.ndim == 2:
        return a
    if a.ndim > 2:
        return None


class RVFLNRegressor(BaseEstimator, RegressorMixin):
    """
    Random Vector Functional Link Network estimator using ridge regression as its fitting procedure.
    The model uses Sklearns API for ease of use.
    ----------
    Attributes:
        n_enhancement: int - Number of enhancement nodes to use for training
        activation: Callable = LeakyReLU - Activation function for the enhancement nodes
        alpha: float = 0.5 - Penalty parameter for the ridge regression

    Training data:
        X: ndarray - Matrix containing the input of the network
        Y: ndarray - Matrix containing the output of the network (if a vector is supplied, it is converted into a column vector)

    Parameters:
        Wh: ndarray - Randomly generated enhancement weights of the network (sampled from standard normal distribution)
        bh: ndarray - Randomly generated enhancement biases of the network (sampled from standard normal distribution)
        W: ndarray - Fitted weights of the model
    """

    def __init__(
        self,
        n_enhancement: int,
        activation: ActivationFunction = LeakyReLU,
        alpha: float = 0.5,
    ):
        self.n_enhancement = n_enhancement
        self.activation = activation
        self.alpha = alpha

    def fit(self, X: np.ndarray, Y: np.ndarray):
        X = matricise(X)
        Y = matricise(Y)
        if X is None:
            raise ValueError(
                "X is not a matrix or vector, please provide a matrix or vector"
            )
        if Y is None:
            raise ValueError(
                "Y is not a matrix or vector, please provide a matrix or vector"
            )
        self.n_input_features = X.shape[1]
        self.n_output_features = Y.shape[1]
        self.Wh = np.random.standard_normal(size=(self.n_enhancement, X.shape[1]))
        self.bh = np.random.standard_normal(size=self.n_enhancement)
        E = X @ self.Wh.T + self.bh
        E = self.activation(E)
        D = np.concatenate((X, E), axis=1)
        m = Ridge(alpha=self.alpha, fit_intercept=False).fit(D, Y)
        self.W = m.coef_.T
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = matricise(X)
        if X is None:
            raise ValueError(
                "X is not a matrix or vector, please provide a matrix or vector"
            )
        if self.W is None:
            raise ValueError(
                "The model has not been fitted yet, please fit the model before trying to use it for prediction."
            )
        if self.n_input_features != X.shape[1]:
            raise ValueError(
                "The input shape of the model is different from that of the supplied matrix. Please make sure that the training data and the data fro prediction has vectors of the same dimensionality."
            )
        E = self.activation(X @ self.Wh.T + self.bh)
        D = np.concatenate((X, E), axis=1)
        Y_hat = D @ self.W
        if self.n_output_features == 1:
            Y_hat = Y_hat.flatten()
        return Y_hat

    def score(self, X: np.ndarray, Y: np.ndarray) -> float:
        return r2_score(self.predict(X), Y)


class RVFLNClassifier(RVFLNRegressor, ClassifierMixin):
    """
    RVFLN Classifier using the Sklearn Classifier API.
    ----
    Attributes:
        n_enhancement: int - Number of enhancement nodes to use for training
        activation: Callable - Activation function for the enhancement nodes
        alpha: float - Penalty parameter for the ridge regression

    Training data:
        X: ndarray - Matrix containing the input of the network
        y: Iterable - List of desired output labels
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def fit(self, X: np.ndarray, y: Iterable):
        self.encoder = LabelBinarizer()
        Y = self.encoder.fit_transform(y)
        super().fit(X, Y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        Y = super().predict(X)
        Y = np.around(Y)
        y = self.encoder.inverse_transform(Y)
        return np.array(y)

    def score(self, X: np.ndarray, y: Iterable) -> float:
        y_hat = self.predict(X)
        y = np.array(y)
        return sum(y == y_hat) / y.size
