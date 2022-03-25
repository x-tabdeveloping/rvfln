from typing import Callable, Iterable

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.preprocessing import LabelBinarizer

from rvfln.activation import ActivationFunction, LeakyReLU
from rvfln.rvfln import RVFLNRegressor, matricise


class BLSRegressor(RVFLNRegressor):
    """
    Broad Learning System estimator using ridge regression as its fitting procedure.
    The model uses Sklearns API for ease of use.
    (note: incremental fitting procedures may be implemented in the future once I'm smart enough to understand the paper)
    ----------
    Attributes:
        n_z: int - Number of mapped features
        n_z_features: int - Number of features in each mapped feature
        n_h: int - Number of enhancement nodes
        activation_mapped_feature: ActivationFunction = LeakyReLU - Activation function for mapped features
        activation_enhancement: ActivationFunction = LeakyReLU - Activation function for enhancement nodes
        alpha: float = 0.5 - Regularization parameter of the ridge regression

    Training data:
        X: ndarray - Matrix containing the input of the network
        Y: ndarray - Matrix containing the output of the network (if a vector is supplied, it is converted into a column vector)

    Parameters:
        We: ndarray - Random weights for the mapped features in the network shape=(n_z, input_features, n_z_features)
        Be: ndarray - Random biases for the mapped features in the network shape=(n_z * n_z_features)
        super(): RVFLNRegressor - The random vector functional link network this model is based on
    """

    def __init__(
        self,
        n_z: int,
        n_z_features: int,
        n_h: int,
        activation_mapped_feature: ActivationFunction = LeakyReLU,
        activation_enhancement: ActivationFunction = LeakyReLU,
        alpha: float = 0.5,
    ):
        self.n_z = n_z
        self.n_z_features = n_z_features
        self.n_h = n_h
        self.activation_mapped_feature = activation_mapped_feature
        self.activation_enhancement = activation_enhancement
        super().__init__(
            n_enhancement=n_h, alpha=alpha, activation=activation_enhancement
        )

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
        self.input_features = X.shape[1]
        self.output_features = Y.shape[1]
        self.We = np.random.standard_normal(
            size=(self.n_z, X.shape[1], self.n_z_features)
        )
        self.Be = np.random.standard_normal(size=(self.n_z_features * self.n_z))
        Z = np.concatenate(X @ self.We, axis=1) + self.Be
        Z = self.activation_mapped_feature(Z)
        super().fit(Z, Y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = matricise(X)
        if X is None:
            raise ValueError(
                "X is not a matrix or vector, please provide a matrix or vector"
            )
        if X.shape[1] != self.input_features:
            raise ValueError(
                "The supplied X has a different number of features from the one the model was fitted on."
            )
        if self.We is None:
            raise ValueError(
                "The model has not been fitted yet, please fit before trying to predict"
            )
        Z = np.concatenate(X @ self.We, axis=1) + self.Be
        Z = self.activation_mapped_feature(Z)
        return super().predict(Z)

    def score(self, X: np.ndarray) -> float:
        Y = matricise(Y)
        if Y is None:
            raise ValueError(
                "Y is not a matrix or vector, please provide a matrix or vector"
            )
        return r2_score(self.predict(X), Y)


class BLSClassifier(BLSRegressor, ClassifierMixin):
    """
    BLS Classifier using the Sklearn Classifier API.
    ----
    Attributes:
        n_z: int - Number of mapped features
        n_z_features: int - Number of features in each mapped feature
        n_h: int - Number of enhancement nodes
        activation_mapped_feature: ActivationFunction = LeakyReLU - Activation function for mapped features
        activation_enhancement: ActivationFunction = LeakyReLU - Activation function for enhancement nodes
        alpha: float = 0.5 - Regularization parameter of the ridge regression

    Training data:
        X: ndarray - Matrix containing the input of the network
        y: Iterable - sequence of labels
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
