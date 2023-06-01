from __future__ import annotations
from typing import Tuple, NoReturn
from ...base import BaseEstimator
import numpy as np
from itertools import product


class DecisionStump(BaseEstimator):
    """
    A decision stump classifier for {-1,1} labels according to the CART algorithm

    Attributes
    ----------
    self.threshold_ : float
        The threshold by which the data is split

    self.j_ : int
        The index of the feature by which to split the data

    self.sign_: int
        The label to predict for samples where the value of the j'th feature is about the threshold
    """

    def __init__(self) -> DecisionStump:
        """
        Instantiate a Decision stump classifier
        """
        super().__init__()
        self.threshold_, self.j_, self.sign_ = None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        Fit a decision stump to the given data. That is, finds the best feature and threshold by which to split

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        error = 2  # The error should range from 0 to 1, so setting it to 2 means it is necessarily larger

        # Find minimum error by iterating over feature and sign combinations
        for f in range(X.shape[1]):
            for s in {-1, 1}:
                thr, thr_error = self._find_threshold(X[:, f], y, s)
                if thr_error < error:
                    self.threshold_ = thr
                    self.j_ = f
                    self.sign_ = s
                    error = thr_error

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict sign responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples

        Notes
        -----
        Feature values strictly below threshold are predicted as `-sign` whereas values which equal
        to or above the threshold are predicted as `sign`
        """
        y_predict = np.empty(X.shape[0])
        bool_arr = X[:, self.j_] < self.threshold_  # Create a boolean array based on the condition
        y_predict[bool_arr] = -1 * self.sign_
        y_predict[~bool_arr] = self.sign_
        return y_predict

    def _find_threshold(self, values: np.ndarray, labels: np.ndarray, sign: int) -> Tuple[float, float]:
        """
        Given a feature vector and labels, find a threshold by which to perform a split
        The threshold is found according to the value minimizing the misclassification
        error along this feature

        Parameters
        ----------
        values: ndarray of shape (n_samples,)
            A feature vector to find a splitting threshold for

        labels: ndarray of shape (n_samples,)
            The labels to compare against

        sign: int
            Predicted label assigned to values equal to or above threshold

        Returns
        -------
        thr: float
            Threshold by which to perform split

        thr_err: float between 0 and 1
            Misclassificaiton error of returned threshold

        Notes
        -----
        For every tested threshold, values strictly below threshold are predicted as `-sign` whereas values
        which equal to or above the threshold are predicted as `sign`
        """

        # Sort the values and rearrange the corresponding labels accordingly
        sorted_indices = np.argsort(values)
        sorted_values = values[sorted_indices]
        sorted_labels = labels[sorted_indices]

        thresholds = np.concatenate([[-np.inf], sorted_values[1:], [np.inf]])

        # Calculate the loss in the scenario where all values are classified as `sign`
        errors = np.sum(np.abs(sorted_labels[sorted_labels * sign > 0]))

        errors = np.append(errors, errors - np.cumsum(sorted_labels * sign))

        min_error_index = np.argmin(errors)

        return float(thresholds[min_error_index]), float(errors[min_error_index])

    # Initialize variables to track the best threshold and error
    # best_threshold = None
    # best_error = float('inf')
    #
    # # Iterate over all possible thresholds
    # for t in np.unique(values):  # Get unique values as potential thresholds
    #     # Predict the labels based on whether the values are less than the threshold (t)
    #     predictions = np.where(values < t, -sign, sign)
    #     # Calculate the misclassification loss by comparing the predicted labels with the true labels
    #     loss = np.sum(labels != predictions)
    #
    #     if loss < best_error:
    #         best_error = loss
    #         best_threshold = t
    #
    # return best_threshold, best_error

    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under misclassification loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        Returns
        -------
        loss : float
            Performance under missclassification loss function
        """
        from ...metrics import misclassification_error
        return misclassification_error(y, self._predict(X))
