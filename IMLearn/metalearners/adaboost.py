import numpy as np
from ..base import BaseEstimator
from typing import Callable, NoReturn


class AdaBoost(BaseEstimator):
    """
    AdaBoost class for boosting a specified weak learner

    Attributes
    ----------
    self.wl_: Callable[[], BaseEstimator]
        Callable for obtaining an instance of type BaseEstimator

    self.iterations_: int
        Number of boosting iterations to perform

    self.models_: List[BaseEstimator]
        List of fitted estimators, fitted along the boosting iterations

    self.weights_: List[float]
        List of weights for each fitted estimator, fitted along the boosting iterations

    self.D_: List[np.ndarray]
        List of weights for each sample, fitted along the boosting iterations
    """

    def __init__(self, wl: Callable[[], BaseEstimator], iterations: int):
        """
        Instantiate an AdaBoost class over the specified base estimator

        Parameters
        ----------
        wl: Callable[[], BaseEstimator]
            Callable for obtaining an instance of type BaseEstimator

        iterations: int
            Number of boosting iterations to perform
        """
        super().__init__()
        self.wl_ = wl
        self.iterations_ = iterations
        self.models_, self.weights_, self.D_ = None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        Fit an AdaBoost classifier over given samples

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        self.models_ = []  # List to store the trained weak learners
        self.weights_ = np.zeros(self.iterations_)  # Array to store the weights for each weak learner
        self.D_ = np.ones(len(y)) / len(y)  # set initial distribution to be uniform

        for i in range(self.iterations_):
            # Invoke the base learner A (e.g., decision stump) with the current distribution to obtain a weak learner
            # The weak learner is trained using the training set weighted according to the current distribution.
            self.models_.append(self.wl_().fit(X, y * self.D_))

            # Predict labels using the most recent weak learner
            y_pred = self.models_[-1].predict(X)

            # Calculate the weighted error - step 5 from restriction
            misclassified_samples = y != y_pred  # Identify misclassified samples
            misclassified_weights = self.D_[misclassified_samples]  # Select weights of misclassified samples
            epsilon = np.sum(misclassified_weights)  # Compute the sum of misclassified weights

            self.weights_[i] = 0.5 * np.log((1 - epsilon) / epsilon)  # step 6 from restriction

            self.D_ *= np.exp(- y * self.weights_[i] * y_pred)  # step 7 from restriction - update sample weights
            self.D_ /= np.sum(self.D_)  # step 8 from restriction - normalize weights

    def _predict(self, X):
        """
        Predict responses for given samples using fitted estimator over all boosting iterations

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples
        """
        return self.partial_predict(X, self.iterations_)

    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under misclassification loss function over all boosting iterations

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
        return self.partial_loss(X, y, self.iterations_)

    def partial_predict(self, X: np.ndarray, T: int) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimators up to T learners

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        T: int
            The number of classifiers (from 1,...,T) to be used for prediction

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples
        """
        weighted_prediction = 0
        # Iterate over the range of learners up to T or the maximum available learners
        for t in range(min(T, self.iterations_)):
            # Calculate the weighted prediction for the current learner
            weighted_prediction += self.weights_[t] * self.models_[t].predict(X)
        # Apply the sign function to convert the weighted prediction into binary responses
        return np.sign(weighted_prediction)

    def partial_loss(self, X: np.ndarray, y: np.ndarray, T: int) -> float:
        """
            Evaluate performance under misclassification loss function using fitted estimators up to T learners

            Parameters
            ----------
            X : ndarray of shape (n_samples, n_features)
                Test samples

            y : ndarray of shape (n_samples, )
                True labels of test samples

            T: int
                The number of classifiers (from 1,...,T) to be used for prediction

            Returns
            -------
            loss : float
                Performance under missclassification loss function
            """
        from ..metrics import misclassification_error
        return misclassification_error(y, self.partial_predict(X, T))



