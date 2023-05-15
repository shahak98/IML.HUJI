from typing import NoReturn
from ...base import BaseEstimator
import numpy as np


class GaussianNaiveBayes(BaseEstimator):
    """
    Gaussian Naive-Bayes classifier
    """

    def __init__(self):
        """
        Instantiate a Gaussian Naive Bayes classifier

        Attributes
        ----------
        self.classes_ : np.ndarray of shape (n_classes,)
            The different labels classes. To be set in `GaussianNaiveBayes.fit`

        self.mu_ : np.ndarray of shape (n_classes,n_features)
            The estimated features means for each class. To be set in `GaussianNaiveBayes.fit`

        self.vars_ : np.ndarray of shape (n_classes, n_features)
            The estimated features variances for each class. To be set in `GaussianNaiveBayes.fit`

        self.pi_: np.ndarray of shape (n_classes)
            The estimated class probabilities. To be set in `GaussianNaiveBayes.fit`
        """
        super().__init__()
        self.classes_, self.mu_, self.vars_, self.pi_ = None, None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits a gaussian naive bayes model

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        # Get unique classes from y and their corresponding counts
        # self.classes_ contains the unique classes
        # self.pi_ contains the counts of each class
        self.classes_, self.pi_ = np.unique(y, return_counts=True)
        self.pi_ = self.pi_ / len(y)

        self.mu_ = np.zeros((len(self.classes_), X.shape[1]))  # Initialize mean vectors
        self.vars_ = np.zeros((len(self.classes_), X.shape[1]))  # Initialize variance vectors

        # Compute mean and variance for each class
        # In this loop, 'i' represents the index and 'c' represents the current class value
        for i, c in enumerate(self.classes_):
            x_class = X[y == c]  # Select rows in X corresponding to the current class
            self.mu_[i] = np.mean(x_class, axis=0)  # Calculate mean of each feature
            self.vars_[i] = np.var(x_class, axis=0, ddof=1)  # Calculate variance of each feature
            # By setting 'ddof=1', it provides an unbiased estimate of the population variance based on a sample.
            # It is used to calculate the sample variance instead of the population variance.

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples
        """
        likelihoods = self.likelihood(X)  # Calculate the likelihood values for each class
        # Find the index of the class with the highest likelihood for each sample
        # along the second axis of likelihood (classes)
        max_indices = np.argmax(likelihoods, axis=1)
        return self.classes_[max_indices]

    def likelihood(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate the likelihood of a given data over the estimated model

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input data to calculate its likelihood over the different classes.

        Returns
        -------
        likelihoods : np.ndarray of shape (n_samples, n_classes)
            The likelihood for each sample under each of the classes

        """
        if not self.fitted_:
            raise ValueError("Estimator must first be fitted before calling `likelihood` function")

        # The likelihood is calculated by applying the formula for the PDF to each feature for each class
        # PDF(x) = (1 / sqrt(2 * pi * variance)) * exp(-(x - mean)^2 / (2 * variance))

        # X has shape (n_samples, n_features), self.mu_ has shape (n_classes, n_features)
        # The resulting array has shape (n_samples, n_classes, n_features) - this is why we add new axis
        difference = (X[:, np.newaxis, :] - self.mu_) ** 2  # Calculate the squared difference between X and self.mu_
        expo = np.exp(difference / (-2 * self.vars_))
        normalization = np.sqrt(2 * np.pi * self.vars_)
        # Calculate the likelihood for each feature
        likelihoods_per_feature = expo / normalization
        # Take the product of the likelihoods across features for each sample
        likelihoods = np.prod(likelihoods_per_feature, axis=2)  # multiplication along the features axis
        # Multiply with the class probabilities
        likelihoods *= self.pi_

        return likelihoods

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
