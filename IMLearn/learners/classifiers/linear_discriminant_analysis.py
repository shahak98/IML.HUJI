from typing import NoReturn
from ...base import BaseEstimator
import numpy as np
from numpy.linalg import det, inv


class LDA(BaseEstimator):
    """
    Linear Discriminant Analysis (LDA) classifier

    Attributes
    ----------
    self.classes_ : np.ndarray of shape (n_classes,)
        The different labels classes. To be set in `LDA.fit`

    self.mu_ : np.ndarray of shape (n_classes,n_features)
        The estimated features means for each class. To be set in `LDA.fit`

    self.cov_ : np.ndarray of shape (n_features,n_features)
        The estimated features covariance. To be set in `LDA.fit`

    self._cov_inv : np.ndarray of shape (n_features,n_features)
        The inverse of the estimated features covariance. To be set in `LDA.fit`

    self.pi_: np.ndarray of shape (n_classes)
        The estimated class probabilities. To be set in `GaussianNaiveBayes.fit`
    """

    def __init__(self):
        """
        Instantiate an LDA classifier
        """
        super().__init__()
        self.classes_, self.mu_, self.cov_, self._cov_inv, self.pi_ = None, None, None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits an LDA model.
        Estimates gaussian for each label class - Different mean vector, same covariance
        matrix with dependent features.

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
        i = 0
        for c in self.classes_:
            x_class = X[y == c]  # Select rows in X corresponding to the current class
            self.mu_[i] = np.mean(x_class, axis=0)  # Calculate mean of each feature
            i += 1

        # Cov = (1/(m-k)) * (xi-mu)(xi-U).T
        c = X - self.mu_[y.astype(int)]
        # Calculate the products of differences between samples and class means,
        # organized in a 3D structure, where each 2D slice represents the products for a specific class.
        # "ki,kj->kij" specifies the desired contraction of indices - class(k) row(i) and col (j)
        c_product = np.einsum("ki,kj->kij", c, c)
        # Sum the products of differences along the first axis (classes)
        cov_sum = np.sum(c_product, axis=0)
        # Divide the sum by the number of samples minus the number of classes
        self.cov_ = cov_sum / (len(X) - len(self.classes_))

        self.cov_inv_ = inv(self.cov_)

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

        # Formula: p(x|k) = (1 / sqrt((2*pi)^d * det(Sigma_k))) * exp(-0.5 * (x - mu_k)^T * Sigma_k^-1 * (x - mu_k))
        # Where: k: Class index, mu_k/Sigma_k: Mean/covariance vector of class k
        normalization_factor = np.sqrt((2 * np.pi) ** X.shape[1] * np.linalg.det(self.cov_))
        # X has shape (n_samples, n_features), self.mu_ has shape (n_classes, n_features)
        # The resulting array has shape (n_samples, n_classes, n_features) - this is why we add new axis
        difference = X[:, np.newaxis, :] - self.mu_  # (x - mu)
        # multiplication along the features axis
        exponent = -0.5 * np.sum(difference.dot(self.cov_inv_) * difference, axis=2)
        likelihoods = np.exp(exponent) / normalization_factor
        return likelihoods * self.pi_  # calculate the likelihoods for each sample and class pair

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
