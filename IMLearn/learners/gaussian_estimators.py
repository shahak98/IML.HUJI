from __future__ import annotations
import numpy as np
from numpy.linalg import inv, det, slogdet


class UnivariateGaussian:
    """
    Class for univariate Gaussian Distribution Estimator
    """

    def __init__(self, biased_var: bool = False) -> UnivariateGaussian:
        """
        Estimator for univariate Gaussian mean and variance parameters

        Parameters
        ----------
        biased_var : bool, default=False
            Should fitted estimator of variance be a biased or unbiased estimator

        Attributes
        ----------
        fitted_ : bool
            Initialized as false indicating current estimator instance has not been fitted.
            To be set as True in `UnivariateGaussian.fit` function.

        mu_: float
            Estimated expectation initialized as None. To be set in `UnivariateGaussian.fit`
            function.

        var_: float
            Estimated variance initialized as None. To be set in `UnivariateGaussian.fit`
            function.
        """
        self.biased_ = biased_var
        self.fitted_, self.mu_, self.var_ = False, None, None

    def fit(self, X: np.ndarray) -> UnivariateGaussian:
        """
        Estimate Gaussian expectation and variance from given samples

        Parameters
        ----------
        X: ndarray of shape (n_samples, )
            Training data

        Returns
        -------
        self : returns an instance of self.

        Notes
        -----
        Sets `self.mu_`, `self.var_` attributes according to calculated estimation (where
        estimator is either biased or unbiased). Then sets `self.fitted_` attribute to `True`
        """
        # raise NotImplementedError()
        self.mu_ = X.mean()
        if self.biased_:  # if biased we're dividing by m, unbiased m-1
            self.var_ = (1 / len(X)) * (np.sum((X - self.mu_) ** 2))
        else:
            self.var_ = (1 / (len(X) - 1)) * (np.sum((X - self.mu_) ** 2))
        self.fitted_ = True
        return self

    def pdf(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate PDF of observations under Gaussian model with fitted estimators

        Parameters
        ----------
        X: ndarray of shape (n_samples, )
            Samples to calculate PDF for

        Returns
        -------
        pdfs: ndarray of shape (n_samples, )
            Calculated values of given samples for PDF function of N(mu_, var_)

        Raises
        ------
        ValueError: In case function was called prior fitting the model
        """
        if not self.fitted_:
            raise ValueError("Estimator must first be fitted before calling `pdf` function")
        # raise NotImplementedError()
        else:
            return (1 / (np.sqrt(self.var_ * 2 * np.pi))) * np.exp(-((X - self.mu_) ** 2) / 2 * self.var_)

    @staticmethod
    def log_likelihood(mu: float, sigma: float, X: np.ndarray) -> float:
        """
        Calculate the log-likelihood of the data under a specified Gaussian model

        Parameters
        ----------
        mu : float
            Expectation of Gaussian
        sigma : float
            Variance of Gaussian
        X : ndarray of shape (n_samples, )
            Samples to calculate log-likelihood with

        Returns
        -------
        log_likelihood: float
            log-likelihood calculated
        """
        # raise NotImplementedError()
        m = len(X)
        return -(m / 2) * np.log((2 * np.pi * sigma)) - ((1 / 2 * sigma) * np.sum((X - mu) ** 2))


class MultivariateGaussian:
    """
    Class for multivariate Gaussian Distribution Estimator
    """

    def __init__(self):
        """
        Initialize an instance of multivariate Gaussian estimator

        Attributes
        ----------
        fitted_ : bool
            Initialized as false indicating current estimator instance has not been fitted.
            To be set as True in `MultivariateGaussian.fit` function.

        mu_: ndarray of shape (n_features,)
            Estimated expectation initialized as None. To be set in `MultivariateGaussian.fit`
            function.

        cov_: ndarray of shape (n_features, n_features)
            Estimated covariance initialized as None. To be set in `MultivariateGaussian.fit`
            function.
        """
        self.mu_, self.cov_ = None, None
        self.fitted_ = False

    def fit(self, X: np.ndarray) -> MultivariateGaussian:
        """
        Estimate Gaussian expectation and covariance from given samples

        Parameters
        ----------
        X: ndarray of shape (n_samples, n_features)
            Training data

        Returns
        -------
        self : returns an instance of self

        Notes
        -----
        Sets `self.mu_`, `self.cov_` attributes according to calculated estimation.
        Then sets `self.fitted_` attribute to `True`
        """
        # raise NotImplementedError()
        self.mu_ = np.mean(X, axis=0)  # axis = 0 means along the column
        m = len(X)
        self.cov_ = (1 / (m - 1)) * (X - self.mu_).T @ (
                X - self.mu_)  # @ symbol matrix multiplication, T = transpose of the array
        self.fitted_ = True
        return self

    def pdf(self, X: np.ndarray):
        """
        Calculate PDF of observations under Gaussian model with fitted estimators

        Parameters
        ----------
        X: ndarray of shape (n_samples, n_features)
            Samples to calculate PDF for

        Returns
        -------
        pdfs: ndarray of shape (n_samples, )
            Calculated values of given samples for PDF function of N(mu_, cov_)

        Raises
        ------
        ValueError: In case function was called prior fitting the model
        """
        if not self.fitted_:
            raise ValueError("Estimator must first be fitted before calling `pdf` function")
        # raise NotImplementedError()
        else:
            features = X.shape[1]
            matrix = X - self.mu_
            exp = np.exp(-0.5 * (matrix.T @ np.linalg.inv(self.cov_) @ matrix))
            return exp / np.sqrt(((2 * np.pi) ** features) * np.linalg.det(self.cov_))

    @staticmethod
    def log_likelihood(mu: np.ndarray, cov: np.ndarray, X: np.ndarray) -> float:
        """
        Calculate the log-likelihood of the data under a specified Gaussian model

        Parameters
        ----------
        mu : ndarray of shape (n_features,)
            Expectation of Gaussian
        cov : ndarray of shape (n_features, n_features)
            covariance matrix of Gaussian
        X : ndarray of shape (n_samples, n_features)
            Samples to calculate log-likelihood with

        Returns
        -------
        log_likelihood: float
            log-likelihood calculated over all input data and under given parameters of Gaussian
        """
        d = X.shape[1]
        m = X.shape[0]
        a = (-m / 2) * np.log((2 * np.pi) ** d)
        b = (-m / 2) * np.log(np.linalg.det(cov))
        c = -0.5 * (np.sum(np.sum((X-mu) @ np.linalg.inv(cov) * (X-mu))))  # sum of the m sample
        return a + b + c

    # m = X.shape[0]
    # d = X.shape[1]
    # cov_det = np.linalg.det(cov)
    # cov_inv = np.linalg.inv(cov)
    # matrix_mult = np.sum((X - mu) @ cov_inv * (X - mu))
    # a = (-m / 2) * np.log((2 * np.pi) ** d)
    # b = (-m / 2) * np.log(cov_det)
    # c = -0.5 * (np.sum(matrix_mult))  # sum of the m sample
    # return a + b + c



