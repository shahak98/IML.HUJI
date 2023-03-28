from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio

pio.templates.default = "simple_white"


def test_univariate_gaussian():
    # Question 1 - Draw samples and print fitted model
    X = np.random.normal(loc=10, scale=1, size=1000)  # create 1000 samples~N(10,1)
    fit_x = UnivariateGaussian().fit(X)  # Sets `self.mu_`, `self.var_` attributes according to calculated estimation
    print(round(fit_x.mu_, 3), round(fit_x.var_, 3))

    # Question 2 - Empirically showing sample mean is consistent
    estimated_mean = []
    samples = []
    for m in range(10, 1010, 10):
        curr_mu = UnivariateGaussian().fit(X[:m])
        estimated_mean.append(
            np.abs(10 - curr_mu.mu_))  # absolute distance between the estimated and true value of exception
        samples.append(m)

    go.Figure([go.Scatter(x=samples, y=estimated_mean, mode='markers', name=r'$\widehat\mu$')],
              layout=go.Layout(title=r"$\text{Variation in the estimation of sample mean based on sample size}$",
                               xaxis_title={"text": "number of samples"},
                               yaxis_title={"text": "r$\hat\mu$"},
                               height=500)).show()

    # Question 3 - Plotting Empirical PDF of fitted model
    # raise NotImplementedError()
    pdf_x = UnivariateGaussian().fit(X).pdf(X)
    go.Figure([go.Scatter(x=X, y=pdf_x, mode='markers', name=r'$\widehat\mu$')],
              layout=go.Layout(title=r"$\text{The Probability Density Function of the Samples Estimated}$",
                               xaxis_title={"text": "Sample Values"},
                               yaxis_title={"text": "PDF"},
                               height=500)).show()


def test_multivariate_gaussian(f1=None):
    # Question 4 - Draw samples and print fitted model
    # raise NotImplementedError()
    mean = np.array([0, 0, 4, 0])
    cov = np.array([[1, 0.2, 0, 0.5], [0.2, 2, 0, 0], [0, 0, 1, 0], [0.5, 0, 0, 1]])
    X = np.random.multivariate_normal(mean=mean, cov=cov, size=1000)  # create 1000 samples
    fit_x = MultivariateGaussian().fit(X)  # Sets `self.mu_`, `self.var_` attributes according to calculated estimation
    print(np.round(fit_x.mu_, 3))
    print(np.round(fit_x.cov_, 3))

    # Question 5 - Likelihood evaluation
    log_likelihood_list = []  # empty list to store the log-likelihood values
    f_vals = np.linspace(-10, 10, 200)
    for f1 in f_vals:  # Calculate log-likelihood for each row
        row = []
        for f3 in f_vals:
            mu = np.array([f1, 0, f3, 0])
            row.append(MultivariateGaussian().log_likelihood(mu, cov, X))
        log_likelihood_list.append(row)
    log_likelihood = np.array(log_likelihood_list)  # make it a numpy array

    fig = go.Figure(go.Heatmap(x=f_vals, y=f_vals, z=log_likelihood),
                    layout=dict(title="The Log-Likelihood Function in terms of the Expected Value of the Features",
                                xaxis_title="f1",
                                yaxis_title="f3"))
    fig.show()

    # Question 6 - Maximum likelihood
    max_index = log_likelihood.argmax()
    max_likelihood_index = np.unravel_index(max_index, log_likelihood.shape)
    max_likelihood_f1 = np.round(f_vals[max_likelihood_index[1]], 3)
    max_likelihood_f3 = np.round(f_vals[max_likelihood_index[0]], 3)
    print("maximum likelihood -features 1,3:", [max_likelihood_f3, max_likelihood_f1])


if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()
    test_multivariate_gaussian()
