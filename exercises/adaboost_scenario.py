import numpy as np
from typing import Tuple
from IMLearn.metalearners.adaboost import AdaBoost
from IMLearn.learners.classifiers import DecisionStump
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def generate_data(n: int, noise_ratio: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a dataset in R^2 of specified size

    Parameters
    ----------
    n: int
        Number of samples to generate

    noise_ratio: float
        Ratio of labels to invert

    Returns
    -------
    X: np.ndarray of shape (n_samples,2)
        Design matrix of samples

    y: np.ndarray of shape (n_samples,)
        Labels of samples
    """
    '''
    generate samples X with shape: (num_samples, 2) and labels y with shape (num_samples).
    num_samples: the number of samples to generate
    noise_ratio: invert the label for this ratio of the samples
    '''
    X, y = np.random.rand(n, 2) * 2 - 1, np.ones(n)
    y[np.sum(X ** 2, axis=1) < 0.5 ** 2] = -1
    y[np.random.choice(n, int(noise_ratio * n))] *= -1
    return X, y


def fit_and_evaluate_adaboost(noise, n_learners=250, train_size=5000, test_size=500):
    (train_X, train_y), (test_X, test_y) = generate_data(train_size, noise), generate_data(test_size, noise)

    # Question 1: Train- and test errors of AdaBoost in noiseless case
    model = AdaBoost(DecisionStump, n_learners).fit(train_X, train_y)
    train_loss = []
    test_loss = []

    # Calculate training and test misclassification at the t-th iteration
    iteration_number = range(1, n_learners + 1)
    for t in iteration_number:
        train_loss.append(model.partial_loss(train_X, train_y, t))
        test_loss.append(model.partial_loss(test_X, test_y, t))

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(iteration_number), y=train_loss, mode='lines', name='Training Error'))
    fig.add_trace(go.Scatter(x=list(iteration_number), y=test_loss, mode='lines', name='Test Error'))

    fig.update_layout(title='Training and Test Errors as a Function of the Number of Learners',
                      xaxis_title='Number of Fitted Learners',
                      yaxis_title='Misclassification Error')
    fig.write_image(f"adaboost_{noise}.png", engine='orca')

    # Question 2: Plotting decision surfaces
    T = [5, 50, 100, 250]
    # lims = np.array([np.r_[train_X, test_X].min(axis=0), np.r_[train_X, test_X].max(axis=0)]).T + np.array([-.1, .1])
    # Calculate the minimum and maximum values of train_X and test_X
    min_values = np.min(np.vstack((train_X, test_X)), axis=0)
    max_values = np.max(np.vstack((train_X, test_X)), axis=0)

    # Set the limits for the decision surface plot
    lims = np.column_stack((min_values, max_values)) + np.array([-.1, .1])

    # Create a subplot figure with 1 row and 4 columns
    fig = make_subplots(rows=1, cols=4, subplot_titles=[rf"$\text{{{t} Learners}}$" for t in T])

    # Initialize lists to store traces for decision surfaces and test data
    decision_traces = []
    test_traces = []
    # Iterate over each index and value in T
    for i, t in enumerate(T):
        # Create a decision surface trace using the model's partial_predict function
        decision_trace = decision_surface(lambda X: model.partial_predict(X, t), lims[0], lims[1], showscale=False)

        # Create a scatter plot trace for the test data
        test_trace = go.Scatter(
            x=test_X[:, 0],
            y=test_X[:, 1],
            mode="markers",
            showlegend=False,
            marker=dict(
                color=test_y,
                symbol=np.where(test_y == 1, "circle", "x")))

        # Append the decision surface and test data traces to their respective lists
        decision_traces.append(decision_trace)
        test_traces.append(test_trace)
        # Add the decision surface and test data traces to the figure
        fig.add_traces([decision_trace, test_trace], rows=1, cols=i + 1)

    fig.update_layout(title=dict(
        text="Decision Boundaries of Ensemble at Different Iterations", x=0.5))
    fig.write_image(f"Decision Boundaries of Ensemble at Different Iterations {noise}.png", engine='orca')

    # Question 3: Decision surface of best performing ensemble
    test_error = np.array(test_loss)
    best_t = np.argmin(test_error) + 1

    # Obtain the predictions of the ensemble with the lowest error
    predictions = model.partial_predict(test_X, best_t)

    # Calculate the accuracy of the ensemble with the lowest error
    accuracy = 1 - round(test_error[best_t - 1], 2)

    # Create the decision surface trace using the ensemble with the lowest error
    decision_trace = decision_surface(lambda X: model.partial_predict(X, best_t), lims[0], lims[1], density=60,
                                      showscale=False)

    # Create a scatter plot trace for the test data
    test_trace = go.Scatter(
        x=test_X[:, 0],
        y=test_X[:, 1],
        mode="markers",
        showlegend=False,
        marker=dict(
            color=test_y,
            symbol=np.where(test_y == 1, "circle", "x")))

    # Create the plot figure
    fig = go.Figure(data=[decision_trace, test_trace])
    fig.update_layout(
        title=f"Best Performing Ensemble\nSize: {best_t}, Accuracy: {accuracy}")
    fig.write_image(f"adaboost_{noise}_best_over_test.png", engine='orca')

    # Question 4: Decision surface with weighted samples
    # Normalize and transform the weights
    normalized_weights = 20 * model.D_ / model.D_.max()
    train_trace = go.Scatter(
        x=train_X[:, 0],
        y=train_X[:, 1],
        mode="markers",
        showlegend=False,
        marker=dict(
            size=normalized_weights,
            color=train_y,
            symbol=np.where(train_y == 1, "circle", "x")))

    # Create the figure with decision surface and weighted training data
    fig = go.Figure([decision_surface(model.predict, lims[0], lims[1], showscale=False), train_trace])
    fig.update_layout(title="Final AdaBoost Sample Distribution")
    fig.write_image(f"adaboost_{noise}_weighted_samples.png", engine='orca')


if __name__ == '__main__':
    np.random.seed(0)
    for noise in [0, .4]:
        fit_and_evaluate_adaboost(noise)
