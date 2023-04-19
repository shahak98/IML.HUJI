from IMLearn.utils import split_train_test
from IMLearn.learners.regressors import LinearRegression

from typing import NoReturn, Optional
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio

pio.templates.default = "simple_white"


def preprocess_data_train(X: pd.DataFrame, y: Optional[pd.Series]):
    """
    preprocess of the data train, y is not null

    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector corresponding given samples
    Returns
    -------
    Post-processed design matrix and response vector (prices)
    """
    # delete all rows where y is null, and the corresponding rows in X
    # this rows cannot be used for learning
    null_indices = pd.isnull(y)  # Find indices of null values in y
    X = X.drop(X.index[null_indices])
    y = y.drop(y.index[null_indices])

    # remove duplicate rows
    X.drop_duplicates(inplace=True)  # inplace = duplicates are removed directly from X
    y = y[X.index]  # remove corresponding rows from y

    X = X.drop(["id", "lat", "long", "date"], axis=1)  # remove some col in X

    # remove rows that have null values in one or more features.
    index_to_keep = X.dropna().index  # remember the index of the rows to keep
    X = X.loc[index_to_keep]
    y = y.loc[index_to_keep]  # remove corresponding rows from y

    # remove rows that the num of bedrooms,sqft_living, sqft_lot is not positive
    X = X[(X.bedrooms > 0) & (X.sqft_living > 0) & (X.sqft_lot > 0)]
    y = y.loc[X.index]  # remove corresponding rows from y

    # remove rows that the price of the house is not positive
    index_to_keep = y > 0
    X = X.loc[index_to_keep]  # remove corresponding rows from X
    y = y.loc[index_to_keep]

    # remove rows that the floor,sqft_above,sqft_basement, yr_built, yr_renovated, sqft_living15,sqft_lot15  is negative
    X = X[(X.floors >= 0) & (X.sqft_above >= 0) & (X.sqft_basement >= 0) & (X.yr_built >= 0) & (X.yr_renovated >= 0) &
          (X.sqft_living15 >= 0) & (X.sqft_lot15 >= 0)]
    y = y.loc[X.index]  # remove corresponding rows from y

    # view - An index from 0-4 of how good the view of the property was (remove rows that are not fit to this numbers)
    # condition - An index from 1 to 5 on the condition of the apartment (remove rows that are not fit to this numbers)
    # grade - An index from 1 to 13 (remove rows that are not fit to this numbers)
    X = X[(0 <= X.view) & (X.view <= 4) & (1 <= X.condition) & (X.condition <= 5) & (X.grade <= 13) & (1 <= X.grade)]
    y = y.loc[X.index]

    # Add new col for total square,price per square foot, age of the home, and total number of rooms
    X["total_sqft"] = X["sqft_living"] + X["sqft_lot"]
    X["age_of_home"] = 2023 - X["yr_built"]
    X["rooms"] = X["bedrooms"] + X["bathrooms"]

    # Create separate features for each categorical col
    # The new features will have 'zipcode_' for example as a prefix in their column names
    X = pd.get_dummies(X, prefix='zipcode_', columns=['zipcode'])
    X = pd.get_dummies(X, prefix='waterfront_', columns=['waterfront'])

    return X, y


def preprocess_data_test(X: pd.DataFrame):
    """
    This is the basic preprocessing step on the test data, which only involves dropping columns and adding
    basic columns similar to those in the training data

    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem
    """
    X = X.drop(["id", "lat", "long", "date"], axis=1)  # remove some col in X
    # Add new col for total square,price per square foot, age of the home, and total number of rooms
    X["total_sqft"] = X["sqft_living"] + X["sqft_lot"]
    X["age_of_home"] = 2023 - X["yr_built"]
    X["rooms"] = X["bedrooms"] + X["bathrooms"]
    return X


def preprocess_data(X: pd.DataFrame, y: Optional[pd.Series] = None):
    """
    preprocess data
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector corresponding given samples

    Returns
    -------
    Post-processed design matrix and response vector (prices) - either as a single
    DataFrame or a Tuple[DataFrame, Series]
    """
    # Make preprocessing dependent on whether y is null (test case) or not
    if y is not None:
        return preprocess_data_train(X, y)
    return preprocess_data_test(X)


def fit_cols(X_test: pd.DataFrame, X_train: pd.DataFrame):
    """
     Fits the columns of the training set to the columns of the test set
     Parameters
     ----------
     X_test : DataFrame of shape (n_samples, n_features)
         Design matrix of regression problem
     X_train : DataFrame of shape (n_samples, n_features)
         Design matrix of regression problem
    Returns
    -------
    The new fitted X_test
    """
    # Ensure that the column names with 'zip_' in X_train are the same as the column names with 'zip_' in X_test
    zip_cols = X_train.filter(like='zipcode_').columns  # Find all column  in X_train that contain "zip_"
    X_test_copy = pd.get_dummies(X_test, prefix='zipcode_', columns=['zipcode'])
    # make X_test to match cols in X_train, fill in any missing columns with 0
    X_test_copy = X_test_copy.reindex(columns=zip_cols, fill_value=0)
    X_test = pd.concat([X_test, X_test_copy], axis=1)

    # Ensure that the column names with 'waterfront_' in X_train are the same as the column names in X_test
    w_cols = X_train.filter(like='waterfront_').columns
    X_test_copy = pd.get_dummies(X_test, prefix='waterfront_', columns=['waterfront'])
    # make X_test to match cols in X_train, fill in any missing columns with 0
    X_test_copy = X_test_copy.reindex(columns=w_cols, fill_value=0)
    X_test = pd.concat([X_test, X_test_copy], axis=1)

    X_test = X_test.drop(["waterfront", "zipcode"], axis=1)  # remove this col from the original test X (we did dummis)
    # cols = X_test.columns
    # for c in cols:
    #     print(c)
    # cols = X_train.columns
    # for c in cols:
    #     print(c)
    # Replace not valid values in X_test with their corresponding column mean from X_train
    for f in X_test.columns:
        # Replace null values
        if X_test[f].isnull().any():  # there is at least one missing value in the column
            X_test[f].fillna(X_train[f].mean(),
                             inplace=True)  # inplace=True specifies that the operation should be in place
        # Check and replace values in view/condition/grade column that are not valid
        if f == 'view':
            values_to_replace = (X_test['view'] < 0) | (X_test['view'] > 4)
            if values_to_replace.any():  # check if any value in 'view' col fails condition
                X_test.loc[values_to_replace, 'view'] = X_train[f].mean()
        elif f == 'condition':
            values_to_replace = (X_test[f] < 1) | (X_test[f] > 5)
            if values_to_replace.any():
                X_test.loc[values_to_replace, f] = X_train[f].mean()
        elif f == 'grade':
            values_to_replace = (X_test[f] < 1) | (X_test[f] > 13)
            if values_to_replace.any():
                X_test.loc[values_to_replace, f] = X_train[f].mean()
        # Check and replace values in bedrooms,sqft_living, sqft_lot column to be positive
        elif f in ['bedrooms', 'sqft_living', 'sqft_lot']:
            values_to_replace = (X_test[f] <= 0)
            if values_to_replace.any():
                X_test.loc[values_to_replace, f] = X_train[f].mean()
        # Check and replace values in specific column to be non-negative
        elif f in ['floors', 'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated', 'sqft_living15', 'sqft_lot15']:
            values_to_replace = (X_test[f] < 0)
            if values_to_replace.any():
                X_test.loc[values_to_replace, f] = X_train[f].mean()

    return X_test


def feature_evaluation(X: pd.DataFrame, y: pd.Series, output_path: str = ".") -> NoReturn:
    """
    Create scatter plot between each feature and the response.
        - Plot title specifies feature name
        - Plot title specifies Pearson Correlation between feature and response
        - Plot saved under given folder with file name including feature name
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector to evaluate against

    output_path: str (default ".")
        Path to folder in which plots are saved
    """
    # Remove dummy columns from X
    dummy_cols = ['zipcode_', 'waterfront_']
    dummy_columns = X.filter(regex='|'.join(dummy_cols))  # select all columns that contain any of those prefixes
    X = X.drop(dummy_columns.columns, axis=1)

    # p.cov(X[i], y) returns a 2X2 covariance matrix
    # cov(x[i], x[i])  cov(x[i], y)
    # cov(y,x[i])      cov(y,y)
    # we are interested in the covariance between feature-y only, therefor we only take the element at position [0, 1]
    for i in X:
        pearson_correlation = np.cov(X[i], y)[0, 1] / (np.std(X[i]) * np.std(y))
        # Create scatter plot
        fig = px.scatter(pd.DataFrame({'x': X[i], 'y': y}), x="x", y="y", trendline="ols",
                         title=f" Pearson Correlation- {pearson_correlation} <br> Between {i} and response",
                         labels={"x": f"{i} ", "y": "y value"})
        fig.write_image(output_path + f"/pearson.correlation.{i}.png", engine='orca')  # save the image


if __name__ == '__main__':
    np.random.seed(0)
    df = pd.read_csv("../datasets/house_prices.csv")

    # Question 1 - split data into train and test sets

    # create DataFrame without price column - X
    df_no_price = df.drop('price', axis=1)
    # create DataFrame with only price column - y
    df_price = df['price']
    train_X, train_y, test_X, test_y = split_train_test(df_no_price, df_price, train_proportion=0.75)

    # Question 2 - Preprocessing of housing prices dataset
    train_X, train_y = preprocess_data_train(train_X, train_y)

    # Question 3 - Feature evaluation with respect to response
    feature_evaluation(train_X, train_y)  # TODO do i need to do it also on the test set?

    # Question 4 - Fit model over increasing percentages of the overall training data
    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    #   4) Store average and variance of loss over test set
    # Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)

    # Remove rows that doesn't have a value in y in the test set
    null_indices = pd.isnull(test_y)  # Find indices of null values in y
    test_X = test_X.drop(test_X.index[null_indices])
    test_y = test_y.drop(test_y.index[null_indices])

    test_X = preprocess_data(test_X)  # basic preprocessing
    test_X = fit_cols(test_X, train_X)  # fit the cols, and fill the missing\invalid values

    # cols = test_X.columns
    # for c in cols:
    #     print(c)

    # percentage = list(range(10, 101))  # list of percentage values from 10 to 100 (inclusive)
    # num_runs = 10  # number of times the inner loop will be executed for each percentage value
    # results = np.zeros((len(percentage), num_runs))  # This array will be used to store the results
    #
    # for i in range(len(percentage)):
    #     p = percentage[i]
    #     for j in range(results.shape[1]):
    #         X_percentage = train_X.sample(frac=p / 100.0)  # take the p% of the train set
    #         y_percentage = train_y.loc[X_percentage.index]  # corresponding y values for the p% of the training data
    #         model = LinearRegression(include_intercept=True).fit(X_percentage, y_percentage)  # Fit linear model
    #         y_hat = model.predict(test_X)
    #         results[i, j] = model.loss(test_X, y_hat)
    #
    # mean = results.mean(axis=1)  # mean loss values for each percentage
    # sdt_2 = 2 * results.std(axis=1)  # (standard deviation of the loss values for each percentage) *2
    # confidence_interval_plus = mean + sdt_2
    # confidence_interval_minus = mean - sdt_2
    # fig = go.Figure([go.Scatter(x=percentage, y=confidence_interval_minus, mode="lines", line=dict(color='blue')),
    #                  go.Scatter(x=percentage, y=confidence_interval_plus, mode="lines", line=dict(color='blue')),
    #                  go.Scatter(x=percentage, y=mean, mode="markers+lines")],
    #                 layout=go.Layout(title="Effect of training size on mean squared error",
    #                                  xaxis=dict(title="Percentage of Training Set"),
    #                                  yaxis=dict(title="mean squared error")))
    # fig.write_image("mse.png")
