from IMLearn.utils import split_train_test
from IMLearn.learners.regressors import LinearRegression

from typing import NoReturn, Optional
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio

pio.templates.default = "simple_white"


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
    # remove duplicate rows
    index_to_keep = X.drop_duplicates().index  # remember the index of the rows to keep
    X = X.loc[index_to_keep]
    if y is not None:
        y = y.loc[index_to_keep]  # remove corresponding rows from y

    X = X.drop(["id", "lat", "long", "date"], axis=1)  # remove some col in X

    # remove rows that have null values in one or more features.
    index_to_keep = X.dropna().index  # remember the index of the rows to keep
    X = X.loc[index_to_keep]
    if y is not None:
        y = y.loc[index_to_keep]  # remove corresponding rows from y

    # remove rows that the price of the house,num of bedrooms,sqft_living, sqft_lot is positive
    X = X[(X.bedrooms > 0) & (X.sqft_living > 0) & (X.sqft_lot > 0)]
    if y is not None:
        y = y[X.index]  # remove corresponding rows from y
        y = y[(y.price > 0)]
        X = X[y.index]  # remove corresponding rows from X

    # remove rows that the floor,sqft_above,sqft_basement, yr_built, yr_renovated, sqft_living15,sqft_lot15  is negative
    X = X[(X.floor >= 0) & (X.sqft_above >= 0) & (X.sqft_basement >= 0) & (X.yr_built >= 0) & (X.yr_renovated >= 0) &
          (X.sqft_living15 >= 0) & (X.sqft_lot15 >= 0)]
    if y is not None:
        y = y[X.index]  # remove corresponding rows from y

    # view - An index from 0-4 of how good the view of the property was (remove rows that are not fit to this numbers)
    # condition - An index from 1 to 5 on the condition of the apartment (remove rows that are not fit to this numbers)
    # grade - An index from 1 to 13 (remove rows that are not fit to this numbers)
    X = X[(0 <= X.view <= 4) & (1 <= X.condition <= 5) & (1 <= X.grade <= 13)]
    if y is not None:
        y = y[X.index]

    # Create separate features for each categorical col
    # The new features will have 'zipcode_' as a prefix in their column names
    X = pd.get_dummies(X, prefix='zipcode_', columns=['zipcode'])
    X = pd.get_dummies(X, prefix='waterfront_', columns=['waterfront'])
    X = pd.get_dummies(X, prefix='view_', columns=['view'])
    X = pd.get_dummies(X, prefix='condition_', columns=['condition'])
    X = pd.get_dummies(X, prefix='grade_', columns=['grade'])

    # Add new col for total square,price per square foot, age of the home, and total number of rooms
    X["total_sqft"] = X["sqft_living"] + X["sqft_lot"]
    if y is not None:
        X["price_per_sqft"] = X["price"] / X["total_sqft"]
    X["age_of_home"] = 2023 - X["yr_built"]
    X["rooms"] = X["bedrooms"] + X["showers"]


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
    raise NotImplementedError()


if __name__ == '__main__':
    np.random.seed(0)
    df = pd.read_csv("../datasets/house_prices.csv")

    # Question 1 - split data into train and test sets

    # create DataFrame without price column - X
    df_no_price = df.drop('price', axis=1)
    # create DataFrame with only price column - y
    df_price = df[['price']]
    train_X, train_y, test_X, test_y = split_train_test(df_no_price, df_price, train_proportion=0.75)

    # Question 2 - Preprocessing of housing prices dataset
    raise NotImplementedError()

    # Question 3 - Feature evaluation with respect to response
    raise NotImplementedError()

    # Question 4 - Fit model over increasing percentages of the overall training data
    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    #   4) Store average and variance of loss over test set
    # Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)
    raise NotImplementedError()
