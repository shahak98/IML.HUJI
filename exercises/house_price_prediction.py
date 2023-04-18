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
    index_to_keep = X.drop_duplicates().index  # remember the index of the rows to keep
    X = X.loc[index_to_keep]
    y = y.loc[index_to_keep]  # remove corresponding rows from y

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

    # Create separate features for each categorical col
    # The new features will have 'zipcode_' for example as a prefix in their column names

    X = pd.get_dummies(X, prefix='zipcode_', columns=['zipcode'])
    X = pd.get_dummies(X, prefix='waterfront_', columns=['waterfront'])
    # for view,grade,condition this will be in addition to the original cols
    view_dummies = pd.get_dummies(pd.cut(X['view'], bins=[0, 1, 2, 3, 4], labels=['view_0_1', 'view_1_2', 'view_2_3',
                                                                                  'view_3_4'], include_lowest=True),
                                  prefix='view')
    condition_dummies = pd.get_dummies(pd.cut(X['condition'],
                                              bins=[1, 2, 3, 4, 5], labels=['condition_1_2', 'condition_2_3',
                                                                            'condition_3_4', 'condition_4_5'],
                                              include_lowest=True), prefix='condition')
    grade_dummies = pd.get_dummies(
        pd.cut(X['grade'], bins=[1, 3, 7, 11, 13], labels=['grade_1_3', 'grade_3_7', 'grade_7_11', 'grade_11_13'],
               include_lowest=True),prefix='grade')
    X = pd.concat([X, view_dummies, condition_dummies, grade_dummies], axis=1)

    # Add new col for total square,price per square foot, age of the home, and total number of rooms
    X["total_sqft"] = X["sqft_living"] + X["sqft_lot"]
    X["price_per_sqft"] = y / X["total_sqft"]
    X["age_of_home"] = 2023 - X["yr_built"]
    X["rooms"] = X["bedrooms"] + X["bathrooms"]

    return X, y


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
    # If y is not null, delete all rows where y is null, and the corresponding rows in X
    # this rows cannot be used for learning
    if y is not None:
        null_indices = pd.isnull(y)  # Find indices of null values in y
        X = X.drop(X.index[null_indices])
        y = y.drop(y.index[null_indices])

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
        y = y.loc[X.index]  # remove corresponding rows from y
        y = y[(y.price > 0)]
        X = X.loc[y.index]  # remove corresponding rows from X

    # remove rows that the floor,sqft_above,sqft_basement, yr_built, yr_renovated, sqft_living15,sqft_lot15  is negative
    X = X[(X.floors >= 0) & (X.sqft_above >= 0) & (X.sqft_basement >= 0) & (X.yr_built >= 0) & (X.yr_renovated >= 0) &
          (X.sqft_living15 >= 0) & (X.sqft_lot15 >= 0)]
    if y is not None:
        y = y.loc[X.index]  # remove corresponding rows from y

    # view - An index from 0-4 of how good the view of the property was (remove rows that are not fit to this numbers)
    # condition - An index from 1 to 5 on the condition of the apartment (remove rows that are not fit to this numbers)
    # grade - An index from 1 to 13 (remove rows that are not fit to this numbers)
    X = X[(0 <= X.view) & (X.view <= 4) & (1 <= X.condition) & (X.condition <= 5) & (X.grade <= 13) & (1 <= X.grade)]
    if y is not None:
        y = y.loc[X.index]

    # Create separate features for each categorical col
    # The new features will have 'zipcode_' for example as a prefix in their column names
    X = pd.get_dummies(X, prefix='zipcode_', columns=['zipcode'])
    X = pd.get_dummies(X, prefix='waterfront_', columns=['waterfront'])

    X['view'] = pd.cut(X['view'], bins=[0, 1, 2, 3, 4], labels=['view_0_1', 'view_1_2', 'view_2_3', 'view_3_4'],
                       include_lowest=True)  # create separate features for each range of values
    X = pd.get_dummies(X, prefix='', columns=['view'])
    X['condition'] = pd.cut(X['condition'], bins=[1, 2, 3, 4, 5], labels=['condition_1_2', 'condition_2_3',
                                                                          'condition_3_4', 'condition_4_5'],
                            include_lowest=True)
    X = pd.get_dummies(X, prefix='', columns=['condition'])
    X['grade'] = pd.cut(X['grade'], bins=[1, 3, 7, 11, 13], labels=['grade_1_3', 'grade_3_7', 'grade_7_11',
                                                                    'grade_11_13'], include_lowest=True)
    X = pd.get_dummies(X, prefix='', columns=['grade'])

    # Add new col for total square,price per square foot, age of the home, and total number of rooms
    X["total_sqft"] = X["sqft_living"] + X["sqft_lot"]
    if y is not None:
        X["price_per_sqft"] = y["price"] / X["total_sqft"]
    X["age_of_home"] = 2023 - X["yr_built"]
    X["rooms"] = X["bedrooms"] + X["bathrooms"]

    return X, y


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
    dummy_cols = ['zipcode_', 'waterfront_', 'view_', 'condition_', 'grade_']
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
    feature_evaluation(train_X, train_y)

    # Question 4 - Fit model over increasing percentages of the overall training data
    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    #   4) Store average and variance of loss over test set
    # Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)
