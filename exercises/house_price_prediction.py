from IMLearn.utils import split_train_test
from IMLearn.learners.regressors import LinearRegression

from typing import NoReturn, Optional
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio

pio.templates.default = "simple_white"
Mean_train_col = {}
Names_col_train = []


def preprocess_train_valid_values(X: pd.DataFrame, y: Optional[pd.Series]):
    """
    preprocess of the data train, y is not null

    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector corresponding given samples
    Returns
    -------
    remove rows that the values in them not valid return new X and new Y
    """
    # remove rows that the num of bedrooms,sqft_living, sqft_lot is not positive
    X = X[(X.sqft_living > 0) & (X.sqft_lot > 0)]
    y = y.loc[X.index]  # remove corresponding rows from y

    # remove rows that the price of the house is not positive
    index_to_keep = y > 0
    X = X.loc[index_to_keep]  # remove corresponding rows from X
    y = y.loc[index_to_keep]

    # remove rows that the value  is negative
    X = X[(X.floors >= 0) & (X.sqft_above >= 0) & (X.sqft_basement >= 0) & (X.yr_built >= 0) & (X.yr_renovated >= 0) &
          (X.sqft_living15 >= 0) & (X.sqft_lot15 >= 0) & (X.bathrooms >= 0) & (X.bedrooms >= 0)]
    y = y.loc[X.index]  # remove corresponding rows from y

    # view - An index from 0-4 of how good the view of the property was (remove rows that are not fit to this numbers)
    # condition - An index from 1 to 5 on the condition of the apartment (remove rows that are not fit to this numbers)
    # grade - An index from 1 to 13 (remove rows that are not fit to this numbers)
    X = X[(0 <= X.view) & (X.view <= 4) & (1 <= X.condition) & (X.condition <= 5) & (X.grade <= 13) & (1 <= X.grade)]
    y = y.loc[X.index]

    return X, y


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
    # remove duplicate rows
    X.drop_duplicates(inplace=True)  # inplace = duplicates are removed directly from X
    y = y[X.index]  # remove corresponding rows from y

    # remove irrelevant col in X
    X = X.drop(["id", "lat", "long", "date"], axis=1)

    # Convert columns to numeric data types
    # If it is unable to convert a value to a numeric type, it will replace that value with null
    y = pd.to_numeric(y, errors='coerce')
    X['bedrooms'] = pd.to_numeric(X['bedrooms'], errors='coerce')
    X['sqft_living'] = pd.to_numeric(X['sqft_living'], errors='coerce')
    X['sqft_lot'] = pd.to_numeric(X['sqft_lot'], errors='coerce')
    X['floors'] = pd.to_numeric(X['floors'], errors='coerce')
    X['sqft_above'] = pd.to_numeric(X['sqft_above'], errors='coerce')
    X['sqft_basement'] = pd.to_numeric(X['sqft_basement'], errors='coerce')
    X['yr_built'] = pd.to_numeric(X['yr_built'], errors='coerce')
    X['yr_renovated'] = pd.to_numeric(X['yr_renovated'], errors='coerce')
    X['sqft_living15'] = pd.to_numeric(X['sqft_living15'], errors='coerce')
    X['sqft_lot15'] = pd.to_numeric(X['sqft_lot15'], errors='coerce')
    X['view'] = pd.to_numeric(X['view'], errors='coerce')
    X['condition'] = pd.to_numeric(X['condition'], errors='coerce')
    X['grade'] = pd.to_numeric(X['grade'], errors='coerce')
    X['bathrooms'] = pd.to_numeric(X['bathrooms'], errors='coerce')


    # remove rows that have null values in one or more features
    null_rows = X.isnull().any(axis=1)  # find rows with null values in X
    index_to_remove = null_rows[null_rows].index.tolist()  # get the indices of those rows
    X = X.drop(index_to_remove)
    y = y.drop(index_to_remove)

    X, y = preprocess_train_valid_values(X, y)  # remove rows that the values in them not valid

    # Add new col for total square,price per square foot, age of the home, and total number of rooms
    X["renovation_age"] = X["yr_renovated"] - X["yr_built"]
    X["rooms"] = X["bedrooms"] + X["bathrooms"]

    # Create separate features for each categorical col
    # The new features will have 'zipcode_' for example as a prefix in their column names
    X = pd.get_dummies(X, prefix='zipcode', columns=['zipcode'])
    X = pd.get_dummies(X, prefix='waterfront', columns=['waterfront'])

    # Convert the date column to a datetime format
    # X['date'] = pd.to_datetime(X['date'])

    # save the mean of each col in global variable (as dict), and the name of the col - will be use in preprocess test
    global Mean_train_col
    Mean_train_col = X.mean().to_dict()
    global Names_col_train
    Names_col_train = X.columns.tolist()

    return X, y


def preprocess_data_test(X: pd.DataFrame):
    """
    This is the basic preprocessing step on the test data, which only involves dropping columns and adding
    basic columns similar to those in the training data

    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem
    """
    # Make dummies for categorical col
    X = pd.get_dummies(X, prefix='zipcode', columns=['zipcode'])
    X = pd.get_dummies(X, prefix='waterfront', columns=['waterfront'])

    # make X_test to match cols in X_train, fill in any missing columns with 0
    X = X.reindex(columns=Names_col_train, fill_value=0)

    # Convert columns to numeric data types
    # If it is unable to convert a value to a numeric type, it will replace that value with null
    X['bedrooms'] = pd.to_numeric(X['bedrooms'], errors='coerce')
    X['sqft_living'] = pd.to_numeric(X['sqft_living'], errors='coerce')
    X['sqft_lot'] = pd.to_numeric(X['sqft_lot'], errors='coerce')
    X['floors'] = pd.to_numeric(X['floors'], errors='coerce')
    X['sqft_above'] = pd.to_numeric(X['sqft_above'], errors='coerce')
    X['sqft_basement'] = pd.to_numeric(X['sqft_basement'], errors='coerce')
    X['yr_built'] = pd.to_numeric(X['yr_built'], errors='coerce')
    X['yr_renovated'] = pd.to_numeric(X['yr_renovated'], errors='coerce')
    X['sqft_living15'] = pd.to_numeric(X['sqft_living15'], errors='coerce')
    X['sqft_lot15'] = pd.to_numeric(X['sqft_lot15'], errors='coerce')
    X['view'] = pd.to_numeric(X['view'], errors='coerce')
    X['condition'] = pd.to_numeric(X['condition'], errors='coerce')
    X['grade'] = pd.to_numeric(X['grade'], errors='coerce')
    X['bathrooms'] = pd.to_numeric(X['bathrooms'], errors='coerce')

    # Replace not valid values in X_test with their corresponding column mean from X_train
    for f in X.columns:
        # Replace null values
        if X[f].isnull().any():  # there is at least one missing value in the column
            X[f].fillna(Mean_train_col[f],
                        inplace=True)  # inplace=True specifies that the operation should be in place
        # print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)
        # Check and replace values in view/condition/grade column that are not valid
        if f == 'view':
            values_to_replace = (X['view'] < 0) | (X['view'] > 4)
            if values_to_replace.any():  # check if any value in 'view' col fails condition
                X.loc[values_to_replace, 'view'] = Mean_train_col[f]
        elif f == 'condition':
            values_to_replace = (X[f] < 1) | (X[f] > 5)
            if values_to_replace.any():
                X.loc[values_to_replace, f] = Mean_train_col[f]
        elif f == 'grade':
            values_to_replace = (X[f] < 1) | (X[f] > 13)
            if values_to_replace.any():
                X.loc[values_to_replace, f] = Mean_train_col[f]
        # Check and replace values in bedrooms,sqft_living, sqft_lot column to be positive
        elif f in ['sqft_living', 'sqft_lot']:
            values_to_replace = (X[f] <= 0)
            if values_to_replace.any():
                X.loc[values_to_replace, f] = Mean_train_col[f]
        # Check and replace values in specific column to be non-negative
        elif f in ['floors', 'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated', 'sqft_living15', 'sqft_lot15',
                   'bathrooms', 'bedrooms']:
            values_to_replace = (X[f] < 0)
            if values_to_replace.any():
                X.loc[values_to_replace, f] = Mean_train_col[f]

    # Add new col for total square,price per square foot, age of the home, and total number of rooms
    X["renovation_age"] = X["yr_renovated"] - X["yr_built"]
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

    # Remove rows that doesn't have a value in price col
    df = df.dropna(subset=['price'])

    # create DataFrame without price column - X
    df_no_price = df.drop('price', axis=1)
    # create DataFrame with only price column - y
    df_price = df['price']
    train_X, train_y, test_X, test_y = split_train_test(df_no_price, df_price, train_proportion=0.75)

    # Question 2 - Preprocessing of housing prices dataset
    train_X, train_y = preprocess_data(train_X, train_y)
    test_X = preprocess_data(test_X)



    # Question 3 - Feature evaluation with respect to response
    feature_evaluation(train_X, train_y)

    # Question 4 - Fit model over increasing percentages of the overall training data
    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    #   4) Store average and variance of loss over test set
    # Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)

    percentage = list(range(10, 101))  # list of percentage values from 10 to 100 (inclusive)
    num_runs = 10  # number of times the inner loop will be executed for each percentage value
    results = np.zeros((len(percentage), num_runs))  # This array will be used to store the results

    for i in range(len(percentage)):
        p = percentage[i]
        for j in range(results.shape[1]):
            X_percentage = train_X.sample(frac=p / 100.0)  # take the p% of the train set
            y_percentage = train_y.loc[X_percentage.index]  # corresponding y values for the p% of the training data
            model = LinearRegression(include_intercept=True).fit(X_percentage, y_percentage)  # Fit linear model
            results[i, j] = model.loss(test_X, test_y)

    mean = results.mean(axis=1)  # mean loss values for each percentage
    sdt_2 = 2 * results.std(axis=1)  # (standard deviation of the loss values for each percentage) *2
    confidence_interval_plus = mean + sdt_2
    confidence_interval_minus = mean - sdt_2
    fig = go.Figure(
        [go.Scatter(x=percentage, y=confidence_interval_minus, mode="lines", line=dict(color='grey', width=1)),
         go.Scatter(x=percentage, y=confidence_interval_plus, fill='tonexty', mode="lines",
                    line=dict(color='grey', width=1)),
         go.Scatter(x=percentage, y=mean, mode="markers+lines",
                    line=dict(color='light blue', width=2), marker=dict(color='light blue', size=4))],
        layout=go.Layout(title="Effect of training size on mean squared error",
                         xaxis=dict(title="Percentage of Training Set"),
                         yaxis=dict(title="mean squared error")))
    fig.write_image("mse.png", engine='orca')
