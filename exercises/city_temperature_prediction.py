import IMLearn.learners.regressors.linear_regression
from IMLearn.learners.regressors import PolynomialFitting
from IMLearn.utils import split_train_test

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio

pio.templates.default = "simple_white"


def load_data(filename: str) -> pd.DataFrame:
    """
    Load city daily temperature dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (Temp)
    """
    # Use the parse_dates argument of the pandas.read_csv to set the type of the `Date` column
    df = pd.read_csv(filename, parse_dates=["Date"]).dropna().drop_duplicates()
    df["DayOfYear"] = df["Date"].dt.dayofyear  # add "dayofyear" col
    df = df[(df.Year > 0)]  # Keep only the rows where the year is greater than zero
    df = df[(1 <= df.Month) & (df.Month <= 12)]  # Keep only the rows where the month value is valid
    df = df[(1 <= df.Day) & (df.Day <= 31)]  # Keep only the rows where the Day value is valid
    df = df[df.Temp > -70]
    df = df[df.Temp < 70]
    df["Year"] = df["Year"].astype(str)
    return df


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of city temperature dataset
    df = load_data("../datasets/city_temperature.csv")

    # Question 2 - Exploring data for specific country
    df_israel = df[df['Country'] == 'Israel']  # samples only from the country of Israel
    px.scatter(df_israel, x="DayOfYear", y="Temp", color="Year",
               title="Average daily temperature change as a function of DayOfYear") \
        .write_image("temp_vs_DayOfYear.png", engine='orca')

    df_israel_month = df_israel.groupby(["Month"], as_index=False)  # as_index=False making Month to be a regular column
    # compute the standard deviation of temperature for each month
    df_israel_month = df_israel_month.agg(std=("Temp", "std"))
    px.bar(df_israel_month,
           title="Monthly Standard Deviation of Daily Temperatures", x="Month", y="std") \
        .write_image("israel.monthly.std.temperature.png", engine='orca')

    # Question 3 - Exploring differences between countries

    # agg() adds suffixes to the original column('Temp') to indicate the operation performed.
    # _x is added to the Temp column in df_monthly_avg, while _y is added to the Temp column in df_monthly_std
    df_monthly_avg = df.groupby(['Country', 'Month'], as_index=False).agg({'Temp': 'mean'})
    df_monthly_std = df.groupby(['Country', 'Month'], as_index=False).agg({'Temp': 'std'})
    df_monthly = df_monthly_avg.merge(df_monthly_std, on=['Country', 'Month'])  # Merge the dataframes
    fig = px.line(df_monthly, x='Month', y='Temp_x', error_y='Temp_y', color='Country',
                  title="average and standard deviation of the temperature")
    fig.update_layout(xaxis_title="Month", yaxis_title="Temp")
    fig.write_image("avg.std.tmp.png", engine='orca')

    # Question 4 - Fitting model for different values of `k`

    train_X, train_y, test_X, test_y = split_train_test(df_israel.DayOfYear, df_israel.Temp, train_proportion=0.75)
    # Convert the training and testing data to NumPy arrays
    train_X, train_y, test_X, test_y = train_X.to_numpy(), train_y.to_numpy(), test_X.to_numpy(), test_y.to_numpy()
    k_degree = list(range(1, 11))  # k ∈ [1,10] degree of polynomial
    loss = np.zeros_like(k_degree,
                         dtype=float)  # Return an array of zeros with the same shape and type as a given array
    for i in range(len(k_degree)):
        model = PolynomialFitting(k=k_degree[i]).fit(train_X, train_y)  # Fit a polynomial model to the training data
        loss[i] = np.round(model.loss(test_X, test_y), 2)

    # Plot the loss values for each degree
    loss = pd.DataFrame(dict(k=k_degree, loss=loss))
    px.bar(loss, x="k", y="loss", text="loss",
           title="MSE For Different Values of degree") \
        .write_image("israel.different.degree.png", engine='orca')
    print(loss)  # Print the test error recorded for each value of k

    # Question 5 - Evaluating fitted model on different countries

    #  fit a model using a polynomial of degree 5 to the temperature data of Israel
    model = PolynomialFitting(k=5).fit(df_israel.DayOfYear.to_numpy(), df_israel.Temp.to_numpy())
    # make list of all the countries without Israel
    countries = df.Country.unique().tolist()
    countries.remove('Israel')

    data = {}
    for c in countries:
        # Compute the model error for the current country and store it in the data dictionary
        data[c] = round(model.loss(df[df.Country == c].DayOfYear, df[df.Country == c].Temp), 2)

    fig = px.bar(x=countries, y=data, labels={"x": "Country", "y": "Error"}, color=countries)
    fig.update_layout(title_text="Model's Error by Country")
    fig.write_image("test.other.countries.png",  engine='orca')




