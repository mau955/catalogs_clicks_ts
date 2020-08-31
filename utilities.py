#!/usr/bin/env python
# coding: utf-8


# -------------------------------------------------------
# PREDICTING CUSTOMER WEBSITE CLICKS
# -------------------------------------------------------


# importing libraries
import pandas as pd
import matplotlib.pyplot as plt

from fbprophet import Prophet
from scipy import stats
import statsmodels.formula.api as smf


# parameters
fs = (12, 6)
plt.style.use("ggplot")


# read catalogs data set
def read_catalogs_data(cat_file_path, f="W"):
    # importing the data set
    nbf_catalogs = pd.read_excel(cat_file_path)

    # cleaning the data set
    nbf_catalogs.columns = nbf_catalogs.columns.str.lower()
    nbf_catalogs = nbf_catalogs[~nbf_catalogs["date"].isna()]
    nbf_catalogs["date"] = pd.to_datetime(nbf_catalogs["date"], format="%d-%m-%y")
    nbf_catalogs.set_index("date", inplace=True)
    nbf_catalogs.columns = ["no_catalogs"]

    # group the data set
    nbf_catalogs_ts = nbf_catalogs.resample(f).sum()

    # explore the data set

    # print(nbf_catalogs.dtypes)
    # print(nbf_catalogs.head())
    # print(nbf_catalogs.sort_index().tail())

    # print("\n")
    # print("-" * 70)
    # print("Catalogs Descriptive Statistics:")
    # print(nbf_catalogs.describe())
    #
    # print("-" * 70)
    # print("Grouped Catalogs Descriptive Statistics:")
    # print(nbf_catalogs_ts.describe())
    #
    # print("-" * 70)
    # print("Catalogs Data Range:", str(nbf_catalogs.index.min()), str(nbf_catalogs.index.max()))
    # print("-" * 70)
    # print("Grouped Catalogs Data Range:", str(nbf_catalogs_ts.index.min()), str(nbf_catalogs_ts.index.max()))
    # print("-" * 70)

    return nbf_catalogs_ts


# read clicks data
def read_clicks_data(click_file_path, f="W"):
    # importing the data set
    nbf_click = pd.read_excel(click_file_path, sheet_name="Dataset1", parse_dates=["Date"], index_col=1)

    # cleaning the data set
    nbf_click.columns = nbf_click.columns.str.lower()
    nbf_click.index.name = nbf_click.index.name.lower()
    nbf_click = nbf_click[~nbf_click["source"].isna()]

    # group the data set
    nbf_click_ts = nbf_click["users"].resample(f).sum().to_frame("no_clicks")

    # explore the data set

    print("-" * 70)
    print(nbf_click["source"].unique())

    # print("-" * 70)
    # print(nbf_click.head())
    #
    # print("-" * 70)
    # print(nbf_click["source"].isna().sum())
    #
    # print("-" * 70)
    # print(str(nbf_click.index.min()), str(nbf_click.index.max()))
    # print("-" * 70)
    # print(str(nbf_click_ts.index.min()), str(nbf_click_ts.index.max()))

    return nbf_click_ts


# merge catalogs data and clicks data
def merge_catalogs_clicks(cat_ts, click_ts):
    cat_click_ts = cat_ts.merge(click_ts, left_index=True, right_index=True, how="right")

    # cleaning the data set
    cat_click_ts.fillna(0, inplace=True)

    # correlation of the variables
    corr, p = stats.pearsonr(cat_click_ts['no_clicks'], cat_click_ts['no_catalogs'])

    # OLS regression
    model = smf.ols(formula='no_clicks ~ no_catalogs', data=cat_click_ts).fit()

    # explore the data set

    # print("-" * 70)
    # print(cat_click_ts.isna().sum())
    #
    # print("-" * 70)
    # print(cat_click_ts.describe())
    #
    # print("-" * 70)
    # print(cat_click_ts.tail())
    #
    # print("-" * 70)
    # print(str(cat_click_ts.index.min()), str(cat_click_ts.index.max()))

    print("-" * 70)
    print("corr", corr)
    print("p-value", p)

    print("-" * 70)
    print(model.summary())

    return cat_click_ts


# time-series analysis
def clicks_ts(cat_click_ts, n=13, f="W"):
    # select target column
    click_ts = cat_click_ts["no_clicks"].to_frame("y")
    click_ts.index.name = "ds"

    # plot time-series
    # fig, ax = plt.subplots(figsize=(12, 6))
    # click_ts.plot(kind="line", ax=ax, color="r")
    # ax.set_title("Direct Clicks Time-series")
    # ax.set_xlabel("Time")
    # ax.set_ylabel("# Clicks")
    # ax.legend("")

    # wrangle data set
    click_ts = click_ts.reset_index()

    # fit model
    m = Prophet(weekly_seasonality=False, daily_seasonality=False)
    m.fit(click_ts)

    # extend into the future
    future = m.make_future_dataframe(periods=n, freq=f, include_history=True)

    # make prediction
    forecast = m.predict(future)

    # visualize forecast
    fig1 = m.plot(forecast)

    # visualize components
    # fig2 = m.plot_components(forecast)

    # explore the data set

    # print("-" * 70)
    # print(future.tail())

    print("-" * 70)
    print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']][-n:])
