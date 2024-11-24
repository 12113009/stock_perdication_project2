import streamlit as st
from datetime import date
import yfinance as yf
import statsmodels.api as sm
import pandas as pd
import numpy as np
from plotly import graph_objs as go


# Set the title of the Streamlit app
st.title("Stock Prediction App")

# User input for stock ticker
stock_ticker = st.text_input("Enter stock ticker:", "AAPL")

# User input for date range
start_date = st.date_input("Start Date", value=pd.to_datetime("2015-01-01"))
end_date = st.date_input("End Date", value=date.today())

# Convert date inputs to strings
start = start_date.strftime("%Y-%m-%d")
today = end_date.strftime("%Y-%m-%d")

# Function to load data
def load_data(ticker, start, end):
    data = yf.download(ticker, start, end)
    data.reset_index(inplace=True)
    return data

# Load and display raw data
data_load_state = st.text("Loading data...")
data = load_data(stock_ticker, start, today)
data_load_state.text("Loading data... done!")
st.subheader("Raw data")
st.write(data.tail())

# Function to plot raw data
def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name='stock_open',line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name='stock_close',line=dict(color='red')))
    fig.layout.update(title_text="Time Series Data", xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

# Plot raw data
plot_raw_data()

# Forecasting with `statsmodels`
df_train = data[['Date', 'Close']]
df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

# Set the date column as index
df_train.set_index('ds', inplace=True)

# Perform time series forecasting using `statsmodels`
model = sm.tsa.statespace.SARIMAX(df_train['y'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
results = model.fit()

# Predict future values
n_years = st.slider("Years of prediction:", 1, 4)
period = n_years * 365

future_dates = pd.date_range(df_train.index[-1], periods=period, freq='D')
future_dates = future_dates[1:]  # Exclude the start date
forecast = results.get_forecast(steps=period).summary_frame()

# Plot the forecasted data
st.subheader('Forecast data')
st.write(forecast.tail())

# Plot forecast data
fig1 = go.Figure()
fig1.add_trace(go.Scatter(x=df_train.index, y=df_train['y'], name='Original'))
fig1.add_trace(go.Scatter(x=future_dates, y=forecast['mean'], name='Future Prediction',line=dict(color='green')))
fig1.layout.update(title_text='Time Series Forecast', xaxis_rangeslider_visible=True)
st.plotly_chart(fig1)









