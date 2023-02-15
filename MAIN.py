import streamlit as st 

st.set_page_config(
    page_title= "Prophet model",
    page_icon= "chart_with_upwards_trend",
    )
st.title("Stock Prediction App")
st.subheader("***Prophet model***")
st.sidebar.success("select a page above.")

import numpy as np
import pandas as pd
from pandas_datareader import data as pdr
import streamlit as st 
from datetime import date
import datetime
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
import warnings
warnings.simplefilter("ignore")
import plotly.express as px 

# Set date range for data
st.sidebar.subheader('Parameters')
START = st.sidebar.date_input("Start date", datetime.date(2015, 1, 1))
TODAY = st.sidebar.date_input("End date", datetime.date(2022, 12, 1))
# 
# st.title("Stock Prediction App")
# Create the slider
num_months = st.slider('Number of months to predict:', min_value=1, max_value=12, value=1)
period = num_months*30

# Load data
stocks = pd.read_csv("EQUITY_L.csv")
selected_stock = st.sidebar.selectbox("Select Dataset for Prediction,", stocks)
tickerData = yf.Ticker(selected_stock) # Get ticker data

@st.cache(allow_output_mutation=True)
# @st.cache
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data

data = load_data(selected_stock)

# # Ticker information
# string_logo = '<img src=%s>' % tickerData.info['logo_url']
# st.markdown(string_logo, unsafe_allow_html=True)

# string_name = tickerData.info['longName']
# st.header('**%s**' % string_name)

# string_summary = tickerData.info['longBusinessSummary']
# st.info(string_summary)


# Raw Data Showing
st.subheader('Raw Data')
st.write(data) 


data['MA10'] = data['Close'].rolling(window=10).mean()
data['MA20'] = data['Close'].rolling(window=20).mean()
data['MA100'] = data['Close'].rolling(window=100).mean()

# Plot raw data
def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name='Stock_open'))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name='Stock_close'))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['High'], name='Stock_High'))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Low'], name='Stock_Low'))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Adj Close'], name='Stock_Adj Close'))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['MA100'], name='Moving Avrages - 100'))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['MA10'], name='Moving Avrages - 10'))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['MA20'], name='Moving Avrages - 20'))
    fig.layout.update(title_text="Time Series Data", xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)
      
plot_raw_data()

# Forcasting the data
df_train = data[['Date','Close']]
df_train = df_train[df_train['Date'].dt.dayofweek < 5]  # Remove Saturdays and Sundays
df_train = df_train.rename(columns={"Date":"ds", "Close": "y"})

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(df_train.iloc[:, :-1], df_train.iloc[:, -1], test_size=0.2)

# Fit ARIMA model on training data
arima_model = sm.tsa.ARIMA(y_train, order=(1, 1, 1)).fit()

# Make predictions on test data
arima_predictions = arima_model.predict(start=len(X_train), end=len(X_train)+len(X_test)-1, dynamic=False)

# Evaluate model performance
mse = np.mean((arima_predictions - y_test)**2)
rmse = np.sqrt(mse)
st.subheader("ARIMA model performance: Root Mean Squared Error = " + str(rmse))


# Use Prophet model
m = Prophet()
m.fit(df_train)
future = m.make_future_dataframe(periods=period, freq='D')
forecast = m.predict(future)

# Plot Prophet predictions
st.subheader("Prophet Model")
fig = plot_plotly(m, forecast)
st.plotly_chart(fig)

# Compare predictions
st.subheader("Comparison of Prophet and ARIMA models")
df = pd.DataFrame({'Actual': y_test, 'Prophet': forecast['yhat'][-len(y_test):], 'ARIMA': arima_predictions})
df = df.fillna(0)  # Replace NaN values with 0
st.line_chart(df)


# Compare predictions
st.subheader("Comparison of Actual, Prophet and ARIMA models")
df = pd.DataFrame({'Actual': df_train['y'], 'Prophet': forecast['yhat'], 'ARIMA': arima_predictions})
df = df.fillna(0)  # Replace NaN values with 0

# Customize chart
chart = go.Figure(data=[
    go.Scatter(name='Actual', x=df.index, y=df['Actual'], hovertext='Actual', line=dict(color='blue')),
    go.Scatter(name='Prophet', x=df.index, y=df['Prophet'], hovertext='Prophet', line=dict(color='red')),
    go.Scatter(name='ARIMA', x=df.index, y=df['ARIMA'], hovertext='ARIMA', line=dict(color='green'))
])

# Add hover text
chart.update_traces(hovertemplate='<b>%{Clsoe}</b><br>Value: %{y:,.2f}')

# Add axis labels
chart.update_layout(xaxis_title='Date', yaxis_title='Price')

st.plotly_chart(chart)
