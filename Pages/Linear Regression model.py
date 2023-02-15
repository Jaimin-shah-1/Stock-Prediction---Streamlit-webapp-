import streamlit as st 

st.set_page_config(
    page_title= " Linear Regression model",
    page_icon= "chart_with_upwards_trend",
    )

st.title("Linear Regression model")

st.sidebar.success("select a page above.")

import pandas as pd
import numpy as np
import datetime as dt
import yfinance as yf
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import plotly.express as px
from prophet.plot import plot_plotly
from plotly import graph_objs as go
from pandas_datareader import data as pdr
from datetime import date
import datetime
import warnings
warnings.simplefilter("ignore")

# Set date range for data
st.sidebar.subheader('Parameters')
start = st.sidebar.date_input("Start date", datetime.date(2001, 1, 1))
end = st.sidebar.date_input("End date", datetime.date(2021, 1, 1))

st.subheader("Stock Prediction App")

# Create the slider
# num_months = st.slider('Number of months to predict:', min_value=1, max_value=12, value=1)
# period = num_months*30

# Load data
stocks = pd.read_csv("EQUITY_L.csv")
selected_stock = st.sidebar.selectbox("Select Dataset for Prediction,", stocks)
tickerData = yf.Ticker(selected_stock) # Get ticker data

@st.cache(allow_output_mutation=True)
def load_data(ticker):
    df = yf.download(ticker, start, end)
    df.reset_index(inplace=True)
    return df

df = load_data(selected_stock)

#Raw Data Showing
st.subheader('Raw Data')
st.write(df) 
#-------------------------------------------------------------------
# From here the  jupyter notebook code will be continued 
# # User-inputted stock ticker
# stock_ticker = input("Enter the stock ticker: ")
# # Define a start date and End Date
# start = dt.datetime(2001,1,1)
# end =  dt.datetime(2021,1,1)
# # Read Stock Price Data 
# df = yf.download(stock_ticker, start , end)
# df.reset_index(inplace=True)


# Use only the 'Close' data as the target
y = df['Close']

# Scale the data if necessary
scaler = StandardScaler()
y = scaler.fit_transform(y.values.reshape(-1, 1))

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df.index.to_frame(), y, test_size=0.3, random_state=0)

# Create a Ridge Regression model and fit it to the training data
regressor = Ridge(alpha=0.5)
regressor.fit(X_train, y_train)

# Use the model to make predictions on the test data
y_pred_ridge = regressor.predict(X_test)

# Calculate the mean squared error, mean absolute error, and R2 score to evaluate the accuracy of the model
mse_ridge = mean_squared_error(y_test, y_pred_ridge)
mae_ridge = mean_absolute_error(y_test, y_pred_ridge)
rmse_ridge = np.sqrt(mse_ridge)
r2_ridge = r2_score(y_test, y_pred_ridge)

st.subheader("#Ridge Regression Results:#")
st.write("Mean Squared Error:", mse_ridge)
st.write("Root Mean Squared Error:", rmse_ridge)
st.write("Mean Absolute Error:", mae_ridge)
st.write("R2 Score:", r2_ridge)

# Create a Random Forest Regressor and fit it to the training data
regressor = RandomForestRegressor(n_estimators=100)
regressor.fit(X_train, y_train.flatten())

# Use the model to make predictions on the test data
y_pred_rf = regressor.predict(X_test)

# Calculate the mean squared error, mean absolute error, and R2 score to evaluate the accuracy of the model
mse_rf = mean_squared_error(y_test, y_pred_rf)
mae_rf = mean_absolute_error(y_test, y_pred_rf)
rmse_rf = np.sqrt(mse_rf)

from sklearn.ensemble import RandomForestRegressor
# Create a Random Forest Regression model and fit it to the training data
regressor_rf = RandomForestRegressor(n_estimators=100, random_state=0)
regressor_rf.fit(X_train, y_train.flatten())

# Use the model to make predictions on the test data
y_pred_rf = regressor_rf.predict(X_test)

# Calculate the mean squared error, mean absolute error, and R2 score to evaluate the accuracy of the model
mse_rf = mean_squared_error(y_test, y_pred_rf)
mae_rf = mean_absolute_error(y_test, y_pred_rf)
rmse_rf = np.sqrt(mse_rf)
r2_rf = r2_score(y_test, y_pred_rf)

st.write('-'*50)
st.subheader("#Random Forest Regression Results:#")
st.write("Mean Squared Error:", mse_rf)
st.write("Root Mean Squared Error:", rmse_rf)
st.write("Mean Absolute Error:", mae_rf)
st.write("R2 Score:", r2_rf)

from sklearn.linear_model import LinearRegression
# Create a Linear Regression model and fit it to the training data
linear_regressor = LinearRegression()
linear_regressor.fit(X_train, y_train)

# Use the model to make predictions on the test data
y_pred_linear = linear_regressor.predict(X_test)

# Calculate the mean squared error, mean absolute error, and R2 score to evaluate the accuracy of the linear regression model
mse_linear = mean_squared_error(y_test, y_pred_linear)
mae_linear = mean_absolute_error(y_test, y_pred_linear)
rmse_linear = np.sqrt(mse_linear)
r2_linear = r2_score(y_test, y_pred_linear)

st.write('-'*50)
st.subheader("#Linear Regression Results:#")
st.write("Linear Regression Model:")
st.write("Mean Squared Error:", mse_linear)
st.write("Root Mean Squared Error:", rmse_linear)
st.write("Mean Absolute Error:", mae_linear)
st.write("R2 Score:", r2_linear)

from sklearn.tree import DecisionTreeRegressor
# Create a Decision Tree Regression model and fit it to the training data
dtr = DecisionTreeRegressor(random_state=0)
dtr.fit(X_train, y_train)

# Use the model to make predictions on the test data
y_dtr_pred = dtr.predict(X_test)

# Calculate the mean squared error, mean absolute error, and R2 score to evaluate the accuracy of the model
mse_dtr = mean_squared_error(y_test, y_dtr_pred)
mae_dtr = mean_absolute_error(y_test, y_dtr_pred)
rmse_dtr = np.sqrt(mse_dtr)
r2_dtr = r2_score(y_test, y_dtr_pred)

st.write('-'*50)
st.subheader("#Decision Tree Regression Results:#")
st.write("Decision Tree Regression")
st.write("Mean Squared Error:", mse_dtr)
st.write("Root Mean Squared Error:", rmse_dtr)
st.write("Mean Absolute Error:", mae_dtr)
st.write("R2 Score:", r2_dtr)

# # Plot the actual vs predicted stock prices for the linear regression model
# fig = px.scatter(x=y_test.flatten(), y=y_pred_linear.flatten(), title='Linear Regression Model - Actual vs Predicted Stock Prices')
# fig.show()

st.write('-'*50)
import matplotlib.pyplot as plt
# Plot the Ridge Regression Predictions
fig = plt.figure(figsize=(10,5))
plt.plot(y_test, label='True')
plt.plot(y_pred_ridge, label='Ridge Regression')
plt.legend()
plt.grid()
plt.title('Ridge Regression Predictions')
plt.show()
st.pyplot(fig)

# Plot the Random Forest Regression Predictions
fig1 = plt.figure(figsize=(10,5))
plt.plot(y_test, label='True')
plt.plot(y_pred_rf, label='Random Forest Regression')
plt.legend()
plt.title('Random Forest Regression Predictions')
plt.show()
plt.grid()
st.pyplot(fig1)

# Plot the Linear Regression Predictions
fig2 = plt.figure(figsize=(10,5))
plt.plot(y_test, label='True')
plt.plot(y_pred_linear, label='Linear Regression')
plt.legend()
plt.title('Linear Regression Predictions')
plt.show()
plt.grid()
st.pyplot(fig2)


df['MA10'] = df['Close'].rolling(window=10).mean()
df['MA20'] = df['Close'].rolling(window=20).mean()
df['MA100'] = df['Close'].rolling(window=100).mean()

# Plot raw data
def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['Date'], y=df['Open'], name='Stock_open'))
    fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], name='Stock_close'))
    fig.add_trace(go.Scatter(x=df['Date'], y=df['High'], name='Stock_High'))
    fig.add_trace(go.Scatter(x=df['Date'], y=df['Low'], name='Stock_Low'))
    fig.add_trace(go.Scatter(x=df['Date'], y=df['Adj Close'], name='Stock_Adj Close'))
    fig.add_trace(go.Scatter(x=df['Date'], y=df['MA100'], name='Moving Avrages - 100'))
    fig.add_trace(go.Scatter(x=df['Date'], y=df['MA10'], name='Moving Avrages - 10'))
    fig.add_trace(go.Scatter(x=df['Date'], y=df['MA20'], name='Moving Avrages - 20'))
    fig.layout.update(title_text="Time Series Data", xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)
      
plot_raw_data()
