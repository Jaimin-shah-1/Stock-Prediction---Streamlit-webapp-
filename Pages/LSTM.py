# Creating the webpage for showing in streamlit web app.
import streamlit as st 

st.set_page_config(
    page_title= "LSTM model",
    page_icon= "📊",
)
st.title('Stock Trend Predecion')
st.subheader("***LSTM model***")
st.sidebar.success("select a page above.")

# Importing nessary libraries 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as data
import keras.models 
from tensorflow import keras
from keras.layers import Dense, Dropout, LSTM
from keras.models import Sequential
# import streamlit as st 
import warnings
import datetime
import yfinance as yf
warnings.filterwarnings("ignore", category=DeprecationWarning) 
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go


st.sidebar.subheader('Parameters')
start = st.sidebar.date_input("Start date", datetime.date(2010, 1, 1))
end = st.sidebar.date_input("End date", datetime.date(2020, 1, 1))

# Loading the stock data form csv file.
stocks = pd.read_csv("EQUITY_L.csv")
selected_stock = st.sidebar.selectbox("Select Dataset for Prediction,", stocks)
tickerData = yf.Ticker(selected_stock) # Get ticker data

@st.cache(allow_output_mutation=True)
# @st.cache
def load_data(ticker):
    df = yf.download(ticker, start, end)
    df.reset_index(inplace=True)
    return df

df = load_data(selected_stock)
#---------------------------------------------------------------------------------------------------------"
#ORIGINAL DATA#

# start = '2015-01-01'
# end = '2020-12-31'

# user_input = st.text_input('Enter Stock Ticker', 'ITC.NS')
# df = data.DataReader(user_input, 'yahoo', start, end) #fetching the data of user_input


#Describing the data 
st.subheader('Data from 2010 - 2020')
st.write(df.describe())

#Visualizations
st.subheader('Closing Price vs Time Chart')
fig = plt.figure(figsize = (12,6))
plt.plot(df.Close)
plt.grid()
st.pyplot(fig)



st.subheader('Closing Price vs Time Chart with MA100')
ma100 = df.Close.rolling(100).mean()
fig = plt.figure(figsize = (12,6))
plt.plot(ma100)
plt.plot(df.Close)
plt.grid()
plt.legend(["Original closing data", "MA of 100"],loc ="upper left")
st.pyplot(fig)

st.subheader('Closing Price vs Time Chart with MA 10d')
ma10 = df.Open.rolling(10).mean()
fig=plt.figure(figsize = (12,6))
plt.plot(df.Close)
plt.title("closing data")
plt.plot(ma10, "r")
plt.grid()
plt.legend(["Original closing data", "MA of 10"],loc ="upper left")
st.pyplot(fig)



st.subheader('Closing Price vs Time Chart with 100day MA & 200days MA')
ma100 = df.Close.rolling(100).mean()
ma200 = df.Close.rolling(200).mean()
fig = plt.figure(figsize = (12,6))
plt.plot(ma100)
plt.plot(ma200)
plt.grid()
plt.plot(df.Close)
plt.legend(["Original closing data", "MA of 100d & 200d"],loc ="upper left")
st.pyplot(fig)

st.subheader('Closing Price vs Time Chart with 10,20,50,100, 200 days MA')
ma10 = df.Close.rolling(10).mean()
ma20 = df.Close.rolling(20).mean()
ma50 = df.Close.rolling(50).mean()
ma100 = df.Close.rolling(100).mean()
ma200 = df.Close.rolling(200).mean()
fig = plt.figure(figsize = (12,6))
plt.plot(ma10)
plt.plot(ma20)
plt.plot(ma50)
plt.plot(ma100)
plt.plot(ma200)
plt.grid()
plt.plot(df.Close)
plt.legend(["Original closing data","MA of 10","MA of 20","MA of 50","MA of 100","MA of 200"],loc ="upper left")
st.pyplot(fig)


df['MA10'] = df['Close'].rolling(window=10).mean()
df['MA20'] = df['Close'].rolling(window=20).mean()
df['MA100'] = df['Close'].rolling(window=100).mean()
df['MA200'] = df['Close'].rolling(window=200).mean()

# Plot raw data
def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['Date'], y=df['Open'], name='Stock_open'))
    fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], name='Stock_close'))
    fig.add_trace(go.Scatter(x=df['Date'], y=df['High'], name='Stock_High'))
    fig.add_trace(go.Scatter(x=df['Date'], y=df['Low'], name='Stock_Low'))
    fig.add_trace(go.Scatter(x=df['Date'], y=df['Adj Close'], name='Stock_Adj Close'))
    fig.add_trace(go.Scatter(x=df['Date'], y=df['MA10'], name='Moving Avrages - 10'))
    fig.add_trace(go.Scatter(x=df['Date'], y=df['MA20'], name='Moving Avrages - 20'))
    fig.add_trace(go.Scatter(x=df['Date'], y=df['MA100'], name='Moving Avrages - 100'))
    fig.add_trace(go.Scatter(x=df['Date'], y=df['MA200'], name='Moving Avrages - 200'))
    fig.layout.update(title_text="Time Series Data", xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)
      
plot_raw_data()


#Splitting Data into Traning & Testing

data_train = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
data_test = pd.DataFrame(df['Close'][int(len(df)*0.70): int(len(df))])


#Min-Max Scaling

from sklearn.preprocessing import MinMaxScaler #for the stacked LSTM model we have to scale down the data 
scalar = MinMaxScaler(feature_range=(0,1)) #scaling down the data between (0,1)

#Fitting the traning data into min-max scaler & also will be convertig into array
data_training_array = scalar.fit_transform(data_train) #scaler.fit_transform will automatically give an array



#load my model
#model=load_models('keras_model.h5')
model = keras.models.load_model('keras_model.h5')


#Testing part
past_100_days = data_train.tail(100)
final_df = past_100_days.append(data_test, ignore_index=True) #in past_10_days i have append the data testing so now last 100days dt & dt are connected
input_data = scalar.fit_transform(final_df)

x_test = []
y_test = []

for i in range(100, input_data.shape[0]): #the range will go till '630'
    x_test.append(input_data[i-100: i])
    y_test.append(input_data[i, 0]) # it is the '0'th coloumn, that is closing price coloumn
   
x_test, y_test = np.array(x_test), np.array(y_test)#converting to numpy array
y_predicated = model.predict(x_test)
scalar = scalar.scale_


# so i need to divide my y_predicated and y_test value by the factor 

scale_factor = 1/scalar[0]
y_predicated = y_predicated * scale_factor
y_test = y_test * scale_factor



#Final Graph
st.subheader('Prediction vs Original of Closing Price ')
fig2 = plt.figure(figsize=(12,6))
plt.plot(y_test, 'b', label = 'Original Price')
plt.plot(y_predicated, 'r', label = 'Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.grid()
plt.legend()
st.pyplot(fig2)

#*******************************************************************************************************
#******************************************************************************************************

# Visulization for open 

st.subheader('Opening Price vs Time Chart')
fig = plt.figure(figsize = (12,6))
plt.plot(df.Open)
st.pyplot(fig)



st.subheader('Opening Price vs Time Chart with MA100')
ma100 = df.Open.rolling(100).mean()
fig = plt.figure(figsize = (12,6))
plt.plot(ma100)
plt.plot(df.Open)
st.pyplot(fig)


st.subheader('opening Price vs Time Chart with 100day MA & 200days MA')
ma100 = df.Open.rolling(100).mean()
ma200 = df.Open.rolling(200).mean()
fig = plt.figure(figsize = (12,6))
plt.plot(ma100, 'r')
plt.plot(ma200, 'g')
plt.plot(df.Open, 'b')
st.pyplot(fig)


#Splitting the data into traning and testing
data_train_Open = pd.DataFrame(df['Open'][0:int(len(df)*0.70)])
data_test_Open = pd.DataFrame(df['Open'][int(len(df)*0.70): int(len(df))])



from sklearn.preprocessing import MinMaxScaler #for the stacked LSTM model we have to scale down the data 
scalar_Open = MinMaxScaler(feature_range=(0,1)) #scaling down the data between (0,1)


#Fitting the traning data into min-max scaler & also will be convertig into array
data_training_array_Open = scalar_Open.fit_transform(data_train_Open)


#load my model
#model=load_models('keras_model.h5')
model_Open = keras.models.load_model('keras_model.h5')


#Testing part
past_100_days_Open = data_train_Open.tail(100)
final_df_Open = past_100_days_Open.append(data_test_Open, ignore_index=True)
input_data_Open = scalar_Open.fit_transform(final_df_Open)
 
#Testing
x_test = []
y_test = []

for i in range(100, input_data_Open.shape[0]):
    x_test.append(input_data_Open[i-100: i])
    y_test.append(input_data_Open[i, 0])   
   

x_test, y_test = np.array(x_test), np.array(y_test)
y_predicated = model_Open.predict(x_test)
scalar_open = scalar_Open.scale_

scale_factor = 1/scalar_open[0]
y_predicated = y_predicated * scale_factor
y_test = y_test * scale_factor


st.subheader('Predictions vs original for Open Price')
fig3 = plt.figure(figsize=(12,6))
plt.plot(y_test, 'b', label = 'Original Price')
plt.plot(y_predicated, 'r', label = 'Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig3)