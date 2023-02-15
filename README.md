# Stock-Prediction---Streamlit-webapp-

Stock Price/Trend Prediction Web Application
This is a web application that provides stock price and trend prediction using the FB Prophet tool, LSTM, and Linear Regression. The user can input the stock symbol and select the desired model, and the web app will display the predicted stock prices and trends for the next period.

Installation
To run this application, you need to have Python 3 installed on your machine. You can clone this repository and install the dependencies using pip. First, navigate to the project directory and run:

bash
Copy code
git clone https://github.com/yourusername/stock-prediction-webapp.git
cd stock-prediction-webapp
pip install -r requirements.txt
Usage
To run the application, navigate to the project directory and run the following command:

Copy code
streamlit run app.py
This will open the web application in your default browser. You can enter the stock symbol and select the desired model from the dropdown list. Click on the "Predict" button to generate the predictions. The predicted prices and trends will be displayed in a line chart.

Models
The web application provides three models for stock price and trend prediction:

FB Prophet
LSTM
Linear Regression
FB Prophet
Prophet is a procedure for forecasting time series data based on an additive model where non-linear trends are fit with yearly, weekly, and daily seasonality, plus holiday effects. It works best with time series that have strong seasonal effects and several seasons of historical data.

LSTM
Long Short-Term Memory (LSTM) is a type of recurrent neural network (RNN) architecture that is capable of learning long-term dependencies. It has been successfully applied to various sequence prediction problems, including stock price prediction.

Linear Regression
Linear regression is a simple approach for modeling the relationship between a dependent variable and one or more independent variables. It can be used for stock price prediction by modeling the relationship between the stock price and various economic and financial factors.

Data
The web application uses stock price data from Yahoo Finance. It retrieves the historical prices for the selected stock symbol and uses them to train the selected model. The data is updated daily and covers the last 5 years of trading.

Acknowledgments
This web application was developed by [Jaymin Shah]. It is based on the FB Prophet, Keras, and scikit-learn libraries.
