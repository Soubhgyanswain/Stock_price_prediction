import numpy as np
import pandas as pd
import yfinance as yf
from keras.models import load_model
import streamlit as st
import matplotlib.pyplot as plt
from datetime import datetime

# Load the trained model
model = load_model(r"C:\Users\KIIT\Desktop\STOCK PP\Stock Predictions Model1.keras")

# Streamlit App Header
st.header('Stock Market Predictor')

# User Input for Stock Symbol
stock = st.text_input('Enter Stock Symbol', 'GOOG')

# Define Date Range (Updated to fetch latest data till 23 March 2025)
start = '2012-01-01'
end = datetime.today().strftime('%Y-%m-%d')  # Automatically fetches today's date

# Fetch Stock Data
data = yf.download(stock, start, end)

# Display Stock Data
st.subheader('Stock Data')
st.write(data)

# Splitting Data into Training and Testing Sets
data_train = pd.DataFrame(data.Close[:int(len(data)*0.80)])
data_test = pd.DataFrame(data.Close[int(len(data)*0.80):])

# Normalize Data for Prediction
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))

past_100_days = data_train.tail(100)
data_test = pd.concat([past_100_days, data_test], ignore_index=True)
data_test_scale = scaler.fit_transform(data_test)

# Moving Averages (50, 100, 200 days)
st.subheader('Price vs MA50')
ma_50_days = data.Close.rolling(50).mean()
fig1 = plt.figure(figsize=(8,6))
plt.plot(ma_50_days, 'r')
plt.plot(data.Close, 'g')
st.pyplot(fig1)

st.subheader('Price vs MA50 vs MA100')
ma_100_days = data.Close.rolling(100).mean()
fig2 = plt.figure(figsize=(8,6))
plt.plot(ma_50_days, 'r')
plt.plot(ma_100_days, 'b')
plt.plot(data.Close, 'g')
st.pyplot(fig2)

st.subheader('Price vs MA100 vs MA200')
ma_200_days = data.Close.rolling(200).mean()
fig3 = plt.figure(figsize=(8,6))
plt.plot(ma_100_days, 'r')
plt.plot(ma_200_days, 'b')
plt.plot(data.Close, 'g')
st.pyplot(fig3)

# Preparing Data for Prediction
x, y = [], []
for i in range(100, data_test_scale.shape[0]):
    x.append(data_test_scale[i-100:i])
    y.append(data_test_scale[i, 0])

x, y = np.array(x), np.array(y)

# Predicting with the Model
predict = model.predict(x)

# Rescale Predictions
scale = 1 / scaler.scale_
predict = predict * scale
y = y * scale

# Plot Original vs Predicted Price
st.subheader('Original Price vs Predicted Price')
fig4 = plt.figure(figsize=(8,6))
plt.plot(predict, 'r', label='Predicted Price')
plt.plot(y, 'g', label='Original Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig4)
