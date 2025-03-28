# 📈 Stock Price Prediction

This project aims to predict stock prices using machine learning models trained on historical stock data. It leverages deep learning techniques and data visualization to analyze market trends.

## 🚀 Features
- Predicts stock prices based on historical data.
- Implements LSTM-based deep learning models.
- Visualizes trends with interactive plots.
- Uses moving averages and other indicators.

## 📂 Project Structure
├── app1.py                 # Alternative or additional script
├── STOCK PP.ipynb          # Jupyter Notebook (Data analysis & model training)
├── updated stock.ipynb     # Updated version of stock prediction notebook
├── Stock Predictions Model.keras     # Pre-trained Keras model
├── Stock Predictions Model1.keras    # Alternative model version
├── Figure_1.png            # Visualization of stock trends
├── Figure_3.png            # Additional stock price graphs
├── Figure_4.png            # Moving averages & predictions


## 🛠️ Installation & Setup

First you have to install Anaconda Navigator  -> Launch Jupyter Notebook -> Create New file (shown)
![image](https://github.com/user-attachments/assets/44c97eb0-ace3-4845-afd2-0b23ab1768af)

COPY CODE (.ipynb file):
``` bash
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
```
``` bash
start = '2012-01-01'
end = '2022-12-21'
stock = 'GOOG'
data = yf.download(stock, start, end)
```
``` bash
data.reset_index(inplace=True)
```
``` bash
data
```
``` bash
ma_100_days = data.Close.rolling(100).mean()
```
``` bash
plt.figure(figsize=(8,6))
plt.plot(ma_100_days, 'r')
plt.plot(data.Close, 'g')
plt.show()
```
``` bash
ma_200_days = data.Close.rolling(200).mean()
```
``` bash
plt.figure(figsize=(8,6))
plt.plot(ma_100_days, 'r')
plt.plot(ma_200_days,'b')
plt.plot(data.Close,'g')
plt.show()
```
``` bash
data.dropna(inplace=True)
```
``` bash
data_train = pd.DataFrame(data.Close[0: int(len(data)*0.80)])
data_test = pd.DataFrame(data.Close[int(len(data)*0.80): len(data)])
```
``` bash
data_train.shape[0]
```
``` bash
data_test.shape[0]
```
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))

data_train_scale = scaler.fit_transform(data_train)

x = []
y = []

for i in range(100, data_train_scale.shape[0]):
    x.append(data_train_scale[i-100:i])
    y.append(data_train_scale[i,0])

x, y = np.array(x), np.array(y)

from keras.layers import Dense, Dropout, LSTM
from keras.models import Sequential

model = Sequential()
model.add(LSTM(units = 50, activation = 'relu', return_sequences = True,
               input_shape = ((x.shape[1],1))))
model.add(Dropout(0.2))

model.add(LSTM(units = 60, activation='relu', return_sequences = True))
model.add(Dropout(0.3))

model.add(LSTM(units = 80, activation = 'relu', return_sequences = True))
model.add(Dropout(0.4))

model.add(LSTM(units = 120, activation = 'relu'))
model.add(Dropout(0.5))

model.add(Dense(units =1))

model.compile(optimizer = 'adam', loss = 'mean_squared_error')

model.fit(x,y, epochs = 50, batch_size =32, verbose =1)

model.summary()

pas_100_days = data_train.tail(100)

data_test = pd.concat([pas_100_days, data_test], ignore_index=True)

data_test_scale  =  scaler.fit_transform(data_test)

x = []
y = []

for i in range(100, data_test_scale.shape[0]):
    x.append(data_test_scale[i-100:i])
    y.append(data_test_scale[i,0])
x, y = np.array(x), np.array(y)

y_predict = model.predict(x)

scale =1/scaler.scale_

y_predict = y_predict*scale

y = y*scale

plt.figure(figsize=(10,8))
plt.plot(y_predict, 'r', label = 'Predicted Price')
plt.plot(y, 'g', label = 'Original Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()

model.save('Stock Predictions Model.keras')

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/stock-price-prediction.git
   cd stock-price-prediction
   ```
2. Activate environment:
   ```bash
   venv\Scripts\activate


🧠 Model Training
The model is trained using LSTM (Long Short-Term Memory) networks.

Uses datasets from Yahoo Finance (or any stock data provider).

Hyperparameters are tuned for optimal prediction accuracy.

📊 Visualization
Stock price trends are displayed using Matplotlib.

Moving Averages and price predictions are plotted.

📸 Sample Results

🤝 Contributing
Feel free to open issues or contribute with pull requests.

📜 License
This project is licensed under the MIT License.
