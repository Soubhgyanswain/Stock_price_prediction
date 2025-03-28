# 📈 Stock Price Prediction

This project aims to predict stock prices using machine learning models trained on historical stock data. It leverages deep learning techniques and data visualization to analyze market trends.

## 🚀 Features
- Predicts stock prices based on historical data.
- Implements LSTM-based deep learning models.
- Visualizes trends with interactive plots.
- Uses moving averages and other indicators.

## 📸 Results:

![image](https://github.com/user-attachments/assets/2b7d59bf-934b-402c-8cb1-f6f5a5d85fac)
![image](https://github.com/user-attachments/assets/41a33070-e8e2-47b2-a10a-81bbac14069a)
![image](https://github.com/user-attachments/assets/63beb568-0e2d-4b1f-af3f-908b8d5e1df5)
![image](https://github.com/user-attachments/assets/8903037c-c71b-4624-ba2b-5aa99a8acfe8)
![image](https://github.com/user-attachments/assets/49c5a548-e6ab-42aa-9b31-31cf2842bf2c)


## 🛠️ INSTALLATION & SETUP

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
``` bash
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))
```
``` bash
data_train_scale = scaler.fit_transform(data_train)
```
``` bash
x = []
y = []
```
``` bash
for i in range(100, data_train_scale.shape[0]):
    x.append(data_train_scale[i-100:i])
    y.append(data_train_scale[i,0])
```
``` bash
x, y = np.array(x), np.array(y)
```
``` bash
from keras.layers import Dense, Dropout, LSTM
from keras.models import Sequential
```
``` bash
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
```
``` bash
model.compile(optimizer = 'adam', loss = 'mean_squared_error')
```
``` bash
model.fit(x,y, epochs = 50, batch_size =32, verbose =1)
```
``` bash
model.summary()
```
``` bash
pas_100_days = data_train.tail(100)
```
``` bash
data_test = pd.concat([pas_100_days, data_test], ignore_index=True)
```
``` bash
data_test_scale  =  scaler.fit_transform(data_test)
```
``` bash
x = []
y = []

for i in range(100, data_test_scale.shape[0]):
    x.append(data_test_scale[i-100:i])
    y.append(data_test_scale[i,0])
x, y = np.array(x), np.array(y)
```
``` bash
y_predict = model.predict(x)
```
``` bash
scale =1/scaler.scale_
```
``` bash
y_predict = y_predict*scale
```
``` bash
y = y*scale
```
``` bash
plt.figure(figsize=(10,8))
plt.plot(y_predict, 'r', label = 'Predicted Price')
plt.plot(y, 'g', label = 'Original Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()
```
``` bash
model.save('Stock Predictions Model.keras')
```

1. Activate environment:
   ```bash
   python -m venv venv
   venv\Scripts\activate
   ```
2. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/stock-price-prediction.git
   cd stock-price-prediction
   ```
3. install all packages
   ``` bash
   pip install numpy pandas yfinance tensorflow scikit-learn matplotlib streamlit
   ```
4. Run the app:
   ```bash
   streamlit run app1.py
   ```


🧠 Model Training
- The model is trained using LSTM (Long Short-Term Memory) networks.
- Uses datasets from Yahoo Finance (or any stock data provider).
- Hyperparameters are tuned for optimal prediction accuracy.

📊 Visualization
Stock price trends are displayed using Matplotlib.
Moving Averages and price predictions are plotted.

🤝 Contributing
Feel free to open issues or contribute with pull requests.

📜 License
This project is licensed under the MIT License.
