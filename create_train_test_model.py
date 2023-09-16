import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf 
import datetime as dt
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from keras.layers import Dense, Dropout, LSTM
from keras.models import Sequential

# Specify the cryptocurrency symbol and date range for data retrieval
crypto_currency = 'BTC-USD'
start = dt.datetime(2020,1,1)
end = dt.datetime(2022,1,1)

# Use yfinance to fetch cryptocurrency price data
data = yf.download(crypto_currency, start=start, end=end)

# Initialize a Min-Max scaler to scale data between 0 and 1
scaler = MinMaxScaler(feature_range=(0,1))
scaler_data = scaler.fit_transform(data['Close'].values.reshape(-1,1))

# Set the number of days for price prediction
prediction_input_days = 60

# Prepare training data for the LSTM model
x_train, y_train = [], []
for x in range(prediction_input_days, len(scaler_data)):
    x_train.append(scaler_data[x-prediction_input_days:x, 0])
    y_train.append(scaler_data[x, 0])

x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

# Create a Sequential Neural Network model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))
model.add(Dense(units=1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(x_train, y_train, epochs=25, batch_size=32)

# Testing The Model

# Define the testing date range
test_start = dt.datetime(2022,1,1)
test_end =dt.datetime.now()
test_data = yf.download(crypto_currency, start=test_start, end=test_end)
actual_prices = test_data['Close'].values

total_dataset = pd.concat((data['Close'], test_data['Close']), axis=0)
model_inputs = total_dataset[len(total_dataset) - len(test_data) - prediction_input_days:].values
model_inputs = model_inputs.reshape(-1, 1)
model_inputs = scaler.fit_transform(model_inputs)

x_test = []
for x in range(prediction_input_days, len(model_inputs)):
    x_test.append(model_inputs[x-prediction_input_days:x, 0])

x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

prediction_prices = model.predict(x_test)
prediction_prices = scaler.inverse_transform(prediction_prices)

plt.plot(actual_prices, color='black', label='Actual Prices')
plt.plot(prediction_prices, color='green', label='Predicted Prices')
plt.title(f'{crypto_currency} price prediction')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend(loc='upper right')
plt.show()


# Save the model to a file
model.save("crypto_price_prediction_model.h5")

# Predict Next Day
real_data = [model_inputs[len(model_inputs) - prediction_input_days:len(model_inputs), 0]]
real_data = np.array(real_data)
real_data = np.reshape(real_data, (real_data.shape[0], real_data.shape[1], 1))

prediction = model.predict(real_data)
prediction = scaler.inverse_transform(prediction)
print('Prediciton:',prediction)


# Test Predict 60 days

prediction_result_days = 60

current_date = dt.datetime.now()
start_prediction_result_days = current_date - dt.timedelta(days=prediction_result_days)
end_prediction_result_days =dt.datetime.now()
data_prediction_result_days = yf.download(crypto_currency, start=start_prediction_result_days, end=end_prediction_result_days)
actual_price_of_prediction_result_days = data_prediction_result_days['Close'].values

# Initialize an array to store the predicted prices
predicted_prices = []


# Start with the most recent 60 days of data as input
initial_input = model_inputs[len(model_inputs) - 60 -1 - prediction_result_days:len(model_inputs)-60 -1, 0]

# Predict prices for 60 days
for i in range(prediction_result_days):
    # Reshape the input for the model
    input_data = np.reshape(initial_input, (1, prediction_result_days, 1))
    
    # Use the model to predict the next day's price
    next_day_prediction = model.predict(input_data)
    
    # Inverse transform the prediction to get the actual price
    next_day_price = scaler.inverse_transform(next_day_prediction)
    
    # Add the predicted price to the list
    predicted_prices.append(next_day_price[0, 0])
    
    # Update the input for the next iteration
    initial_input = np.roll(initial_input, shift=-1)
    initial_input[-1] = next_day_prediction[0, 0]

plt.plot(actual_price_of_prediction_result_days, color='black', label='Actual Prices')
plt.plot(predicted_prices, color='green', label='Predicted Prices')
plt.title(f'{crypto_currency} price prediction')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend(loc='upper right')
plt.show()