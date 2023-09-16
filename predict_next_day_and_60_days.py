import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf
import datetime as dt
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from keras.layers import Dense, Dropout, LSTM

model = keras.models.load_model("crypto_price_prediction_model.h5")

crypto_currency = 'BTC-USD'

prediction_days = 60

start = dt.datetime(2020,1,1)
end = dt.datetime.now()

# Use yfinance to fetch data
data = yf.download(crypto_currency, start=start, end=end)

scaler = MinMaxScaler(feature_range=(0,1))
scaler_data = scaler.fit_transform(data['Close'].values.reshape(-1,1))

# Testing The Model

test_start = dt.datetime(2022,1,1)
test_end =dt.datetime.now()
test_data = yf.download(crypto_currency, start=test_start, end=test_end)
actual_prices = test_data['Close'].values

total_dataset = pd.concat((data['Close'], test_data['Close']), axis=0)
model_inputs = total_dataset[len(total_dataset) - len(test_data) - prediction_days:].values
model_inputs = model_inputs.reshape(-1, 1)
model_inputs = scaler.fit_transform(model_inputs)

x_test = []

for x in range(prediction_days, len(model_inputs)):
    x_test.append(model_inputs[x-prediction_days:x, 0])

x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

prediction_prices = model.predict(x_test)
prediction_prices = scaler.inverse_transform(prediction_prices)

plt.plot(actual_prices, color='black', label='Actual Prices')
plt.plot(prediction_prices, color='green', label='Predicted Prices')
plt.title(f'{crypto_currency} price prediction')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend(loc='upper left')
plt.show()


# Save the model to a file
model.save("crypto_price_prediction_model.h5")

# Predict Next Day

real_data = [model_inputs[len(model_inputs) - prediction_days:len(model_inputs), 0]]
real_data = np.array(real_data)
real_data = np.reshape(real_data, (real_data.shape[0], real_data.shape[1], 1))

prediction = model.predict(real_data)
prediction = scaler.inverse_transform(prediction)
print('Prediciton:',prediction)


# Test Predict last 60 days

delay_days = 14

# Get the current date
current_date = dt.datetime.now()

# Calculate the start date that is 60 days before now
start_delay_days_last_days = current_date - dt.timedelta(days=delay_days)

end_delay_days_last_days =dt.datetime.now()
data_delay_days_last_days = yf.download(crypto_currency, start=start_delay_days_last_days, end=end_delay_days_last_days)
actual_delay_days_last_days_prices = data_delay_days_last_days['Close'].values

# Initialize an array to store the predicted prices
predicted_prices = []


# Start with the most recent 60 days of data as input
initial_input = model_inputs[len(model_inputs) - delay_days -1 - prediction_days:len(model_inputs)-delay_days -1, 0]

# Predict prices for 30 days
for i in range(delay_days):
    # Reshape the input for the model
    input_data = np.reshape(initial_input, (1, prediction_days, 1))
    
    # Use the model to predict the next day's price
    next_day_prediction = model.predict(input_data)
    
    # Inverse transform the prediction to get the actual price
    next_day_price = scaler.inverse_transform(next_day_prediction)
    
    # Add the predicted price to the list
    predicted_prices.append(next_day_price[0, 0])
    
    # Update the input for the next iteration
    initial_input = np.roll(initial_input, shift=-1)
    initial_input[-1] = next_day_prediction[0, 0]

plt.plot(actual_delay_days_last_days_prices, color='black', label='Actual Prices')
plt.plot(predicted_prices, color='green', label='Predicted Prices')
plt.title(f'{crypto_currency} price prediction')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend(loc='upper left')
plt.show()