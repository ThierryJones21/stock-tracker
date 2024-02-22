# # Install necessary libraries
# !pip install alpaca-trade-api
# !pip install tensorflow
# !pip install scikit-learn

# Import libraries
import os
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import layers
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import requests
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from tabulate import tabulate


#FMP API for training data
api_key_fmp = "dc16ae239a0a90cc7f039177aa18aa33"

### TRAINING CODE

def fetch_stock_data(symbol, from_date, to_date):
    api_url = f"https://financialmodelingprep.com/api/v3/historical-price-full/{symbol}?apikey={api_key_fmp}&from={from_date}&to={to_date}"
    response = requests.get(api_url)
    data = response.json()

    historical_data = data['historical']

    print(historical_data)

    reversed_data = historical_data[::-1]
    return reversed_data

# Define function for data preprocessing
def preprocess_data(data):
    closing_prices = [day['close'] for day in data]
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(np.array(closing_prices).reshape(-1, 1))
    return scaled_data, scaler, [day['date'] for day in data]

# Define function to create sequences for LSTM
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:(i + seq_length)])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)

# Define function to build LSTM model
def build_lstm_model(seq_length):

    model = Sequential([layers.Input((seq_length, 1)),
                    layers.LSTM(64),
                    layers.Dense(32, activation='relu'),
                    layers.Dense(32, activation='relu'),
                    layers.Dropout(0.2),
                    layers.Dense(1)])

    model.compile(loss='mse', 
              optimizer=Adam(learning_rate=0.001),
              metrics=['mean_absolute_error'])
    
    # model = Sequential()
    # model.add(LSTM(units=50, return_sequences=True, input_shape=(seq_length, 1)))
    # model.add(Dropout(0.2))
    # model.add(LSTM(units=50, return_sequences=True))
    # model.add(Dropout(0.2))
    # model.add(LSTM(units=50))
    # model.add(Dropout(0.2))
    # model.add(Dense(units=1))
    # model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Define function for training the model
def train_model(model, X_train, y_train, X_val, y_val, epochs, batch_size):
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, batch_size=batch_size, verbose=0)
    return model

# Define function to save the trained model
def save_model(model, filename):
    model.save(filename)

# Define function to calculate model accuracy
def calculate_accuracy(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    return rmse

def run_code(symbol, start_date, end_date):

    seq_length = 20
    epochs = 100
    batch_size = 32
    model_filename = 'stock_prediction_model.h5'

    # Fetch data
    stock_data = fetch_stock_data(symbol, start_date, end_date)

    # Preprocess data
    scaled_data, scaler, dates = preprocess_data(stock_data)

    # Create sequences
    X, y = create_sequences(scaled_data, seq_length)

    # Split data into training, validation, and testing sets
    q_80 = int(len(X) * 0.8)
    q_90 = int(len(X) * 0.9)

    X_train, y_train = X[:q_80], y[:q_80]
    X_val, y_val = X[q_80:q_90], y[q_80:q_90]
    X_test, y_test = X[q_90:], y[q_90:]

    total_length_combined = len(X_train) + len(X_val) + len(X_test)
    print("Total length of X_train + X_val + X_test:", total_length_combined)
    print("Total of X", len(X))

    total_length_y_combined = len(y_train) + len(y_val) + len(y_test)
    print("Total length of y_train + y_val + y_test:", total_length_y_combined)
    print("Total of y:", len(y))



    # dates_train = dates[:q_80]
    # dates_val = dates[q_80:q_90]
    # dates_test = dates[q_90:]

    dates_train = [dates[i] for i in range(q_80)]
    dates_val = [dates[q_80 + i] for i in range(len(y_val))]
    dates_test = [dates[q_90 + i] for i in range(len(y_test))]



    dates_train = [datetime.strptime(date_str, '%Y-%m-%d').date() for date_str in dates_train]
    dates_val = [datetime.strptime(date_str, '%Y-%m-%d').date() for date_str in dates_val]
    dates_test = [datetime.strptime(date_str, '%Y-%m-%d').date() for date_str in dates_test]

    total_length_dates_combined = len(dates_train) + len(dates_val) + len(dates_test)
    print("Total length of dates_train + dates_val + dates_test:", total_length_dates_combined)
    print("Total of dates:", len(dates))

    # print("Length of entire dataset:", len(X))
    # print("Length of X_test:", len(X_test))
    print("Length of y_test:", len(y_test))
    print("Length of dates_test:", len(dates_test))

    # Ensure the lengths of dates_test and y_test are the same
    # y_test = y_test[:len(dates_test)]
    # y_test = y_test.reshape(-1)

    # print(tabulate(zip(dates_test, y_test), headers=['Date', 'Y_Test']))

    # Plotting
    plt.plot(dates_train, y_train)
    plt.plot(dates_val, y_val)
    plt.plot(dates_test, y_test)

    plt.legend(['Train', 'Validation', 'Test'])
    plt.show()

    # Reshape data for LSTM
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

    # Build LSTM model
    model = build_lstm_model(seq_length)

    # Train model
    model = train_model(model, X_train, y_train, X_val, y_val, epochs, batch_size)

    # Save model
    save_model(model, model_filename)

    # Predict using the trained model
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    y_val_pred = model.predict(X_val)  

    # Inverse transform the predicted and actual values
    y_train_actual = scaler.inverse_transform(y_train.reshape(-1, 1))
    y_train_pred_actual = scaler.inverse_transform(y_train_pred)
    y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))
    y_test_pred_actual = scaler.inverse_transform(y_test_pred)
    y_val_actual = scaler.inverse_transform(y_val.reshape(-1, 1))
    y_val_pred_actual = scaler.inverse_transform(y_val_pred)  


    plt.figure(figsize=(10, 6))

    # Plot training data
    plt.plot(dates_train, y_train_actual, label='Training Actual')
    plt.plot(dates_train, y_train_pred_actual, color='orange', label='Training Predicted')

    # Plot testing data
    plt.plot(dates_test, y_test_actual, label='Testing Actual')
    plt.plot(dates_test, y_test_pred_actual, color='green', label='Testing Predicted')

    # Plot validation data
    plt.plot(dates_val, y_val_actual, label='Validation Actual')
    plt.plot(dates_val, y_val_pred_actual, color='blue', label='Validation Predicted')


    # # Extend predictions for additional days
    # future_prices = []
    # X_extend = X_test[-1:].copy()
    # for _ in range(7):  # Predict for the next 7 days
    #     future_price = model.predict(X_extend)[0][0]
    #     future_prices.append(future_price)
    #     X_extend = np.roll(X_extend, -1)
    #     X_extend[-1][-1] = future_price

    # future_prices_actual = scaler.inverse_transform(np.array(future_prices).reshape(-1, 1))



    # # # Extend predictions
    # future_dates = generate_date_range(test_dates[-1] + timedelta(days=1), test_dates[-1] + timedelta(days=seq_length))
    # plt.plot(future_dates, future_prices_actual, color='purple', label='Future Predictions')

    plt.title('Actual vs. Predicted Closing Prices')
    plt.xlabel('Date')
    plt.ylabel('Closing Price $')
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.show()


def generate_date_range(start_date, end_date):
    return [start_date + timedelta(days=i) for i in range((end_date - start_date).days + 1)]



def main():
    symbol = 'TSLA'
    start_date = '2023-01-01'
    end_date= '2024-01-01'

    run_code(symbol, start_date, end_date)


if __name__ == "__main__":
    main()

"""Imports and Model"""