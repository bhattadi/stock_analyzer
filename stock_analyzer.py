# # Goal: Build a neural network to learn the fluctuations of the stock market

# # Step 1: Load in data from APIs and store into data structures
# # Step 2: Define network Architecture
# # Step 3: Train a network on the data
# # Step 4: Evaluate performance of the network

import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np
from torch import nn, optim
import torch.nn.functional as F
from sklearn.svm import SVR

inputs = 5
ticker = 'MSFT'
# ticker = input('Enter Stock Ticker: ')
training_history = yf.download(ticker, period= "max", interval='1mo')
print('Number of history points: ', len(training_history))

# Assign values to each date
start = 0
input_to_model = []
output_to_model = []

# Preprocess all historical data
for i, row in training_history.iterrows():
    # add all the numbers for num rows into a vector
    input_to_model.append(start)
    start += 1

    # For the first data point assume no change
    if start == 1:
        output_to_model.append(0)
        continue

    # Find percent change in adjusted closing price
    initial = row['Adj Close']
    new_val = training_history['Adj Close'][start-2]
    percent_change = (initial - new_val) / new_val
    print('initial: ' , initial, 'new val: ', new_val, 'percent_change: ', percent_change)
    output_to_model.append(percent_change)

# Split the output-input pairs into training and testing data (80% train & 20% test)
X_train = input_to_model[:round(len(input_to_model)*0.8)]
Y_train = output_to_model[:round(len(output_to_model)*0.8)]

X_test = input_to_model[round(len(input_to_model)*0.8):]
Y_test = output_to_model[round(len(output_to_model)*0.8):]

# Define the model
model = SVR()

# clean up the containers
X_train = np.asarray(X_train).reshape(-1, 1)
Y_train = np.asarray(Y_train)

# fit the data to the model
model.fit(X_train, Y_train)

print("Number of " + ticker + " stock dates trained on: " + str(start))

# Run Predictions on the last 12 records
results = model.predict(X_test)
print(results)