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

# Plot the data of this stock over 2020
# print(len(training_history['Adj Close']))
# training_history['Adj Close'].plot(label=ticker)
# plt.xlabel("Date")
# plt.ylabel("Adjusted")
# plt.title("Microsoft Price data")
# plt.legend(loc="upper right")
# plt.show()
# plt.savefig('microsoft_details.png')

# Assign values to each date
start = 0
X_train = []
Y_train = []

# Construct the training data
for i, row in training_history.iterrows():
    # add all the numbers for num rows into a vector
    X_train.append(start)
    start += 1

    # For the first data point assume no change
    if start == 1:
        Y_train.append(0)
        continue

    # Find percent change in adjusted closing price
    percent_change = (row['Adj Close'] - training_history['Adj Close'][start-2]) / training_history['Adj Close'][start-2]
    Y_train.append(percent_change)
    print(i)

# Creating Testing Data
print('Creating test data')
X_test = np.arange(start, start+10, 1).reshape(-1,1)

# Construct the testing data
for i, row in training_history.iterrows():
    print(i)
    # add all the numbers for num rows into a vector
    X_train.append(start)
    start += 1

    # For the first data point assume no change
    if start == 1:
        Y_train.append(0)
        continue

    # Find percent change in adjusted closing price
    percent_change = (row['Adj Close'] - training_history['Adj Close'][start-2]) / training_history['Adj Close'][start-2]
    Y_train.append(percent_change)
Y_test = []

# Define the model
model = SVR()

# clean up the containers
X_train = np.asarray(X_train).reshape(-1, 1)
Y_train = np.asarray(Y_train)

# fit the data to the model
model.fit(X_train, Y_train)

print("Number of " + ticker + " stock dates trained on: " + str(start))

results = model.predict(X_test)

print(results)