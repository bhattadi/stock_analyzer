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
ticker = "MSFT"
training_history = yf.download(ticker, start = "2019-01-01", end = "2020-12-31", period = "1d")

# Plot the data of this stock over 2020
training_history['Adj Close'].plot(label=ticker)
plt.xlabel("Date")
plt.ylabel("Adjusted")
plt.title("Microsoft Price data")
plt.legend(loc="upper right")
plt.show()
plt.savefig('microsoft_details.png')

# Assign values to each date
start = 0

X_train = []
Y_train = []


for i, row in training_history.iterrows():
    # add all the numbers for num rows into a vector
    X_train.append(start)
    start += 1

    # add all the adjacent closing price
    avg_price = row['Adj Close']
    Y_train.append(avg_price)

X_test = np.arange(start, start+30, 1).reshape(-1,1)
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