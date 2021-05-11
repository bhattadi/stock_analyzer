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
from sklearn.neural_network import MLPRegressor
import math

def train_model_and_predict_last_date(ticker):
    training_history = yf.download(ticker, period= "max")
    # print('Number of history points: ', len(training_history))

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

        # Error handling for data that is not available
        if math.isnan(initial) or math.isnan(new_val):
            output_to_model.append(0)
            continue

        # print('initial: ' , initial, 'new val: ', new_val, 'percent_change: ', percent_change)
        output_to_model.append(percent_change)

    # Split the output-input pairs into training and testing data --> testing data is the last day
    split_index = len(input_to_model)-1
    X_train = input_to_model[:split_index]
    Y_train = output_to_model[:split_index]

    X_test = input_to_model[split_index:]
    Y_test = output_to_model[split_index:]

    # Define the model
    model = MLPRegressor(hidden_layer_sizes=(25, 50), max_iter=100, activation='relu')

    # model = SVR()

    # clean up the containers
    X_train = np.asarray(X_train).reshape(-1, 1)
    Y_train = np.asarray(Y_train)
    X_test = np.asarray(X_test).reshape(-1, 1)
    Y_test= np.asarray(Y_test)


    # fit the data to the model
    model.fit(X_train, Y_train)
    # print("Number of " + ticker + " stock dates trained on: " + str(start))

    # Run Predictions on the 20% testing data
    correct = 0
    total = 0
    for i in X_test:
        # print(i)
        results = model.predict([i])
        expected = Y_test[list(X_test).index(i)]
        # print("Predicted: ", results, ' || Expected: ', expected)
        if (results < 0 and expected < 0) or (results > 0 and expected > 0):
            correct += 1
        total += 1

    # print('Accuracy: ', correct / total)
    return correct / total

# Grab all the stock tickers on Stock Exchange
def grab_tickers():
    df = pd.read_excel('stock_tickers.xlsx')
    tickers = df['Ticker']
    return tickers
    
# Evaluate model performance on different stocks' last day 
def evaluate_model():
    tickers = grab_tickers()
    scores = []
    for ticker in tickers:
        # Use try except to skip over stocks that cause an error
        try:
            score = train_model_and_predict_last_date(ticker)
            scores.append(score)
        except:
            print('An error occured on Ticker: ', ticker)

    # Compute average of averages
    average = sum(scores) / len(scores)
    print('Model Performance Average: ', average)
    return average



# Main Driver to run all 5 parts
def main():
    evaluate_model()

if __name__ == "__main__":
    main()