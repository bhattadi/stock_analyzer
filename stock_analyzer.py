# # Goal: Build a neural network to learn the fluctuations of the stock market

# # Step 1: Load in data from APIs and store into data structures
# # Step 2: Define network Architecture
# # Step 3: Train a network on the data
# # Step 4: Evaluate performance of the network

import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

ticker = "MSFT"
newtime_daily = yf.download(ticker, start = "2020-01-01", end = "2020-12-31", period = "1d")
test = yf.download("FB", start = "2020-01-01", end = "2020-12-31", period = "1d")

newtime_daily['Adj Close'].plot(label="MSFT")
test['Adj Close'].plot(label="FB")
plt.xlabel("Date")
plt.ylabel("Adjusted")
plt.title("Miscrosoft Price data")
plt.legend(loc="upper right")
plt.show()

plt.savefig('microsoft_details.png')
