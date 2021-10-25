# -*- coding: utf-8 -*-
# Author: jS1ngle
# License: MIT License (http://opensource.org/licenses/MIT)
import pandas as pd
from SimulationHelperFunctions import get_hist_price_data
import numpy as np
from scipy.optimize import minimize


# -----Functions---------------------------------------------------------
def evaluate_portfolio(weights):
    weights = np.array(weights)
    ret = (np.sum(np.array(logReturns.mean(axis=0)) * weights)) * timeFrame
    vol = np.sqrt(np.dot(weights.T, np.dot(cov, weights))) * np.sqrt(timeFrame)
    sharpe = ret / vol
    return np.array([ret, vol, sharpe])


# -----Function to be minimized
def objective_function(weights, sign=1):
    return sign * evaluate_portfolio(weights)[1]


# -----Additional criteria
def sum_criteria(weights):
    return np.sum(weights) - 1


# -----Input-------------------------------------------------------------
timeFrame = 100

# -----Exemplary
coinDataStruct = [['BNB', 'Binance', 'BTC'],
                  ['ADA', 'Binance', 'BTC'],
                  ['ETH', 'Binance', 'BTC'],
                  ['SOL', 'Binance', 'BTC'],
                  ['DOGE', 'Binance', 'BTC'],
                  ['DOT', 'Binance', 'BTC']]


# -----Function calls----------------------------------------------------
data = pd.DataFrame({})
coins = []

for i in range(len(coinDataStruct)):
    coin = coinDataStruct[i][0]
    coins.append(coin)
    exchange = coinDataStruct[i][1]
    tradeCurrency = coinDataStruct[i][2]
    data[coin] = get_hist_price_data(coin, tradeCurrency, timeFrame, exchange)['close']

logReturns = np.log(data / data.shift(1))
logReturns = logReturns[1:]
cov = logReturns.cov().values
logReturns = logReturns.values

# Guess values
weights0 = [1 / len(coinDataStruct)] * len(coinDataStruct)

# Bounds of each asset (equal weights)
bounds = tuple((0, 1) for _ in range(len(coinDataStruct)))

# Constraint
constraint = ({'type': 'eq', 'fun': sum_criteria})

optimalPortfolio = minimize(objective_function, weights0, method='SLSQP', bounds=bounds, constraints=constraint)

print("Optimal weight allocation: ")
for coin, alloc in zip(coinDataStruct, optimalPortfolio.x):
    print(coin[0], ": ", alloc)


# -----Testing-----------------------------------------------------------
test = evaluate_portfolio(optimalPortfolio.x)
print("Return: {}, Volatility: {}, Sharpe ratio: {}".format(test[0], test[1], test[2]))
