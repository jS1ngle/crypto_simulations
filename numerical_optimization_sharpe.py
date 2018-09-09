# -*- coding: utf-8 -*-
#Author: johannes <info@numex-blog.com>, 01.09.18
#License: MIT License (http://opensource.org/licenses/MIT)
import pandas as pd
from SimulationHelperFunctions import getHistPriceData
import numpy as np
import scipy
from scipy.optimize import minimize

#-----------------------------------------------------------------------
#-----Functions---------------------------------------------------------
def evaluatePortfolio(weights):
    weights = np.array(weights)
    ret = (np.sum(np.array(logReturns.mean(axis=0)) * weights)) * timeFrame
    #ret = np.sum(np.array(logReturns.mean().tolist()) * weights) * timeFrame
    vol = np.sqrt(np.dot(weights.T, np.dot(cov, weights))) * np.sqrt(timeFrame)
    sharpe = ret/vol
    return np.array([ret, vol, sharpe])

#-----Function to be minimized
def objectiveFunction(weights, sign=1):
    return sign*evaluatePortfolio(weights)[1]

#-----Additional criteria
def sumCriteria(weights):
    return np.sum(weights) - 1

#-----------------------------------------------------------------------
#-----Input-------------------------------------------------------------
timeFrame = 30

#-----Exemplary
coinDataStruct = [['XMR', 'Binance', 'BTC'],
                  ['VET', 'Binance', 'BTC'],
                  ['OMG', 'Binance', 'BTC'],
                  ['NEO', 'Binance', 'BTC']]

#-----------------------------------------------------------------------
#-----Function calls----------------------------------------------------
data = pd.DataFrame({})
coins = []

for i in range(len(coinDataStruct)):
    coin = coinDataStruct[i][0]
    coins.append(coin)
    exchange = coinDataStruct[i][1]
    tradeCurrency = coinDataStruct[i][2]
    data[coin] = getHistPriceData(coin, tradeCurrency, timeFrame, exchange)['close']

logReturns = np.log(data / data.shift(1))
logReturns = logReturns[1:]
cov = logReturns.cov().as_matrix()
logReturns = logReturns.values

#Guess values
weights0 = [0.25, 0.25, 0.25, 0.25]

#Bounds of each asset
bounds = ((0, 1), (0, 1), (0, 1), (0, 1))

#Constraint
constraint = ({'type':'eq','fun': sumCriteria})

optimimalPortfolio = minimize(objectiveFunction, weights0, method='SLSQP', bounds=bounds, constraints=constraint)
print(optimimalPortfolio)

#-----------------------------------------------------------------------
#-----Testing-----------------------------------------------------------
test = evaluatePortfolio(optimimalPortfolio.x)
print(test)
