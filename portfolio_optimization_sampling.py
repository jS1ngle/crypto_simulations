# -*- coding: utf-8 -*-
#Author: johannes <info@numex-blog.com>, 22.07.18
#License: MIT License (http://opensource.org/licenses/MIT)

import pandas as pd
import requests

from SimulationHelperFunctions import *

import matplotlib.pyplot as plt
import numpy as np
import datetime
import scipy
import scipy.stats

#-----------------------------------------------------------------------
#-----Functions---------------------------------------------------------

def getHistPriceData(coinTo, ccFrom, limit, exchange):
    curl = 'https://min-api.cryptocompare.com/data/histoday?fsym={}&tsym={}&limit={}&e={}'\
            .format(coinTo.upper(), ccFrom.upper(), limit, exchange)
    data = requests.get(curl, headers={'User-Agent': 'Mozilla/5.0'}).json()['Data']
    df = pd.DataFrame(data)
    df['date'] = [datetime.datetime.fromtimestamp(d).strftime('%Y-%m-%d') for d in df.time]
    return df

def evaluatePortfolios(coinDataStruct, numberPortfolio, portfolioSamples, logReturns, cov, timeFrame):
    #Initialize results matrix
    results = np.zeros((3+len(coinDataStruct), numberPortfolio))

    for i in range(numberPortfolio):
        weights = portfolioSamples[i]
        #portfolio return
        results[0,i] = np.sum(np.array(logReturns.mean().tolist()) * weights) * timeFrame
        #portfolio volatility
        results[1,i] = np.sqrt(np.dot(weights.T, np.dot(cov, weights))) * np.sqrt(timeFrame)

        #Calc Sortino ratio and add weights
        tmp = []
        for col in range(len(weights)):
            tmp.append(calcSortino(logReturns[coins[col]].tolist()[1:] * timeFrame, 0.0))
            results[col+3,i] = weights[col]

        #Total sortino
        results[2,i] = np.sum(np.multiply(tmp, weights)) * np.sqrt(timeFrame)

    return results

#-----------------------------------------------------------------------
#-----Input-------------------------------------------------------------

timeFrame = 30
numberPortfolio = 20000
coinDataStruct = [['BTC', 'Binance', 'BTC'],
                  ['ETH', 'Binance', 'BTC'],
                  ['XRP', 'Binance', 'BTC'],
                  ['BCH', 'Binance', 'BTC']]

df2 = pd.DataFrame({})

for i in range(len(coinDataStruct)):
    coin = coinDataStruct[i][0]
    exchange = coinDataStruct[i][1]
    tradeCurrency = coinDataStruct[i][2]

    #Pass closing price to new dataframe
    df2[coin] = getHistPriceData(coin, tradeCurrency, timeFrame, exchange)['close']

coins = []
for i in range(len(coinDataStruct)):
    coins.append(coinDataStruct[i][0])

logReturns = np.log(df2 / df2.shift(1))
cov = logReturns.cov().as_matrix()

#Create portfolios with different coin allocations
portfolioSamples = np.random.dirichlet(np.ones(len(coinDataStruct)), numberPortfolio)

results = evaluatePortfolios(coinDataStruct,
                             numberPortfolio,
                             portfolioSamples,
                             logReturns,
                             cov,
                             timeFrame)

#Convert results array to Pandas DataFrame
results_frame = pd.DataFrame(results.T)
resColumn = ['ret', 'stdev', 'Sortino'] + coins
results_frame.columns = results_frame.columns[:0].tolist()  + resColumn

#print results_frame
results_frame.to_csv('portfolioOptimization.csv')

#Process efficient frontier data
pf = getParetoFrontier(results_frame.stdev, results_frame.ret)

x = [x[0] for x in pf]
y = [y[1] for y in pf]

pfFrame = results_frame[(results_frame.stdev.isin(x)) & (results_frame.ret.isin(y))]
pfFrame.set_index('Sortino', inplace=True)

pfFrame.to_csv('paretoFrontier.csv')
