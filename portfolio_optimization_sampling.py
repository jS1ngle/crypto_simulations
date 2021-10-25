# -*- coding: utf-8 -*-
# Author: jS1ngle
# License: MIT License (http://opensource.org/licenses/MIT)

from SimulationHelperFunctions import *
import numpy as np


# -----Functions---------------------------------------------------------

def evaluate_portfolios(coin_data_struct, number_portfolio, portfolio_samples, logReturns, cov, timeFrame):
    # Initialize results matrix
    results = np.zeros((3 + len(coin_data_struct), number_portfolio))

    for i in range(number_portfolio):
        weights = portfolio_samples[i]
        # portfolio return
        results[0, i] = np.sum(np.array(logReturns.mean().tolist()) * weights) * timeFrame
        # portfolio volatility
        results[1, i] = np.sqrt(np.dot(weights.T, np.dot(cov, weights))) * np.sqrt(timeFrame)

        # Calc Sortino ratio and add weights
        tmp = []
        for col in range(len(weights)):
            tmp.append(calc_sortino(logReturns[coins[col]].tolist()[1:] * timeFrame, 0.0))
            results[col + 3, i] = weights[col]

        # Total Sortino ratio
        results[2, i] = np.sum(np.multiply(tmp, weights)) * np.sqrt(timeFrame)

    return results


# -----Input-------------------------------------------------------------

# In time periods (usually days)
timeFrame = 30
numberPortfolio = 20000
coinDataStruct = [['BNB', 'Binance', 'BTC'],
                  ['ADA', 'Binance', 'BTC'],
                  ['ETH', 'Binance', 'BTC'],
                  ['SOL', 'Binance', 'BTC'],
                  ['DOGE', 'Binance', 'BTC'],
                  ['DOT', 'Binance', 'BTC']]

df2 = pd.DataFrame({})

for i in range(len(coinDataStruct)):
    coin = coinDataStruct[i][0]
    exchange = coinDataStruct[i][1]
    tradeCurrency = coinDataStruct[i][2]

    # Pass closing price to new dataframe
    df2[coin] = get_hist_price_data(coin, tradeCurrency, timeFrame, exchange)['close']

coins = []
for i in range(len(coinDataStruct)):
    coins.append(coinDataStruct[i][0])

logReturns = np.log(df2 / df2.shift(1))
cov = logReturns.cov().as_matrix()

# Create portfolios with different coin allocations
portfolioSamples = np.random.dirichlet(np.ones(len(coinDataStruct)), numberPortfolio)

results = evaluate_portfolios(coinDataStruct,
                              numberPortfolio,
                              portfolioSamples,
                              logReturns,
                              cov,
                              timeFrame)

# Convert results array to Pandas DataFrame
results_frame = pd.DataFrame(results.T)
resColumn = ['ret', 'stdev', 'Sortino'] + coins
results_frame.columns = results_frame.columns[:0].tolist() + resColumn

# print results_frame
results_frame.to_csv('portfolioOptimization.csv')

# Process efficient frontier data
pf = get_pareto_frontier(results_frame.stdev, results_frame.ret)

x = [x[0] for x in pf]
y = [y[1] for y in pf]

pfFrame = results_frame[(results_frame.stdev.isin(x)) & (results_frame.ret.isin(y))]
pfFrame.set_index('Sortino', inplace=True)

pfFrame.to_csv('paretoFrontier.csv')
