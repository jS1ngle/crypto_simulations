# -*- coding: utf-8 -*-
#Author: johannes <info@numex-blog.com>, 08.09.18
#License: MIT License (http://opensource.org/licenses/MIT)
import pandas as pd
from SimulationHelperFunctions import getHistPriceData
import matplotlib.pyplot as plt
import numpy as np

#-----------------------------------------------------------------------
#-----Functions---------------------------------------------------------
def evaluatePortfolio(weights):
    weights = np.array(weights)
    ret = (np.sum(np.array(logReturns.mean(axis=0)) * weights)) * timeFrame
    #ret = np.sum(np.array(logReturns.mean().tolist()) * weights) * timeFrame
    vol = np.sqrt(np.dot(weights.T, np.dot(cov, weights))) * np.sqrt(timeFrame)
    sharpe = ret/vol
    return np.array([ret, vol, sharpe])


def evaluateAllPortfolios(portfolioWeights, numberPortfolio):
    res = []
    for i in range(numberPortfolio):
        weights = portfolioWeights[i]
        tmpEval = evaluatePortfolio(weights)
        tmpRes = np.concatenate((tmpEval, weights), axis=0)
        res.append(tmpRes)
    return res

def getParetoFrontier(x, y):
    data = sorted([[x[i], y[i]] for i in range(len(x))])
    front = [data[0]]

    for pair in data[1:]:
        if pair[1] >= front[-1][1]:
            front.append(pair)

    return list(front)

#-----------------------------------------------------------------------
#-----Input---------------------------------------------------------
timeFrame = 30
numberPortfolio = 1000000
coinDataStruct = [['XMR', 'Binance', 'BTC'],
                  ['VET', 'Binance', 'BTC'],
                  ['OMG', 'Binance', 'BTC'],
                  ['NEO', 'Binance', 'BTC']]


#df = getHistPriceData(coinDataStruct[0][0], coinDataStruct[0][2], timeFrame, coinDataStruct[0][1])

#print(df)

df2 = pd.DataFrame({})
coins = []

for i in range(len(coinDataStruct)):
    coin = coinDataStruct[i][0]
    coins.append(coin)
    exchange = coinDataStruct[i][1]
    tradeCurrency = coinDataStruct[i][2]
    #Pass closing price to new dataframe
    df2[coin] = getHistPriceData(coin, tradeCurrency, timeFrame, exchange)['close']


logReturns = np.log(df2 / df2.shift(1))
logReturns = logReturns[1:]
cov = logReturns.cov().as_matrix()
logReturns = logReturns.values


#Create portfolios with different coin allocations
portfolioSamples = np.random.dirichlet(np.ones(len(coinDataStruct)), numberPortfolio)

#Get results
results = evaluateAllPortfolios(portfolioSamples, numberPortfolio)

results_frame = pd.DataFrame(results)
resColumn = ['ret','stdev','Sharpe'] + coins
results_frame.columns = results_frame.columns[:0].tolist()  + resColumn

#print results_frame
results_frame.to_csv('portfolioOptimization.csv')

#Highest (Sharpe) Ratio
maxSharpe = results_frame.loc[results_frame['Sharpe'].idxmax()]

print(maxSharpe)

explode = [0.1] * len(maxSharpe[3:])

plt.pie(maxSharpe[3:],
        explode = explode,
        pctdistance=1.1,
        labeldistance=0.9,
        labels=maxSharpe[3:].index,
        autopct='%1.1f%%',
        radius=1)

plt.title('Max Sharpe ratio  ' + str(round(maxSharpe[2], 3)))
plt.show()

#Lowest standard deviation (volatility)
minVolatility = results_frame.loc[results_frame['stdev'].idxmin()]

print(minVolatility)

plt.pie(minVolatility[3:],
        explode = explode,
        pctdistance=1.1,
        labeldistance=0.9,
        labels=minVolatility[3:].index,
        autopct='%1.1f%%',
        radius=1)

plt.title('Min volatility  ' + str(round(minVolatility[2], 3)))
plt.show()

#create scatter plot coloured by Sharpe Ratio
plt.scatter(results_frame.stdev, results_frame.ret, c=results_frame.Sharpe, cmap='RdYlBu')
plt.xlabel('Volatility')
plt.ylabel('Returns')
cbar = plt.colorbar()
cbar.set_label('Sharpe ratio')

#Plot highest Sharpe Ratio
plt.scatter(maxSharpe[1], maxSharpe[0], color='r', s=100)

#Plot lowest variance portfolio
plt.scatter(minVolatility[1], minVolatility[0], color='g', s=100)

#Process efficient frontier data
pf = getParetoFrontier(results_frame.stdev, results_frame.ret)

x = [x[0] for x in pf]
y = [y[1] for y in pf]

pfFrame = results_frame[(results_frame.stdev.isin(x)) & (results_frame.ret.isin(y))]
pfFrame.set_index('Sharpe', inplace=True)

pfFrame.to_csv('paretoFrontier.csv')

#print(pfFrame)

plt.plot(x, y, '-r', linewidth=2.0)

plt.show()
