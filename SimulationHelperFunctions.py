import numpy as np
import requests
import pandas as pd
import datetime

def calcSortino(returns, rMar):
    ner = []
    er = []

    for i in range(len(returns)):
        er.append(returns[i] - rMar)
        if((returns[i] - rMar) < 0):
            ner.append((returns[i] - rMar)**2)
        else:
            ner.append(0)

    downRisk = np.sqrt(sum(ner)/len(returns))
    meanReturn = np.mean(er)
    return meanReturn/downRisk


def calcSharpe(dailyReturn, annualBenchReturn):
    rMar = annualBenchReturn
    exDailyRet = [r - rMar for r in dailyReturn]
    sharpe = np.mean(exDailyRet) / np.std(exDailyRet, ddof=1)
    return sharpe


def getHistPriceData(coinTo, ccFrom, limit, exchange):
    curl = 'https://min-api.cryptocompare.com/data/histoday?fsym={}&tsym={}&limit={}&e={}'\
            .format(coinTo.upper(), ccFrom.upper(), limit, exchange)
    data = requests.get(curl, headers={'User-Agent': 'Mozilla/5.0'}).json()['Data']
    df = pd.DataFrame(data)
    df['date'] = [datetime.datetime.fromtimestamp(d).strftime('%Y-%m-%d') for d in df.time]
    return df
