# -*- coding: utf-8 -*-
#Author: johannes <info@numex-blog.com>, 01.11.18
#License: MIT License (http://opensource.org/licenses/MIT)

import pandas as pd
import requests
import matplotlib.pyplot as plt
from pytrends.request import TrendReq
import datetime
from datetime import timedelta
from scipy.stats.stats import pearsonr
from SimulationHelperFunctions import getHistPriceData


#===Input data
timeWindow = 100
timeWindowMovingAvg = 5
coinTo = "BTC"
ccFrom = "EUR"
exchange = "Kraken"
keywords = ["bitcoin"]

#===Get raw data
df = getHistPriceData(coinTo, ccFrom, timeWindow, exchange)
df.columns = [['close', 'high', 'low', 'open', 'time', 'volumefrom', 'volumeto', 'date']]

#===Starting point to have the exact amount of data
sp = len(df.time[timeWindowMovingAvg-1:])

#===Pass data to pytrend and execute it
beginDateWindow = datetime.datetime.now().date() - timedelta(days=timeWindow)
pytrend = TrendReq()
dataWindow =str(beginDateWindow) + " " + str(datetime.datetime.now().date())
pytrend.build_payload(keywords, cat=0, timeframe=dataWindow)
dfTrend = pytrend.interest_over_time()  #using interest over time function
dfTrend.columns = ['keyword', 't']

#===Moving average
maTrendPrice = df.close.rolling(center=False, window=timeWindowMovingAvg).mean()
maTrendVolume = df.volumeto.rolling(center=False, window=timeWindowMovingAvg).mean()
maTrendGoogle = dfTrend.keyword.rolling(center=False, window=timeWindowMovingAvg).mean()

#===Fatten lists
trend = dfTrend.keyword.tolist()
close = [y for x in df['close'].values.tolist() for y in x]
volume = [y for x in df['volumeto'].values.tolist() for y in x]
date = [y for x in df['date'].values.tolist() for y in x]

#===Drop last 3 entries since google is 3 days delayed
close = close[:-3]
volume = volume[:-3]
date = date[:-3]

#===Calc Pearson correlation coefficients
priceCorrelationCoefficient = pearsonr(trend, close)[0]
volumeCorrelationCoefficient = pearsonr(trend, volume)[0]

print(priceCorrelationCoefficient)
print(volumeCorrelationCoefficient)

#===Plot data price and trend
fig, ax1 = plt.subplots()
ax1.plot(date, close, 'r', label='Price', linewidth=1.5)
ax1.set_ylabel('Price in Euro', color='r')

ax2 = ax1.twinx()
ax2.plot(trend, 'b', label='Google trend', linewidth=1.5)
ax2.set_ylabel('Google trend', color='b')

for label in ax1.xaxis.get_ticklabels():
    label.set_rotation(45)

ax1.xaxis.set_major_locator(plt.MaxNLocator(10))

fig.tight_layout()

fig.savefig('correlating_price_trend.png')

#===Plot volume and trend
fig2, ax3 = plt.subplots()
ax3.plot(date, volume, 'g', label='Volume', linewidth=1.5)
ax3.set_ylabel('Volume', color='g')

ax4 = ax3.twinx()
ax4.plot(trend, 'b', label='Google trend', linewidth=1.5)
ax4.set_ylabel('Google trend', color='b')

for label in ax3.xaxis.get_ticklabels():
    label.set_rotation(45)

ax3.xaxis.set_major_locator(plt.MaxNLocator(10))

fig2.tight_layout()
plt.show()

fig2.savefig('correlating_volume_trend.png')
