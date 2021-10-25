import numpy as np
import requests
import pandas as pd
import datetime


def calc_sortino(returns, r_mar):
    ner = []
    er = []

    for i in range(len(returns)):
        er.append(returns[i] - r_mar)
        if (returns[i] - r_mar) < 0:
            ner.append((returns[i] - r_mar) ** 2)
        else:
            ner.append(0)

    down_risk = np.sqrt(sum(ner) / len(returns))
    mean_return = np.mean(er)
    return mean_return / down_risk


def calc_sharpe(daily_return, annual_bench_return):
    r_mar = annual_bench_return
    ex_daily_ret = [r - r_mar for r in daily_return]
    sharpe = np.mean(ex_daily_ret) / np.std(ex_daily_ret, ddof=1)
    return sharpe


def get_hist_price_data(coin_to, cc_from, limit, exchange):
    curl = 'https://min-api.cryptocompare.com/data/histoday?fsym={}&tsym={}&limit={}&e={}' \
        .format(coin_to.upper(), cc_from.upper(), limit, exchange)
    data = requests.get(curl, headers={'User-Agent': 'Mozilla/5.0'}).json()['Data']
    df = pd.DataFrame(data)
    df['date'] = [datetime.datetime.fromtimestamp(d).strftime('%Y-%m-%d') for d in df.time]
    return df


def get_pareto_frontier(x, y):
    data = sorted([[x[i], y[i]] for i in range(len(x))])
    front = [data[0]]
    for pair in data[1:]:
        if pair[1] >= front[-1][1]:
            front.append(pair)
    return list(front)
