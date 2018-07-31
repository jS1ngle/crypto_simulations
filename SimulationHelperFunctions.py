import numpy as np

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
