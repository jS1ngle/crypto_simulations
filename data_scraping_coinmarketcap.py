#Author: jS1ngle
#License: MIT License (http://opensource.org/licenses/MIT)

import urllib.request
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime

url = 'https://coinmarketcap.com/currencies/bitcoin/historical-data/?start=20170615&end=20180615'
page = urllib.request.urlopen(url)
soup = BeautifulSoup(page, 'html.parser')

priceDiv = soup.find('div', attrs={'class':'table-responsive'})
rows = priceDiv.find_all('tr')

data = []
i = 0;

for row in rows:
    tmp = []
    tds = row.findChildren()

    for td in tds:
        tmp.append(td.text)

    if(i > 0):
        tmp[0] = tmp[0].replace(',','')
        tmp[5] = tmp[5].replace(',','')
        tmp[6] = tmp[6].replace(',','')
        data.append({'date':datetime.strptime(tmp[0], '%b %d %Y'),
                     'open':float(tmp[1]),
                     'high':float(tmp[2]),
                     'low':float(tmp[3]),
                     'close':float(tmp[4]),
                     'volume':float(tmp[5]),
                     'mcap':float(tmp[6])})

    i = i + 1;

df = pd.DataFrame(data)
