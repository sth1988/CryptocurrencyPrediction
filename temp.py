import json
import ssl
import pandas as pd
from urllib.request import urlopen

start_time = '1439014500'  ## From 2015

coins = ['BTC', 'LTC', 'ETH', 'XMR']
df_list=[]
for coin in coins:
    url = 'https://poloniex.com/public?command=returnChartData&currencyPair=USDT_' \
          +coin+'&start='\
          +start_time+\
          '&end=9999999999&resolution=auto'
    context = ssl._create_unverified_context()
    openUrl = urlopen(url, context=context)
    r = openUrl.read()
    openUrl.close()
    d = json.loads(r.decode())
    print(pd.DataFrame(d).shape)
    
url = 'https://poloniex.com/public?command=returnChartData&currencyPair=USDT_ETH' \
      '&start='+start_time+\
      '&end=9999999999&resolution=auto'
context = ssl._create_unverified_context()
openUrl = urlopen(url, context=context)
r = openUrl.read()
openUrl.close()
d = json.loads(r.decode())

df = pd.DataFrame(d)
original_columns=[u'close', u'date', u'high', u'low', u'open']
new_columns = ['Close','Timestamp','High','Low','Open']
df = df.loc[:,original_columns]
df.columns = new_columns
df.head()

df.to_csv('data/bitcoin2015to2019.csv',index=None)
