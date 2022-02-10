import pandas as pd
import numpy as np
import datetime

from_date = '2020-01-01'
to_date = '2021-12-31'
file_path = 'C:/Users/noldec/Desktop/binance'
coins =['EOS','ADA', 'SOL', 'TRX']
coin_dataframes = {}

# Helper
def encode(data, col, max_val):
    ohclv[col + '_sin'] = np.sin(2 * np.pi * ohclv[col]/max_val)
    ohclv[col + '_cos'] = np.cos(2 * np.pi * ohclv[col]/max_val)
    return data


#FGI
fng_index = pd.read_csv('%s/data.csv' % file_path)
fng_index['date'] = fng_index['date'].apply(lambda x: datetime.datetime.strptime(x, "%d-%m-%Y").strftime("%Y-%m-%d"))
fng_index['date'] = pd.to_datetime(fng_index['date'])
fng_index = fng_index[(fng_index['date'] > pd.to_datetime(from_date)) & (fng_index['date'] < pd.to_datetime(to_date)) ]
fng_index = fng_index.set_index('date')
fng_index = fng_index.drop('classification', axis=1)

for coin in coins:
    df_name = 'df_'+ coin
    ohclv = pd.read_json('%s/%s_USDT-5m.json' % (file_path, coin))
    ohclv.columns =['date_raw','open', 'high', 'low', 'close','volume']


    #OHCLV
    ohclv['date_time'] = ohclv['date_raw'].apply(lambda x: datetime.datetime.fromtimestamp(x/1000).strftime('%Y-%m-%d %H:%M:%S.%f'))
    ohclv['date_time'] = pd.to_datetime(ohclv['date_time'])
    ohclv['date'] = ohclv['date_raw'].apply(lambda x: datetime.datetime.fromtimestamp(x/1000).strftime('%Y-%m-%d'))
    ohclv['date'] = pd.to_datetime(ohclv['date'])
    ohclv = ohclv[(ohclv['date'] > pd.to_datetime(from_date)) & (ohclv['date'] < pd.to_datetime(to_date))  ]

    #Cylicial Feature Encoding
    #https://towardsdatascience.com/cyclical-features-encoding-its-about-time-ce23581845ca

    #Weekday cos & sin
    ohclv['dayOfWeek'] = ohclv['date'].dt.dayofweek
    ohclv = encode(ohclv, 'dayOfWeek', ohclv["dayOfWeek"].max())
    #Date cos & sin
    ohclv['dayOfYear'] = ohclv['date_time'].apply(lambda x: int(x.strftime('%j')))
    ohclv = encode(ohclv, 'dayOfYear', ohclv["dayOfYear"].max())
    #Time cos & sin
    ohclv["minute"] = (ohclv['date_time'].dt.hour*60) + ohclv['date_time'].dt.minute
    ohclv = encode(ohclv, 'minute', ohclv["minute"].max())
    #Month cos & sin
    ohclv["month"] = ohclv['date_time'].dt.month
    ohclv = encode(ohclv, 'month', ohclv["month"].max())
    ohclv = ohclv.set_index('date')

    #Marketcap
    marketcap = pd.read_csv('%s/%s-usd-max.csv' % (file_path, coin))
    marketcap['date'] = marketcap['snapped_at'].apply(lambda x: datetime.datetime.strptime(x[:-4], '%Y-%m-%d %H:%M:%S'))
    marketcap['date'] = pd.to_datetime(marketcap['date'])
    marketcap = marketcap.set_index('date')

    # Merge OHCLV & FNG
    df=pd.merge((pd.merge(ohclv,fng_index, how='inner', left_index=True, right_index=True)), marketcap, how='inner', left_index=True, right_index=True)

    # Clean dataframe
    df = df.drop('snapped_at', axis=1)
    df = df.drop('total_volume', axis=1)
    df = df.drop('price', axis=1)
    df = df.drop('date_raw', axis=1)
    df = df.drop('minute', axis=1)
    df = df.drop('dayOfWeek', axis=1)
    df = df.drop('month', axis=1)
    df = df.drop('dayOfYear', axis=1)
    df = df.drop('date_time', axis=1)

    df['Pair'] = '%s_USDT' % coin
    coin_dataframes[df_name] = df

frames = [coin_dataframes['df_EOS'], coin_dataframes['df_ADA'], coin_dataframes['df_TRX'], coin_dataframes['df_SOL']]
result = pd.concat(frames, ignore_index=False)
coin_dataframes['complete_df'] = result


# Twitter sentiment
# reddit sentiment
# google sentiment
# fear and greed stocks
# Bitcoin dominance
# Weeks after Listing
# Upcoming events
