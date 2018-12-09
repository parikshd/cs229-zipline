import pandas as pd
import numpy as np
import json
from pandas.io.json import json_normalize

with open('testinput/flight_failure/flight_failed.json') as f:
    data_1 = json.load(f)

data_df1 = json_normalize(data_1['flights'])
print(data_df1.shape)

data_df1.drop(data_df1.select_dtypes(['object']), inplace=True, axis=1)
print("dropped string fields")
print(data_df1.shape)

df_all = data_df1[np.isfinite(data_df1['highest_failure_level.id'])]
df_all.dropna(axis=1, how='all', inplace=True)

print("dropped string fields")
print(df_all.shape)

df_all.fillna(0, inplace=True)
df_all.to_csv('testinput/flights_failed.csv', encoding='utf-8', index=False)
