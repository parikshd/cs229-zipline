import pandas as pd
import numpy as np
df1 = pd.read_csv("../milestone_data.csv")
cols = list(df1.columns.values)

df1500 = pd.read_csv("output/flights0-1500.csv", low_memory=False)
df3000 = pd.read_csv("output/flights1500-3000.csv", low_memory=False)

print(df1500.shape)
print(df3000.shape)

df1500_reduced = df1500[cols]
df3000_reduced = df3000[cols]

print(df1500_reduced.shape)
print(df3000_reduced.shape)

df_all_reduced = pd.concat([df1500_reduced,df3000_reduced], axis=0)
print(df_all_reduced.shape)

df_all_reduced.replace(r'\s*',np.nan,regex=True).replace('',np.nan)
data_new_dropped = df_all_reduced.dropna()

print(data_new_dropped.shape)
data_new_dropped.to_csv('output/all_flights_data_new_dropped_na.csv', index=False)