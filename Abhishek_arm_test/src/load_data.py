import json
from pandas.io.json import json_normalize
import pandas as pd

df_1 = pd.read_csv("testinput/flights_new_till_03dec.csv", low_memory=False)
print(df_1.shape)

df_2 = pd.read_csv("testinput/flights_03dec_08_dec.csv", low_memory=False)
print(df_2.shape)

df_3 = pd.read_csv("testinput/flights_failed.csv", low_memory=False)
print(df_3.shape)

combined_df = pd.concat([df_1, df_2,df_3])
print(combined_df.shape)

combined_df.to_csv('testinput/all_test_with_failures.csv', encoding='utf-8', index=False)