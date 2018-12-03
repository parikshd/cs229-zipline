import json
from pandas.io.json import json_normalize
import pandas as pd
with open('../all_logs/flights0_1500.json') as f:
    data_1 = json.load(f)

with open('../all_logs/flights1500_3000.json') as f:
    data_2 = json.load(f)

data_df1 = json_normalize(data_1['flights'])
print(data_df1.shape)
data_df2 = json_normalize(data_2['flights'])
print(data_df2.shape)

combined_df = pd.concat([data_df1, data_df2])
print(combined_df.shape)

combined_df.to_csv('output/flights_raw.csv', encoding='utf-8', index=False)