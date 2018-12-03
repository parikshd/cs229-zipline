import pandas as pd
import numpy as np

df1 = pd.read_csv("../milestone_data.csv")
cols = list(df1.columns.values)

df_all = pd.read_csv("output/flights_raw.csv", low_memory=False)
print(df_all.shape)

df_all.drop(df_all.select_dtypes(['object']), inplace=True, axis=1)
print(df_all.shape)

df_all = df_all[np.isfinite(df_all['highest_failure_level.id'])]
df_all.dropna(axis=1, how='all', inplace=True)

print(df_all.shape)
df_all_copy = df_all

df_all.fillna(df_all.mean(), inplace=True)
df_all.to_csv('output/flights_pass_1.csv', encoding='utf-8', index=False)

df_all_copy.fillna(0, inplace=True)
df_all_copy.to_csv('output/flights_pass_1_na_0.csv', encoding='utf-8', index=False)

df_all_copy_pass_fail = df_all_copy
for idx, row in df_all_copy_pass_fail.iterrows():
    if  df_all_copy_pass_fail.loc[idx,'highest_failure_level.id'] == 1:
        df_all_copy_pass_fail.loc[idx,'highest_failure_level.id'] = 0
    if  df_all_copy_pass_fail.loc[idx,'highest_failure_level.id'] == 2:
        df_all_copy_pass_fail.loc[idx,'highest_failure_level.id'] = 1
    if  df_all_copy_pass_fail.loc[idx,'highest_failure_level.id'] == 4:
        df_all_copy_pass_fail.loc[idx,'highest_failure_level.id'] = 1

df_all_copy_pass_fail.to_csv('output/flights_pass_fail.csv', encoding='utf-8', index=False)

df_all_copy_read = pd.read_csv('output/flights_pass_1_na_0.csv', low_memory=False)
print(df_all_copy_read.shape)
df_all_copy_fail_crash = df_all_copy_read[df_all_copy_read['highest_failure_level.id'] != 1]
print(df_all_copy_fail_crash.shape)

for idx, row in df_all_copy_fail_crash.iterrows():
    if  df_all_copy_fail_crash.loc[idx,'highest_failure_level.id'] == 2:
        df_all_copy_fail_crash.loc[idx,'highest_failure_level.id'] = 0
    if  df_all_copy_fail_crash.loc[idx,'highest_failure_level.id'] == 4:
        df_all_copy_fail_crash.loc[idx,'highest_failure_level.id'] = 1

df_all_copy_fail_crash.to_csv('output/flights_fail_crash.csv', encoding='utf-8', index=False)

# df1500_reduced = df1500[cols]
# df3000_reduced = df3000[cols]
#
# print(df1500_reduced.shape)
# print(df3000_reduced.shape)
#
# df_all_reduced = pd.concat([df1500_reduced,df3000_reduced], axis=0)
# print(df_all_reduced.shape)
#
# df_all_reduced.replace(r'\s*',np.nan,regex=True).replace('',np.nan)
# data_new_dropped = df_all_reduced.dropna()
#
# print(data_new_dropped.shape)
# data_new_dropped.to_csv('output/all_flights_data_new_dropped_na.csv', index=False)