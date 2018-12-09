import pandas as pd
import numpy as np

df_all = pd.read_csv("testinput/all_test_with_failures.csv", low_memory=False)
print(df_all.shape)

df_all.drop(df_all.select_dtypes(['object']), inplace=True, axis=1)
print(df_all.shape)

df_all = df_all[np.isfinite(df_all['highest_failure_level.id'])]
df_all.dropna(axis=1, how='all', inplace=True)

print(df_all.shape)

df_all.fillna(0, inplace=True)
df_all.to_csv('testinput/all_test_with_failures_clean.csv', encoding='utf-8', index=False)
