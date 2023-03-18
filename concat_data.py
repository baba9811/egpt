import pandas as pd

df1 = pd.read_csv("data.csv", encoding='cp949')
df2 = pd.read_csv("wellness.csv", encoding='cp949')
df2['score'] = 1

df3 = pd.concat([df1, df2]).reset_index(drop=True)

df3.to_csv('train_data.csv', encoding='cp949', index=False)

