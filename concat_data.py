import pandas as pd

df1 = pd.read_csv("./data/data.csv", encoding='cp949')
df2 = pd.read_csv("./data/wellness.csv", encoding='cp949')
df3 = pd.read_csv("./data/chatbot_data.csv", encoding='cp949')

df2['score'] = 1

df_con = pd.concat([df1, df2]).reset_index(drop=True)
df_con = pd.concat([df_con, df3]).reset_index(drop=True)

df_con.to_csv('./data/train_data.csv', encoding='cp949', index=False)

# 송영숙님 데이터 제외한 데이터셋 만들기
df_con2 = pd.concat([df1, df2]).reset_index(drop=True)
df_con2.to_csv("./data/train_data2.csv", encoding='cp949', index=False)