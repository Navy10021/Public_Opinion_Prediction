import pandas as pd
from preprocessing.twitter_preprocessing import *

# Twitter examples
data_path_1 = './data/임대차3법/임대차3법_2021년1월~2022년6월.xlsx'
data_path_2 = './data/임대차3법/임대차3법_2020년7월~12월.xlsx'

col_name = ['Date', 'Id', 'Text']
df_1 = pd.read_excel(data_path_1, sheet_name = 2, names = col_name, header = None)[2:]
df_2 = pd.read_excel(data_path_2, sheet_name = 2, names = col_name, header = None)[2:]
df = pd.concat([df_1, df_2], ignore_index = True)

# Drop noise data
drop_idx = df[df['Date'] == '기간 : 2021/01/01~2022/06/30'].index
df = df.drop(drop_idx)
drop_idx = df[df['Date'] == '작성일'].index
df = df.drop(drop_idx)
df = df.dropna()
df = df.drop_duplicates(subset=['Text'])

# Datetime
df['Date'] = pd.to_datetime(df['Date'])

print(">> Data Size :", len(df))

new_df = df[['Date', 'Text']]

# Text Preprocessing
cleaned_text = []
for text in new_df.Text:
    new_text = clean_multi_space(text)
    new_text = clean_chinese(new_text)
    new_text = filter_chars(new_text)
    new_text = clean_hashtage(new_text)
    new_text = clean_text(new_text)
    new_text = clean_emoji(new_text)
    new_text = new_text.strip()
    cleaned_text.append(new_text)

new_df['Text'] = cleaned_text
new_df['Sentiment'] = [0]*len(new_df)  # 부정 : 0 , 긍정 : 1
new_df = new_df.dropna()
print(">> Total Number of Twitters : ", len(new_df))
print(new_df.head())

# Save dataFrame
save_path = './data/clean_twitter_1.csv'
new_df.to_csv(save_path, mode = 'w')