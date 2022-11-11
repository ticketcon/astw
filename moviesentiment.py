import pandas as pd
import nltk
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# reading and wragling data
df_avatar = pd.read_csv('avatar.csv',encoding = 'unicode_escape', engine='python')
df_avt1 = df_avatar.groupby('character').count()
df_avt1 = df_avt1.sort_values(by=['character_words'], ascending=False)[:10]
top_character_names = df_avt1.index.values

# filtering out non-top characters
df_character_sentiment = df_avatar[df_avatar['character'].isin(top_character_names)]
df_character_sentiment = df_character_sentiment[['character', 'character_words']]

# calculating sentiment score
sid = SentimentIntensityAnalyzer()
df_character_sentiment.reset_index(inplace=True, drop=True)
df_character_sentiment[['neg', 'neu', 'pos', 'compound']] = df_character_sentiment['character_words'].apply(sid.polarity_scores).apply(pd.Series)
df_character_sentiment
