"""
To train this model, in your terminal:
> python train_sentiment_model.py
"""
from sklearn.externals import joblib
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import sklearn.feature_selection as skfs
from nltk.sentiment.vader import SentimentIntensityAnalyzer as SIA

print('Loading the dataset')
us_data = pd.read_csv('~/Desktop/data-miners/project/data/USvideos.csv')
ca_data = pd.read_csv('~/Desktop/data-miners/project/data/CAvideos.csv')
gb_data = pd.read_csv('~/Desktop/data-miners/project/data/GBvideos.csv')
frames = [us_data, ca_data, gb_data]
whole_data = pd.concat(frames)
whole_data.drop_duplicates(keep=False, inplace=True)
print('Training SentimentIntensityAnalyzer')

sentiment_analysis = SIA()

sentiment_results_dict = {}
countries = {'us': us_data, 'ca': ca_data, 'gb': gb_data, 'all': whole_data}
col = ['tags', 'description', 'title']
for country in countries.items():
    sentiment_results_dict[country[0]] = {}
    for col_name in col:
        percentage = []
        sentiment_results = []
        for i in country[1][col_name]:
            ps = sentiment_analysis.polarity_scores(str(i))
            ps[col_name] = i
            sentiment_results.append(ps)
        df_sentiment = pd.DataFrame.from_records(sentiment_results)
        df_sentiment['Sentiment'] = 'Neutral'
        df_sentiment.loc[df_sentiment['compound']
                         > 0.3, 'Sentiment'] = 'Positive'
        df_sentiment.loc[df_sentiment['compound']
                         < -0.3, 'Sentiment'] = 'Negative'
        pos_lam = df_sentiment.apply(
            lambda x: True if x['Sentiment'] == 'Positive' else False, axis=1)
        neg_lam = df_sentiment.apply(
            lambda x: True if x['Sentiment'] == 'Negative' else False, axis=1)
        neu_lam = df_sentiment.apply(
            lambda x: True if x['Sentiment'] == 'Neutral' else False, axis=1)
        pos_count = len(pos_lam[pos_lam == True].index)
        neg_count = len(neg_lam[neg_lam == True].index)
        neu_count = len(neu_lam[neu_lam == True].index)
        total_count = pos_count + neg_count + neu_count
        pos_percent = (pos_count/total_count) * 100
        neg_percent = (neg_count/total_count) * 100
        neu_percent = (neu_count/total_count) * 100
        percentage.append(pos_percent)
        percentage.append(neg_percent)
        percentage.append(neu_percent)
        sentiment_results_dict[country[0]][col_name] = percentage

print('Exporting the trained model')
joblib.dump(sentiment_analysis, 'model/sentiment_analysis.joblib')
joblib.dump(sentiment_results_dict, 'model/sentiment_results_dict.joblib')
