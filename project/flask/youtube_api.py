"""
To run this app, in your terminal:
python youtube_api.py
"""
import connexion
from sklearn.externals import joblib
import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import sklearn.feature_selection as skfs
from pathlib import Path
from nltk.sentiment.vader import SentimentIntensityAnalyzer as SIA

app = connexion.FlaskApp(__name__, port=8080, specification_dir='swagger/')
application = app.app

# Load the followint pre-trained models:
filepath = './model/'

# Categories and counts models
us_cat_count = joblib.load(filepath + 'us_cat_count.joblib')
ca_cat_count = joblib.load(filepath + 'ca_cat_count.joblib')
gb_cat_count = joblib.load(filepath + 'gb_cat_count.joblib')

# Overall trending videos like/dislike engagement (over 10%) models
us_ld_engagement = joblib.load(filepath + 'us_ld_engagement.joblib')
ca_ld_engagement = joblib.load(filepath + 'ca_ld_engagement.joblib')
gb_ld_engagement = joblib.load(filepath + 'gb_ld_engagement.joblib')

# Overall trending videos comment engagement (over 2%) models
us_comm_engagement = joblib.load(filepath + 'us_comm_engagement.joblib')
ca_comm_engagement = joblib.load(filepath + 'ca_comm_engagement.joblib')
gb_comm_engagement = joblib.load(filepath + 'gb_comm_engagement.joblib')

# Trending videos like/dislike engagement by category models
us_cat_ld_engagement = joblib.load(filepath + 'us_cat_ld_engagement.joblib')
ca_cat_ld_engagement = joblib.load(filepath + 'ca_cat_ld_engagement.joblib')
gb_cat_ld_engagement = joblib.load(filepath + 'gb_cat_ld_engagement.joblib')

# Trending videos comment engagement by category models
us_cat_comm_engagement = joblib.load(filepath + 'us_cat_comm_engagement.joblib')
ca_cat_comm_engagement = joblib.load(filepath + 'ca_cat_comm_engagement.joblib')
gb_cat_comm_engagement = joblib.load(filepath + 'gb_cat_comm_engagement.joblib')

# Trending videos' title frequent itemsets models
freq_one_itemsets = joblib.load(filepath + 'itemset1.joblib')
freq_two_itemsets = joblib.load(filepath + 'itemset2.joblib')
freq_three_itemsets = joblib.load(filepath + 'itemset3.joblib')

# Trending videos' title association rules model
assoc_rules = joblib.load(filepath + 'assoc_rules.joblib')

# naive bayes model for predicting category
text_clf = joblib.load('./model/naive_bayes_classifier.joblib')
nb_count_vect = joblib.load('./model/naive_bayes_clf_count_vect.joblib')

# linear regression model for predicting views, likes, comments
linear_reg_dict = joblib.load('./model/linear_reg.joblib')

# sentiment analysis model
sentiment = joblib.load('./model/sentiment_analysis.joblib')
us_data = joblib.load('./model/us_data.joblib')
sentiment_results_dict = joblib.load('./model/sentiment_results_dict.joblib')

# Implement health function
def health():
  try:
    get_cat_count('us')
    get_ld_engagement('ca')
    get_comm_engagement('gb')
    get_cat_ld_engagement('us')
    get_cat_comm_engagement('gb')
    get_freq_1_itemsets()
    get_freq_2_itemsets()
    get_freq_3_itemsets()
    get_assoc_rules()
  except:
    return {"Message": "Service is unhealthy"}, 500
  
  return {"Message": "Service is OK"}

# Implement describe functions
def get_cat_count(country):
  if country == 'us':
    return us_cat_count
  elif country == 'ca':
    return ca_cat_count
  else:
    return gb_cat_count

def get_ld_engagement(country):
  if country == 'us':
    return us_ld_engagement
  elif country == 'ca':
    return ca_ld_engagement
  else:
    return gb_ld_engagement

def get_comm_engagement(country):
  if country == 'us':
    return us_comm_engagement
  elif country == 'ca':
    return ca_comm_engagement
  else:
    return gb_comm_engagement

def get_cat_ld_engagement(country):
  if country == 'us':
    return us_cat_ld_engagement
  elif country == 'ca':
    return ca_cat_ld_engagement
  else:
    return gb_cat_ld_engagement

def get_cat_comm_engagement(country):
  if country == 'us':
    return us_cat_comm_engagement
  elif country == 'ca':
    return ca_cat_comm_engagement
  else:
    return gb_cat_comm_engagement

def get_freq_1_itemsets():
  return freq_one_itemsets

def get_freq_2_itemsets():
  return freq_two_itemsets

def get_freq_3_itemsets():
  return freq_three_itemsets

def get_assoc_rules():
  return assoc_rules

# predicting category_id and tags based on title, description
# and channel_title.
# generate_tag() is a sub function of predict_category_tags()
# returns json_format: category_id and possible tags
def predict_category_tags(title, description, channel_title):
    text = title + description + channel_title
    text = pd.Series(text)
    X_test = nb_count_vect.transform(text)
    X_test = X_test.todense()
    predicted = np.array(text_clf.predict(X_test))[0]
    tags = generate_tag(predicted, text)
    return {"category_id": int(predicted),
            "possible tags": list(tags)}

# this function generates tags based on category_id and a text
# text is combined title, description, channel_title
# returns 5 tags according to the information given
def generate_tag(category_id, text):
    cv_name = './model/category/count_vect'+str(int(category_id)) + '.joblib'
    mir_name = './model/category/mutual_info_reg'+str(int(category_id)) + '.joblib'
    cv = joblib.load(cv_name)
    mir = joblib.load(mir_name)
    res = dict(zip(cv.get_feature_names(), mir))
    possible_words = cv.get_feature_names()
    vectorizer2 = CountVectorizer(stop_words='english')
    text = vectorizer2.fit_transform(text)
    text = vectorizer2.get_feature_names()
    words = {}
    for word in text:
        if word in res.keys():
            words[word] = res[word]
    words = sorted(words.items(), key=lambda x: x[1], reverse=True)[:5]
    tags = [tag[0] for tag in words]
    return tags

# This functin predicts number of views, likes, or comments
# based on country, title, channel_title, tags and 
# feature(likes, views, comments) selected by user.
# returns number of views, likes, or comments
def prediction_engagement(country, title, channel_title, tags, feature):
    text = title + channel_title + tags
    user_text = linear_reg_dict[country]['cv'].transform([text])
    return abs(linear_reg_dict[country][feature].predict(user_text)[0])

# sentiment analyzer based on video_id
# returns sentiment analysis result based on video_id
def sentiment_analyzer(video_id):
    sentiment_results = []
    data = us_data.loc[us_data['video_id'] == video_id]
    if len(data) == 0:
        return "video_id does not exist."
    text = data["tags"] + data["description"] + data["title"]
    text = text.str.cat(sep=', ')
    sentiment_dict = sentiment.polarity_scores(text)
    if sentiment_dict['compound'] >= 0.3:  # 0.05
        sentiment_dict['result'] = 'Positive'
    elif sentiment_dict['compound'] <= -0.3:
        sentiment_dict['result'] = 'Negative'
    else:
        sentiment_dict['result'] = 'Neutral'
    return sentiment_dict

# sentiment analyzer based on country and features(views, likes, comments)
def sentiment_analyzer_feature(country, text_feature):
    percentage = sentiment_results_dict[country][text_feature]
    return {"The percentage of positive sentiment text features is: ": str(round(percentage[0], 2)) + "%",
            "The percentage of negative sentiment text features is: ": str(round(percentage[1], 2)) + "%",
            "The percentage of neutral sentiment text features is: ": str(round(percentage[2], 2)) + "%"}

# get channel information based on channel_name
# gives basic statistics of the input channel_name
def get_channel_info(channel_name):
    found = us_data[us_data['channel_title'].str.contains(channel_name)]
    if len(found) == 0:
        return "this channel_name does not exist."
    channel = us_data.groupby("channel_title").get_group(channel_name)
    most_pop_video = channel.sort_values(
        by='views', ascending=False).iloc[0]['title']
    max_views = channel.sort_values(
        by='views', ascending=False).iloc[0]['views']
    max_likes = channel.sort_values(
        by='likes', ascending=False).iloc[0]['likes']
    return {"channel_name": channel_name,
            "num_contents": int(len(channel)),
            "most_pop_video": most_pop_video,
            "max_view": int(max_views),
            "total_views": int(sum(channel['views'])),
            "avg_views": channel['views'].mean(),
            "max_likes": int(max_likes),
            "total_likes": int(sum(channel['likes'])),
            "avg_likes": channel['likes'].mean(),
            "total_dislikes": int(sum(channel['dislikes'])),
            "avg_dislikes": channel['dislikes'].mean(),
            "total_comments": int(sum(channel['comment_count'])),
            "avg_comments": channel['comment_count'].mean()}

# get top 10 tags based on category_id.
def get_top_10_tags_in_category(category_id):
    cv_name = './model/category/count_vect'+str(int(category_id)) + '.joblib'
    mir_name = './model/category/mutual_info_reg'+str(int(category_id)) + '.joblib'
    cv = joblib.load(cv_name)
    mir = joblib.load(mir_name)
    res = dict(zip(cv.get_feature_names(), mir))
    res = sorted(res.items(), key=lambda x: x[1], reverse=True)[0: 10]
    tags = []
    for tag in res:
        tags.append(tag[0])
    return {"tags": list(tags)}

# Read the API definition for our service from the yaml file
app.add_api("youtube_api.yaml")

# Start the app
if __name__ == "__main__":
    app.run()