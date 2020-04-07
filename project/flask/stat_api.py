"""
To run this app, in your terminal:
> python stat_api.py
"""
import connexion
import pandas as pd
import re
import matplotlib.pyplot as plt
import numpy as np
from sklearn.externals import joblib
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import sklearn.feature_selection as skfs

# Instantiate our Flask app object
app = connexion.FlaskApp(__name__, port=8080, specification_dir='swagger/')
application = app.app


def bring_whole_dataset():
    us_data = pd.read_csv('~/Desktop/data-miners/project/data/USvideos.csv')
    ca_data = pd.read_csv('~/Desktop/data-miners/project/data/CAvideos.csv')
    gb_data = pd.read_csv('~/Desktop/data-miners/project/data/GBvideos.csv')
    frames = [us_data, ca_data, gb_data]
    whole_data = pd.concat(frames)
    whole_data.drop_duplicates(keep=False, inplace=True)
    return whole_data


def get_channel_info(channel_name):
    whole_data = bring_whole_dataset()
    channel = whole_data.groupby("channel_title").get_group(channel_name)
    if len(channel) == 0:
        return "this channel_name does not exist."
    num_content = len(channel)
    most_pop_video = channel.sort_values(
        by='views', ascending=False).iloc[0]['title']
    max_views = channel.sort_values(
        by='views', ascending=False).iloc[0]['views']
    total_views = sum(channel['views'])
    avg_views = channel['views'].mean()
    max_likes = channel.sort_values(
        by='likes', ascending=False).iloc[0]['likes']
    total_likes = sum(channel['likes'])
    avg_likes = channel['likes'].mean()
    total_dislikes = sum(channel['dislikes'])
    avg_dislikes = channel['dislikes'].mean()
    total_comment = sum(channel['comment_count'])
    avg_comment = channel['comment_count'].mean()
    print(num_content)
    print(max_views)
    print(avg_views)
    print(avg_comment)
    return {"channel_name": channel_name,
            "num_contents": int(num_content),
            "most_pop_video": most_pop_video,
            "max_view": int(max_views),
            "total_views": int(total_views),
            "avg_views": avg_views,
            "max_likes": int(max_likes),
            "total_likes": int(total_likes),
            "avg_likes": avg_likes,
            "total_dislikes": int(total_dislikes),
            "avg_dislikes": avg_dislikes,
            "total_comments": int(total_comment),
            "avg_comments": avg_comment}


# Not used
def generate_tags_count_dict(data):
    tags = data['tags']
    tags_count_dict = {}
    for line in tags:
        line = line.lower()
        line = line.split("|")
        for word in line:
            word = re.sub("(\")", "", word)
            if word not in tags_count_dict.keys():
                tags_count_dict[word] = 0
            tags_count_dict[word] = tags_count_dict[word] + 1
    return tags_count_dict


def generate_info_dict(data):
    info = {}
    info['count'] = len(data)
    info['most_pop_video'] = data.sort_values(
        by='views', ascending=False).iloc[0]['video_id']
    info['max_views'] = data.sort_values(
        by='views', ascending=False).iloc[0]['views']
    info['total_views'] = sum(data['views'])
    info['avg_views'] = data['views'].mean()
    info['max_likes'] = data.sort_values(
        by='likes', ascending=False).iloc[0]['likes']
    info['total_likes'] = sum(data['likes'])
    info['avg_likes'] = data['likes'].mean()
    info['total_dislikes'] = sum(data['dislikes'])
    info['avg_dislikes'] = data['dislikes'].mean()
    info['total_comment'] = sum(data['comment_count'])
    info['avg_comment'] = data['comment_count'].mean()
    return info


def generate_category_dicts(data):
    category_tags_dict = {}
    category_video_dict = {}
    category_channel_dict = {}
    category_info_dict = {}
    category_list = set(data['category_id'])
    print(category_list)
    for category_id in category_list:
        cate_data = data.groupby("category_id").get_group(category_id)
        category_info_dict[category_id] = generate_info_dict(cate_data)
        category_video_dict[category_id] = cate_data['video_id'].tolist()
        category_tags_dict[category_id] = generate_tags_count_dict(cate_data)
        category_channel_dict[category_id] = set(cate_data['channel_title'])
        # data = data.groupby("category_id").get_group(category_id)
    return category_info_dict, category_tags_dict, category_video_dict, category_channel_dict


def generate_top_10_tags_in_category(category_id):
    us_data = pd.read_csv('~/Desktop/data-miners/project/data/USvideos.csv')
    vectorizer = CountVectorizer()
    data = us_data['tags'][us_data['category_id'] == category_id]
    if len(data) == 0:
        return "This category_id does not exist."
    X = vectorizer.fit_transform(data)
    print("2")
    Y = us_data['views'][us_data['category_id'] == category_id]
    res = dict(zip(vectorizer.get_feature_names(),
                   skfs.mutual_info_regression(X, Y, discrete_features=True)
                   ))
    res = sorted(res.items(), key=lambda x: x[1], reverse=True)[0:10]
    tags = []
    for tag in res:
        tags.append(tag[0])
    return {"tags": list(tags)}


# Read the API definition for our service from the yaml file
app.add_api("stat_api.yaml")

# Start the app
if __name__ == "__main__":
    app.run()
