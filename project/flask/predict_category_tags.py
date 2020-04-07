"""
To run this app, in your terminal:
> python predict_category_tags.py
"""
import connexion
import pandas as pd
import numpy as np
from sklearn.externals import joblib
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import sklearn.feature_selection as skfs

# Instantiate our Flask app object
app = connexion.FlaskApp(__name__, port=8080, specification_dir='swagger/')
application = app.app
text_clf = joblib.load('./model/naive_bayes_classifier.joblib')
count_vect = joblib.load('./model/naive_bayes_clf_count_vect.joblib')


def predict(title, description, channel_title):
    text = title + description + channel_title
    text = pd.Series(text)
    X_test = count_vect.transform(text)
    X_test = X_test.todense()
    predicted = np.array(text_clf.predict(X_test))[0]
    print(predicted)
    tags = generate_tag(predicted, text)
    print(tags)
    print("8")
    return {"category_id": int(predicted),
            "possible tags": list(tags)}


def generate_tag(category_id, text):
    us_data = pd.read_csv('~/Desktop/data-miners/project/data/USvideos.csv')
    vectorizer = CountVectorizer()
    # us_data['text'] = us_data[['title', 'channel_title',
    #                            'tags', 'description']].astype(str).agg(''.join, axis=1)
    print("1")
    data = us_data['tags'][us_data['category_id'] == category_id]
    if len(data) == 0:
        return "This category_id does not exist."
    X = vectorizer.fit_transform(data)
    print("2")
    Y = us_data['views'][us_data['category_id'] == category_id]
    res = dict(zip(vectorizer.get_feature_names(),
                   skfs.mutual_info_regression(X, Y, discrete_features=True)
                   ))
    print("3")
    possible_words = vectorizer.get_feature_names()
    vectorizer2 = CountVectorizer()
    print("4")
    text = vectorizer2.fit_transform(text)
    text = vectorizer2.get_feature_names()
    words = {}
    for word in text:
        if word in res.keys():
            words[word] = res[word]
    print("5")
    print("6")
    words = sorted(words.items(), key=lambda x: x[1], reverse=True)[:5]
    tags = []
    for tag in words:
        tags.append(tag[0])
    print(tags)
    print("7")
    return tags


# Read the API definition for our service from the yaml file
app.add_api("predict_category_tags.yaml")

# Start the app
if __name__ == "__main__":
    app.run()
