"""
To train this model, in your terminal:
> python train_naive_bayes_model.py
"""

from sklearn.externals import joblib
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split


print('Loading the US dataset')
us_data = pd.read_csv('~/Desktop/data-miners/project/data/USvideos.csv')
us_data['text'] = us_data[['title', 'channel_title',
                           'tags', 'description']].astype(str).agg(''.join, axis=1)
Y = us_data['category_id']

print('Training a Naive Bayes classifier')
count_vect = CountVectorizer(stop_words='english')
X = count_vect.fit_transform(us_data['text'])
X_train, X_dummy_test, y_train, y_dummy_test = train_test_split(
    X, Y, test_size=0.30)

X_train = X_train.todense()
text_clf = MultinomialNB().fit(X_train, y_train)
print("done training")

print('Exporting the trained model')
joblib.dump(count_vect, 'model/naive_bayes_clf_count_vect.joblib')
joblib.dump(text_clf, 'model/naive_bayes_classifier.joblib')
