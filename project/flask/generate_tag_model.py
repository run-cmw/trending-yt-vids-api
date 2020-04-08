"""
To train this model, in your terminal:
> python generate_tag_model.py
"""

from sklearn.externals import joblib
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import sklearn.feature_selection as skfs

print('Loading the dataset')
us_data = pd.read_csv('~/Desktop/data-miners/project/data/USvideos.csv')
ca_data = pd.read_csv('~/Desktop/data-miners/project/data/CAvideos.csv')
gb_data = pd.read_csv('~/Desktop/data-miners/project/data/GBvideos.csv')
frames = [us_data, ca_data, gb_data]
whole_data = pd.concat(frames)
whole_data.drop_duplicates(keep=False, inplace=True)

print('Training a mutual_info_regression classifier')
for i in us_data['category_id'].unique():
    print(i)
    vectorizer = CountVectorizer()
    data = us_data['tags'][us_data['category_id'] == i]
    X = vectorizer.fit_transform(data)
    Y = us_data['views'][us_data['category_id'] == i]
    mir = skfs.mutual_info_regression(X, Y, discrete_features=True)
    print('Exporting the trained model')
    name_mir = "model/category/mutual_info_reg" + str(i) + ".joblib"
    name_cv = "model/category/count_vect" + str(i) + ".joblib"
    joblib.dump(vectorizer, name_cv)
    joblib.dump(mir, name_mir)


print("done training")
