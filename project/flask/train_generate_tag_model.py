"""
To train this model, in your terminal:
> python train_generate_tag_model.py
"""

from sklearn.externals import joblib
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import sklearn.feature_selection as skfs

print('Loading the dataset')

us_data = pd.read_csv('~/Desktop/data-miners/project/data/USvideos.csv')

print('Training a mutual_info_regression classifier')
for i in us_data['category_id'].unique():
    print(i)
    vectorizer = CountVectorizer(stop_words='english')
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
