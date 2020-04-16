"""
To train this model, in your terminal:
> python train_linear_reg_model.py
"""

from sklearn.externals import joblib
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

print("loading dataset")
us_data = pd.read_csv('~/Desktop/data-miners/project/data/USvideos.csv')
ca_data = pd.read_csv('~/Desktop/data-miners/project/data/CAvideos.csv')
gb_data = pd.read_csv('~/Desktop/data-miners/project/data/GBvideos.csv')
frames = [us_data, ca_data, gb_data]
whole_data = pd.concat(frames)
whole_data.drop_duplicates(keep=False, inplace=True)

print("training a linear regression model")
linear_reg_dict = {}
countries = {'us': us_data, 'ca': ca_data, 'gb': gb_data, 'all': whole_data}

for country in countries.items():
    linear_reg_dict[country[0]] = {}
    count_vectorizer = CountVectorizer(stop_words='english')
    df_res = country[1].copy(deep=True)
    df_res['title_channel_tags'] = df_res['title'] + \
        df_res['tags'] + df_res['channel_title']
    X = count_vectorizer.fit_transform(df_res['title_channel_tags'])
    linear_reg_dict[country[0]]['cv'] = count_vectorizer
    features = ['likes', 'comment_count', 'views']
    for feature in features:
        Y = df_res[feature]
        X_train, X_test, y_train, y_test = train_test_split(
            X, Y, test_size=0.20)
        regression = LinearRegression()
        regression.fit(X_train, y_train)
        # print("The R squared of the model is: " +
        #       str(regression.score(X_test, y_test)))
        linear_reg_dict[country[0]][feature] = regression

print('Exporting the trained model')
joblib.dump(linear_reg_dict, 'model/linear_reg.joblib')
