"""
To train this model, in your terminal:
> python load_dataset.py
"""
from sklearn.externals import joblib
import pandas as pd

print("loading dataset")
us_data = pd.read_csv('~/Desktop/data-miners/project/data/USvideos.csv')


print('Exporting the trained model')
joblib.dump(us_data, 'model/us_data.joblib')