# python time_association_api.py to run in terminal
import connexion
from sklearn.externals import joblib

# Instantiate Flask app object
app = connexion.FlaskApp(__name__, port=8080, specification_dir="swagger/")
application = app.app

# Load pre-trained model
filepath = './model/'

# Categories and counts
us_cat_count = joblib.load(filepath + 'us_cat_count.joblib')
ca_cat_count = joblib.load(filepath + 'ca_cat_count.joblib')
gb_cat_count = joblib.load(filepath + 'gb_cat_count.joblib')

# Overall trending videos like/dislike engagement (over 10%)
us_ld_engagement = joblib.load(filepath + 'us_ld_engagement.joblib')
ca_ld_engagement = joblib.load(filepath + 'ca_ld_engagement.joblib')
gb_ld_engagement = joblib.load(filepath + 'gb_ld_engagement.joblib')

# Overall trending videos comment engagement (over 2%)
us_comm_engagement = joblib.load(filepath + 'us_comm_engagement.joblib')
ca_comm_engagement = joblib.load(filepath + 'ca_comm_engagement.joblib')
gb_comm_engagement = joblib.load(filepath + 'gb_comm_engagement.joblib')

# Trending videos like/dislike engagement by category
us_cat_ld_engagement = joblib.load(filepath + 'us_cat_ld_engagement.joblib')
ca_cat_ld_engagement = joblib.load(filepath + 'ca_cat_ld_engagement.joblib')
gb_cat_ld_engagement = joblib.load(filepath + 'gb_cat_ld_engagement.joblib')

# Trending videos comment engagement by category
us_cat_comm_engagement = joblib.load(filepath + 'us_cat_comm_engagement.joblib')
ca_cat_comm_engagement = joblib.load(filepath + 'ca_cat_comm_engagement.joblib')
gb_cat_comm_engagement = joblib.load(filepath + 'gb_cat_comm_engagement.joblib')

# Trending videos' title frequent itemsets
freq_one_itemsets = joblib.load(filepath + 'itemset1.joblib')
freq_two_itemsets = joblib.load(filepath + 'itemset2.joblib')
freq_three_itemsets = joblib.load(filepath + 'itemset3.joblib')

# Trending videos' title association rules
assoc_rules = joblib.load(filepath + 'assoc_rules.joblib')

# Implement functions
def get_cat_count(country):
  if country == us:
    counts = us_cat_count
  elif country == ca:
    counts = ca_cat_count
  else:
    counts = gb_cat_count
  return {"category counts": counts}

def get_freq_1_itemsets():
  return freq_one_itemsets

def get_freq_2_itemsets():
  return freq_two_itemsets

def get_freq_3_itemsets():
  return freq_three_itemsets

def get_assoc_rules():
  return assoc_rules

# Read API definition for service from yaml file
app.add_api("time_association_api.yaml")

# Start app
if __name__ == "__main__":
  app.run()