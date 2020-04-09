# python time_association_api.py to run in terminal
import connexion
from sklearn.externals import joblib

# Instantiate Flask app object
app = connexion.FlaskApp(__name__, port=8080, specification_dir="swagger/")
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

# Read API definition for service from yaml file
app.add_api("time_association_api.yaml")

# Start app
if __name__ == "__main__":
  app.run()