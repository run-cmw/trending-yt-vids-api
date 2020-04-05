# python time_association_geo_api.py to run in terminal
import connexion
from sklearn.externals import joblib

# Instantiate Flask app object
app = connexion.FlaskApp(__name__, port=8080, specification_dif="swagger/")
application = app.app

# Load pre-trained model

# Implement functions

# Read API definition for service from yaml file
app.add_api("time_association_geo_api.yaml")

# Start app
if __name__ == "__main__":
  app.run()