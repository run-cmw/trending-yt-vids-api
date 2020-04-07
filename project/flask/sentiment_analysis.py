"""
To run this app, in your terminal:
> python sentiment_analysis.py
"""
import connexion
import pandas as pd
import sklearn.feature_selection as skfs
from nltk.sentiment.vader import SentimentIntensityAnalyzer as SIA

app = connexion.FlaskApp(__name__, port=8080, specification_dir='swagger/')
application = app.app


def sentiment_analyzer(video_id):
    us_data = pd.read_csv('~/Desktop/data-miners/project/data/USvideos.csv')
    sentiment_analysis = SIA()
    sentiment_results = []
    data = us_data.loc[us_data['video_id'] == video_id]
    if len(data) == 0:
        return "video_id does not exist."
    text = data["tags"] + data["description"] + data["title"]
    text = text.str.cat(sep=', ')
    sentiment = SIA()
    sentiment_dict = sentiment.polarity_scores(text)
    if sentiment_dict['compound'] >= 0.05:
        sentiment_dict['result'] = 'Positive'
    elif sentiment_dict['compound'] <= - 0.05:
        sentiment_dict['result'] = 'Negative'
    else:
        sentiment_dict['result'] = 'Neutral'
    return sentiment_dict


# Read the API definition for our service from the yaml file
app.add_api("sentiment_analysis.yaml")

# Start the app
if __name__ == "__main__":
    app.run()
