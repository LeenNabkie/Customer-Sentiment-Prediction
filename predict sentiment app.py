from flask import Flask, render_template, request
from sklearn.externals import joblib
import os
import pandas as pd
import numpy as np

app = Flask(__name__, static_url_path='/static/')


@app.route('/')
def form():
    return render_template('index.html')


@app.route('/predict_sentiment', methods=['POST', 'GET'])
def predict_sentiment():
    # get the parameters
    women_comment = str(request.form['comment'])
    women_age = int(request.form['age'])
    clothes_type = int(request.form['clothes_type'])

    # load the X_columns file
    X_columns = joblib.load('model/X_columns.joblib')
    print(X_columns)

    # generate a dataframe with zeros
    df_prediction = pd.DataFrame(np.zeros((1, len(X_columns))), columns=X_columns)
    print(df_prediction)

    # change the dataframe according to the inputs
    df_prediction.at[0, 'comment_length'] = len(women_comment)
    df_prediction.at[0, 'Age'] = women_age
    df_prediction.at[0, 'Type_'+str(clothes_type)] = 1.0
    print(df_prediction)

    # load the model and predict
    model = joblib.load('model/model.joblib')
    prediction = model.predict(df_prediction.head(1).values)
    predicted_sentiment = prediction.round(1)[0]

    if clothes_type == 1:
      clothes_type = 'dress'
    elif clothes_type == 2:
      clothes_type = 'Pants'
    elif clothes_type == 3:
      clothes_type = 'Blouses'
    elif clothes_type == 4:
      clothes_type = 'shirt'
    elif clothes_type == 5:
      clothes_type = 'skirt'


    if predicted_sentiment == 0:
      predicted_sentiment = 'very positive'
    elif predicted_sentiment == 1:
      predicted_sentiment = 'Positive'
    elif predicted_sentiment == 2:
      predicted_time = 'Neutral'
    elif predicted_sentiment == 3:
      predicted_time = 'Negative'
    elif predicted_sentiment == 4:
      predicted_time = 'Very negative'

    return render_template('results.html',
                           women_comment=str(women_comment),
                           women_age=int(women_age),
                           clothes_type=str(clothes_type),
                           predicted_sentiment=str(predicted_sentiment)
                           )


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)

