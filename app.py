{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "from flask import Flask, render_template, request\n",
    "from sklearn.externals import joblib\n",
    "import os\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "\n",
    "app = Flask(__name__, static_url_path='/static/')\n",
    "\n",
    "\n",
    "@app.route('/')\n",
    "def form():\n",
    "    return render_template('index.html')\n",
    "\n",
    "\n",
    "@app.route('/predict_sentiment', methods=['POST', 'GET'])\n",
    "def predict_sentiment():\n",
    "    # get the parameters\n",
    "    women_age = int(request.form['age'])\n",
    "    clothes_type = int(request.form['clothes_type'])\n",
    "\n",
    "    # load the X_columns file\n",
    "    X_columns = joblib.load('model/X_columns.joblib')\n",
    "    print(X_columns)\n",
    "\n",
    "    # generate a dataframe with zeros\n",
    "    df_prediction = pd.DataFrame(np.zeros((1, len(X_columns))), columns=X_columns)\n",
    "    print(df_prediction)\n",
    "\n",
    "    # change the dataframe according to the inputs\n",
    "    df_prediction.at[0, 'Age'] = women_age\n",
    "    df_prediction.at[0, 'Type_'+str(clothes_type)] = 1.0\n",
    "    print(df_prediction)\n",
    "\n",
    "    # load the model and predict\n",
    "    model = joblib.load('model/model.joblib')\n",
    "    prediction = model.predict(df_prediction.head(1).values)\n",
    "    predicted_sentiment = prediction.round(1)[0]\n",
    "\n",
    "    if clothes_type == 1:\n",
    "      clothes_type = 'dress'\n",
    "    elif clothes_type == 2:\n",
    "      clothes_type = 'Pants'\n",
    "    elif clothes_type == 3:\n",
    "      clothes_type = 'Blouses'\n",
    "    elif clothes_type == 4:\n",
    "      clothes_type = 'shirt'\n",
    "    elif clothes_type == 5:\n",
    "      clothes_type = 'skirt'\n",
    "\n",
    "\n",
    "    if predicted_time == 0:\n",
    "      predicted_sentiment = 'very positive'\n",
    "    elif predicted_time == 1:\n",
    "      predicted_sentiment = 'Positive'\n",
    "    elif predicted_time == 2:\n",
    "      predicted_time = 'Neutral'\n",
    "    elif predicted_time == 3:\n",
    "      predicted_time = 'Negative'\n",
    "    elif predicted_time == 4:\n",
    "      predicted_time = 'Very negative'\n",
    "\n",
    "    return render_template('results.html',\n",
    "                           women_age=int(women_age),\n",
    "                           clothes_type=str(clothes_type),\n",
    "                           predicted_sentiment=str(predicted_sentiment)\n",
    "                           )\n",
    "\n",
    "\n",
    "if __name__ == '__main__':\n",
    "    port = int(os.environ.get('PORT', 5000))\n",
    "    app.run(host='0.0.0.0', port=port)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.6.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
