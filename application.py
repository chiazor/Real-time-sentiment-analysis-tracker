from flask import Flask, render_template, url_for, request, jsonify
#from pymongo import MongoClient
import joblib
from transformers import TextClassificationPipeline, AutoModelForSequenceClassification

#from pymongo.errors import BulkWriteError


application = Flask(__name__)
# application.config["MONGO_URI"] = "mongodb://localhost:27017/sentiment_db"
# mongodb_client = PyMongo(application)
# db = mongodb_client.db


@application.route('/')
def home():
    return render_template('index.html')


@application.route('/predict', methods=['POST'])
def predict():
    tokenizer = open('tokenizer.pkl','rb')
    tokenizer = joblib.load(tokenizer)
    model = open('model.pkl','rb')
    model = joblib.load(model) #my model is too large for github.
    _model = AutoModelForSequenceClassification.from_pretrained("yosemite/autonlp-imdb-sentiment-analysis-english-470512388")
    pipe = TextClassificationPipeline(model=_model, tokenizer=tokenizer)

    if request.method == 'POST':
        message = request.form['message']
        data = [message]
        m_pipe = pipe(data)
        my_prediction = m_pipe[0]['label']

    return jsonify({'message': data, 'sentiment': m_pipe[0]['label'], 'score': m_pipe[0]['score']})

# @application.route('/query')
# def querydb():
#     keywords = db.sentiment.find()
#     return flask




if __name__ == '__main__':
    #application.run(debug=True)
    application.run(host='0.0.0.0', port=80)
