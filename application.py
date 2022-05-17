from flask import Flask, render_template, url_for, request
from pymongo import MongoClient
import joblib
from sklearn.feature_extraction.text import TfidfTransformer

application = Flask(__name__)


@application.route('/')
def home():
    return render_template('index.html')


@application.route('/predict', methods=['POST'])
def predict():
    vectorizer = open('vectorizer.pkl','rb')
    cv = joblib.load(vectorizer)
    _model = open('model.pkl','rb')
    clf = joblib.load(_model)

    if request.method == 'POST':
        message = request.form['message']
        data = [message]
        tf_transformer = TfidfTransformer()
        vect = tf_transformer.fit_transform(data).toarray()
        my_prediction = clf.predict(vect)

    return render_template('result.html', prediction=my_prediction)




if __name__ == '__main__':
    # app.run(debug=True)
    application.run(host='0.0.0.0', port=80)
