from flask import Flask, request, jsonify
from flask_cors import CORS

from preprocess import Preprocess
from process import Process

app = Flask(__name__)
CORS(app)

preprocess = Preprocess()
process = Process()

@app.route('/hello', methods=['GET'])
def hello():
    return jsonify({'message': 'Hello World!'})

@app.route('/predict', methods=['POST'])
def predict():
    body = request.get_json()
    inputs = body.get('inputs')
    tokenized = preprocess.preprocessing(inputs)
    predictions = process.rounded_predictions(tokenized)
    print(predictions)
    return jsonify({'data': "OK"})

@app.route('/summary-predict', methods=['POST'])
def summary_predict():
    body = request.get_json()
    inputs = body.get('inputs')
    tokenized = preprocess.preprocessing(inputs)
    predictions = process.summary_predict(tokenized)
    print(predictions)
    return jsonify({'data': "OK"})
    
@app.route('/predict-token', methods=['POST'])
def predict_token():
    body = request.get_json()
    inputs = body.get('inputs')
    tokenized = preprocess.preprocess_get_token(inputs)
    return jsonify({'data': {
        'tokenized': tokenized,
    }})

if __name__ == '__main__':
    app.run(debug=False)