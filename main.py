from flask import Flask, request, jsonify

from preprocess import Preprocess
from process import Process

app = Flask(__name__)

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
    predictions = process.predict(tokenized)
    return jsonify({'data': predictions})
    
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