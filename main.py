from flask import Flask, request, jsonify
from flask_cors import CORS

from process import Process

app = Flask(__name__)
CORS(app)

process = Process()

@app.route('/hello', methods=['GET'])
def hello():
    return jsonify({'message': 'Hello World!'})

@app.route('/predict', methods=['POST'])
def predict():
    params = request.args
    body = request.get_json()

    if params.get('type') == "ids":
        inputs = body.get('inputs')
        predictions = process.rounded_predictions(inputs)
        return jsonify({'data': predictions})
    else:
        labels = body.get('labels')
        inputs = body.get('inputs')
        predictions = process.predict(labels, inputs)
        return jsonify({'data': predictions})

if __name__ == '__main__':
    app.run(debug=False)