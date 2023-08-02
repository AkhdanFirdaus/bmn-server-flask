import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify
from keras.models import load_model
from transformers import BertTokenizer, TFBertModel
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

from preprocess import Preprocessing
from process import Process

app = Flask(__name__)

stemmer = StemmerFactory().create_stemmer()
stopword = StopWordRemoverFactory().create_stop_word_remover()
tokenizer = BertTokenizer.from_pretrained('indobenchmark/indobert-base-p1')
preprocess = Preprocessing(stemmer, stopword, tokenizer, 128)

global loaded_model
loaded_model = load_model(
    'klasifikasi.h5',
    custom_objects={'TFBertModel': TFBertModel},
    compile=False
)
process = Process(loaded_model)

@app.route('/hello', methods=['GET'])
def hello():
    return jsonify({'message': 'Hello World!'})

@app.route('/predict', methods=['POST'])
def predict():
    body = request.get_json()
    inputs = body.get('inputs')
    tokenized = preprocess.preprocessing(inputs)
    predictions = process.rounded_predictions(tokenized)
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
