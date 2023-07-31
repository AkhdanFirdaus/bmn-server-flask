import numpy as np
from flask import Flask, request, jsonify
import tensorflow as tf
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

loaded_model = tf.keras.models.load_model(
    './model/klasifikasi2.h5',
    custom_objects={'TFBertModel': TFBertModel},
    compile=False
)
process = Process(loaded_model)

@app.route('/hello', methods=['GET'])
def hello():
    return jsonify({'message': 'Hello World!'})

@app.route('/predict', methods=['POST'])
def predict():
    body_request = request.get_json()
    inputs = body_request.get('inputs')
    tokenized = preprocess.preprocessing(inputs)
    predictions = process.rounded_predictions(tokenized)
    return jsonify({'data': predictions})

if __name__ == '__main__':
    app.run(debug=True)
