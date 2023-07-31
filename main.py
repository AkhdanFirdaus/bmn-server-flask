import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify
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

@app.route('/hello', methods=['GET'])
def hello():
    return jsonify({'message': 'Hello World!'})

@app.route('/predict', methods=['POST'])
def predict():
    body = request.get_json()
    inputs = body.get('inputs')
    tokenized = preprocess.preprocessing(inputs)
    return jsonify({'data': {
        'input_ids': tokenized['input_ids'].numpy().tolist(),
        'attention_mask': tokenized['attention_mask'].numpy().tolist(),
    }})

if __name__ == '__main__':
    app.run(debug=True)
