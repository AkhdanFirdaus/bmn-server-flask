import tensorflow as tf
import numpy as np
from transformers import BertTokenizer
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

BERT_NAME = 'indobenchmark/indobert-lite-base-p1'

class Preprocess():
    def __init__(self):
        self.stemmer = StemmerFactory().create_stemmer()
        self.stopword = StopWordRemoverFactory().create_stop_word_remover()
        self.tokenizer = BertTokenizer.from_pretrained(BERT_NAME)
        self.max_len = 128

    def casefolding(self, val):
        return str(val).lower()

    def stemming(self, val):
        return self.stemmer.stem(str(val))

    def stopwordremove(self, val):
        return self.stopword.remove(str(val))

    def tokenizing(self, val):
        return self.tokenizer.encode_plus(
            val,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_token_type_ids=False,
            return_tensors='tf'
        )

    def preprocessing(self, sentences):
        input_ids, attention_mask = [], []
        for sentence in sentences:
            input = self.casefolding(sentence)
            input = self.stemming(input)
            input = self.stopwordremove(input)
            output = self.tokenizing(input)
            input_ids.append(output['input_ids'])
            attention_mask.append(output['attention_mask'])

        return {
            'input_ids': tf.convert_to_tensor(np.asarray(input_ids).squeeze(), dtype=tf.int32),
            'attention_mask': tf.convert_to_tensor(np.asarray(attention_mask).squeeze(), dtype=tf.int32)
        }

    def preprocess_get_token(self, sentences, display_len=20):
        tokenized = self.preprocessing(sentences)
        return [self.tokenizer.convert_ids_to_tokens(tokenized['input_ids'][i][:display_len]) for i in range(len(sentences))]