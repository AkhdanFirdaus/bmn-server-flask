import numpy as np
import tensorflow as tf
from transformers import TFBertModel

BERT_NAME = 'indobenchmark/indobert-lite-base-p1'
MODEL_PATH = 'model_3.keras'

class Process():
    def __init__(self):
        self.model = tf.keras.models.load_model(
            MODEL_PATH,
            custom_objects={'TFBertModel': TFBertModel.from_pretrained(BERT_NAME)},
            compile=False
        )
        self.threshold = 0.5

    def rounded_predictions(self, inputs):
        predictions = self.model.predict(inputs)
        return np.where(predictions > self.threshold, 1, 0)

    def measure_severity(self, predictions):
        return {
            'label': 'Parah',
            'index': 0.8,
        }

    def predict(self, inputs):
        example_output = [0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0]
        predictions = example_output
        return {
            'outputs': predictions,
            'severity': self.measure_severity(predictions)
        }

    def summary_predict(self, inputs):
        example_output = [
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            [1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            [1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0],
            [1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0],
        ]
        # predictions = self.model.predict(inputs)
        predictions = example_output
        return {
            'outputs': predictions,
            'severity': self.measure_severity(predictions)
        }