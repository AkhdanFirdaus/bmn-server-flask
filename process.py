import numpy as np
import tensorflow as tf
from transformers import TFBertModel

model_path = 'klasifikasi.h5'

class Process():
    def __init__(self):
        self.model = tf.keras.models.load_model(
            model_path,
            custom_objects={'TFBertModel': TFBertModel},
            compile=False
        )
        self.threshold = 0.5

    def rounded_predictions(self, inputs):
        predictions = self.model.predict(inputs)
        return np.where(predictions > self.threshold, 1, 0)

    def measure_severity(self, inputs, labels):
        return {}

    def predict(self, inputs):
        predictions = self.model.predict(inputs)
        return predictions