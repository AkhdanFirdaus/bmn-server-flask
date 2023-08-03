import numpy as np
import tensorflow as tf
from transformers import TFBertModel
from preprocess import Preprocess

BERT_NAME = 'indobenchmark/indobert-lite-base-p1'
MODEL_PATH = 'model_3.keras'

class Process():
    def __init__(self):
        self.model = tf.keras.models.load_model(
            MODEL_PATH,
            custom_objects={'TFBertModel': TFBertModel.from_pretrained(BERT_NAME)},
            compile=False
        )
        self.preprocess = Preprocess()
        self.threshold = 0.5

    def rounded_predictions(self, inputs):
        tokenized = self.preprocess.preprocessing(inputs)
        predictions = self.model.predict(tokenized)
        return np.where(predictions > self.threshold, 1, 0).tolist()

    def predict(self, labels, inputs):
        predictions = self.rounded_predictions(inputs)
        
        outputs = []

        for i in range(len(predictions)):
            prediction = predictions[i]
            total_bobot = 0
            total_masalah = 0
            label_detected = []

            for j in range(len(prediction)):
                if prediction[j] == 1:
                    total_bobot += labels[j]['bobot']
                    total_masalah += 1
                    label_detected.append(labels[j]['label'])

            output = {
                'laporan': inputs[i],
                'label': label_detected,
                'prediction': prediction,
                'severity': total_bobot / total_masalah,
                'accuracy': 0
            }
            outputs.append(output)

        return {
            'outputs': outputs,
            'sum_severity': sum([output['severity'] for output in outputs]) / len(outputs)
        }