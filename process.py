class Process():
    def __init__(self, model):
        self.model = model
        self.threshold = 0.5

    def rounded_predictions(self, inputs):
        predictions = self.model.predict(inputs)
        return np.where(predictions > self.threshold, 1, 0)

    def measure_severity(self, inputs, labels):
        return {}

    def predict(self, inputs, labels):
        return {
            'data': {
                'severity': NULL,
                'results': [],
            }
        }