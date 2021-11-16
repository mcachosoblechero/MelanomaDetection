import tensorflow as tf
import keras


class MelanomaClassifier():
    def __init__(self, model_file):
        self.model = keras.models.load_model(model_file)

    def predict(self, x):
        return self.model.predict(x)
