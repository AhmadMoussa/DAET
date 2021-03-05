import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
import keras
from sklearn.metrics import classification_report, confusion_matrix, plot_confusion_matrix

class Classifier():
    def __init__(self, model_path):
        self.model = keras.models.load_model(model_path)

    def predict_with_model(self, sample):
        print(sample.shape)
        return self.model.predict(np.array([sample]))




