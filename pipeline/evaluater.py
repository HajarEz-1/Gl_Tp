import joblib
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

from core.dataset import DatasetLoader
from core.model import ClinicalPredictor


class ModelEvaluator:
    
    def __init__(self, model_path='model.pkl'):
        self.model_path = model_path
        self.model = None
        self.predictor = None
        self.loader = DatasetLoader()
        self.rmse = None
        self.r2 = None
        self.accuracy = None
    
    def evaluate(self):
        _, X_test, _, y_test = self.loader.preprocess()
        
        self.model = joblib.load(self.model_path)
        predictions = self.model.predict(X_test)
        
        self.rmse = np.sqrt(mean_squared_error(y_test, predictions))
        self.r2 = r2_score(y_test, predictions)
        
        self.predictor = ClinicalPredictor(self.model)
        true_diagnoses = ["infecte" if y >= 5 else "Sain" for y in y_test]
        pred_diagnoses = [self.predictor.diagnose(x.reshape(1, -1)) for x in X_test]
        self.accuracy = np.mean(np.array(true_diagnoses) == np.array(pred_diagnoses))
        
        print(f"RMSE: {self.rmse:.2f}")
        print(f"R²: {self.r2:.2f}")
        print(f"Diagnosis Accuracy: {self.accuracy:.2f}")
    
    def get_metrics(self):
        return {
            'rmse': self.rmse,
            'r2': self.r2,
            'diagnosis_accuracy': self.accuracy
        }