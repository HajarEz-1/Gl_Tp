
import numpy as np

class ClinicalPredictor:

    def __init__(self, model):
        if model is None:
            raise ValueError("A pre-trained model is required.")
        self.model = model
    
    def diagnose(self, patient_data):
        if patient_data is None or len(patient_data) == 0:
            raise ValueError("Patient data cannot be empty.")
        
        if len(patient_data.shape) == 1:
            patient_data = patient_data.reshape(1, -1)
        
        prediction = self.model.predict(patient_data)[0]  
        
        if prediction >= 5:
            return "infecte"
        else:
            return "Sain"
