# core/dataset.py (Updated: Changed default data_path to 'data/patient_data.csv' for consistency)
"""
Module for loading and preprocessing the clinical dataset.
Handles data from data/patient_data.csv, prepares features and target for training.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


class DatasetLoader:
    """
    Class to load and preprocess the patient dataset for clinical prediction.
    Features: age, bmi, smoker (encoded); Target: risk_score.
    """
    
    def __init__(self, data_path='data/patient_data.csv'):
        self.data_path = data_path
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
    
    def load_data(self):
        """Load the CSV data into a DataFrame."""
        self.df = pd.read_csv(self.data_path)
        print(f"Dataset loaded: {self.df.shape[0]} samples, {self.df.shape[1]} features.")
        return self.df
    
    def preprocess(self, test_size=0.2, random_state=42):
        if self.df is None:
            self.load_data()
        
        # Select features: age, bmi, smoker
        features = ['age', 'bmi', 'smoker']
        target = 'risk_score'
        
        # Encode smoker (yes/no -> 1/0)
        self.df['smoker_encoded'] = self.label_encoder.fit_transform(self.df['smoker'])
        X = self.df[features[:-1] + ['smoker_encoded']]  # age, bmi, smoker_encoded
        y = self.df[target]
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X_scaled, y, test_size=test_size, random_state=random_state
        )
        
        print(f"Train set: {self.X_train.shape}, Test set: {self.X_test.shape}")
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def get_preprocessor(self):
        """Return the fitted scaler and encoder for inference."""
        return self.scaler, self.label_encoder