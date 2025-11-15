
import joblib
from sklearn.linear_model import LinearRegression
from core.dataset import DatasetLoader

class ModelTrainer:
    
    def __init__(self, model_path='model.pkl'):
        self.model_path = model_path
        self.model = LinearRegression()
        self.trained = False
    
    def train(self):
        loader = DatasetLoader()
        X_train, _, y_train, _ = loader.preprocess()
        
        self.model.fit(X_train, y_train)
        joblib.dump(self.model, self.model_path)
        self.trained = True
        print(f"Model trained and saved to {self.model_path}")
        return self.model
    
    def get_model(self):
        if not self.trained:
            try:
                self.model = joblib.load(self.model_path)
                self.trained = True
            except FileNotFoundError:
                print("No saved model found. Training a new one...")
                self.train()
        return self.model