import os
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

os.chdir(project_root)

from core.dataset import DatasetLoader
from core.model import ClinicalPredictor
from pipeline.evaluater import ModelEvaluator
from pipeline.trainer import ModelTrainer


def main():
    
    print("  - Initializing trainer...")
    trainer = ModelTrainer(model_path=str(project_root / 'model.pkl'))
    print("  - Retrieving model (trains if needed)...")
    model = trainer.get_model()
    print("  ✓ Model ready!")
    
    
    print("  - Initializing evaluator...")
    evaluator = ModelEvaluator(model_path=str(project_root / 'model.pkl'))
    print("  - Running evaluation...")
    evaluator.evaluate()
    print("  - Retrieving metrics...")
    metrics = evaluator.get_metrics()
    print(f"  ✓ Evaluation complete. Metrics: {metrics}")
    
    print("\n[STEP 3: Example Diagnosis]")
    print("  - Loading and preprocessing dataset for tools...")
    loader = DatasetLoader()  # Now works with relative path due to os.chdir
    loader.preprocess()  # Load, preprocess, and fit scaler/encoder
    scaler, encoder = loader.get_preprocessor()
    print("  - Preparing example patient data...")
    # Example patient: age=45, bmi=25.0, smoker='yes'
    patient_raw = [45, 25.0, 'yes']
    patient_features = scaler.transform([[patient_raw[0], patient_raw[1], encoder.transform([patient_raw[2]])[0]]])
    print("  - Initializing predictor...")
    predictor = ClinicalPredictor(model)
    print("  - Performing diagnosis...")
    diagnosis = predictor.diagnose(patient_features)
    print(f"  ✓ Patient (age={patient_raw[0]}, bmi={patient_raw[1]}, smoker={patient_raw[2]}): {diagnosis}")
    
    print("\n=== App Complete - All Steps Successful! ===")

if __name__ == "__main__":
    main()