# tests/test_predictor.py
"""Unit tests for ClinicalPredictor class."""

import numpy as np
import pytest

from core.model import ClinicalPredictor


class DummyModel:
    """Mock model for testing."""

    def __init__(self, value):
        self.value = value

    def predict(self, X):
        return np.array([self.value] * len(X))


class TestClinicalPredictor:
    """Tests for ClinicalPredictor class."""

    def test_predict_infected_when_threshold_or_above(self):
        """Test diagnosis returns 'infecte' when prediction >= 5."""
        model = DummyModel(5)
        predictor = ClinicalPredictor(model)
        x = np.array([[1.0, 2.0, 3.0]])
        assert predictor.diagnose(x) == "infecte"

    def test_predict_healthy_below_threshold(self):
        """Test diagnosis returns 'Sain' when prediction < 5."""
        model = DummyModel(4)
        predictor = ClinicalPredictor(model)
        x = np.array([[1.0, 2.0, 3.0]])
        assert predictor.diagnose(x) == "Sain"

    def test_predict_infected_at_high_value(self):
        """Test diagnosis for high risk score."""
        model = DummyModel(10)
        predictor = ClinicalPredictor(model)
        x = np.array([[1.0, 2.0, 3.0]])
        assert predictor.diagnose(x) == "infecte"

    def test_predict_healthy_at_zero(self):
        """Test diagnosis for zero risk score."""
        model = DummyModel(0)
        predictor = ClinicalPredictor(model)
        x = np.array([[1.0, 2.0, 3.0]])
        assert predictor.diagnose(x) == "Sain"

    def test_none_model_raises_error(self):
        """Test that None model raises ValueError."""
        with pytest.raises(ValueError, match="pre-trained model is required"):
            ClinicalPredictor(None)

    def test_empty_patient_data_raises_error(self):
        """Test that empty patient data raises ValueError."""
        model = DummyModel(5)
        predictor = ClinicalPredictor(model)
        with pytest.raises(ValueError, match="Patient data cannot be empty"):
            predictor.diagnose(np.array([]))

    def test_none_patient_data_raises_error(self):
        """Test that None patient data raises ValueError."""
        model = DummyModel(5)
        predictor = ClinicalPredictor(model)
        with pytest.raises(ValueError, match="Patient data cannot be empty"):
            predictor.diagnose(None)

    def test_1d_array_reshaping(self):
        """Test that 1D array is reshaped to 2D."""
        model = DummyModel(5)
        predictor = ClinicalPredictor(model)
        x = np.array([1.0, 2.0, 3.0])  # 1D array
        # Should not raise, should reshape internally
        result = predictor.diagnose(x)
        assert result in ["infecte", "Sain"]

