# tests/test_evaluator.py
"""Unit tests for ModelEvaluator class."""

from unittest.mock import MagicMock, patch

import numpy as np

from pipeline.evaluater import ModelEvaluator


class TestModelEvaluator:
    """Tests for ModelEvaluator class."""

    def test_init_default_path(self):
        """Test ModelEvaluator initializes with default model path."""
        evaluator = ModelEvaluator()
        assert evaluator.model_path == 'model.pkl'
        assert evaluator.model is None
        assert evaluator.rmse is None
        assert evaluator.r2 is None
        assert evaluator.accuracy is None

    def test_init_custom_path(self):
        """Test ModelEvaluator accepts custom model path."""
        evaluator = ModelEvaluator(model_path='custom_model.pkl')
        assert evaluator.model_path == 'custom_model.pkl'

    def test_evaluate_produces_metrics(self):
        """Test evaluation produces RMSE, R2, and accuracy."""
        evaluator = ModelEvaluator.__new__(ModelEvaluator)
        evaluator.model_path = 'model.pkl'
        evaluator.model = None
        evaluator.predictor = None
        evaluator.rmse = None
        evaluator.r2 = None
        evaluator.accuracy = None

        # Mock loader
        mock_loader = MagicMock()
        mock_loader.preprocess.return_value = (
            None,
            np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
            None,
            np.array([3, 6, 9])
        )
        evaluator.loader = mock_loader

        # Mock model
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([3.1, 5.9, 9.1])

        with patch('joblib.load', return_value=mock_model):
            evaluator.evaluate()

        assert evaluator.rmse is not None
        assert evaluator.r2 is not None
        assert evaluator.accuracy is not None

    def test_get_metrics_structure(self):
        """Test get_metrics returns correct dictionary structure."""
        evaluator = ModelEvaluator()
        evaluator.rmse = 1.5
        evaluator.r2 = 0.85
        evaluator.accuracy = 0.9

        metrics = evaluator.get_metrics()

        assert 'rmse' in metrics
        assert 'r2' in metrics
        assert 'diagnosis_accuracy' in metrics
        assert metrics['rmse'] == 1.5
        assert metrics['r2'] == 0.85
        assert metrics['diagnosis_accuracy'] == 0.9

    def test_get_metrics_returns_none_before_evaluate(self):
        """Test that metrics are None before evaluation."""
        evaluator = ModelEvaluator()
        metrics = evaluator.get_metrics()

        assert metrics['rmse'] is None
        assert metrics['r2'] is None
        assert metrics['diagnosis_accuracy'] is None
