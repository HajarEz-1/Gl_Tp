# tests/test_trainer.py
"""Unit tests for ModelTrainer class."""

import os
import tempfile
from unittest.mock import MagicMock, patch

import numpy as np

from pipeline.trainer import ModelTrainer


class TestModelTrainer:
    """Tests for ModelTrainer class."""

    def test_init_default_path(self):
        """Test ModelTrainer initializes with default model path."""
        trainer = ModelTrainer()
        assert trainer.model_path == 'model.pkl'
        assert trainer.trained is False

    def test_init_custom_path(self):
        """Test ModelTrainer accepts custom model path."""
        trainer = ModelTrainer(model_path='custom_model.pkl')
        assert trainer.model_path == 'custom_model.pkl'

    @patch('pipeline.trainer.DatasetLoader')
    @patch('joblib.dump')
    def test_train_saves_model(self, mock_dump, mock_loader_class):
        """Test that training saves a model file."""
        # Mock dataset loader
        mock_loader = MagicMock()
        mock_loader.preprocess.return_value = (
            np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
            np.array([[10, 11, 12]]),
            np.array([1, 2, 3]),
            np.array([4])
        )
        mock_loader_class.return_value = mock_loader

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, 'test_model.pkl')
            trainer = ModelTrainer(model_path=model_path)
            model = trainer.train()

            assert trainer.trained is True
            assert model is not None
            mock_dump.assert_called_once()

    @patch('pipeline.trainer.DatasetLoader')
    def test_train_returns_model(self, mock_loader_class):
        """Test that train() returns the trained model."""
        mock_loader = MagicMock()
        mock_loader.preprocess.return_value = (
            np.array([[1, 2, 3], [4, 5, 6]]),
            np.array([[7, 8, 9]]),
            np.array([1, 2]),
            np.array([3])
        )
        mock_loader_class.return_value = mock_loader

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, 'test_model.pkl')
            trainer = ModelTrainer(model_path=model_path)
            model = trainer.train()

            assert model is not None
            assert hasattr(model, 'predict')

    @patch('joblib.load')
    def test_get_model_loads_existing(self, mock_load):
        """Test get_model() loads existing model."""
        mock_model = MagicMock()
        mock_load.return_value = mock_model

        trainer = ModelTrainer(model_path='existing_model.pkl')
        result = trainer.get_model()

        assert result == mock_model
        assert trainer.trained is True
        mock_load.assert_called_once_with('existing_model.pkl')

    @patch('joblib.load')
    @patch('pipeline.trainer.DatasetLoader')
    def test_get_model_trains_if_not_found(self, mock_loader_class, mock_load):
        """Test get_model() trains new model if file not found."""
        mock_load.side_effect = FileNotFoundError()

        mock_loader = MagicMock()
        mock_loader.preprocess.return_value = (
            np.array([[1, 2, 3]]),
            np.array([[4, 5, 6]]),
            np.array([1]),
            np.array([2])
        )
        mock_loader_class.return_value = mock_loader

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, 'new_model.pkl')
            trainer = ModelTrainer(model_path=model_path)
            result = trainer.get_model()

            assert result is not None
            assert trainer.trained is True
