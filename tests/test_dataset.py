# tests/test_dataset.py
"""Unit tests for DatasetLoader class."""

from unittest.mock import patch

import pandas as pd
import pytest

from core.dataset import DatasetLoader


class TestDatasetLoader:
    """Tests for DatasetLoader class."""

    def test_init_default_path(self):
        """Test that DatasetLoader initializes with default path."""
        loader = DatasetLoader()
        assert loader.data_path == 'data/patient_data.csv'
        assert loader.df is None

    def test_init_custom_path(self):
        """Test that DatasetLoader accepts custom path."""
        loader = DatasetLoader(data_path='custom/path.csv')
        assert loader.data_path == 'custom/path.csv'

    @patch('pandas.read_csv')
    def test_load_data_success(self, mock_read_csv):
        """Test successful CSV loading."""
        mock_df = pd.DataFrame({
            'age': [25, 30, 35],
            'bmi': [22.0, 25.0, 28.0],
            'smoker': ['no', 'yes', 'no'],
            'risk_score': [3, 7, 4]
        })
        mock_read_csv.return_value = mock_df

        loader = DatasetLoader()
        loader.load_data()

        assert loader.df is not None
        assert len(loader.df) == 3
        mock_read_csv.assert_called_once_with('data/patient_data.csv')

    def test_load_data_file_not_found(self):
        """Test loading with non-existent file raises error."""
        loader = DatasetLoader(data_path='nonexistent.csv')
        with pytest.raises(FileNotFoundError):
            loader.load_data()

    @patch('pandas.read_csv')
    def test_preprocess_returns_correct_shapes(self, mock_read_csv):
        """Test preprocessing returns correctly shaped data."""
        mock_df = pd.DataFrame({
            'age': [25, 30, 35, 40, 45, 50, 55, 60, 65, 70],
            'bmi': [22.0, 25.0, 28.0, 30.0, 24.0, 26.0, 29.0, 31.0, 23.0, 27.0],
            'smoker': ['no', 'yes', 'no', 'yes', 'no', 'yes', 'no', 'yes', 'no', 'yes'],
            'risk_score': [3, 7, 4, 8, 2, 6, 5, 9, 3, 7]
        })
        mock_read_csv.return_value = mock_df

        loader = DatasetLoader()
        X_train, X_test, y_train, y_test = loader.preprocess(test_size=0.2)

        # Check shapes (80% train, 20% test)
        assert X_train.shape[0] == 8
        assert X_test.shape[0] == 2
        assert len(y_train) == 8
        assert len(y_test) == 2

    @patch('pandas.read_csv')
    def test_get_preprocessor(self, mock_read_csv):
        """Test that preprocessor tools are returned after preprocessing."""
        mock_df = pd.DataFrame({
            'age': [25, 30, 35, 40, 45],
            'bmi': [22.0, 25.0, 28.0, 30.0, 24.0],
            'smoker': ['no', 'yes', 'no', 'yes', 'no'],
            'risk_score': [3, 7, 4, 8, 2]
        })
        mock_read_csv.return_value = mock_df

        loader = DatasetLoader()
        loader.preprocess()
        scaler, encoder = loader.get_preprocessor()

        assert scaler is not None
        assert encoder is not None
