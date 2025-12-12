# tests/test_utils.py
"""Unit tests for utility functions."""

import pandas as pd

from utils.preprocess_utils import encode_smoker, select_features


class TestPreprocessUtils:
    """Tests for preprocess_utils module."""

    def test_encode_smoker_yes_no(self):
        """Test encode_smoker encodes yes/no correctly."""
        df = pd.DataFrame({
            'smoker': ['yes', 'no', 'yes', 'no']
        })

        result_df, encoder = encode_smoker(df)

        assert 'smoker_encoded' in result_df.columns
        assert encoder is not None
        # Verify encoding is consistent (0 or 1)
        assert set(result_df['smoker_encoded'].unique()).issubset({0, 1})

    def test_encode_smoker_returns_encoder(self):
        """Test that encode_smoker returns a fitted encoder."""
        df = pd.DataFrame({
            'smoker': ['yes', 'no']
        })

        _, encoder = encode_smoker(df)

        # Encoder should be able to transform new values
        assert hasattr(encoder, 'transform')
        assert hasattr(encoder, 'inverse_transform')

    def test_select_features_default(self):
        """Test select_features with default feature list."""
        df = pd.DataFrame({
            'age': [25, 30, 35],
            'bmi': [22.0, 25.0, 28.0],
            'smoker_encoded': [0, 1, 0],
            'extra_column': ['a', 'b', 'c']
        })

        result = select_features(df)

        assert list(result.columns) == ['age', 'bmi', 'smoker_encoded']
        assert 'extra_column' not in result.columns

    def test_select_features_custom(self):
        """Test select_features with custom feature list."""
        df = pd.DataFrame({
            'age': [25, 30, 35],
            'bmi': [22.0, 25.0, 28.0],
            'smoker_encoded': [0, 1, 0]
        })

        result = select_features(df, features=['age', 'bmi'])

        assert list(result.columns) == ['age', 'bmi']
        assert 'smoker_encoded' not in result.columns

    def test_select_features_preserves_data(self):
        """Test that select_features preserves data values."""
        df = pd.DataFrame({
            'age': [25, 30],
            'bmi': [22.0, 25.0],
            'smoker_encoded': [0, 1]
        })

        result = select_features(df)

        assert result['age'].tolist() == [25, 30]
        assert result['bmi'].tolist() == [22.0, 25.0]
