"""Tests for validator module."""

import os
import tempfile
import numpy as np
import pytest
from pathlib import Path

from validator import DatasetValidator, ModelValidator
from exceptions import (
    DatasetError, EmptyDatasetError, InsufficientSamplesError
)


class TestDatasetValidator:
    """Test dataset validation."""

    def test_validate_existing_path(self):
        """Test validating an existing directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = DatasetValidator.validate_dataset_path(tmpdir)
            assert result == str(Path(tmpdir).resolve())

    def test_validate_nonexistent_path(self):
        """Test validating a non-existent path."""
        with pytest.raises(DatasetError):
            DatasetValidator.validate_dataset_path("/nonexistent/path/12345")

    def test_validate_empty_path(self):
        """Test validating an empty path."""
        with pytest.raises(DatasetError):
            DatasetValidator.validate_dataset_path("")

    def test_validate_file_instead_of_directory(self):
        """Test validating a file instead of directory."""
        with tempfile.NamedTemporaryFile() as tmp:
            with pytest.raises(DatasetError):
                DatasetValidator.validate_dataset_path(tmp.name)

    def test_validate_image_file_valid(self):
        """Test validating a valid image file."""
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            tmp.write(b'dummy image data')
            tmp.close()
            try:
                assert DatasetValidator.validate_image_file(
                    tmp.name, ('.png', '.jpg')
                )
            finally:
                if os.path.exists(tmp.name):
                    os.unlink(tmp.name)

    def test_validate_image_file_invalid_extension(self):
        """Test validating a file with invalid extension."""
        with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as tmp:
            tmp.write(b'dummy data')
            tmp.close()
            try:
                assert not DatasetValidator.validate_image_file(
                    tmp.name, ('.png', '.jpg')
                )
            finally:
                if os.path.exists(tmp.name):
                    os.unlink(tmp.name)

    def test_validate_image_file_hidden(self):
        """Test validating a hidden file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            hidden_file = Path(tmpdir) / '.hidden.png'
            hidden_file.touch()
            assert not DatasetValidator.validate_image_file(
                str(hidden_file), ('.png',)
            )

    def test_validate_class_distribution_valid(self):
        """Test validating sufficient samples per class."""
        labels = ['a'] * 10 + ['b'] * 15 + ['c'] * 20
        result = DatasetValidator.validate_class_distribution(labels, min_samples=5)
        assert result == {'a': 10, 'b': 15, 'c': 20}

    def test_validate_class_distribution_insufficient(self):
        """Test validating with insufficient samples."""
        labels = ['a'] * 2 + ['b'] * 15
        with pytest.raises(InsufficientSamplesError) as exc_info:
            DatasetValidator.validate_class_distribution(labels, min_samples=5)
        assert exc_info.value.class_name == 'a'
        assert exc_info.value.count == 2

    def test_validate_class_distribution_empty(self):
        """Test validating empty labels."""
        with pytest.raises(EmptyDatasetError):
            DatasetValidator.validate_class_distribution([])

    def test_validate_features_valid(self):
        """Test validating valid features."""
        features = np.random.rand(100, 1764)
        result = DatasetValidator.validate_features(features, expected_length=1764)
        assert result.shape == (100, 1764)

    def test_validate_features_with_nans(self):
        """Test validating features with NaN values."""
        features = np.random.rand(100, 1764).astype(np.float32)
        features[0, 0] = np.nan
        result = DatasetValidator.validate_features(features)
        assert not np.any(np.isnan(result))

    def test_validate_features_with_infs(self):
        """Test validating features with Inf values."""
        features = np.random.rand(100, 1764).astype(np.float32)
        features[0, 0] = np.inf
        result = DatasetValidator.validate_features(features)
        assert not np.any(np.isinf(result))

    def test_validate_features_empty(self):
        """Test validating empty features."""
        with pytest.raises(EmptyDatasetError):
            DatasetValidator.validate_features(np.array([]))

    def test_validate_features_wrong_length(self):
        """Test validating features with wrong length."""
        features = np.random.rand(100, 100)
        with pytest.raises(DatasetError):
            DatasetValidator.validate_features(features, expected_length=1764)


class TestModelValidator:
    """Test model validation."""

    def test_validate_train_test_split_valid(self):
        """Test validating valid train/test split."""
        X = np.random.rand(100, 10)
        y = np.random.randint(0, 5, 100)
        ModelValidator.validate_train_test_split(X, y, test_size=0.2)

    def test_validate_train_test_split_mismatched(self):
        """Test validating mismatched X and y."""
        X = np.random.rand(100, 10)
        y = np.random.randint(0, 5, 50)
        with pytest.raises(ValueError):
            ModelValidator.validate_train_test_split(X, y, test_size=0.2)

    def test_validate_train_test_split_too_few_samples(self):
        """Test validating with too few samples."""
        X = np.random.rand(5, 10)
        y = np.random.randint(0, 2, 5)
        with pytest.raises(ValueError):
            ModelValidator.validate_train_test_split(X, y, test_size=0.5)

    def test_validate_saved_model_exists(self):
        """Test validating existing model file."""
        with tempfile.NamedTemporaryFile(suffix='.joblib', delete=False) as tmp:
            tmp.write(b'dummy model data')
            tmp.close()
            try:
                assert ModelValidator.validate_saved_model(tmp.name)
            finally:
                if os.path.exists(tmp.name):
                    os.unlink(tmp.name)

    def test_validate_saved_model_nonexistent(self):
        """Test validating non-existent model file."""
        assert not ModelValidator.validate_saved_model('/nonexistent/model.joblib')

    def test_validate_saved_model_empty(self):
        """Test validating empty model file."""
        with tempfile.NamedTemporaryFile(suffix='.joblib', delete=False) as tmp:
            tmp.close()
            try:
                assert not ModelValidator.validate_saved_model(tmp.name)
            finally:
                if os.path.exists(tmp.name):
                    os.unlink(tmp.name)
