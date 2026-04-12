"""
Data validation utilities for the Leap Gesture Recognition project.
Ensures data integrity before processing.
"""

import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
from collections import Counter

from exceptions import (
    DatasetError, EmptyDatasetError, InsufficientSamplesError,
    CorruptImageError
)
from logger import logger


class DatasetValidator:
    """Validates dataset structure and contents."""

    @staticmethod
    def validate_dataset_path(path: str) -> str:
        """
        Validate that a dataset path exists and is accessible.

        Args:
            path: Path to validate

        Returns:
            Validated absolute path

        Raises:
            DatasetError: If path doesn't exist or isn't accessible
        """
        if not path:
            raise DatasetError("Dataset path is empty")

        abs_path = Path(path).resolve()

        if not abs_path.exists():
            raise DatasetError(f"Dataset path does not exist: {abs_path}")

        if not abs_path.is_dir():
            raise DatasetError(f"Dataset path is not a directory: {abs_path}")

        if not os.access(abs_path, os.R_OK):
            raise DatasetError(f"Dataset path is not readable: {abs_path}")

        logger.debug(f"Validated dataset path: {abs_path}")
        return str(abs_path)

    @staticmethod
    def validate_image_file(filepath: str, valid_extensions: Tuple[str, ...]) -> bool:
        """
        Validate that a file is a valid image.

        Args:
            filepath: Path to file
            valid_extensions: Tuple of valid extensions

        Returns:
            True if valid image file
        """
        path = Path(filepath)

        # Check extension
        if path.suffix.lower() not in valid_extensions:
            return False

        # Check it's not a hidden file
        if path.name.startswith('.'):
            return False

        # Check file exists and is readable
        if not path.exists():
            return False

        if not os.access(path, os.R_OK):
            return False

        # Check file size (skip empty files)
        if path.stat().st_size == 0:
            logger.warning(f"Empty file skipped: {filepath}")
            return False

        return True

    @staticmethod
    def validate_class_distribution(
        labels: List[str],
        min_samples: int = 10
    ) -> Dict[str, int]:
        """
        Validate that each class has sufficient samples.

        Args:
            labels: List of class labels
            min_samples: Minimum samples required per class

        Returns:
            Dictionary of class counts

        Raises:
            EmptyDatasetError: If no labels provided
            InsufficientSamplesError: If any class has too few samples
        """
        if not labels:
            raise EmptyDatasetError("No labels provided")

        counts = Counter(labels)

        logger.info(f"Found {len(counts)} classes with distribution:")
        for class_name, count in sorted(counts.items()):
            logger.info(f"  {class_name}: {count} samples")

            if count < min_samples:
                raise InsufficientSamplesError(class_name, count, min_samples)

        return dict(counts)

    @staticmethod
    def validate_features(
        features: np.ndarray,
        expected_length: Optional[int] = None
    ) -> np.ndarray:
        """
        Validate feature array integrity.

        Args:
            features: Feature array to validate
            expected_length: Expected feature vector length

        Returns:
            Validated features

        Raises:
            DatasetError: If features are invalid
        """
        if features is None:
            raise DatasetError("Features array is None")

        if len(features) == 0:
            raise EmptyDatasetError("Features array is empty")

        if np.any(np.isnan(features)):
            nan_count = np.sum(np.isnan(features))
            logger.warning(f"Found {nan_count} NaN values in features, replacing with 0")
            features = np.nan_to_num(features, nan=0.0)

        if np.any(np.isinf(features)):
            inf_count = np.sum(np.isinf(features))
            logger.warning(f"Found {inf_count} Inf values in features, replacing with 0")
            features = np.nan_to_num(features, posinf=0.0, neginf=0.0)

        if expected_length and features.shape[1] != expected_length:
            raise DatasetError(
                f"Feature length mismatch: expected {expected_length}, "
                f"got {features.shape[1]}"
            )

        logger.debug(f"Validated features shape: {features.shape}")
        return features


class ModelValidator:
    """Validates model and training configuration."""

    @staticmethod
    def validate_train_test_split(
        X: np.ndarray,
        y: np.ndarray,
        test_size: float
    ) -> None:
        """
        Validate train/test split parameters.

        Args:
            X: Feature array
            y: Label array
            test_size: Test size ratio

        Raises:
            ValueError: If parameters are invalid
        """
        if len(X) != len(y):
            raise ValueError(
                f"X and y have different lengths: {len(X)} vs {len(y)}"
            )

        n_samples = len(X)
        n_test = int(n_samples * test_size)
        n_train = n_samples - n_test

        if n_train < 10:
            raise ValueError(
                f"Too few training samples: {n_train}. "
                f"Need at least 10, have {n_samples} total."
            )

        if n_test < 2:
            raise ValueError(
                f"Too few test samples: {n_test}. "
                f"Increase total samples or decrease test_size."
            )

        logger.debug(f"Train/test split validated: {n_train} train, {n_test} test")

    @staticmethod
    def validate_saved_model(model_path: str) -> bool:
        """
        Check if a saved model exists and is readable.

        Args:
            model_path: Path to model file

        Returns:
            True if model exists and is readable
        """
        path = Path(model_path)

        if not path.exists():
            logger.debug(f"Model file not found: {model_path}")
            return False

        if not path.is_file():
            logger.warning(f"Model path is not a file: {model_path}")
            return False

        if not os.access(path, os.R_OK):
            logger.warning(f"Model file not readable: {model_path}")
            return False

        if path.stat().st_size == 0:
            logger.warning(f"Model file is empty: {model_path}")
            return False

        logger.debug(f"Model file validated: {model_path}")
        return True
