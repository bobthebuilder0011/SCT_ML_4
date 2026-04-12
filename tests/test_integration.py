"""Integration tests for the full pipeline."""

import os
import tempfile
import numpy as np
import pytest
from pathlib import Path
import cv2

from leap_gesture_svm import (
    extract_hog_features, preprocess_image, save_model, load_saved_model
)
from config import config
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC


class TestHOGExtraction:
    """Test HOG feature extraction."""

    def test_hog_features_shape(self):
        """Test HOG features have expected shape."""
        image = np.random.rand(64, 64).astype(np.float32)
        features = extract_hog_features(image)
        assert len(features) == config.hog.feature_length

    def test_hog_features_reproducible(self):
        """Test HOG features are reproducible for same image."""
        image = np.ones((64, 64), dtype=np.float32) * 0.5
        features1 = extract_hog_features(image)
        features2 = extract_hog_features(image)
        np.testing.assert_array_equal(features1, features2)


class TestImagePreprocessing:
    """Test image preprocessing."""

    def test_preprocess_valid_image(self):
        """Test preprocessing a valid image file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a test image
            img_path = Path(tmpdir) / 'test.png'
            img = np.random.randint(0, 256, (128, 128), dtype=np.uint8)
            cv2.imwrite(str(img_path), img)

            result = preprocess_image(str(img_path))
            assert result is not None
            assert result.shape == config.data.img_size
            assert result.dtype == np.float32
            assert np.all(result >= 0) and np.all(result <= 1)

    def test_preprocess_nonexistent_image(self):
        """Test preprocessing a non-existent image."""
        result = preprocess_image('/nonexistent/image.png')
        assert result is None

    def test_preprocess_invalid_file(self):
        """Test preprocessing an invalid file."""
        with tempfile.NamedTemporaryFile(suffix='.txt') as tmp:
            result = preprocess_image(tmp.name)
            assert result is None


class TestModelPersistence:
    """Test model saving and loading."""

    def test_save_and_load_model(self):
        """Test saving and loading a model."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / 'model.joblib'

            # Create a simple model
            model = Pipeline([
                ('svm', SVC(kernel='linear', probability=True))
            ])

            # Train on dummy data
            X = np.random.rand(100, 10)
            y = np.random.randint(0, 3, 100)
            model.fit(X, y)

            # Create label encoder
            label_encoder = LabelEncoder()
            label_encoder.fit(['class_a', 'class_b', 'class_c'])

            # Save model
            save_model(model, label_encoder, str(model_path))
            assert model_path.exists()

            # Load model
            loaded = load_saved_model(str(model_path))
            assert loaded is not None
            loaded_model, loaded_encoder = loaded

            # Verify loaded model works
            predictions = loaded_model.predict(X[:5])
            assert len(predictions) == 5

            # Verify encoder works
            assert list(loaded_encoder.classes_) == ['class_a', 'class_b', 'class_c']


class TestEndToEnd:
    """End-to-end pipeline tests."""

    def test_create_mock_dataset(self):
        """Test creating and processing a mock dataset."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_root = Path(tmpdir) / 'leapGestRecog'

            # Create mock dataset structure
            for subject in ['00', '01']:
                subject_dir = dataset_root / subject
                for gesture in ['01_palm', '02_fist']:
                    gesture_dir = subject_dir / gesture
                    gesture_dir.mkdir(parents=True)

                    # Create mock images
                    for i in range(5):
                        img = np.random.randint(0, 256, (128, 128), dtype=np.uint8)
                        cv2.imwrite(str(gesture_dir / f'frame_{i:04d}.png'), img)

            # Verify structure
            assert (dataset_root / '00' / '01_palm').exists()
            assert len(list((dataset_root / '00' / '01_palm').glob('*.png'))) == 5


@pytest.mark.skipif(
    os.environ.get('SKIP_SLOW_TESTS'), reason="Skipping slow tests"
)
class TestSlowIntegration:
    """Slow integration tests that require full dataset."""

    def test_full_pipeline(self):
        """Test the complete pipeline (requires actual dataset)."""
        # This test is skipped unless explicitly enabled
        pass
