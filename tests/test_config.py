"""Tests for configuration module."""

import os
import pytest
from config import Config, DataConfig, HOGConfig, SVMConfig, config
from exceptions import ConfigurationError


class TestDataConfig:
    """Test data configuration."""

    def test_default_values(self):
        """Test default configuration values."""
        cfg = DataConfig()
        assert cfg.img_size == (64, 64)
        assert cfg.test_size == 0.2
        assert cfg.random_state == 42

    def test_valid_extensions(self):
        """Test valid extensions are defined."""
        cfg = DataConfig()
        assert '.png' in cfg.valid_extensions
        assert '.jpg' in cfg.valid_extensions


class TestHOGConfig:
    """Test HOG configuration."""

    def test_feature_length_calculation(self):
        """Test HOG feature length calculation."""
        cfg = HOGConfig()
        # For 64x64 with 8x8 cells and 2x2 blocks
        assert cfg.feature_length == 1764

    def test_custom_feature_length(self):
        """Test feature length with different parameters."""
        cfg = HOGConfig(pixels_per_cell=(16, 16))
        # For 64x64 with 16x16 cells and 2x2 blocks
        # cells = 4x4 = 16, blocks = 3x3 = 9, features = 9*4*9 = 324
        assert cfg.feature_length == 324


class TestConfigValidation:
    """Test configuration validation."""

    def test_valid_config(self):
        """Test validation passes with valid config."""
        cfg = Config()
        cfg.validate()  # Should not raise

    def test_invalid_test_size(self):
        """Test validation fails with invalid test_size."""
        cfg = Config()
        cfg.data.test_size = 1.5
        with pytest.raises(ValueError):
            cfg.validate()

    def test_zero_test_size(self):
        """Test validation fails with zero test_size."""
        cfg = Config()
        cfg.data.test_size = 0
        with pytest.raises(ValueError):
            cfg.validate()

    def test_invalid_cv_folds(self):
        """Test validation fails with insufficient CV folds."""
        cfg = Config()
        cfg.svm.cv_folds = 1
        with pytest.raises(ValueError):
            cfg.validate()


class TestEnvironmentConfig:
    """Test loading config from environment variables."""

    def test_env_override(self, monkeypatch):
        """Test environment variable overrides."""
        monkeypatch.setenv('TEST_SIZE', '0.3')
        monkeypatch.setenv('RANDOM_STATE', '123')

        cfg = Config.from_env()
        assert cfg.data.test_size == 0.3
        assert cfg.data.random_state == 123

    def test_env_img_size(self, monkeypatch):
        """Test IMG_SIZE environment variable."""
        monkeypatch.setenv('IMG_SIZE', '32')
        cfg = Config.from_env()
        assert cfg.data.img_size == (32, 32)
