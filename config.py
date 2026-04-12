"""
Configuration management for the Leap Gesture Recognition project.
Centralizes all settings and provides validation.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple


@dataclass
class DataConfig:
    """Data loading and preprocessing configuration."""
    img_size: Tuple[int, int] = (64, 64)
    test_size: float = 0.2
    random_state: int = 42
    valid_extensions: Tuple[str, ...] = field(
        default_factory=lambda: ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif')
    )
    min_samples_per_class: int = 10
    max_samples: Optional[int] = None  # Limit for testing (None = no limit)
    dataset_cache_dir: Optional[str] = None  # Custom cache dir (None = use default kagglehub)
    cleanup_dataset: bool = False  # Delete dataset after training (for temporary storage)


@dataclass
class HOGConfig:
    """HOG feature extraction configuration."""
    orientations: int = 9
    pixels_per_cell: Tuple[int, int] = (8, 8)
    cells_per_block: Tuple[int, int] = (2, 2)
    block_norm: str = 'L2-Hys'
    visualize: bool = False
    feature_vector: bool = True

    @property
    def feature_length(self) -> int:
        """Calculate expected HOG feature vector length."""
        # For 64x64 image: (64/8) * (64/8) * 9 * 4 = 1764
        cells_x = 64 // self.pixels_per_cell[0]
        cells_y = 64 // self.pixels_per_cell[1]
        blocks_x = cells_x - self.cells_per_block[0] + 1
        blocks_y = cells_y - self.cells_per_block[1] + 1
        return (self.orientations *
                self.cells_per_block[0] * self.cells_per_block[1] *
                blocks_x * blocks_y)


@dataclass
class SVMConfig:
    """SVM model configuration."""
    kernel: str = 'rbf'
    C_values: Tuple[float, ...] = (0.1, 1, 10, 100)
    gamma_values: Tuple[str, float, ...] = ('scale', 'auto', 0.001, 0.01)
    cv_folds: int = 3
    n_jobs: int = -1
    probability: bool = True
    tune_hyperparams: bool = True
    cache_size: int = 500  # MB


@dataclass
class OutputConfig:
    """Output and logging configuration."""
    model_path: str = "models/svm_gesture_model.joblib"
    encoder_path: str = "models/label_encoder.joblib"
    log_path: str = "logs/training.log"
    results_dir: str = "results"
    figures_dir: str = "results/figures"
    verbose: int = 1

    def __post_init__(self):
        """Ensure directories exist."""
        Path(self.results_dir).mkdir(parents=True, exist_ok=True)
        Path(self.figures_dir).mkdir(parents=True, exist_ok=True)
        Path(self.model_path).parent.mkdir(parents=True, exist_ok=True)
        Path(self.log_path).parent.mkdir(parents=True, exist_ok=True)


class Config:
    """Main configuration class combining all sub-configs."""

    def __init__(self):
        self.data = DataConfig()
        self.hog = HOGConfig()
        self.svm = SVMConfig()
        self.output = OutputConfig()

    @classmethod
    def from_env(cls) -> 'Config':
        """Load configuration from environment variables."""
        config = cls()

        # Override with environment variables if set
        if os.getenv('IMG_SIZE'):
            size = int(os.getenv('IMG_SIZE'))
            config.data.img_size = (size, size)

        if os.getenv('TEST_SIZE'):
            config.data.test_size = float(os.getenv('TEST_SIZE'))

        if os.getenv('RANDOM_STATE'):
            config.data.random_state = int(os.getenv('RANDOM_STATE'))

        if os.getenv('CV_FOLDS'):
            config.svm.cv_folds = int(os.getenv('CV_FOLDS'))

        if os.getenv('MODEL_PATH'):
            config.output.model_path = os.getenv('MODEL_PATH')

        if os.getenv('TUNE_HYPERPARAMS'):
            config.svm.tune_hyperparams = os.getenv('TUNE_HYPERPARAMS').lower() == 'true'

        if os.getenv('DATASET_CACHE_DIR'):
            config.data.dataset_cache_dir = os.getenv('DATASET_CACHE_DIR')

        if os.getenv('CLEANUP_DATASET'):
            config.data.cleanup_dataset = os.getenv('CLEANUP_DATASET').lower() == 'true'

        return config

    def validate(self) -> None:
        """Validate configuration parameters."""
        errors = []

        # Data validation
        if self.data.test_size <= 0 or self.data.test_size >= 1:
            errors.append(f"TEST_SIZE must be between 0 and 1, got {self.data.test_size}")

        if self.data.img_size[0] <= 0 or self.data.img_size[1] <= 0:
            errors.append(f"IMG_SIZE must be positive, got {self.data.img_size}")

        if self.data.img_size[0] % self.hog.pixels_per_cell[0] != 0:
            errors.append(f"Image width must be divisible by pixels_per_cell")

        # SVM validation
        if self.svm.cv_folds < 2:
            errors.append(f"CV_FOLDS must be at least 2, got {self.svm.cv_folds}")

        if errors:
            raise ValueError("Configuration errors:\n" + "\n".join(errors))


# Global config instance
config = Config.from_env()
