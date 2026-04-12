"""
Custom exceptions for the Leap Gesture Recognition project.
Provides detailed error messages and proper exception hierarchy.
"""


class GestureRecognitionError(Exception):
    """Base exception for all gesture recognition errors."""
    pass


class DatasetError(GestureRecognitionError):
    """Raised when dataset operations fail."""
    pass


class DatasetNotFoundError(DatasetError):
    """Raised when the dataset cannot be found or downloaded."""
    pass


class EmptyDatasetError(DatasetError):
    """Raised when no valid images are found in the dataset."""
    pass


class InsufficientSamplesError(DatasetError):
    """Raised when there are too few samples for a class."""
    def __init__(self, class_name: str, count: int, min_required: int):
        super().__init__(
            f"Class '{class_name}' has only {count} samples, "
            f"minimum required: {min_required}"
        )
        self.class_name = class_name
        self.count = count
        self.min_required = min_required


class ImageProcessingError(GestureRecognitionError):
    """Raised when image loading or preprocessing fails."""
    pass


class CorruptImageError(ImageProcessingError):
    """Raised when an image file is corrupt or unreadable."""
    def __init__(self, filepath: str, reason: str = ""):
        super().__init__(f"Corrupt image: {filepath}" + (f" ({reason})" if reason else ""))
        self.filepath = filepath


class FeatureExtractionError(GestureRecognitionError):
    """Raised when HOG feature extraction fails."""
    pass


class ModelError(GestureRecognitionError):
    """Raised when model operations fail."""
    pass


class ModelNotTrainedError(ModelError):
    """Raised when trying to use a model that hasn't been trained."""
    pass


class ModelSaveError(ModelError):
    """Raised when saving the model fails."""
    pass


class ModelLoadError(ModelError):
    """Raised when loading a saved model fails."""
    pass


class ConfigurationError(GestureRecognitionError):
    """Raised when configuration is invalid."""
    pass


class ValidationError(GestureRecognitionError):
    """Raised when validation fails."""
    pass


class KaggleAuthError(GestureRecognitionError):
    """Raised when Kaggle authentication fails."""
    def __init__(self, message: str = "Kaggle authentication failed"):
        super().__init__(
            f"{message}. Please ensure you have:\n"
            "  1. A Kaggle account at https://www.kaggle.com\n"
            "  2. Downloaded your API token from https://www.kaggle.com/account\n"
            "  3. Placed kaggle.json in ~/.kaggle/ (Linux/Mac) or C:/Users/<user>/.kaggle/ (Windows)"
        )
