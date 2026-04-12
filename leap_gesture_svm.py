"""
Leap Gesture Recognition using SVM with HOG Features

This script classifies hand gestures from the leapGestRecog dataset using
Support Vector Machines (SVM) with Histogram of Oriented Gradients (HOG) features.

Usage:
    python leap_gesture_svm.py [--tune] [--no-plots] [--limit N]

Author: Senior ML Engineer
License: MIT
"""

import os
import sys
import argparse
import traceback
import logging
from pathlib import Path
from typing import Tuple, List, Optional, Dict, Any
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

import cv2
import numpy as np
import matplotlib

# Bulletproof Matplotlib backend selection
try:
    if os.environ.get('DISPLAY', '') == '' and os.name != 'nt':
        matplotlib.use('Agg')
    else:
        # Try to use a GUI backend, fallback to Agg
        for backend in ['TkAgg', 'Qt5Agg', 'MacOSX']:
            try:
                matplotlib.use(backend)
                break
            except Exception:
                continue
except Exception:
    matplotlib.use('Agg')

import matplotlib.pyplot as plt
import seaborn as sns

try:
    import kagglehub
except ImportError:
    kagglehub = None

import joblib
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_recall_fscore_support
)
from skimage.feature import hog

# Local imports
from config import config, Config
from logger import setup_logger, ProgressLogger
from exceptions import (
    DatasetNotFoundError, EmptyDatasetError, ImageProcessingError,
    ModelError, ModelSaveError, KaggleAuthError
)
from validator import DatasetValidator, ModelValidator

# Setup logger
logger = setup_logger(
    name="gesture_recognition",
    log_file=config.output.log_path,
    level=logging.INFO
)


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Leap Gesture Recognition using SVM with HOG features",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python leap_gesture_svm.py                    # Run with defaults
    python leap_gesture_svm.py --tune             # Enable hyperparameter tuning
    python leap_gesture_svm.py --no-plots         # Skip visualizations
    python leap_gesture_svm.py --limit 100        # Process only 100 samples
        """
    )

    parser.add_argument(
        '--tune', action='store_true',
        help='Enable hyperparameter tuning with GridSearchCV'
    )
    parser.add_argument(
        '--no-plots', action='store_true',
        help='Disable matplotlib visualizations'
    )
    parser.add_argument(
        '--limit', type=int, default=None,
        help='Limit number of samples to process (for testing)'
    )
    parser.add_argument(
        '--dataset-path', type=str, default=None,
        help='Path to existing dataset (skip download)'
    )
    parser.add_argument(
        '--model-path', type=str, default=config.output.model_path,
        help='Path to save/load model'
    )
    parser.add_argument(
        '--skip-training', action='store_true',
        help='Skip training, only load and evaluate existing model'
    )
    parser.add_argument(
        '--cache-dir', type=str, default=None,
        help='Directory to cache downloaded dataset (default: system temp)'
    )
    parser.add_argument(
        '--cleanup', action='store_true',
        help='Delete dataset after training (keeps only the trained model)'
    )

    return parser.parse_args()


def download_dataset_with_retry(max_retries: int = 3, cache_dir: Optional[str] = None) -> str:
    """
    Download dataset with retry logic and proper error handling.

    Args:
        max_retries: Maximum number of download attempts
        cache_dir: Custom directory to store downloaded dataset

    Returns:
        Path to downloaded dataset

    Raises:
        DatasetNotFoundError: If download fails after all retries
        KaggleAuthError: If authentication fails
    """
    if kagglehub is None:
        raise ImportError(
            "kagglehub not installed. Run: pip install kagglehub"
        )

    # Set custom cache directory if provided
    if cache_dir:
        os.environ['KAGGLEHUB_CACHE_DIR'] = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        logger.info(f"Using custom cache directory: {cache_dir}")

    # Log cache location if available
    try:
        cache_info = kagglehub.get_dataset_path('gti-upm/leapgestrecog')
        logger.info(f"Dataset cache location: {cache_info}")
    except AttributeError:
        pass  # Older versions don't have this function

    for attempt in range(max_retries):
        try:
            logger.info(f"Downloading dataset (attempt {attempt + 1}/{max_retries})...")
            path = kagglehub.dataset_download("gti-upm/leapgestrecog")
            logger.info(f"Dataset downloaded to: {path}")
            return path

        except Exception as e:
            error_msg = str(e).lower()

            if '403' in error_msg or 'forbidden' in error_msg or 'unauthorized' in error_msg:
                logger.error("Kaggle authentication failed. Ensure kaggle.json is in ~/.kaggle/ (Linux/Mac) or C:\\Users\\<user>\\.kaggle\\ (Windows)")
                print("\n" + "!" * 60)
                print("KAGLE AUTHENTICATION ERROR")
                print("1. Go to https://www.kaggle.com/settings")
                print("2. Click 'Create New API Token' to download kaggle.json")
                print("3. Move it to your home directory's .kaggle folder")
                print("!" * 60 + "\n")
                raise KaggleAuthError()
            elif 'not found' in error_msg:
                raise DatasetNotFoundError(f"Dataset not found on Kaggle: {e}")

            logger.warning(f"Download attempt {attempt + 1} failed: {e}")

            if attempt < max_retries - 1:
                import time
                time.sleep(2 ** attempt)  # Exponential backoff
            else:
                raise DatasetNotFoundError(
                    f"Failed to download dataset after {max_retries} attempts: {e}"
                )

    raise DatasetNotFoundError("Unexpected error in download")


def find_dataset_root(base_path: str) -> Optional[str]:
    """
    Find the actual dataset root containing the 'leapGestRecog' folder.

    Args:
        base_path: The path returned by kagglehub or provided by user

    Returns:
        Path to the dataset root or None if not found
    """
    if not base_path or not os.path.exists(base_path):
        return None

    # Check if base_path directly contains leapGestRecog
    direct_path = os.path.join(base_path, 'leapGestRecog')
    if os.path.isdir(direct_path):
        return direct_path

    # Search recursively
    for root, dirs, _ in os.walk(base_path):
        if 'leapGestRecog' in dirs:
            return os.path.join(root, 'leapGestRecog')

        # Check if current directory matches pattern
        if os.path.basename(root) == 'leapGestRecog':
            return root

    return None


def extract_hog_features(image: np.ndarray) -> np.ndarray:
    """
    Extract HOG (Histogram of Oriented Gradients) features from an image.

    For a 64x64 image with default parameters (9 orientations, 8x8 pixels/cell,
    2x2 cells/block): Feature vector length = 9 × 4 × 49 = 1764 features

    Args:
        image: Grayscale image (should be 64x64)

    Returns:
        HOG feature vector

    Raises:
        ImageProcessingError: If feature extraction fails
    """
    try:
        if image is None or image.size == 0:
            raise ImageProcessingError("Empty image provided to HOG extractor")

        features = hog(
            image,
            orientations=config.hog.orientations,
            pixels_per_cell=config.hog.pixels_per_cell,
            cells_per_block=config.hog.cells_per_block,
            visualize=False,
            block_norm=config.hog.block_norm
        )

        if len(features) == 0:
            raise ImageProcessingError("HOG extraction returned empty features")

        return features

    except Exception as e:
        raise ImageProcessingError(f"HOG feature extraction failed: {e}")


def preprocess_image(image_path: str) -> Optional[np.ndarray]:
    """
    Load and preprocess a single image with CLAHE enhancement.

    Args:
        image_path: Path to image file

    Returns:
        Preprocessed image array or None if processing fails
    """
    try:
        # Load image
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            logger.debug(f"Could not load image: {image_path}")
            return None

        # Resize
        image_resized = cv2.resize(image, config.data.img_size)

        # Apply CLAHE (Contrast Enhancement) - matching realtime UI
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(image_resized)

        # Normalize
        image_normalized = enhanced.astype(np.float32) / 255.0

        return image_normalized

    except Exception as e:
        logger.debug(f"Error preprocessing {image_path}: {e}")
        return None


def load_and_preprocess_data(dataset_path: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Load images from nested folders and extract HOG features.

    Expected structure: dataset_path/subject/gesture_class/image_files

    Args:
        dataset_path: Root path to the dataset

    Returns:
        Tuple of (features, labels, gesture_names)

    Raises:
        EmptyDatasetError: If no valid images found
    """
    # Validate dataset path
    validated_path = DatasetValidator.validate_dataset_path(dataset_path)

    features: List[np.ndarray] = []
    labels: List[str] = []
    gesture_names: set = set()
    skipped_files: int = 0
    processed_files: int = 0

    logger.info(f"Loading data from: {validated_path}")

    with ProgressLogger(logger, "Data loading") as progress:
        # Count total for progress
        total_estimate = sum(
            len(files) for _, _, files in os.walk(validated_path)
        )

        # Walk through subject folders
        for subject_folder in sorted(os.listdir(validated_path)):
            subject_path = os.path.join(validated_path, subject_folder)

            if not os.path.isdir(subject_path):
                continue

            # Walk through gesture folders
            for gesture_folder in sorted(os.listdir(subject_path)):
                gesture_path = os.path.join(subject_path, gesture_folder)

                if not os.path.isdir(gesture_path):
                    continue

                # Extract gesture name
                gesture_name = (
                    gesture_folder.split('_', 1)[1]
                    if '_' in gesture_folder else gesture_folder
                )
                gesture_names.add(gesture_name)

                # Process images
                for filename in os.listdir(gesture_path):
                    file_path = os.path.join(gesture_path, filename)

                    # Validate file
                    if not DatasetValidator.validate_image_file(
                        file_path, config.data.valid_extensions
                    ):
                        skipped_files += 1
                        continue

                    # Process image
                    image = preprocess_image(file_path)
                    if image is None:
                        skipped_files += 1
                        continue

                    # Extract features
                    try:
                        hog_features = extract_hog_features(image)
                        features.append(hog_features)
                        labels.append(gesture_name)
                        processed_files += 1

                        # Apply limit if set
                        if (config.data.max_samples and
                                len(features) >= config.data.max_samples):
                            logger.info(f"Reached sample limit: {config.data.max_samples}")
                            break

                    except ImageProcessingError as e:
                        logger.debug(f"Feature extraction failed for {file_path}: {e}")
                        skipped_files += 1

                    # Progress update
                    progress.log_progress(
                        processed_files + skipped_files,
                        f"processed {processed_files} images"
                    )

    # Validate results
    if len(features) == 0:
        raise EmptyDatasetError(
            f"No valid images found in {validated_path}. "
            f"Checked {processed_files + skipped_files} files, "
            f"skipped {skipped_files}. "
            f"Valid extensions: {config.data.valid_extensions}"
        )

    # Validate class distribution
    class_counts = DatasetValidator.validate_class_distribution(
        labels, config.data.min_samples_per_class
    )

    logger.info(f"Successfully loaded {len(features)} images across "
                f"{len(class_counts)} classes")
    logger.info(f"HOG feature vector length: {len(features[0])} dimensions")
    logger.info(f"Skipped {skipped_files} invalid/corrupt files")

    return np.array(features), np.array(labels), sorted(list(gesture_names))


def plot_sample_images(dataset_path: str, gesture_names: List[str],
                       samples_per_class: int = 2) -> None:
    """Display sample images with HOG visualizations."""
    try:
        n_classes = min(len(gesture_names), 5)
        n_cols = samples_per_class * 2

        fig, axes = plt.subplots(n_classes, n_cols, figsize=(n_cols * 3, n_classes * 2))

        if n_classes == 1:
            axes = axes.reshape(1, -1)

        for i, gesture in enumerate(gesture_names[:n_classes]):
            found_samples = 0

            for subject in os.listdir(dataset_path):
                subject_path = os.path.join(dataset_path, subject)
                if not os.path.isdir(subject_path):
                    continue

                gesture_folder = None
                for folder in os.listdir(subject_path):
                    if gesture in folder:
                        gesture_folder = os.path.join(subject_path, folder)
                        break

                if gesture_folder and os.path.exists(gesture_folder):
                    images = [
                        f for f in os.listdir(gesture_folder)
                        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
                    ][:samples_per_class]

                    for j, img_name in enumerate(images):
                        if found_samples >= samples_per_class:
                            break

                        img_path = os.path.join(gesture_folder, img_name)
                        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                        if img is None:
                            continue

                        img = cv2.resize(img, config.data.img_size)

                        # Original
                        axes[i, j * 2].imshow(img, cmap='gray')
                        axes[i, j * 2].set_title(f"{gesture}", fontsize=10)
                        axes[i, j * 2].axis('off')

                        # HOG visualization
                        from skimage.feature import hog as sk_hog
                        _, hog_img = sk_hog(
                            img, orientations=9, pixels_per_cell=(8, 8),
                            cells_per_block=(2, 2), visualize=True
                        )
                        axes[i, j * 2 + 1].imshow(hog_img, cmap='hot')
                        axes[i, j * 2 + 1].set_title(f"{gesture} HOG", fontsize=10)
                        axes[i, j * 2 + 1].axis('off')

                        found_samples += 1

                    if found_samples >= samples_per_class:
                        break

        plt.suptitle("Sample Images and HOG Features", fontsize=14, y=1.02)
        plt.tight_layout()
        plt.savefig(f"{config.output.figures_dir}/sample_images.png", dpi=150, bbox_inches='tight')
        logger.info(f"Saved sample images to {config.output.figures_dir}/sample_images.png")
        plt.show()

    except Exception as e:
        logger.error(f"Failed to plot sample images: {e}")


def plot_class_distribution(labels: np.ndarray) -> None:
    """Plot class distribution."""
    try:
        unique, counts = np.unique(labels, return_counts=True)

        plt.figure(figsize=(12, 6))
        bars = plt.bar(unique, counts, color='steelblue', edgecolor='black')
        plt.xlabel('Gesture Class', fontsize=12)
        plt.ylabel('Number of Samples', fontsize=12)
        plt.title('Class Distribution', fontsize=14)
        plt.xticks(rotation=45, ha='right')

        for bar, count in zip(bars, counts):
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 10,
                str(count), ha='center', va='bottom', fontsize=9
            )

        plt.tight_layout()
        plt.savefig(f"{config.output.figures_dir}/class_distribution.png", dpi=150, bbox_inches='tight')
        logger.info(f"Saved class distribution plot")
        plt.show()

    except Exception as e:
        logger.error(f"Failed to plot class distribution: {e}")


def train_svm_model(X_train: np.ndarray, y_train: np.ndarray,
                    tune: bool = True) -> Pipeline:
    """
    Train SVM model with optional hyperparameter tuning.

    Args:
        X_train: Training features
        y_train: Training labels
        tune: Whether to perform hyperparameter tuning

    Returns:
        Trained Pipeline
    """
    # Validate split
    ModelValidator.validate_train_test_split(
        X_train, y_train, config.data.test_size
    )

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('svm', SVC(
            kernel=config.svm.kernel,
            probability=config.svm.probability,
            cache_size=config.svm.cache_size,
            class_weight='balanced',  # Added to prevent bias (e.g., towards 'L')
            random_state=config.data.random_state,
            verbose=config.output.verbose
        ))
    ])

    if tune and config.svm.tune_hyperparams:
        logger.info("Starting hyperparameter tuning with GridSearchCV...")

        param_grid = {
            'svm__C': config.svm.C_values,
            'svm__gamma': config.svm.gamma_values
        }

        cv = StratifiedKFold(
            n_splits=min(config.svm.cv_folds, len(np.unique(y_train))),
            shuffle=True,
            random_state=config.data.random_state
        )

        grid_search = GridSearchCV(
            pipeline,
            param_grid,
            cv=cv,
            scoring='accuracy',
            n_jobs=config.svm.n_jobs,
            verbose=config.output.verbose,
            return_train_score=True
        )

        with ProgressLogger(logger, "Hyperparameter tuning") as _:
            grid_search.fit(X_train, y_train)

        logger.info(f"Best parameters: {grid_search.best_params_}")
        logger.info(f"Best CV accuracy: {grid_search.best_score_:.4f}")

        # Save CV results
        cv_results_path = f"{config.output.results_dir}/cv_results.txt"
        with open(cv_results_path, 'w') as f:
            f.write("Cross-Validation Results:\n")
            f.write("=" * 50 + "\n")
            for mean, std, params in zip(
                grid_search.cv_results_['mean_test_score'],
                grid_search.cv_results_['std_test_score'],
                grid_search.cv_results_['params']
            ):
                f.write(f"{params}: {mean:.4f} (+/- {std*2:.4f})\n")

        return grid_search.best_estimator_

    else:
        logger.info("Training SVM with default parameters...")
        pipeline.fit(X_train, y_train)
        return pipeline


def evaluate_model(model: Pipeline, X_test: np.ndarray, y_test: np.ndarray,
                   label_encoder: LabelEncoder) -> Dict[str, Any]:
    """
    Evaluate model and return metrics.

    Returns:
        Dictionary of evaluation metrics
    """
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_test, y_pred, average='weighted'
    )

    logger.info(f"\n{'='*60}")
    logger.info(f"EVALUATION RESULTS")
    logger.info(f"{'='*60}")
    logger.info(f"Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    logger.info(f"Precision: {precision:.4f}")
    logger.info(f"Recall:    {recall:.4f}")
    logger.info(f"F1-Score:  {f1:.4f}")
    logger.info(f"{'='*60}")

    # Classification report
    report = classification_report(
        y_test, y_pred, target_names=label_encoder.classes_
    )
    logger.info(f"\nClassification Report:\n{report}")

    # Save report
    report_path = f"{config.output.results_dir}/classification_report.txt"
    with open(report_path, 'w') as f:
        f.write("Classification Report\n")
        f.write("=" * 60 + "\n")
        f.write(report)
        f.write(f"\nAccuracy: {accuracy:.4f}\n")

    # Confusion matrix
    try:
        cm = confusion_matrix(y_test, y_pred)

        plt.figure(figsize=(12, 10))
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_
        )
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        plt.title('Confusion Matrix', fontsize=14)
        plt.tight_layout()
        plt.savefig(f"{config.output.figures_dir}/confusion_matrix.png", dpi=150, bbox_inches='tight')
        logger.info(f"Saved confusion matrix")
        plt.show()

    except Exception as e:
        logger.error(f"Failed to plot confusion matrix: {e}")

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'predictions': y_pred
    }


def save_model(model: Pipeline, label_encoder: LabelEncoder,
               model_path: str) -> None:
    """Save model and label encoder."""
    try:
        Path(model_path).parent.mkdir(parents=True, exist_ok=True)

        joblib.dump(model, model_path)
        encoder_path = model_path.replace('.joblib', '_encoder.joblib')
        joblib.dump(label_encoder, encoder_path)

        logger.info(f"Model saved to: {model_path}")
        logger.info(f"Encoder saved to: {encoder_path}")

    except Exception as e:
        raise ModelSaveError(f"Failed to save model: {e}")


def load_saved_model(model_path: str) -> Optional[Tuple[Pipeline, LabelEncoder]]:
    """Load saved model and label encoder."""
    if not ModelValidator.validate_saved_model(model_path):
        return None

    try:
        model = joblib.load(model_path)
        encoder_path = model_path.replace('.joblib', '_encoder.joblib')

        if not os.path.exists(encoder_path):
            logger.error(f"Encoder file not found: {encoder_path}")
            return None

        label_encoder = joblib.load(encoder_path)
        logger.info(f"Loaded model from: {model_path}")
        return model, label_encoder

    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return None


def predict_image(model: Pipeline, label_encoder: LabelEncoder,
                  image_path: str) -> Tuple[str, float]:
    """Predict gesture for a single image."""
    image = preprocess_image(image_path)
    if image is None:
        raise ImageProcessingError(f"Could not load image: {image_path}")

    features = extract_hog_features(image).reshape(1, -1)
    prediction = model.predict(features)[0]
    probabilities = model.predict_proba(features)[0]
    confidence = np.max(probabilities)

    predicted_class = label_encoder.inverse_transform([prediction])[0]
    return predicted_class, confidence


def cleanup_dataset(dataset_path: str) -> None:
    """
    Clean up downloaded dataset to save disk space.

    Args:
        dataset_path: Path to the dataset directory
    """
    import shutil
    try:
        if os.path.exists(dataset_path):
            logger.info(f"Cleaning up dataset at: {dataset_path}")
            shutil.rmtree(dataset_path)
            logger.info("Dataset cleaned up successfully")
    except Exception as e:
        logger.warning(f"Failed to cleanup dataset: {e}")


def main():
    """Main execution function."""
    print("=" * 70)
    print("Leap Gesture Recognition - SVM with HOG Features")
    print("=" * 70)

    # Parse arguments
    args = parse_arguments()

    # Update config from args
    if args.limit:
        config.data.max_samples = args.limit
    if args.tune:
        config.svm.tune_hyperparams = True

    # Validate config
    try:
        config.validate()
    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        sys.exit(1)

    logger.info("Starting gesture recognition pipeline")

    # Track if we need to cleanup
    should_cleanup = args.cleanup or config.data.cleanup_dataset
    downloaded_path = None

    # Load or download dataset
    dataset_path = None
    if args.dataset_path:
        dataset_path = args.dataset_path
        logger.info(f"Using provided dataset path: {dataset_path}")
    else:
        try:
            base_path = download_dataset_with_retry(cache_dir=args.cache_dir)
            downloaded_path = base_path
            dataset_path = find_dataset_root(base_path)
        except Exception as e:
            logger.error(f"Dataset acquisition failed: {e}")
            sys.exit(1)

    if not dataset_path:
        logger.error("Could not locate dataset root (leapGestRecog folder)")
        sys.exit(1)

    # Load and preprocess data
    try:
        features, labels, gesture_names = load_and_preprocess_data(dataset_path)
    except Exception as e:
        logger.error(f"Data loading failed: {e}")
        sys.exit(1)

    # Visualizations
    if not args.no_plots:
        plot_class_distribution(labels)
        plot_sample_images(dataset_path, gesture_names[:5])

    # Encode labels
    label_encoder = LabelEncoder()
    labels_encoded = label_encoder.fit_transform(labels)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels_encoded,
        test_size=config.data.test_size,
        random_state=config.data.random_state,
        stratify=labels_encoded
    )

    logger.info(f"Training set: {len(X_train)} samples")
    logger.info(f"Test set: {len(X_test)} samples")

    # Train or load model
    if args.skip_training:
        model_data = load_saved_model(args.model_path)
        if model_data is None:
            logger.error("No saved model found and --skip-training specified")
            sys.exit(1)
        model, label_encoder = model_data
    else:
        model = train_svm_model(X_train, y_train, tune=config.svm.tune_hyperparams)

        # Evaluate
        evaluate_model(model, X_test, y_test, label_encoder)

        # Save model
        save_model(model, label_encoder, args.model_path)

    logger.info("\n" + "=" * 70)
    logger.info("Pipeline completed successfully!")
    logger.info("=" * 70)

    # Cleanup downloaded dataset if requested
    if should_cleanup and downloaded_path and os.path.exists(downloaded_path):
        cleanup_dataset(downloaded_path)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\nProcess interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Unhandled exception: {e}")
        logger.debug(traceback.format_exc())
        sys.exit(1)
