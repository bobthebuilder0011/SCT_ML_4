"""
Real-Time Hand Gesture Recognition UI
Uses trained SVM model with HOG features for live webcam inference.

Usage:
    python realtime_gesture_ui.py          # Run with default model path
    python realtime_gesture_ui.py --train  # Force retrain model first
    python realtime_gesture_ui.py --model path/to/model.joblib

Controls:
    'q' - Quit application
    's' - Save screenshot
    'c' - Toggle confidence display
    'f' - Toggle FPS display
"""

import os
import sys
import argparse
import time
import logging
import traceback
import shutil
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
import warnings
warnings.filterwarnings('ignore')

import cv2
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline

# Import from existing project modules
try:
    from config import config
    from logger import setup_logger
    from exceptions import ModelLoadError, ImageProcessingError
except ImportError as e:
    print(f"Error: Could not import project modules: {e}")
    print("Make sure you're running from the project root directory.")
    sys.exit(1)

# Setup logger
logger = setup_logger(
    name="realtime_gesture",
    log_file="logs/realtime.log",
    level=logging.INFO
)

# UI Configuration
WINDOW_NAME = "Advanced Gesture Analysis - SVM/HOG"
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.5
FONT_THICKNESS = 1
TEXT_COLOR = (255, 255, 255)
ACCENT_COLOR = (0, 255, 0)    # Green for matches
SCAN_COLOR = (0, 165, 255)    # Orange for scanning
GUIDE_COLOR = (0, 255, 255)
FPS_COLOR = (0, 255, 255)


def draw_text_with_background(
    img: np.ndarray,
    text: str,
    pos: Tuple[int, int],
    font: int = FONT,
    scale: float = FONT_SCALE,
    color: Tuple[int, int, int] = TEXT_COLOR,
    thickness: int = FONT_THICKNESS,
    bg_color: Tuple[int, int, int] = (0, 0, 0),
    alpha: float = 0.6
) -> None:
    """Draw text with semi-transparent background."""
    (w, h), baseline = cv2.getTextSize(text, font, scale, thickness)
    x, y = pos
    
    # Create overlay for semi-transparent background
    overlay = img.copy()
    cv2.rectangle(overlay, (x - 5, y - h - 10), (x + w + 5, y + baseline + 5), bg_color, -1)
    
    # Blend overlay with original image
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    
    # Draw text
    cv2.putText(img, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Real-time hand gesture recognition using trained SVM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python realtime_gesture_ui.py              # Run inference with saved model
    python realtime_gesture_ui.py --train    # Train new model first
    python realtime_gesture_ui.py --model models/custom.joblib
        """
    )

    parser.add_argument(
        '--model', '-m', type=str,
        default=config.output.model_path,
        help='Path to trained model file (default: models/svm_gesture_model.joblib)'
    )
    parser.add_argument(
        '--train', action='store_true',
        help='Force training even if model exists'
    )
    parser.add_argument(
        '--camera', '-c', type=int, default=0,
        help='Camera device index (default: 0)'
    )
    parser.add_argument(
        '--roi-size', type=int, default=300,
        help='Size of center ROI box in pixels (default: 300)'
    )
    parser.add_argument(
        '--confidence', action='store_true',
        help='Show confidence scores'
    )
    parser.add_argument(
        '--fps', action='store_true',
        help='Show FPS counter'
    )

    return parser.parse_args()


def extract_hog_features(image: np.ndarray) -> np.ndarray:
    """
    Extract HOG features from image - EXACT same as training.

    Args:
        image: Grayscale image (64x64)

    Returns:
        HOG feature vector (1764 dimensions)
    """
    from skimage.feature import hog

    features = hog(
        image,
        orientations=config.hog.orientations,
        pixels_per_cell=config.hog.pixels_per_cell,
        cells_per_block=config.hog.cells_per_block,
        visualize=False,
        block_norm=config.hog.block_norm
    )
    return features


def preprocess_frame(frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Preprocess frame for model inference.
    Applies SAME preprocessing as during training, but with enhanced contrast
    for better feature extraction in varying lighting conditions.

    Args:
        frame: Raw BGR frame from camera

    Returns:
        Tuple of (resized image, features, hog_visualization)
    """
    from skimage.feature import hog

    # Convert to grayscale (same as training)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Resize to 64x64 (same as training)
    resized = cv2.resize(gray, config.data.img_size)

    # ENHANCEMENT: Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    # This makes 'lines' and edges much sharper for HOG regardless of lighting.
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(resized)

    # Normalize to [0, 1] (same as training)
    normalized = enhanced.astype(np.float32) / 255.0

    # Extract HOG features with visualization
    features, hog_image = hog(
        normalized,
        orientations=config.hog.orientations,
        pixels_per_cell=config.hog.pixels_per_cell,
        cells_per_block=config.hog.cells_per_block,
        visualize=True,
        block_norm=config.hog.block_norm
    )

    return enhanced, features.reshape(1, -1), hog_image


def load_trained_model(model_path: str) -> Tuple[Pipeline, LabelEncoder]:
    """
    Load trained SVM model and label encoder.

    Args:
        model_path: Path to saved model

    Returns:
        Tuple of (model, label_encoder)

    Raises:
        ModelLoadError: If model cannot be loaded
    """
    # Handle Windows paths correctly
    model_path = os.path.normpath(model_path)

    if not os.path.exists(model_path):
        raise ModelLoadError(f"Model file not found: {model_path}")

    try:
        model = joblib.load(model_path)
        encoder_path = model_path.replace('.joblib', '_encoder.joblib')
        encoder_path = os.path.normpath(encoder_path)

        if not os.path.exists(encoder_path):
            raise ModelLoadError(f"Label encoder not found: {encoder_path}")

        label_encoder = joblib.load(encoder_path)

        logger.info(f"Loaded model from: {model_path}")
        logger.info(f"Classes: {label_encoder.classes_}")

        return model, label_encoder

    except Exception as e:
        raise ModelLoadError(f"Failed to load model: {e}")


def train_model() -> Tuple[Pipeline, LabelEncoder]:
    """
    Train the SVM model from scratch.
    Imports training functions from leap_gesture_svm.
    """
    logger.info("Training new model...")

    # Import training functions
    try:
        from leap_gesture_svm import (
            download_dataset_with_retry,
            find_dataset_root,
            load_and_preprocess_data,
            train_svm_model,
            save_model
        )
    except ImportError as e:
        logger.error(f"Cannot import training functions: {e}")
        raise

    # Download and load data
    try:
        base_path = download_dataset_with_retry()
        dataset_path = find_dataset_root(base_path)
    except Exception as e:
        logger.error(f"Failed to download dataset: {e}")
        raise

    if not dataset_path:
        raise RuntimeError("Could not find leapGestRecog folder")

    features, labels, _ = load_and_preprocess_data(dataset_path)

    # Encode labels
    label_encoder = LabelEncoder()
    labels_encoded = label_encoder.fit_transform(labels)

    # Train model (no split needed for deployment training)
    model = train_svm_model(features, labels_encoded, tune=False)

    # Save model
    save_model(model, label_encoder, config.output.model_path)

    return model, label_encoder


def draw_guide_box(frame: np.ndarray, roi_size: int) -> Tuple[int, int, int, int]:
    """
    Draw guide box in center of frame for hand placement.

    Args:
        frame: Frame to draw on
        roi_size: Size of ROI box

    Returns:
        Tuple of (x1, y1, x2, y2) coordinates
    """
    h, w = frame.shape[:2]
    cx, cy = w // 2, h // 2
    half = roi_size // 2

    x1 = cx - half
    y1 = cy - half
    x2 = cx + half
    y2 = cy + half

    # Draw very thin guide rectangle (dashed-like)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (100, 100, 100), 1)

    # Draw technical corner markers
    corner_len = 20
    thickness = 1

    # Top-left corner
    cv2.line(frame, (x1, y1), (x1 + corner_len, y1), GUIDE_COLOR, thickness)
    cv2.line(frame, (x1, y1), (x1, y1 + corner_len), GUIDE_COLOR, thickness)

    # Top-right corner
    cv2.line(frame, (x2, y1), (x2 - corner_len, y1), GUIDE_COLOR, thickness)
    cv2.line(frame, (x2, y1), (x2, y1 + corner_len), GUIDE_COLOR, thickness)

    # Bottom-left corner
    cv2.line(frame, (x1, y2), (x1 + corner_len, y2), GUIDE_COLOR, thickness)
    cv2.line(frame, (x1, y2), (x1, y2 - corner_len), GUIDE_COLOR, thickness)

    # Bottom-right corner
    cv2.line(frame, (x2, y2), (x2 - corner_len, y2), GUIDE_COLOR, thickness)
    cv2.line(frame, (x2, y2), (x2, y2 - corner_len), GUIDE_COLOR, thickness)

    # Add label in a finer font
    cv2.putText(
        frame, "ALIGN HAND",
        (x1, y1 - 10),
        FONT, 0.4, GUIDE_COLOR, 1, cv2.LINE_AA
    )

    return x1, y1, x2, y2


def predict_gesture(model: Pipeline, label_encoder: LabelEncoder,
                   features: np.ndarray) -> Tuple[str, float]:
    """
    Predict gesture from features.

    Args:
        model: Trained SVM pipeline
        label_encoder: Fitted label encoder
        features: HOG features (1, 1764)

    Returns:
        Tuple of (predicted_class, confidence)
    """
    prediction = model.predict(features)[0]
    probabilities = model.predict_proba(features)[0]
    confidence = np.max(probabilities)

    gesture = label_encoder.inverse_transform([prediction])[0]
    return gesture, confidence


def initialize_camera(camera_index: int) -> cv2.VideoCapture:
    """
    Initialize camera with proper error handling.

    Args:
        camera_index: Camera device index

    Returns:
        Opened VideoCapture object

    Raises:
        RuntimeError: If camera cannot be opened
    """
    logger.info(f"Initializing camera (device {camera_index})...")

    try:
        cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)  # CAP_DSHOW for Windows compatibility

        # Wait a moment for camera to initialize
        time.sleep(0.5)

        if not cap.isOpened():
            # Try alternative backends
            cap = cv2.VideoCapture(camera_index)
            time.sleep(0.5)

        if not cap.isOpened():
            raise RuntimeError(
                f"Cannot open camera {camera_index}. "
                f"Common causes:\n"
                f"  - Camera is in use by another application (Teams, Zoom, etc.)\n"
                f"  - Camera driver not installed\n"
                f"  - Wrong camera index (try --camera 1 or --camera 2)\n"
                f"\nTo list available cameras, run:\n"
                f"  python -c \"import cv2; [print(f'Camera {i}: {cv2.VideoCapture(i).isOpened()}') for i in range(3)]\""
            )

        # Set camera resolution
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        # Set buffer size to reduce latency
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # Get actual resolution
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if width == 0 or height == 0:
            raise RuntimeError("Camera opened but returned invalid dimensions")

        logger.info(f"Camera initialized: {width}x{height}")
        return cap

    except Exception as e:
        if isinstance(e, RuntimeError):
            raise
        raise RuntimeError(f"Camera initialization failed: {e}")


def run_realtime_ui(model: Pipeline, label_encoder: LabelEncoder,
                   args: argparse.Namespace) -> None:
    """
    Run real-time gesture recognition UI.

    Args:
        model: Trained SVM model
        label_encoder: Label encoder for class names
        args: Command line arguments
    """
    # Initialize camera with error handling
    try:
        cap = initialize_camera(args.camera)
    except RuntimeError as e:
        logger.error(f"Camera error: {e}")
        print(f"\nERROR: {e}")
        sys.exit(1)

    # Get actual resolution
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Create window
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, 800, 600)

    # State variables
    show_confidence = args.confidence
    show_fps = args.fps
    fps_counter = []
    screenshot_count = 0
    frame_count = 0
    
    current_gesture = "Wait..."
    current_confidence = 0.0
    status_msg = "Ready"
    status_time = 0

    # Initialize preview variables to avoid scope issues
    processed = np.zeros(config.data.img_size, dtype=np.uint8)
    hog_image = np.zeros(config.data.img_size, dtype=np.float32)

    print("\n" + "=" * 60)
    print("Real-Time Gesture Recognition Started!")
    print("=" * 60)
    print("Controls:")
    print("  'q' - Quit")
    print("  's' - Save screenshot")
    print("  'c' - Toggle confidence display")
    print("  'f' - Toggle FPS display")
    print("=" * 60 + "\n")

    try:
        while True:
            # Read frame
            ret, frame = cap.read()
            if not ret:
                logger.warning("Failed to read frame, retrying...")
                time.sleep(0.1)
                continue

            frame_count += 1
            display_frame = frame.copy()

            # Calculate FPS
            current_time = time.time()
            fps_counter.append(current_time)
            fps_counter = [t for t in fps_counter if current_time - t < 1.0]
            fps = len(fps_counter)

            # Get center ROI coordinates
            cx, cy = width // 2, height // 2
            half = args.roi_size // 2
            x1, y1 = cx - half, cy - half
            x2, y2 = cx + half, cy + half

            # Ensure ROI is within frame bounds
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(width, x2), min(height, y2)

            # Extract ROI
            roi = frame[y1:y2, x1:x2]

            if roi.size > 0 and roi.shape[0] >= 10 and roi.shape[1] >= 10:
                # Preprocess ROI (same as training)
                try:
                    processed, features, hog_image = preprocess_frame(roi)
                    
                    # Predict gesture (every 2 frames for performance)
                    if frame_count % 2 == 0:
                        current_gesture, current_confidence = predict_gesture(
                            model, label_encoder, features
                        )
                except Exception as e:
                    logger.error(f"Prediction error: {e}")
                    current_gesture = "ERR: ANALYSIS FAIL"
            else:
                current_gesture = "SCANNING..."
                current_confidence = 0.0

            # Draw guide box
            draw_guide_box(display_frame, args.roi_size)

            # --- TOP UI PANEL ---
            # Status Indicator (Flashing for scanning, solid for match)
            indicator_color = SCAN_COLOR
            indicator_text = "STATUS: MONITORING"
            
            if current_gesture != "SCANNING..." and current_gesture != "ERR: ANALYSIS FAIL":
                if current_confidence > 0.6:  # High confidence match
                    indicator_color = ACCENT_COLOR
                    indicator_text = "MATCH FOUND"
                else:
                    indicator_text = "ANALYZING..."
            
            # Simple scanning pulse
            if indicator_text == "STATUS: MONITORING" and (frame_count // 5) % 2 == 0:
                indicator_color = (100, 100, 100) # Dim

            draw_text_with_background(display_frame, indicator_text, (20, 30), scale=0.35, color=indicator_color, thickness=1)

            # --- GESTURE DISPLAY (Prominent) ---
            if indicator_text == "MATCH FOUND":
                # Big bold identification text
                gesture_display = current_gesture.upper()
                draw_text_with_background(
                    display_frame, f"GESTURE IDENTIFIED: {gesture_display}", (20, 70), 
                    scale=0.8, color=ACCENT_COLOR, thickness=2, bg_color=(0, 40, 0)
                )
                if show_confidence:
                    draw_text_with_background(
                        display_frame, f"CONFIDENCE: {current_confidence:.1%}", (20, 100), 
                        scale=0.4, color=(200, 255, 200), thickness=1
                    )
            elif indicator_text == "ANALYZING...":
                # Faded Identification text
                gesture_display = current_gesture.upper()
                draw_text_with_background(
                    display_frame, f"SEARCHING MATCH: {gesture_display}", (20, 70), 
                    scale=0.7, color=(180, 180, 180), thickness=1
                )
            else:
                # Scanning text
                draw_text_with_background(
                    display_frame, "READY FOR INPUT", (20, 70), 
                    scale=0.7, color=SCAN_COLOR, thickness=1
                )

            # FPS counter (Smaller)
            if show_fps:
                fps_text = f"FPS: {fps}"
                draw_text_with_background(
                    display_frame, fps_text, (width - 70, 30), 
                    scale=0.3, color=FPS_COLOR, thickness=1
                )

            # --- BOTTOM UI PANEL ---
            # Instructions (Minimalist)
            instructions = "Q:QUIT | S:SAVE | C:CONF | F:FPS"
            draw_text_with_background(
                display_frame, instructions, (width - 180, height - 15),
                scale=0.3, color=(200, 200, 200), thickness=1
            )

            # --- PREVIEW PANELS (Advanced Features) ---
            try:
                # Processed ROI and HOG Preview
                preview_size = 70
                gap = 10
                
                # 1. Input Preview
                processed_bgr = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)
                preview_in = cv2.resize(processed_bgr, (preview_size, preview_size))
                
                # 2. HOG Preview (the 'lines')
                # Use MAGMA for a more advanced feel
                hog_norm = cv2.normalize(hog_image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                hog_bgr = cv2.applyColorMap(hog_norm, cv2.COLORMAP_MAGMA)
                preview_hog = cv2.resize(hog_bgr, (preview_size, preview_size))
                
                # Positions (Bottom-Left)
                py = height - preview_size - 40
                
                # Draw Input Preview
                px1 = 20
                display_frame[py:py+preview_size, px1:px1+preview_size] = preview_in
                cv2.rectangle(display_frame, (px1, py), (px1+preview_size, py+preview_size), (255, 255, 255), 1)
                cv2.putText(display_frame, "RAW INPUT", (px1, py - 5), FONT, 0.3, (255, 255, 255), 1, cv2.LINE_AA)
                
                # Draw HOG Preview (The 'Lines')
                px2 = px1 + preview_size + gap
                display_frame[py:py+preview_size, px2:px2+preview_size] = preview_hog
                cv2.rectangle(display_frame, (px2, py), (px2+preview_size, py+preview_size), (255, 255, 255), 1)
                cv2.putText(display_frame, "HOG ANALYSIS", (px2, py - 5), FONT, 0.3, (255, 255, 255), 1, cv2.LINE_AA)
            except Exception as e:
                logger.debug(f"Preview error: {e}")

            # Display frame
            cv2.imshow(WINDOW_NAME, display_frame)

            # Handle key presses
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q') or key == 27:
                logger.info("Quit requested by user")
                break
            elif key == ord('s'):
                screenshot_count += 1
                filename = f"screenshot_{screenshot_count:03d}.png"
                cv2.imwrite(filename, display_frame)
                status_msg = f"SAVED: {filename}"
                status_time = time.time()
                logger.info(f"Screenshot saved: {filename}")
            elif key == ord('c'):
                show_confidence = not show_confidence
                status_msg = f"CONFIDENCE {'ON' if show_confidence else 'OFF'}"
                status_time = time.time()
            elif key == ord('f'):
                show_fps = not show_fps
                status_msg = f"FPS {'ON' if show_fps else 'OFF'}"
                status_time = time.time()

    except KeyboardInterrupt:
        logger.info("Interrupted by user (Ctrl+C)")

    finally:
        # Cleanup
        logger.info("Releasing camera and closing windows...")
        cap.release()
        cv2.destroyAllWindows()

        # Force window close on Windows
        for _ in range(5):
            cv2.waitKey(1)
            time.sleep(0.01)

        print("\nCamera released. Goodbye!")


def main():
    """Main entry point."""
    args = parse_arguments()

    # Check if we need to train
    if args.train or not os.path.exists(args.model):
        if not args.train:
            logger.info(f"No model found at {args.model}, training new model...")
        try:
            model, label_encoder = train_model()
        except Exception as e:
            logger.error(f"Training failed: {e}")
            traceback.print_exc()
            sys.exit(1)
    else:
        # Load existing model
        try:
            model, label_encoder = load_trained_model(args.model)
        except ModelLoadError as e:
            logger.error(f"Failed to load model: {e}")
            print(f"\nError: {e}")
            print("\nTo train a new model, run:")
            print("  python realtime_gesture_ui.py --train")
            sys.exit(1)

    # Run real-time UI
    try:
        run_realtime_ui(model, label_encoder, args)
    except Exception as e:
        logger.error(f"Runtime error: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
