"""
Command-line interface for Leap Gesture Recognition.
Provides additional utilities and commands.
"""

import argparse
import sys
import os
from pathlib import Path

import numpy as np
import cv2

from leap_gesture_svm import (
    load_saved_model, predict_image, download_dataset_with_retry,
    find_dataset_root, load_and_preprocess_data
)
from config import config
from logger import logger


def create_parser() -> argparse.ArgumentParser:
    """Create CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog='leap-gesture',
        description='Leap Gesture Recognition CLI',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  leap-gesture predict image.png              # Predict single image
  leap-gesture predict-batch images/          # Predict batch
  leap-gesture validate-model                 # Validate saved model
  leap-gesture info                           # Show dataset/model info
        '''
    )

    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Predict command
    predict_parser = subparsers.add_parser('predict', help='Predict gesture for image')
    predict_parser.add_argument('image', help='Path to image file')
    predict_parser.add_argument('--model', '-m', default=config.output.model_path,
                               help='Path to model file')
    predict_parser.add_argument('--confidence', '-c', action='store_true',
                               help='Show confidence scores')

    # Batch predict command
    batch_parser = subparsers.add_parser('predict-batch', help='Predict batch of images')
    batch_parser.add_argument('directory', help='Directory containing images')
    batch_parser.add_argument('--model', '-m', default=config.output.model_path,
                             help='Path to model file')
    batch_parser.add_argument('--output', '-o', help='Output file for results')

    # Validate model command
    validate_parser = subparsers.add_parser('validate-model',
                                           help='Validate saved model integrity')
    validate_parser.add_argument('--model', '-m', default=config.output.model_path,
                                help='Path to model file')

    # Info command
    info_parser = subparsers.add_parser('info', help='Show project information')
    info_parser.add_argument('--dataset-path', help='Show dataset statistics')

    return parser


def cmd_predict(args):
    """Predict gesture for single image."""
    if not os.path.exists(args.image):
        logger.error(f"Image not found: {args.image}")
        return 1

    result = load_saved_model(args.model)
    if result is None:
        logger.error(f"Could not load model from {args.model}")
        return 1

    model, encoder = result

    try:
        prediction, confidence = predict_image(model, encoder, args.image)
        print(f"Image: {args.image}")
        print(f"Prediction: {prediction}")

        if args.confidence:
            print(f"Confidence: {confidence:.4f}")

        return 0

    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        return 1


def cmd_predict_batch(args):
    """Predict gestures for batch of images."""
    if not os.path.isdir(args.directory):
        logger.error(f"Directory not found: {args.directory}")
        return 1

    result = load_saved_model(args.model)
    if result is None:
        logger.error(f"Could not load model from {args.model}")
        return 1

    model, encoder = result

    # Find all images
    valid_exts = ('.png', '.jpg', '.jpeg', '.bmp')
    images = [
        os.path.join(args.directory, f)
        for f in os.listdir(args.directory)
        if f.lower().endswith(valid_exts)
    ]

    if not images:
        logger.error(f"No valid images found in {args.directory}")
        return 1

    results = []
    for img_path in images:
        try:
            prediction, confidence = predict_image(model, encoder, img_path)
            results.append({
                'image': os.path.basename(img_path),
                'prediction': prediction,
                'confidence': confidence
            })
            print(f"{os.path.basename(img_path)}: {prediction} ({confidence:.2%})")
        except Exception as e:
            logger.warning(f"Failed to predict {img_path}: {e}")

    # Save results
    if args.output:
        import json
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {args.output}")

    return 0


def cmd_validate_model(args):
    """Validate saved model."""
    from validator import ModelValidator

    print(f"Validating model: {args.model}")

    if ModelValidator.validate_saved_model(args.model):
        print("✓ Model file exists and is readable")
    else:
        print("✗ Model file not found or not readable")
        return 1

    result = load_saved_model(args.model)
    if result is None:
        print("✗ Failed to load model")
        return 1

    model, encoder = result
    print("✓ Model loaded successfully")
    print(f"  Classes: {encoder.classes_}")
    print(f"  Number of classes: {len(encoder.classes_)}")

    # Test prediction
    print("\nTesting prediction with random input...")
    test_features = np.random.rand(1, 1764)
    prediction = model.predict(test_features)[0]
    probabilities = model.predict_proba(test_features)[0]
    print(f"✓ Prediction successful: {encoder.inverse_transform([prediction])[0]}")
    print(f"  Top class probability: {np.max(probabilities):.4f}")

    return 0


def cmd_info(args):
    """Show project information."""
    print("=" * 60)
    print("Leap Gesture Recognition - Project Information")
    print("=" * 60)

    print("\nConfiguration:")
    print(f"  Image size: {config.data.img_size}")
    print(f"  Test size: {config.data.test_size}")
    print(f"  Random state: {config.data.random_state}")
    print(f"  HOG feature length: {config.hog.feature_length}")

    print("\nPaths:")
    print(f"  Model: {config.output.model_path}")
    print(f"  Logs: {config.output.log_path}")
    print(f"  Results: {config.output.results_dir}")

    if args.dataset_path:
        if os.path.exists(args.dataset_path):
            print("\nDataset Statistics:")
            print(f"  Path: {args.dataset_path}")
            # Quick scan
            total_images = 0
            for root, _, files in os.walk(args.dataset_path):
                total_images += len([f for f in files if f.endswith('.png')])
            print(f"  Total PNG images: {total_images}")
        else:
            print(f"\nDataset path not found: {args.dataset_path}")

    # Check for saved model
    print("\nModel Status:")
    if os.path.exists(config.output.model_path):
        size_mb = os.path.getsize(config.output.model_path) / (1024 * 1024)
        print(f"  ✓ Model exists ({size_mb:.2f} MB)")
    else:
        print(f"  ✗ Model not found at {config.output.model_path}")

    print()
    return 0


def main():
    """CLI entry point."""
    parser = create_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    commands = {
        'predict': cmd_predict,
        'predict-batch': cmd_predict_batch,
        'validate-model': cmd_validate_model,
        'info': cmd_info,
    }

    if args.command in commands:
        return commands[args.command](args)
    else:
        print(f"Unknown command: {args.command}")
        return 1


if __name__ == '__main__':
    sys.exit(main())
