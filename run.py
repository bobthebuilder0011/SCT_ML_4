#!/usr/bin/env python
"""
Quick-start script for Leap Gesture Recognition.
Provides interactive prompts for common tasks.
"""

import os
import sys
import argparse


def print_header():
    """Print welcome header."""
    print("=" * 70)
    print("Leap Gesture Recognition - Quick Start")
    print("=" * 70)
    print()


def check_setup():
    """Check if environment is properly configured."""
    print("Checking setup...")

    checks = {
        'Python version': sys.version_info >= (3, 8),
        'kagglehub': False,
        'cv2': False,
        'sklearn': False,
    }

    try:
        import kagglehub
        checks['kagglehub'] = True
    except ImportError:
        pass

    try:
        import cv2
        checks['cv2'] = True
    except ImportError:
        pass

    try:
        import sklearn
        checks['sklearn'] = True
    except ImportError:
        pass

    all_passed = all(checks.values())

    for check, passed in checks.items():
        status = "✓" if passed else "✗"
        print(f"  {status} {check}")

    if not all_passed:
        print("\nSome dependencies are missing. Run:")
        print("  pip install -r requirements.txt")
        return False

    # Check Kaggle credentials
    kaggle_path = os.path.expanduser("~/.kaggle/kaggle.json")
    if sys.platform == "win32":
        kaggle_path = os.path.expanduser("~\\.kaggle\\kaggle.json")

    if os.path.exists(kaggle_path):
        print("  ✓ Kaggle credentials found")
    else:
        print("  ⚠ Kaggle credentials not found")
        print(f"    Expected at: {kaggle_path}")
        print("    Download from: https://www.kaggle.com/account")

    print()
    return all_passed


def interactive_menu():
    """Show interactive menu."""
    print("What would you like to do?")
    print()
    print("  1. Run full training pipeline")
    print("  2. Run quick test (100 samples)")
    print("  3. Run with hyperparameter tuning")
    print("  4. Validate saved model")
    print("  5. Show project info")
    print("  6. Exit")
    print()

    choice = input("Enter choice (1-6): ").strip()

    return choice


def run_command(cmd):
    """Run a shell command."""
    print(f"\nRunning: {cmd}\n")
    result = os.system(cmd)
    return result == 0


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Quick-start script")
    parser.add_argument('--skip-checks', action='store_true',
                       help='Skip environment checks')
    args = parser.parse_args()

    print_header()

    if not args.skip_checks:
        if not check_setup():
            return 1

    while True:
        choice = interactive_menu()

        if choice == '1':
            print("\nStarting full training...")
            run_command("python leap_gesture_svm.py")

        elif choice == '2':
            print("\nRunning quick test...")
            run_command("python leap_gesture_svm.py --limit 100 --no-plots")

        elif choice == '3':
            print("\nRunning with hyperparameter tuning...")
            run_command("python leap_gesture_svm.py --tune")

        elif choice == '4':
            print("\nValidating saved model...")
            run_command("python cli.py validate-model")

        elif choice == '5':
            print("\nProject info...")
            run_command("python cli.py info")

        elif choice == '6':
            print("\nGoodbye!")
            break

        else:
            print("\nInvalid choice. Please try again.")

        print()
        input("Press Enter to continue...")
        print("\n" + "=" * 70 + "\n")

    return 0


if __name__ == '__main__':
    sys.exit(main())
