"""
Environment check script to verify that all required packages are installed correctly.
"""

import sys
print(f"Python version: {sys.version}")

try:
    import numpy as np
    print(f"NumPy version: {np.__version__}")
except ImportError:
    print("NumPy is not installed.")

try:
    import pandas as pd
    print(f"Pandas version: {pd.__version__}")
except ImportError:
    print("Pandas is not installed.")

try:
    import tensorflow as tf
    print(f"TensorFlow version: {tf.__version__}")
    
    # Check if GPU is available
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"GPU is available: {gpus}")
        for gpu in gpus:
            print(f"  Name: {gpu.name}, Type: {gpu.device_type}")
    else:
        print("No GPU found, using CPU instead.")
except ImportError:
    print("TensorFlow is not installed.")

try:
    import flask
    print(f"Flask version: {flask.__version__}")
except ImportError:
    print("Flask is not installed.")

try:
    from PIL import Image, __version__ as pil_version
    print(f"Pillow version: {pil_version}")
except ImportError:
    print("Pillow is not installed.")

try:
    import matplotlib
    print(f"Matplotlib version: {matplotlib.__version__}")
except ImportError:
    print("Matplotlib is not installed.")

try:
    import sklearn
    print(f"Scikit-learn version: {sklearn.__version__}")
except ImportError:
    print("Scikit-learn is not installed.")

print("\nChecking dataset paths...")
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_DIR = os.path.join(BASE_DIR, 'Skin cancer DATA', 'Train')
TEST_DIR = os.path.join(BASE_DIR, 'Skin cancer DATA', 'Test')

if os.path.exists(TRAIN_DIR):
    print(f"Training directory exists: {TRAIN_DIR}")
    train_classes = [d for d in os.listdir(TRAIN_DIR) if os.path.isdir(os.path.join(TRAIN_DIR, d))]
    print(f"Found {len(train_classes)} classes in training directory: {train_classes}")
    
    # Count images in each class
    for class_name in train_classes:
        class_dir = os.path.join(TRAIN_DIR, class_name)
        image_count = len([f for f in os.listdir(class_dir) if os.path.isfile(os.path.join(class_dir, f))])
        print(f"  {class_name}: {image_count} images")
else:
    print(f"Training directory does not exist: {TRAIN_DIR}")

if os.path.exists(TEST_DIR):
    print(f"Test directory exists: {TEST_DIR}")
    test_classes = [d for d in os.listdir(TEST_DIR) if os.path.isdir(os.path.join(TEST_DIR, d))]
    print(f"Found {len(test_classes)} classes in test directory: {test_classes}")
else:
    print(f"Test directory does not exist: {TEST_DIR}")

print("\nEnvironment check completed.")