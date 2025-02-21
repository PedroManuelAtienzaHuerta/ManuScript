import cv2
import numpy as np
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_and_filter_image(image_path):
    """Load the image in gray scale and apply bilateral filter."""
    try:
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        bilateral = cv2.bilateralFilter(image, 9, 75, 75)
        return bilateral
    except Exception as e:
        logging.error(f"Error loading or filtering image {image_path}: {e}")
        return None

def processing_1(image_path, name):
    """Bilateral Filter and Adaptative Binarization"""
    bilateral = load_and_filter_image(image_path)
    if bilateral is not None:
        output_path = os.path.join('Out', name)
        cv2.imwrite(output_path, bilateral)
        logging.info(f"Saved processed image to {output_path}")

def processing_2(image_path, name):
    """Bilateral Filter, Adaptative Binarization and Dilatation"""
    bilateral = load_and_filter_image(image_path)
    if bilateral is not None:
        binarized = cv2.adaptiveThreshold(bilateral, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 4)
        kernel = np.ones((2, 2), np.uint8)
        dilated = cv2.dilate(binarized, kernel, iterations=1)
        output_path = os.path.join('Out', name)
        cv2.imwrite(output_path, dilated)
        logging.info(f"Saved processed image to {output_path}")

def processing_3(image_path, name):
    """Bilateral Filter, Adaptative Binarization and Salt and Pepper Filter"""
    bilateral = load_and_filter_image(image_path)
    if bilateral is not None:
        binarized = cv2.adaptiveThreshold(bilateral, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 4)
        kernel = np.ones((2, 2), np.uint8)
        close = cv2.morphologyEx(binarized, cv2.MORPH_CLOSE, kernel)
        open = cv2.morphologyEx(close, cv2.MORPH_OPEN, kernel)
        output_path = os.path.join('Out', name)
        cv2.imwrite(output_path, open)
        logging.info(f"Saved processed image to {output_path}")

def processing_4(image_path, name):
    """Bilateral Filter and Normal Binarization"""
    bilateral = load_and_filter_image(image_path)
    if bilateral is not None:
        _, binarized = cv2.threshold(bilateral, 185, 255, cv2.THRESH_BINARY)
        output_path = os.path.join('Out', name)
        cv2.imwrite(output_path, binarized)
        logging.info(f"Saved processed image to {output_path}")

# Ensure output directory exists
os.makedirs('Out', exist_ok=True)

# Images' path
Path = '../dataset/A'
Images = [file for file in os.listdir(Path) if file.endswith(('.png', '.jpg', '.jpeg'))]

# Process all images
for img in Images:
    to_proc = os.path.join(Path, img)
    # Select the processing function to apply
    processing_1(to_proc, str(img))
