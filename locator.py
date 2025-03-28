#!/usr/bin/env python3
import sys
import os
import numpy as np
from PIL import Image
from cnn_model import model
from pathlib import Path

# Add the project root to Python path
project_root = str(Path(__file__).parent.parent)
sys.path.append(project_root)

from backend.services.DataTypes import ImageObject

def load_model(model_fname: str) -> None:
    """Load the trained machine learning model."""
    print(f"Loading model from {model_fname}...")
    model.load(model_fname)
    print("Model loaded successfully.")

def preprocess_image(in_fname: str) -> np.ndarray:
    """Load and preprocess the input image (ensuring RGB channels)."""
    print(f"Reading input image from {in_fname}...")
    im = Image.open(in_fname)
    # Force to RGB in case image has an alpha channel or is grayscale
    im = im.convert("RGB")
    arr = np.array(im)
    print(f"Image loaded with shape {arr.shape}.")
    return arr

def sliding_window_detection(arr: np.ndarray, step: int = 10, win: int = 20, threshold: float = 0.5) -> list:
    """
    Perform sliding window detection on the image array.
    Returns a list of detections, each as a tuple: 
        (x_min, y_min, x_max, y_max, confidence)
    """
    detections = []
    height, width, _ = arr.shape
    print("Starting sliding window detection...")
    for i in range(0, height - win, step):
        for j in range(0, width - win, step):
            chip = arr[i:i+win, j:j+win, :]
            # Normalize chip: scale pixel values to [0, 1]
            chip_normalized = chip / 255.0
            # Model expects input shape (1, win, win, 3)
            prediction = model.predict(chip_normalized[np.newaxis, ...])[0]
            # We assume the model returns a probability vector: [prob_background, prob_airplane]
            confidence = prediction[1]
            if confidence >= threshold:
                # Record detection as (x_min, y_min, x_max, y_max, confidence)
                detection = (j, i, j + win, i + win, confidence)
                detections.append(detection)
    print(f"Sliding window detection completed with {len(detections)} detections.")
    return detections

def draw_bounding_boxes(arr: np.ndarray, detections: list, box_color: list = [255, 0, 0], thickness: int = 2) -> np.ndarray:
    """
    Draw bounding boxes on the image array.
    Each detection is a tuple (x_min, y_min, x_max, y_max, confidence).
    """
    output = arr.copy()
    for (x_min, y_min, x_max, y_max, confidence) in detections:
        # Draw top and bottom edges
        output[y_min:y_min+thickness, x_min:x_max, :] = box_color
        output[y_max-thickness:y_max, x_min:x_max, :] = box_color
        # Draw left and right edges
        output[y_min:y_max, x_min:x_min+thickness, :] = box_color
        output[y_min:y_max, x_max-thickness:x_max, :] = box_color
    return output

def convert_detections_to_image_objects(detections: list, image_id: str, class_name: str) -> list:
    """
    Convert each detection (bounding box tuple) into an ImageObject.
    Here we store a simple representation of the bounding box in `related_pixels`
    (for example, the top-left and bottom-right coordinates).
    """
    image_objects = []
    for (x_min, y_min, x_max, y_max, confidence) in detections:
        # For a more refined representation, you might compute all pixels or the connected component.
        related_pixels = [(x_min, y_min), (x_max, y_max)]
        obj = ImageObject(
            ImageObjectID=None,      # Let the constructor assign a new UUID if None
            ImageID=image_id,
            Class=class_name,
            Confidence=confidence,
            related_pixels=related_pixels,
            related_labels=[]        # Populate with associated Label objects if available
        )
        image_objects.append(obj)
    return image_objects

def main():
    """
    Complete workflow:
      1. Load model and image.
      2. Run sliding window detection.
      3. Draw bounding boxes and save output image.
      4. Convert detections to ImageObject instances.
      5. (Optional) Push ImageObjects to your database.
    """
    if len(sys.argv) < 4:
        print("Usage: python object_detection_and_db.py <model_fname> <in_image_fname> <image_id> [<out_image_fname>]")
        sys.exit(1)
    
    model_fname = sys.argv[1]
    in_fname = sys.argv[2]
    image_id = sys.argv[3]      # Unique identifier for the image, e.g. from your DB or project
    out_fname = sys.argv[4] if len(sys.argv) > 4 else os.path.splitext(in_fname)[0] + "_detection.png"
    class_name = "airplane"     # Update as needed

    # Load model and image
    load_model(model_fname)
    arr = preprocess_image(in_fname)
    
    # Perform sliding window detection
    detections = sliding_window_detection(arr, step=10, win=20, threshold=0.5)
    
    # Draw bounding boxes on the image and save output
    output = draw_bounding_boxes(arr, detections)
    Image.fromarray(output).save(out_fname)
    print(f"Output image with detections saved to {out_fname}.")

    # Convert detections to ImageObject instances (for unified DB storage)
    image_objects = convert_detections_to_image_objects(detections, image_id, class_name)
    
    # For demonstration, print the image object details.
    print("Detected ImageObjects:")
    for obj in image_objects:
        print(f"ImageObject - ImageID: {obj.ImageID}, Class: {obj.Class}, "
              f"Confidence: {obj.Confidence:.2f}, Related Pixels: {obj.related_pixels}")
    
    # If desired, use your ImageObjectDatabaseConnector to push these objects to the DB.
    # For example:
    # from ImageObjectDatabaseConnector import MYSQLImageObjectDatabaseConnector
    # db_connector = MYSQLImageObjectDatabaseConnector()
    # for obj in image_objects:
    #     db_connector.push_imageobject(obj)

if __name__ == "__main__":
    main()
