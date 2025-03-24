"""
Apply a trained machine learning model to an entire image scene using a sliding window.
"""

import sys
import os
import numpy as np
from PIL import Image
from scipy import ndimage
# First try the direct import, if that fails use a relative import
try:
    # This works when running as a module
    from model.cnn_model import model
except ImportError:
    # This works when running the script directly
    # Add the parent directory to sys.path
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    # Now try the import again
    from model.cnn_model import model
import time
from dotenv import load_dotenv
from sqlalchemy import text
import datetime
import argparse

# Add the project root directory to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

# Define default model path
DEFAULT_MODEL_PATH = os.path.join("model", "model.tfl")

# Now we can import from backend
from backend.services.LabelDatabaseConnector import MYSQLLabelDatabaseConnector
from backend.services.DataTypes import Label, ImageClassMeasure
from backend.services.ImageClassMeasureDatabaseConnector import MYSQLImageClassMeasureDatabaseConnector
from backend.services.ObjectExtractionService import ObjectExtractionService
from backend.services.LabellerDatabaseConnector import MYSQLLabellerDatabaseConnector

# Load environment variables from backend/.env
backend_env_path = os.path.join(project_root, 'backend', '.env')
print(f"Loading environment variables from: {backend_env_path}")
load_dotenv(backend_env_path)

# Debug: Print environment variables (except password)
print("\nDatabase Connection Info:")
print(f"Host: {os.getenv('DB_HOSTNAME')}")
print(f"User: {os.getenv('DB_USER')}")
print(f"Database: {os.getenv('DB_NAME')}")
print(f"Password: {'*' * len(os.getenv('DB_PASSWORD', ''))}")

def load_model(model_fname):
    """Load the trained machine learning model."""
    print(f"Loading model from {model_fname}...")
    model.load(model_fname)
    print("Model loaded successfully.")

def preprocess_image(in_fname):
    """Load and preprocess the input image."""
    print(f"Reading input image from {in_fname}...")
    im = Image.open(in_fname)
    arr = np.array(im)[:, :, 0:3]  # Extract RGB channels
    print(f"Image loaded with shape {arr.shape}.")
    return arr

def sliding_window_detection(arr, step=10, win=20):
    """Perform sliding window detection on the image array.
    
    Args:
        arr: Input image array
        step: Step size for sliding window (default: 10 pixels)
        win: Window size (default: 20 pixels)
    """
    print("Initializing sliding window detection...")
    shape = arr.shape
    detections = np.zeros((shape[0], shape[1]), dtype='float32')
    confidences = np.zeros((shape[0], shape[1]), dtype='float32')

    # Loop through pixel positions
    for i in range(0, shape[0] - win, step):
        print(f"Processing row {i} of {shape[0] - win}...")
        for j in range(0, shape[1] - win, step):
            # Extract sub-chip
            chip = arr[i:i+win, j:j+win, :]
            
            # Get prediction and confidence
            prediction = model.predict([chip / 255.])[0]
            confidence = prediction[1]  # Probability of being an airplane
            
            # Record detections and confidences
            detections[i + int(win / 2), j + int(win / 2)] = 1 if confidence > 0.5 else 0
            confidences[i + int(win / 2), j + int(win / 2)] = confidence

    print("Sliding window detection completed.")
    return detections, confidences

def draw_bounding_boxes(output, detections, confidences, win, confidence_threshold=0.5):
    """Draw bounding boxes on the detected locations."""
    print("Processing detection locations...")
    dilation = ndimage.binary_dilation(detections, structure=np.ones((3, 3)))
    labels, n_labels = ndimage.label(dilation)
    center_mass = ndimage.center_of_mass(dilation, labels, np.arange(n_labels) + 1)

    # Ensure center_mass is iterable
    if type(center_mass) == tuple:
        center_mass = [center_mass]

    bounding_boxes = []
    for i, j in center_mass:
        i, j = int(i - win / 2), int(j - win / 2)
        
        # Get confidence score for this detection
        confidence = confidences[int(i + win/2), int(j + win/2)]
        
        # Only include detections above confidence threshold
        if confidence >= confidence_threshold:
            # Draw bounding box in output array
            output[i:i+win, j:j+2, 0:3] = [255, 0, 0]  # Left edge
            output[i:i+win, j+win-2:j+win, 0:3] = [255, 0, 0]  # Right edge
            output[i:i+2, j:j+win, 0:3] = [255, 0, 0]  # Top edge
            output[i+win-2:i+win, j:j+win, 0:3] = [255, 0, 0]  # Bottom edge
            
            # Add bounding box data
            bounding_boxes.append({
                'x1': j,
                'y1': i,
                'x2': j + win,
                'y2': i + win,
                'confidence': float(confidence)
            })

    print("Bounding boxes drawn.")
    return output, bounding_boxes

def save_output_image(output, out_fname):
    """Save the processed output image."""
    print(f"Saving output image to {out_fname}...")
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(out_fname), exist_ok=True)
    out_img = Image.fromarray(output)
    out_img.save(out_fname)
    print("Output image saved successfully.")

def update_confidence_scores(image_array, bounding_boxes, image_id):
    """Update confidence scores for detected objects using CNN model.
    
    Args:
        image_array: NumPy array of the original image
        bounding_boxes: List of bounding boxes with detection information
        image_id: ID of the image in the database
    """
    try:
        print("Saving detected objects to ImageObjects table...")
        
        # Import the ImageObject database connector
        from backend.services.ImageObjectDatabaseConnector import MYSQLImageObjectDatabaseConnector
        from DataTypes import ImageObject
        import uuid
        
        # Create a connector
        obj_connector = MYSQLImageObjectDatabaseConnector()
        
        # If bounding_boxes is None or empty, just return
        if bounding_boxes is None or len(bounding_boxes) == 0:
            print("No bounding boxes to save as ImageObjects")
            return
        
        # For each bounding box
        for box in bounding_boxes:
            x1, y1, x2, y2 = int(box['x1']), int(box['y1']), int(box['x2']), int(box['y2'])
            
            # Extract region of interest
            roi = image_array[y1:y2, x1:x2]
            
            # Only process valid ROIs
            if roi.size > 0:
                # Generate a unique ID for this object
                object_id = str(uuid.uuid4())
                
                # Get all pixel coordinates within the bounding box
                pixels = []
                for y in range(y1, y2):
                    for x in range(x1, x2):
                        pixels.append([x, y])
                
                # Create ImageObject instance
                image_object = ImageObject(
                    ImageObjectID=object_id,
                    ImageID=str(image_id),
                    Class='airplane',  # Update this if you have multiple classes
                    Confidence=box.get('confidence', 0.95),  # Use detection confidence or default
                    related_pixels=pixels,
                    related_labels=[]  # No labels initially
                )
                
                # Save to ImageObjects table
                obj_connector.push_imageobject(image_object)
                
                print(f"Saved ImageObject for detection at ({x1},{y1},{x2},{y2}) with confidence {image_object.Confidence}")
            else:
                print(f"Warning: Empty ROI at ({x1},{y1},{x2},{y2}), skipping object creation")
                
        print("ImageObjects creation completed")
        
    except Exception as e:
        print(f"Error creating ImageObjects: {e}")
        import traceback
        traceback.print_exc()

def save_detections_to_db(input_image, bounding_boxes, project_id):
    """Save the detection results to the database."""
    try:
        # Create a database connector
        from backend.services.LabelDatabaseConnector import MYSQLLabelDatabaseConnector
        db_connector = MYSQLLabelDatabaseConnector()
        
        # Push the input image to get an image ID
        image_id = db_connector.push_image(input_image, project_id)
        print(f"Created image entry with ID: {image_id}")
        
        # If no image_id was returned, return None
        if not image_id:
            print("Error: Failed to save image to database")
            return None
        
        # If bounding_boxes is None or empty, just return the image_id
        if bounding_boxes is None or len(bounding_boxes) == 0:
            print("No bounding boxes to save")
            return image_id
        
        # Save each bounding box as a label
        for box in bounding_boxes:
            # Create a label object
            label = {
                "image_id": image_id,
                "x1": float(box['x1']),
                "y1": float(box['y1']),
                "x2": float(box['x2']),
                "y2": float(box['y2']),
                "confidence": 1.0  # Default confidence score
            }
            
            # Push the label to the database
            db_connector.push_label(label)
        
        return image_id
        
    except Exception as e:
        print(f"Error during save_detections_to_db: {e}")
        import traceback
        traceback.print_exc()
        return None

def detector(model_fname, in_fname, out_fname=None):
    """Perform a sliding window detector on an image."""
    if out_fname is None:
        out_fname = os.path.splitext(in_fname)[0] + '_detection.png'

    # Load model and input image
    load_model(model_fname)
    arr = preprocess_image(in_fname)
    output = np.copy(arr)

    # Perform sliding window detection
    detections, confidences = sliding_window_detection(arr)

    # Draw bounding boxes and get bounding box data
    output, bounding_boxes = draw_bounding_boxes(output, detections, confidences, win=20)

    # Save the processed output image
    save_output_image(output, out_fname)
    
    # Print bounding box data
    print("\nDetected Bounding Boxes:")
    print("-" * 50)
    for i, box in enumerate(bounding_boxes, 1):
        print(f"Box {i}:")
        print(f"  Coordinates: ({box['x1']}, {box['y1']}) to ({box['x2']}, {box['y2']})")
        print(f"  Confidence: {box['confidence']:.3f}")
        print("-" * 50)
    print(f"\nTotal detections: {len(bounding_boxes)}")
    
    # Save detections to database using the input image path
    project_id = 5  # Use the project ID from test_upload.py
    image_id = save_detections_to_db(in_fname, bounding_boxes, project_id)
    
    # Update confidence scores
    if image_id:
        update_confidence_scores(np.array(Image.open(in_fname)), bounding_boxes, image_id)
    
    print("Detection script completed.")

def main():
    """Run the airplane detector on an input image."""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run object detection on an image.')
    parser.add_argument('input_image', type=str, help='Path to input image')
    parser.add_argument('--model', type=str, default=DEFAULT_MODEL_PATH, help='Path to model file')
    parser.add_argument('--output', type=str, default=None, help='Path to output image')
    parser.add_argument('--project_id', type=int, default=5, help='Project ID for database')
    
    # Check if we're getting multiple positional arguments (old format)
    if len(sys.argv) > 2 and not sys.argv[2].startswith('--'):
        # Handle old format: script.py model_path input_image [output_image]
        model_path = sys.argv[1]
        input_image = sys.argv[2]
        output_path = sys.argv[3] if len(sys.argv) > 3 else None
        project_id = 5  # Default project ID
    else:
        # New format with named arguments
        args = parser.parse_args()
        input_image = args.input_image
        model_path = args.model
        output_path = args.output
        project_id = args.project_id
    
    print(f"Loading model from {model_path}...")
    print(f"Reading input image from {input_image}...")
    
    # Run the detector
    bounding_boxes = detector(model_path, input_image, output_path)
    
    # Save detections to database
    image_id = save_detections_to_db(input_image, bounding_boxes, project_id)
    
    # If we got a valid image_id, update confidence scores
    if image_id:
        # Load the image as numpy array for update_confidence_scores
        image_array = np.array(Image.open(input_image))
        update_confidence_scores(image_array, bounding_boxes, image_id)
    
    return 0

if __name__ == "__main__":
    main()
