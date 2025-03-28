#!/usr/bin/env python3
import sys
import os
import uuid
import datetime
import numpy as np
from PIL import Image, ImageDraw, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from pathlib import Path

# Ensure the project root is in the PYTHONPATH
project_root = str(Path(__file__).parent.parent)
sys.path.append(project_root)

# Import DB connectors and service classes
from backend.services.ProjectDatabaseConnector import MYSQLProjectDatabaseConnector
from backend.services.LabelDatabaseConnector import MYSQLLabelDatabaseConnector
from backend.services.LabellerDatabaseConnector import MYSQLLabellerDatabaseConnector
from backend.services.ImageObjectDatabaseConnector import MYSQLImageObjectDatabaseConnector
from backend.services.ObjectExtractionService import ObjectExtractionService
from backend.services.ImageClassMeasureDatabaseConnector import MYSQLImageClassMeasureDatabaseConnector
from backend.services.DataTypes import Labeller, Label, Image as DBImage, ImageObject
from backend.services.ObjectExtractionManager import ObjectExtractionManager
from cnn_model import model  # Your trained CNN model module

def load_model(model_fname: str) -> None:
    print(f"Loading model from {model_fname}...")
    model.load(model_fname)
    print("Model loaded successfully.")

def ensure_image_in_db(db_image: DBImage, project_db: MYSQLProjectDatabaseConnector):
    """
    Ensure that the image record (by db_image.ImageID) exists in the Images table.
    If not, insert it.
    If project_db does not support get_images() or push_image(), this function will warn and skip.
    """
    if not hasattr(project_db, "get_images"):
        print("WARNING: ProjectDatabaseConnector has no get_images() method. Skipping image check.")
        return
    query = f"SELECT * FROM Images WHERE id = {db_image.ImageID};"
    images = project_db.get_images(query)
    if not images:
        print(f"Image with ID {db_image.ImageID} not found in Images table. Inserting now.")
        if hasattr(project_db, "push_image"):
            project_db.push_image(db_image)
        else:
            print("WARNING: ProjectDatabaseConnector has no push_image() method.")
    else:
        print(f"Image with ID {db_image.ImageID} already exists in Images table.")

def preprocess_image_from_db(db_image: DBImage) -> np.ndarray:
    """
    Convert the DB image (a PIL Image stored in db_image.image_data) to a numpy array.
    """
    im = db_image.image_data.convert("RGB")
    arr = np.array(im)
    print(f"Image loaded with shape {arr.shape}.")
    return arr

def sliding_window_detection(arr: np.ndarray, step: int = 2, win: int = 20, threshold: float = 0.5) -> list:
    """
    Run sliding window detection over the image.
    Returns a list of detections, where each detection is a tuple:
      (x_min, y_min, x_max, y_max, confidence).
    """
    detections = []
    height, width, _ = arr.shape
    print("Starting sliding window detection...")
    for i in range(0, height - win, step):
        for j in range(0, width - win, step):
            chip = arr[i:i+win, j:j+win, :]
            chip_normalized = chip / 255.0
            prediction = model.predict(chip_normalized[np.newaxis, ...])[0]
            # We assume the model returns a probability vector [prob_background, prob_airplane]
            confidence = prediction[1]
            if confidence >= threshold:
                detections.append((j, i, j+win, i+win, confidence))
    print(f"Sliding window detection completed with {len(detections)} detections.")
    return detections

def consensus_bounding_box(detections: list):
    """
    Compute the union (consensus) bounding box of all detection boxes.
    Returns (x_min, y_min, x_max, y_max) or None if no detections.
    """
    if not detections:
        return None
    x_min = min(det[0] for det in detections)
    y_min = min(det[1] for det in detections)
    x_max = max(det[2] for det in detections)
    y_max = max(det[3] for det in detections)
    return (x_min, y_min, x_max, y_max)

def draw_bounding_box_on_image(arr: np.ndarray, bbox, color: tuple = (0, 255, 0), thickness: int = 3) -> np.ndarray:
    """
    Draw the bounding box (bbox) on the image using PIL's ImageDraw.
    Returns the resulting image as a numpy array.
    """
    if bbox is None:
        return arr
    x_min, y_min, x_max, y_max = bbox
    im = Image.fromarray(arr)
    draw = ImageDraw.Draw(im)
    for t in range(thickness):
        draw.rectangle([x_min+t, y_min+t, x_max-t, y_max-t], outline=color)
    return np.array(im)

def push_detections_as_labels(detections: list, db_image: DBImage, class_name: str, label_db: MYSQLLabelDatabaseConnector, batch_size=100):
    """
    For each detection, create a new Label (with a new LabelID) and push it to the DB in batches.
    This function assumes that label_db has a method push_labels_batch(labels_batch: list).
    """
    batch = []
    for (x_min, y_min, x_max, y_max, confidence) in detections:
        new_label = {
            "LabelID": str(uuid.uuid4()),
            "LabellerID": 0,              # Use a valid LabellerID (as a string)
            "ImageID": 1,    # Must match an existing record in Images!
            "Class": class_name,
            "top_left_x": x_min,
            "top_left_y": y_min,
            "bot_right_x": x_max,
            "bot_right_y": y_max,
            "offset_x": 300,                # Adjust as needed
            "offset_y": 300,                # Adjust as needed
            "creation_time": datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "origImageID": db_image.ImageID
        }
        batch.append(new_label)
        if len(batch) >= batch_size:
            try:
                label_db.push_labels_batch(batch)
                print(f"Pushed batch of {len(batch)} labels.")
            except Exception as e:
                print("Error pushing label batch:", e)
            batch = []
    if batch:
        try:
            label_db.push_labels_batch(batch)
            print(f"Pushed final batch of {len(batch)} labels.")
        except Exception as e:
            print("Error pushing final label batch:", e)

def main():
    # Prompt for project ID and class to extract
    project_id = input("Enter project ID to process: ").strip()
    class_name = input("Enter class to extract (e.g., 'plane'): ").strip()
    
    # Initialize DB connectors and service objects
    project_db = MYSQLProjectDatabaseConnector()
    label_db = MYSQLLabelDatabaseConnector()  # Must implement push_labels_batch()
    labeller_db = MYSQLLabellerDatabaseConnector()
    imageobject_db = MYSQLImageObjectDatabaseConnector()
    icm_db = MYSQLImageClassMeasureDatabaseConnector()
    object_service = ObjectExtractionService(icm_db, labeller_db)
    manager = ObjectExtractionManager(project_db, label_db, labeller_db, imageobject_db, object_service)
    
    # Retrieve the project and choose the first image
    projects = project_db.get_projects(f"SELECT * FROM my_image_db.Projects WHERE ProjectID = {project_id};")
    if not projects:
        print(f"No project found with ID {project_id}")
        return
    project = projects[0]
    if not project.images:
        print("No images in the project.")
        return
    db_image = project.images[0]
    print(f"Processing image with ImageID: {db_image.ImageID}")
    
    # Ensure the image record exists in the Images table (if supported by project_db)
    ensure_image_in_db(db_image, project_db)
    
    # Preprocess the image from the DB into a numpy array
    arr = preprocess_image_from_db(db_image)
    
    # Run sliding window detection on the image
    detections = sliding_window_detection(arr, step=10, win=20, threshold=0.5)
    
    # Compute the consensus bounding box (union of all detections)
    bbox = consensus_bounding_box(detections)
    print("Consensus Bounding Box:", bbox)
    
    # Draw the consensus bounding box on the image and save the result
    arr_with_bbox = draw_bounding_box_on_image(arr, bbox, color=(0, 255, 0), thickness=3)
    out_fname = f"extracted_{db_image.ImageID}.png"
    Image.fromarray(arr_with_bbox).save(out_fname)
    print(f"Output image with consensus bounding box saved as {out_fname}.")
    
    # Push each detection as a new Label in batches
    push_detections_as_labels(detections, db_image, class_name, label_db, batch_size=100)
    
    print("Object extraction completed.")

if __name__ == "__main__":
    model_fname = input("Enter model filename: ").strip()
    load_model(model_fname)
    main()
