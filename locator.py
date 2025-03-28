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

# Import DB connectors and services
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

def preprocess_image_from_db(db_image: DBImage) -> np.ndarray:
    """
    Convert the DB image (a PIL Image stored in db_image.image_data) to a numpy array.
    """
    im = db_image.image_data.convert("RGB")
    arr = np.array(im)
    print(f"Image loaded with shape {arr.shape}.")
    return arr

def sliding_window_detection(arr: np.ndarray, step: int = 20, win: int = 20, threshold: float = 0.5) -> list:
    """
    Run sliding window detection over the image.
    Returns a list of detections: each a tuple (x_min, y_min, x_max, y_max, confidence).
    """
    detections = []
    height, width, _ = arr.shape
    print("Starting sliding window detection...")
    for i in range(0, height - win, step):
        for j in range(0, width - win, step):
            chip = arr[i:i+win, j:j+win, :]
            chip_normalized = chip / 255.0
            prediction = model.predict(chip_normalized[np.newaxis, ...])[0]
            confidence = prediction[1]  # Assuming [prob_background, prob_airplane]
            if confidence >= threshold:
                detections.append((j, i, j + win, i + win, confidence))
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
    Draw the bounding box (bbox) on the image.
    Uses PIL's ImageDraw. Returns a numpy array.
    """
    if bbox is None:
        return arr
    x_min, y_min, x_max, y_max = bbox
    im = Image.fromarray(arr)
    draw = ImageDraw.Draw(im)
    for t in range(thickness):
        draw.rectangle([x_min + t, y_min + t, x_max - t, y_max - t], outline=color)
    return np.array(im)

def push_detections_as_labels(detections: list, db_image: DBImage, class_name: str, label_db) -> None:
    """
    For each detection, create a new Label with a new LabelID and push it to the DB.
    You can adjust the LabellerID, offsets, and Class fields as needed.
    """
    for (x_min, y_min, x_max, y_max, confidence) in detections:
        new_label = Label(
            LabelID=str(uuid.uuid4()),
            LabellerID=0,              # Use an appropriate LabellerID (as string) that exists in your DB
            ImageID=1,    # Ensure this ImageID exists in your Images table
            Class=class_name,
            top_left_x=x_min,
            top_left_y=y_min,
            bot_right_x=x_max,
            bot_right_y=y_max,
            offset_x=300,                # Adjust as needed
            offset_y=300,                # Adjust as needed
            creation_time=datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            origImageID=db_image.ImageID
        )
        try:
            label_db.push_label(new_label)
            print(f"Label pushed to DB with LabelID={new_label.LabelID}.")
        except Exception as e:
            print("Error pushing label:", e)

def main():
    # Prompt for project ID, class, and model filename
    project_id = input("Enter project ID to process: ").strip()
    class_name = input("Enter class to extract (e.g., 'plane'): ").strip()
    
    # Initialize DB connectors and services
    project_db = MYSQLProjectDatabaseConnector()
    label_db = MYSQLLabelDatabaseConnector()
    labeller_db = MYSQLLabellerDatabaseConnector()
    imageobject_db = MYSQLImageObjectDatabaseConnector()
    icm_db = MYSQLImageClassMeasureDatabaseConnector()

    object_service = ObjectExtractionService(icm_db, labeller_db)
    manager = ObjectExtractionManager(project_db, label_db, labeller_db, imageobject_db, object_service)
    
    # Retrieve the project and select the first image
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

    # Convert the DB image to a numpy array
    arr = preprocess_image_from_db(db_image)

    # Run sliding window detection on the image
    detections = sliding_window_detection(arr, step=10, win=20, threshold=0.5)

    # Compute the consensus bounding box
    bbox = consensus_bounding_box(detections)
    print("Consensus Bounding Box:", bbox)

    # Draw the consensus bounding box on the image
    arr_with_bbox = draw_bounding_box_on_image(arr, bbox, color=(0, 255, 0), thickness=3)
    out_fname = f"extracted_{db_image.ImageID}.png"
    Image.fromarray(arr_with_bbox).save(out_fname)
    print(f"Output image with consensus bounding box saved as {out_fname}.")

    # Push all detections as new label entries in the DB
    push_detections_as_labels(detections, db_image, class_name, label_db)

    print("Object extraction completed.")

if __name__ == "__main__":
    model_fname = input("Enter model filename: ").strip()
    load_model(model_fname)
    main()
