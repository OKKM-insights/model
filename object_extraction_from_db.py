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
    Ensure that the image record exists in the Images table.
    If not, insert it.
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

def compute_iou(box1: tuple, box2: tuple) -> float:
    """
    Compute Intersection over Union (IoU) between two boxes.
    Each box is defined as (x_min, y_min, x_max, y_max).
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = area1 + area2 - inter_area
    return inter_area / union_area if union_area > 0 else 0

def non_max_suppression(detections: list, iou_threshold: float = 0.3) -> list:
    """
    Filter overlapping detections using Non-Maximum Suppression.
    Each detection is (x_min, y_min, x_max, y_max, confidence).
    """
    if not detections:
        return []
    # Sort detections by confidence (highest first)
    detections = sorted(detections, key=lambda x: x[4], reverse=True)
    final_detections = []
    while detections:
        best = detections.pop(0)
        final_detections.append(best)
        detections = [d for d in detections if compute_iou(best, d) < iou_threshold]
    return final_detections

def sliding_window_detection_multiscale(arr: np.ndarray, scales: list = [1.0, 0.8, 0.6, 0.4],
                                          step: int = 10, win: int = 20, threshold: float = 0.5) -> list:
    """
    Run sliding window detection over multiple scales.
    Returns a list of detections: each detection is (x_min, y_min, x_max, y_max, confidence).
    """
    detections = []
    orig_h, orig_w, _ = arr.shape
    print("Starting multi-scale sliding window detection...")
    for scale in scales:
        new_w = int(orig_w * scale)
        new_h = int(orig_h * scale)
        scaled_img = np.array(Image.fromarray(arr).resize((new_w, new_h), Image.ANTIALIAS))
        print(f"Processing scale {scale:.2f} with image size ({new_w}, {new_h})")
        for i in range(0, new_h - win, step):
            for j in range(0, new_w - win, step):
                chip = scaled_img[i:i+win, j:j+win, :]
                chip_normalized = chip / 255.0
                prediction = model.predict(chip_normalized[np.newaxis, ...])[0]
                confidence = prediction[1]  # Assuming [prob_background, prob_airplane]
                if confidence >= threshold:
                    # Map detection coordinates back to the original image scale
                    x_min = int(j / scale)
                    y_min = int(i / scale)
                    x_max = int((j + win) / scale)
                    y_max = int((i + win) / scale)
                    detections.append((x_min, y_min, x_max, y_max, confidence))
    print(f"Raw multi-scale detections: {len(detections)}")
    return detections

def draw_bounding_boxes_on_image(arr: np.ndarray, detections: list,
                                 color: tuple = (0, 255, 0), thickness: int = 3) -> np.ndarray:
    """
    Draw all bounding boxes on the image.
    Each detection is (x_min, y_min, x_max, y_max, confidence).
    """
    im = Image.fromarray(arr)
    draw = ImageDraw.Draw(im)
    for (x_min, y_min, x_max, y_max, conf) in detections:
        for t in range(thickness):
            draw.rectangle([x_min+t, y_min+t, x_max-t, y_max-t], outline=color)
    return np.array(im)

def push_detections_as_labels(detections: list, db_image: DBImage, class_name: str,
                              label_db: MYSQLLabelDatabaseConnector, batch_size=100):
    """
    For each detection, create a new Label (with a unique LabelID) and push it to the DB in batches.
    """
    batch = []
    for (x_min, y_min, x_max, y_max, confidence) in detections:
        new_label = {
            "LabelID": str(uuid.uuid4()),
            "LabellerID": 0,              # Use a valid LabellerID
            "ImageID": 1,                 # Must match an existing record in Images
            "Class": class_name,
            "top_left_x": x_min,
            "top_left_y": y_min,
            "bot_right_x": x_max,
            "bot_right_y": y_max,
            "offset_x": 0,              # Adjust as needed
            "offset_y": 0,              # Adjust as needed
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
    
    # Ensure the image record exists in the Images table (if supported)
    ensure_image_in_db(db_image, project_db)
    
    # Preprocess the image from the DB into a numpy array
    arr = preprocess_image_from_db(db_image)
    
    # Run multi-scale sliding window detection on the image
    raw_detections = sliding_window_detection_multiscale(
        arr, scales=[1.0, 0.8, 0.6, 0.4], step=10, win=20, threshold=0.5)
    
    # Apply Non-Maximum Suppression (NMS) to reduce overlapping detections
    final_detections = non_max_suppression(raw_detections, iou_threshold=0.3)
    print(f"Final detections after NMS: {len(final_detections)}")
    
    # Optionally, compute a consensus bounding box (for reference)
    if final_detections:
        x_min = min(det[0] for det in final_detections)
        y_min = min(det[1] for det in final_detections)
        x_max = max(det[2] for det in final_detections)
        y_max = max(det[3] for det in final_detections)
        consensus_box = (x_min, y_min, x_max, y_max)
        print("Consensus Bounding Box:", consensus_box)
    else:
        consensus_box = None
    
    # Draw all final bounding boxes on the image and save the result
    arr_with_boxes = draw_bounding_boxes_on_image(arr, final_detections, color=(0, 255, 0), thickness=3)
    out_fname = f"extracted_{db_image.ImageID}.png"
    Image.fromarray(arr_with_boxes).save(out_fname)
    print(f"Output image with bounding boxes saved as {out_fname}.")
    
    # Push each detection as a new label in batches
    push_detections_as_labels(final_detections, db_image, class_name, label_db, batch_size=100)
    
    print("Object extraction completed.")

if __name__ == "__main__":
    model_fname = input("Enter model filename: ").strip()
    load_model(model_fname)
    main()
