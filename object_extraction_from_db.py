import sys
from pathlib import Path
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Ensure the project root is in the PYTHONPATH
project_root = str(Path(__file__).parent.parent)
sys.path.append(project_root)

#!/usr/bin/env python3
from backend.services.ProjectDatabaseConnector import MYSQLProjectDatabaseConnector
from backend.services.LabelDatabaseConnector import MYSQLLabelDatabaseConnector
from backend.services.LabellerDatabaseConnector import MYSQLLabellerDatabaseConnector
from backend.services.ImageObjectDatabaseConnector import MYSQLImageObjectDatabaseConnector
from backend.services.ObjectExtractionService import ObjectExtractionService
from backend.services.ImageClassMeasureDatabaseConnector import MYSQLImageClassMeasureDatabaseConnector
from backend.services.DataTypes import Labeller, Label, Image as DBImage
from backend.services.ObjectExtractionManager import ObjectExtractionManager

def _simple_consensus_bounding_box(labels, image_shape, consensus_threshold=0.5):
    """
    Computes a simple consensus bounding box (the union of all label boxes).
    'image_shape' is (height, width) but is not used in this simple example.
    """
    if not labels:
        return None
    x_min = min(label.top_left_x for label in labels)
    y_min = min(label.top_left_y for label in labels)
    x_max = max(label.bot_right_x for label in labels)
    y_max = max(label.bot_right_y for label in labels)
    return (x_min, y_min, x_max, y_max)

def get_consensus_bbox(image, labels, threshold=0.5):
    """
    For demonstration, returns the union bounding box of all label boxes.
    """
    # Here we ignore the image beyond using its dimensions.
    image_shape = (image.image_data.height, image.image_data.width)
    return _simple_consensus_bounding_box(labels, image_shape, consensus_threshold=threshold)

def main():
    # Prompt for project ID and class (e.g., 'plane')
    project_id = input("Enter project ID to process: ").strip()
    class_name = input("Enter class to extract (e.g., 'plane'): ").strip()
    
    # Initialize connectors
    project_db = MYSQLProjectDatabaseConnector()
    label_db = MYSQLLabelDatabaseConnector()
    labeller_db = MYSQLLabellerDatabaseConnector()
    imageobject_db = MYSQLImageObjectDatabaseConnector()
    icm_db = MYSQLImageClassMeasureDatabaseConnector()
    
    # Create the ObjectExtractionService and Manager (if needed for more complex operations)
    object_service = ObjectExtractionService(icm_db, labeller_db)
    manager = ObjectExtractionManager(project_db, label_db, labeller_db, imageobject_db, object_service)
    
    # For this simple test, get the project and choose the first image.
    projects = project_db.get_projects(f"SELECT * FROM my_image_db.Projects WHERE ProjectID = {project_id};")
    if not projects:
        print(f"No project found with ID {project_id}")
        return
    project = projects[0]
    if not project.images:
        print("No images in the project.")
        return
    image = project.images[0]
    
    # Query labels for the chosen image and class.
    query_labels = f"SELECT * FROM my_image_db.Labels WHERE OrigImageID = '{image.ImageID}' AND Class = '{class_name}';"
    labels = label_db.get_labels(query_labels)
    print(f"Found {len(labels)} labels for image {image.ImageID} and class '{class_name}'.")
    
'''

     # This call will pull the specified project from the DB,
    # iterate through its images, run the object extraction logic,
    # and push the resulting ImageObjects back into the DB.

    #IMPORTANT - Commented this out as this is pixel wise, for testing purposes following a faster approach using simple bounding.

    # manager.get_objects(project_id, class_name)

     #calling the simple implementation
    image_shape = (image.image_data.height, image.image_data.width)
    bbox = _simple_consensus_bounding_box(labels, image_shape, consensus_threshold=0.5)
    # In your main logic, ensure you pass the correct parameters:
    # For example, if 'image' is loaded and 'labels' is your list of label objects:
    bbox = get_consensus_bbox(image, labels, threshold=0.5)

'''

    # Calculate consensus bounding boxes using two simple implementations.
    bbox_simple = _simple_consensus_bounding_box(labels, (image.image_data.height, image.image_data.width), consensus_threshold=0.5)
    bbox = get_consensus_bbox(image, labels, threshold=0.5)
    
    print("Simple Consensus Bounding Box:", bbox_simple)
    print("Consensus Bounding Box:", bbox)
    print("Object extraction completed.")

if __name__ == "__main__":
    main()
