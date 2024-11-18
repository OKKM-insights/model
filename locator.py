"""
Apply a trained machine learning model to an entire image scene using a sliding window.
"""

import sys
import os
import numpy as np
from PIL import Image
from scipy import ndimage
from cnn_model import model

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

def sliding_window_detection(arr, step=2, win=20):
    """Perform sliding window detection on the image array."""
    print("Initializing sliding window detection...")
    shape = arr.shape
    detections = np.zeros((shape[0], shape[1]), dtype='uint8')

    # Loop through pixel positions
    for i in range(0, shape[0] - win, step):
        print(f"Processing row {i} of {shape[0] - win}...")
        for j in range(0, shape[1] - win, step):
            # Extract sub-chip
            chip = arr[i:i+win, j:j+win, :]
            
            # Predict chip label
            prediction = model.predict_label([chip / 255.])[0][0]

            # Record positive detections
            if prediction == 1:
                detections[i + int(win / 2), j + int(win / 2)] = 1

    print("Sliding window detection completed.")
    return detections

def draw_bounding_boxes(output, detections, win):
    """Draw bounding boxes on the detected locations."""
    print("Processing detection locations...")
    dilation = ndimage.binary_dilation(detections, structure=np.ones((3, 3)))
    labels, n_labels = ndimage.label(dilation)
    center_mass = ndimage.center_of_mass(dilation, labels, np.arange(n_labels) + 1)

    # Ensure center_mass is iterable
    if type(center_mass) == tuple:
        center_mass = [center_mass]

    for i, j in center_mass:
        i, j = int(i - win / 2), int(j - win / 2)

        # Draw bounding box in output array
        output[i:i+win, j:j+2, 0:3] = [255, 0, 0]  # Left edge
        output[i:i+win, j+win-2:j+win, 0:3] = [255, 0, 0]  # Right edge
        output[i:i+2, j:j+win, 0:3] = [255, 0, 0]  # Top edge
        output[i+win-2:i+win, j:j+win, 0:3] = [255, 0, 0]  # Bottom edge

    print("Bounding boxes drawn.")
    return output

def save_output_image(output, out_fname):
    """Save the processed output image."""
    print(f"Saving output image to {out_fname}...")
    out_img = Image.fromarray(output)
    out_img.save(out_fname)
    print("Output image saved successfully.")

def detector(model_fname, in_fname, out_fname=None):
    """Perform a sliding window detector on an image."""
    if out_fname is None:
        out_fname = os.path.splitext(in_fname)[0] + '_detection.png'

    # Load model and input image
    load_model(model_fname)
    arr = preprocess_image(in_fname)
    output = np.copy(arr)

    # Perform sliding window detection
    detections = sliding_window_detection(arr)

    # Draw bounding boxes
    output = draw_bounding_boxes(output, detections, win=20)

    # Save the processed output image
    save_output_image(output, out_fname)

def main():
    """Main function to handle command-line inputs."""
    if len(sys.argv) < 3:
        print("Usage: python locator.py <model_fname> <in_fname> [<out_fname>]")
        sys.exit(1)

    model_fname = sys.argv[1]
    in_fname = sys.argv[2]
    out_fname = sys.argv[3] if len(sys.argv) > 3 else None

    detector(model_fname, in_fname, out_fname)

if __name__ == "__main__":
    print("Starting sliding window detection script...")
    main()
    print("Detection script completed.")
