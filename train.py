"""
Train and export a machine learning model using the PlanesNet dataset.
"""

import sys
import json
import numpy as np
from tflearn.data_utils import to_categorical
from cnn_model import model

def load_dataset(fname):
    """Load and preprocess the PlanesNet dataset.

    Args:
        fname (str): Path to the PlanesNet JSON dataset.

    Returns:
        tuple: Preprocessed image data (X) and labels (Y).
    """
    print(f"Loading dataset from {fname}...")
    with open(fname, 'r') as f:
        planesnet = json.load(f)

    print("Preprocessing image data and labels...")
    X = np.array(planesnet['data']) / 255.0  # Normalize image data
    X = X.reshape([-1, 3, 20, 20]).transpose([0, 2, 3, 1])  # Reshape and transpose for CNN input
    Y = np.array(planesnet['labels'])  # Convert labels to categorical
    Y = to_categorical(Y, 2)
    print(f"Dataset loaded and preprocessed: {X.shape[0]} samples.")
    return X, Y

def train_model(X, Y, n_epoch=50, batch_size=128, validation_split=0.2):
    """Train the CNN model on the PlanesNet dataset.

    Args:
        X (np.ndarray): Preprocessed image data.
        Y (np.ndarray): Preprocessed labels.
        n_epoch (int): Number of training epochs.
        batch_size (int): Batch size for training.
        validation_split (float): Proportion of the dataset for validation.

    Returns:
        None
    """
    print("Starting model training...")
    model.fit(
        X, Y, 
        n_epoch=n_epoch, 
        shuffle=True, 
        validation_set=validation_split,
        show_metric=True, 
        batch_size=batch_size, 
        run_id='planesnet'
    )
    print("Model training completed.")

def save_model(out_fname):
    """Save the trained model to a file.

    Args:
        out_fname (str): Path to save the trained TensorFlow model.

    Returns:
        None
    """
    print(f"Saving trained model to {out_fname}...")
    model.save(out_fname)
    print("Model saved successfully.")

def train(fname, out_fname):
    """Train and save CNN model on the PlanesNet dataset.

    Args:
        fname (str): Path to PlanesNet JSON dataset.
        out_fname (str): Path to output TensorFlow model file (.tfl).

    Returns:
        None
    """
    # Load and preprocess the dataset
    X, Y = load_dataset(fname)

    # Train the model
    train_model(X, Y)

    # Save the trained model
    save_model(out_fname)

def main():
    """Main function to handle command-line inputs."""
    if len(sys.argv) != 3:
        print("Usage: python train_model.py <dataset_fname> <output_model_fname>")
        sys.exit(1)

    dataset_fname = sys.argv[1]
    output_model_fname = sys.argv[2]

    print("Starting training script...")
    train(dataset_fname, output_model_fname)
    print("Training script completed successfully.")

if __name__ == "__main__":
    main()
