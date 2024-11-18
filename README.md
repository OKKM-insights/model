# Object Detector: README

This README provides a comprehensive guide for setting up, training, and running the PlanesNet Detection System. Follow these steps carefully to ensure successful execution.

---

## System Requirements

Ensure that your Python version is compatible with the TensorFlow version required for this project. TensorFlow compatibility is a common issue; please verify and fix mismatches as necessary.

### Recommended Environment:
- **Python Version**: 3.x (preferably 3.8 or later)
- **TensorFlow Version**: Check `requirements.txt` for the specific version.

> *Tip*: Using a virtual environment is highly recommended to avoid conflicts with global Python packages.

---

## Setup Instructions

1. **Install Dependencies**:
    Install all required packages by running the following command:

    ```bash
    pip install -r requirements.txt
    ```

2. **Verify Installation**:
    Ensure that TensorFlow and other dependencies are correctly installed. If you encounter issues, double-check your Python and TensorFlow versions for compatibility.

---

## Required Files

Before proceeding, ensure you have the following files:

1. **`planesnet.json`**:
   - This file contains the data required to train the model.
   - Place it in the root directory of the project.

2. **Folder Structure**:
    ```plaintext
    Project_Directory/
    ├── AI_Models/
    │   └── model.tfl (Generated after training)
    ├── Landscape_Views/
    │   └── (Output images will be saved here)
    ├── planesnet.json (donwload this from the website I gave below)
    ├── train.py
    ├── cnn_model.py
    ├── locator.py
    ├── requirements.txt
    └── README.md
    ```

---

## Dataset Information

The latest version of `planesnet.json` is available through the [PlanesNet Kaggle page](https://www.kaggle.com/rhammell/planesnet). This page also provides detailed information about the dataset layout, including the format and contents. Be sure to download the dataset from there and place the `planesnet.json` file in the root directory of your project.

---

## Step-by-Step Usage

### 1. **Training the Model**
Run the `train.py` script to train the model using `planesnet.json`. This will generate a `.tfl` model file in the `AI_Models/` directory.

```bash
python train.py "planesnet.json" "AI_Models/model.tfl"
```

### 2. **Running the CNN Model Script**
Once the model is trained, execute the `cnn_model.py` script. This script uses the trained model to prepare for object detection tasks.

```bash
python cnn_model.py
```

### 3. **Executing the Locator Script**
To detect planes in landscape images, run the `locator.py` script. Pass the trained model and the image file as arguments.

Example command:

```bash
python locator.py "AI_Models/model.tfl" "Landscape_Views/landscape_A.png"
```

---

## Output Location

After successful execution, the output will be saved in the `Landscape_Views/` folder. The output file will include the processed image with detected planes marked.

---

## Troubleshooting

1. **TensorFlow Version Mismatch**:
    - Verify your Python version matches the TensorFlow requirements.
    - If using a virtual environment, recreate it to avoid conflicts:
      ```bash
      python -m venv venv
      source venv/bin/activate  # On Linux/Mac
      venv\Scripts\activate     # On Windows
      pip install -r requirements.txt
      ```

2. **Missing `planesnet.json`**:
    - Ensure the file is in the root directory.
    - Without this file, the model training will fail.

3. **Model File Not Found**:
    - Ensure the `train.py` script is executed successfully to generate the model file in the `AI_Models/` directory.

4. **Image File Issues**:
    - Check that the input image path passed to `locator.py` exists and is correctly formatted.

---