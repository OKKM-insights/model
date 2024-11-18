import tflearn
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression

def preprocess_images():
    """Set up real-time data preprocessing."""
    print("Setting up real-time data preprocessing...")
    img_prep = ImagePreprocessing()
    img_prep.add_featurewise_zero_center()
    img_prep.add_featurewise_stdnorm()
    print("Preprocessing configuration completed.")
    return img_prep

def augment_images():
    """Set up real-time data augmentation."""
    print("Setting up real-time data augmentation...")
    img_aug = ImageAugmentation()
    img_aug.add_random_flip_leftright()
    img_aug.add_random_flip_updown()
    img_aug.add_random_rotation(max_angle=25.)
    print("Augmentation configuration completed.")
    return img_aug

def build_cnn(input_shape, num_classes, learning_rate=0.001):
    """Builds the CNN model for PlanesNet."""
    print("Building the CNN model...")
    
    # Input layer with preprocessing and augmentation
    print("Initializing input layer...")
    network = input_data(
        shape=input_shape,
        data_preprocessing=preprocess_images(),
        data_augmentation=augment_images()
    )
    print("Input layer initialized.")

    # Convolutional layers
    print("Adding convolutional layers...")
    network = conv_2d(network, 32, 3, activation='relu')
    network = max_pool_2d(network, 2)
    network = conv_2d(network, 64, 3, activation='relu')
    network = conv_2d(network, 64, 3, activation='relu')
    network = max_pool_2d(network, 2)
    print("Convolutional layers added.")

    # Fully connected layers
    print("Adding fully connected layers...")
    network = fully_connected(network, 512, activation='relu')
    network = dropout(network, 0.5)
    network = fully_connected(network, num_classes, activation='softmax')
    print("Fully connected layers added.")

    # Regression layer for training
    print("Adding regression layer...")
    network = regression(
        network, 
        optimizer='adam',
        loss='categorical_crossentropy',
        learning_rate=learning_rate
    )
    print("Regression layer added. CNN model built successfully.")
    return network

def define_model():
    """Define the DNN model with tensorboard logging disabled."""
    print("Defining the DNN model...")
    cnn = build_cnn(input_shape=[None, 20, 20, 3], num_classes=2)
    model = tflearn.DNN(cnn, tensorboard_verbose=0)
    print("Model defined successfully. Ready to train!")
    return model

# Instantiate the model
print("Starting model setup...")
model = define_model()
print("Model setup complete.")
