import os
import numpy as np
import cv2
import yaml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load configuration from config.yaml
def load_config():
    with open('config/config.yaml', 'r') as config_file:
        config = yaml.safe_load(config_file)
    return config

def load_raw_data(config):
    """
    Loads raw image data from directories for each drone type and scenario.
    """
    raw_data_dir = config['paths']['raw_data_dir']

    # Initialize lists to store image data and corresponding labels
    images = []
    labels = []

    # Check if the raw data directory exists
    if not os.path.exists(raw_data_dir):
        raise FileNotFoundError(f"The directory {raw_data_dir} does not exist. Please check the folder structure.")
    
    # Iterate over all drone types
    for drone in os.listdir(raw_data_dir):
        drone_path = os.path.join(raw_data_dir, drone)
        
        if os.path.isdir(drone_path):
            # Iterate over the images subfolder
            images_folder_path = os.path.join(drone_path, 'images')
            
            if os.path.isdir(images_folder_path):
                for scenario in os.listdir(images_folder_path):
                    scenario_path = os.path.join(images_folder_path, scenario)

                    if os.path.isdir(scenario_path):
                        for image_file in os.listdir(scenario_path):
                            image_path = os.path.join(scenario_path, image_file)

                            if image_path.endswith('.jpg') or image_path.endswith('.png'):  # Assuming images are .jpg or .png
                                # Read the image
                                image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Grayscale
                                image = cv2.resize(image, (config['image']['width'], config['image']['height']))  # Resize to config dimensions
                                images.append(image)
                                labels.append(f"{drone}_{scenario}")  # Combine drone type and scenario as label

    # Convert the list of images and labels to numpy arrays
    images = np.array(images)
    labels = np.array(labels)

    # Normalize the images
    images = images.astype('float32') / 255.0
    images = images.reshape((images.shape[0], config['image']['width'], config['image']['height'], config['image']['channels']))  # Reshape to 4D array for CNN input
    
    # Label Encoding for categorical labels
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(labels)
    labels = np.expand_dims(labels, axis=1)  # Ensure labels have the correct shape (N, 1)

    # Split the dataset into training and validation sets (80% train, 20% validation)
    return train_test_split(images, labels, test_size=0.2, random_state=42), label_encoder

def save_processed_data(config):
    """
    Loads raw data, processes it, and saves the images and labels as numpy arrays.
    """
    (X_train, X_val, y_train, y_val), label_encoder = load_raw_data(config)

    # Save processed data
    os.makedirs(config['paths']['processed_data_dir'], exist_ok=True)

    np.save(os.path.join(config['paths']['processed_data_dir'], 'train_images.npy'), X_train)  # Save training images
    np.save(os.path.join(config['paths']['processed_data_dir'], 'train_labels.npy'), y_train)  # Save training labels
    np.save(os.path.join(config['paths']['processed_data_dir'], 'val_images.npy'), X_val)     # Save validation images
    np.save(os.path.join(config['paths']['processed_data_dir'], 'val_labels.npy'), y_val)     # Save validation labels

    # Optionally save label encoder classes
    np.save(os.path.join(config['paths']['processed_data_dir'], 'label_classes.npy'), label_encoder.classes_)

    print("Data preprocessing completed and saved!")

if __name__ == "__main__":
    config = load_config()  # Load the config file
    save_processed_data(config)
