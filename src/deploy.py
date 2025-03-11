import numpy as np
import tensorflow as tf
import cv2
import os
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk

def load_model():
    """
    Load the trained model from the saved directory.
    """
    model = tf.keras.models.load_model('saved_models/drone_model.h5')
    return model

def preprocess_input_image(image_path):
    """
    Preprocess a single image before feeding it to the model.
    Assumes the image is grayscale and resizes it to 224x224.
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (224, 224))  # Resize to 224x224
    img = img.astype('float32') / 255.0  # Normalize
    img = img.reshape((1, 224, 224, 1))  # Add batch dimension (for a single image)
    return img

def load_label_classes():
    """
    Load the label classes from the 'label_classes.npy' file.
    This will be used to map the predicted class index to the corresponding label.
    """
    return np.load('data/processed_data/label_classes.npy', allow_pickle=True)

def predict_image(image_path):
    """
    Make a prediction on a single image. Detects whether a drone is present or not.
    """
    model = load_model()

    # Preprocess the image
    img = preprocess_input_image(image_path)

    # Make prediction
    prediction = model.predict(img)
    predicted_class_index = np.argmax(prediction, axis=1)[0]  # Get the index of the predicted class
    probability = prediction[0][predicted_class_index]  # Get probability of the predicted class

    # Load label classes and map the predicted class index to the actual label
    label_classes = load_label_classes()
    predicted_label = label_classes[predicted_class_index]

    # Check if probability meets the "Drone Detected" threshold
    if probability >= 0.9950:
        message = "Drone Detected"
    else:
        message = "Drone Not Detected"

    print(f"Prediction: {predicted_label}, Probability: {probability:.4f}")
    return message

def upload_image():
    """
    Opens a file dialog to upload an image for prediction.
    """
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.jpeg;*.png")])
    
    if file_path:
        print(f"Processing image: {file_path}")
        result_message = predict_image(file_path)
        
        # Display the result in a message box
        messagebox.showinfo("Prediction Result", result_message)
        display_image(file_path)

def display_image(image_path):
    """
    Display the uploaded image on the Tkinter canvas.
    """
    img = Image.open(image_path)
    img = img.resize((250, 250))  # Resize image for display
    img = ImageTk.PhotoImage(img)
    
    panel = tk.Label(root, image=img)
    panel.image = img
    panel.grid(row=2, column=0, padx=20, pady=20)

# Set up the Tkinter window
root = tk.Tk()
root.title("Drone Detection")
root.geometry("500x500")

# Create upload button
upload_button = tk.Button(root, text="Upload Image", command=upload_image)
upload_button.grid(row=0, column=0, padx=20, pady=20)

# Start the Tkinter event loop
root.mainloop()