import os
import numpy as np
import tensorflow as tf
from model import create_model
from preprocess import load_config
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize

def load_data():
    """
    Load processed data from numpy files.
    """
    X_train = np.load('data/processed_data/train_images.npy')
    y_train = np.load('data/processed_data/train_labels.npy')
    X_val = np.load('data/processed_data/val_images.npy')
    y_val = np.load('data/processed_data/val_labels.npy')
    
    return X_train, X_val, y_train, y_val

def plot_metrics(history):
    """
    Plot accuracy and loss metrics for the model over the training epochs.
    """
    plt.figure(figsize=(12, 6))

    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.legend()
    plt.title('Model Accuracy')

    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.legend()
    plt.title('Model Loss')

    plt.show()

def evaluate_metrics(y_true, y_pred, num_classes):
    """
    Evaluate and plot metrics including confusion matrix and ROC curve.
    """
    precision = precision_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')
    f1 = f1_score(y_true, y_pred, average='macro')
    accuracy = accuracy_score(y_true, y_pred)

    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, Accuracy: {accuracy:.4f}")

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=np.unique(y_true), yticklabels=np.unique(y_true))
    plt.title('Confusion Matrix')
    plt.show()

    # ROC Curve for each class
    y_true_binarized = label_binarize(y_true, classes=range(num_classes))
    y_pred_binarized = label_binarize(y_pred, classes=range(num_classes))

    plt.figure()
    for i in range(num_classes):
        fpr, tpr, _ = roc_curve(y_true_binarized[:, i], y_pred_binarized[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'Class {i} (area = {roc_auc:.2f})')

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) for Each Class')
    plt.legend(loc='lower right')
    plt.show()

def train_model():
    """
    Train the model with specified configurations, and plot metrics after training.
    """
    config = load_config()
    
    # Load training and validation data
    X_train, X_val, y_train, y_val = load_data()
    
    # Create model with specified parameters
    model = create_model(input_shape=(224, 224, 1), 
                         cnn_filters=config['model']['cnn_filters'], 
                         lstm_units=config['model']['lstm_units'], 
                         dense_units=config['model']['dense_units'], 
                         num_classes=config['model']['num_classes'])
    
    # Define early stopping callback
    early_stopping = EarlyStopping(monitor='val_loss', patience=3)
    
    # Train the model
    history = model.fit(
        X_train, y_train,
        batch_size=config['train']['batch_size'],
        epochs=config['train']['epochs'],
        validation_data=(X_val, y_val),
        callbacks=[early_stopping]
    )

    # Evaluate model on validation set
    y_pred = model.predict(X_val)
    y_pred = np.argmax(y_pred, axis=1)  # Convert softmax outputs to class labels

    evaluate_metrics(y_val, y_pred, num_classes=config['model']['num_classes'])
    
    # Save the model
    model_save_dir = config['paths']['model_save_dir']
    os.makedirs(model_save_dir, exist_ok=True)
    model.save(os.path.join(model_save_dir, 'drone_model.h5'))

    # Plot training history
    plot_metrics(history)

if __name__ == "__main__":
    train_model()
