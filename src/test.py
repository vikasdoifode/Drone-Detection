import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from preprocess import load_config

def load_test_data():
    """
    Load test data for evaluation.
    Assumes test data is prepared similar to validation data.
    """
    X_test = np.load('data/processed_data/val_images.npy')
    y_test = np.load('data/processed_data/val_labels.npy')

    return X_test, y_test

def evaluate_test_data():
    # Load the trained model
    model = tf.keras.models.load_model('saved_models/drone_model.h5')

    # Load the test data
    X_test, y_test = load_test_data()

    # Make predictions
    y_pred = model.predict(X_test)
    y_pred = np.argmax(y_pred, axis=1)

    # Calculate evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')

    # Print the metrics
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")

    # Plot confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=np.unique(y_test), yticklabels=np.unique(y_test))
    plt.title('Confusion Matrix')
    plt.show()

if __name__ == "__main__":
    evaluate_test_data()
