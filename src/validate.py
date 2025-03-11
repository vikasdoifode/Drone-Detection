import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

def load_data():
    X_val = np.load('data/processed_data/val_images.npy')
    y_val = np.load('data/processed_data/val_labels.npy')
    return X_val, y_val

def evaluate_model():
    model = tf.keras.models.load_model('saved_models/drone_model.h5')
    X_val, y_val = load_data()
    
    # Predict on validation data
    y_pred = model.predict(X_val)
    y_pred = np.argmax(y_pred, axis=1)
    
    # Evaluate metrics
    accuracy = accuracy_score(y_val, y_pred)
    precision = precision_score(y_val, y_pred, average='macro')
    recall = recall_score(y_val, y_pred, average='macro')
    f1 = f1_score(y_val, y_pred, average='macro')
    
    print(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}")
    
    # Plot Confusion Matrix
    cm = confusion_matrix(y_val, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=np.unique(y_val), yticklabels=np.unique(y_val))
    plt.title('Confusion Matrix')
    plt.show()

if __name__ == "__main__":
    evaluate_model()
