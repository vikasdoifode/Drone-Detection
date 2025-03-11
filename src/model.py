import tensorflow as tf
from tensorflow.keras import layers, models

def create_model(input_shape, cnn_filters=32, cnn_kernel_size=(3, 3), lstm_units=64, dense_units=128, num_classes=15):
    model = models.Sequential()
    
    # CNN Layers
    model.add(layers.Conv2D(cnn_filters, cnn_kernel_size, activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Conv2D(cnn_filters * 2, cnn_kernel_size, activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    
    # Flatten the output of the CNN layers
    model.add(layers.Flatten())
    
    # LSTM Layer (Bi-LSTM)
    model.add(layers.Reshape((-1, cnn_filters * 2)))  # Reshape to 3D for LSTM
    model.add(layers.Bidirectional(layers.LSTM(lstm_units)))
    
    # Dense Layers
    model.add(layers.Dense(dense_units, activation='relu'))
    model.add(layers.Dense(num_classes, activation='softmax'))  # For multi-class classification
    
    # Compile the model with Adam optimizer
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    return model
