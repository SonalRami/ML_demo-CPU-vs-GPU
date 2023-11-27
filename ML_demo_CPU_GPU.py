import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
import numpy as np
import time

# Define Conv1D model for regression task
def create_complex_conv1d_model(input_shape=(1024, 1)):
    model = Sequential([
        Conv1D(128, kernel_size=5, activation='relu', input_shape=input_shape),
        MaxPooling1D(pool_size=2),
        Conv1D(128, kernel_size=5, activation='relu'),
        MaxPooling1D(pool_size=2),
        Conv1D(256, kernel_size=3, activation='relu'),
        MaxPooling1D(pool_size=2),
        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(1)  # Output layer for regression
    ])
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Create a dataset for regression task
def create_large_dataset(n_samples=50000, sequence_length=1024):
    X = np.random.random((n_samples, sequence_length, 1))
    y = np.random.random((n_samples, 1))
    return X, y

# Function to train the model and measure time
def train_model(model, X, y, epochs=5, batch_size=64, use_gpu=True):
    if use_gpu:
        device_name = "/gpu:0"
    else:
        device_name = "/cpu:0"

    start_time = time.time()
    with tf.device(device_name):
        model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=1)
    end_time = time.time()
    return end_time - start_time

# Create large dataset
X, y = create_large_dataset()

# Train on CPU
cpu_model = create_complex_conv1d_model()
cpu_time = train_model(cpu_model, X, y, use_gpu=False)
print(f"Training time on CPU: {cpu_time} seconds")

# Train on GPU
gpu_model = create_complex_conv1d_model()
gpu_time = train_model(gpu_model, X, y, use_gpu=True)
print(f"Training time on GPU: {gpu_time} seconds")
