import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import os
import threading
import matplotlib.pyplot as plt
from collections import deque

# Check if GPU is available and configure it for TensorFlow
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("✅ Using GPU for processing.")
    except RuntimeError as e:
        print(f"⚠️ GPU setup error: {e}")
else:
    print("❌ No GPU found. Running on CPU.")

# Load the pre-trained GAN generator
MODEL_PATH = "gan_model.h5"
generator = load_model(MODEL_PATH)

# System configuration
LATENT_DIM = 100                           # Size of noise vector for GAN
THRESHOLD = 0.5                            # Anomaly detection threshold
FRAME_SIZE = (64, 64)                      # Resize frames to GAN input size
ANOMALY_SCORES = deque(maxlen=50)          # Store last 50 anomaly scores for visualization

# Resize and normalize input frame for GAN comparison
def preprocess_frame(frame):
    frame = cv2.resize(frame, FRAME_SIZE)
    frame = frame.astype(np.float32) / 255.0
    return np.expand_dims(frame, axis=0)

# Use GAN to generate a synthetic image from random noise
def generate_synthetic_image():
    noise = np.random.normal(0, 1, (1, LATENT_DIM))
    generated_image = generator.predict(noise)[0]
    return np.clip(generated_image, 0, 1)

# Compare the real frame with the generated image to detect anomaly
def detect_anomaly(real_frame):
    synthetic_image = generate_synthetic_image()
    real_frame = np.squeeze(real_frame)
    difference = np.mean(np.abs(real_frame - synthetic_image))
    return difference > THRESHOLD, difference

# Continuously update and display a live plot of anomaly scores
def plot_anomaly_score():
    plt.ion()
    fig, ax = plt.subplots()
    
    while True:
        if len(ANOMALY_SCORES) > 0:
            ax.clear()
            above_threshold = [score if score > THRESHOLD else None for score in ANOMALY_SCORES]
            below_threshold = [score if score <= THRESHOLD else None for score in ANOMALY_SCORES]
            
            ax.plot(above_threshold, color='red', marker='o', linestyle='dashed', label='Above Threshold')
            ax.plot(below_threshold, color='green', marker='o', linestyle='dashed', label='Below Threshold')
            ax.axhline(y=THRESHOLD, color='blue', linestyle='--', linewidth=1.5, label='Anomaly Threshold')
            
            ax.set_ylim([0.2, 0.8])
            ax.set_title("Anomaly Score Over Time")
            ax.set_xlabel("Frame")
            ax.set_ylabel("Anomaly Score")
            ax.legend()
            plt.pause(0.1)

# Launch anomaly score graph in a separate thread
graph_thread = threading.Thread(target=plot_anomaly_score, daemon=True)
graph_thread.start()

# Start capturing video from the webcam
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("Error: Could not access the webcam.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame.")
        break

    # Prepare frame for GAN comparison
    processed_frame = preprocess_frame(frame)

    # Perform anomaly detection
    is_anomaly, score = detect_anomaly(processed_frame)
    ANOMALY_SCORES.append(score)

    # Set message and box color based on detection result
    if is_anomaly:
        text = f"Anomaly Detected! Score: {score:.2f}"
        color = (0, 0, 255)  # Red
    else:
        text = f"Normal. Score: {score:.2f}"
        color = (0, 255, 0)  # Green

    # Draw bounding box and display result on frame
    height, width, _ = frame.shape
    cv2.rectangle(frame, (50, 50), (width - 50, height - 50), color, 3)
    cv2.putText(frame, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    # Show video with overlay
    cv2.imshow("Webcam Feed", frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up resources
cap.release()
cv2.destroyAllWindows()
