# Proactive Anomaly Detection in Surveillance using GAN

## 📌 Project Overview

This project implements a **proactive anomaly detection system** in real-time surveillance using **Generative Adversarial Networks (GANs)**. The system is designed to learn normal patterns from surveillance footage and detect anomalies in live video streams. When an anomaly is detected, **YOLO (You Only Look Once)** is triggered to perform object detection, identifying the elements present in the frame.

## 🎯 Key Features

* ✅ Trained on 40 normal surveillance images using GAN to learn typical patterns.
* 🎥 Uses **OpenCV** to process live video input (e.g., from webcam or CCTV).
* 🚨 Anomalies are detected by comparing real-time frames against learned normal behavior.
* 🧠 If anomaly is detected, **YOLO** is activated to detect and label objects in the frame.
* 🔄 Real-time monitoring and processing.

## 🧠 Technologies Used

* **Python**
* **OpenCV** – for video capture and frame manipulation
* **TensorFlow / PyTorch** – for GAN implementation
* **YOLO (v3/v4/v5)** – for object detection on anomaly frames
* **NumPy / Matplotlib** – for data handling and visualization

## 🔍 How It Works

1. **Training Phase**

   * GAN is trained with 40 images containing normal surveillance scenarios.
   * The generator learns to produce images similar to the normal dataset.
   * The discriminator learns to distinguish between real and generated (normal) images.

2. **Detection Phase**

   * Real-time video frames are captured using OpenCV.
   * Each frame is compared against the learned normal patterns.
   * If the frame significantly deviates (i.e., potential anomaly), it's flagged.
   * YOLO is then executed to identify objects in the anomalous frame for further analysis.

## 🛠️ Setup Instructions

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/surveillance-anomaly-detection.git
   cd surveillance-anomaly-detection
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Add your **training images** to the `/training_data` folder.

4. Run the training script for the GAN:

   ```bash
   python train_gan.py
   ```

5. Start live anomaly detection:

   ```bash
   python live_detection.py
   ```

## 📂 Folder Structure

```
surveillance-anomaly-detection/
│
├── training_data/         # Folder containing normal images
├── gan_model/             # Trained GAN model weights
├── yolo/                  # YOLO configuration and weights
├── train_gan.py           # GAN training script
├── live_detection.py      # Real-time anomaly detection script
├── utils.py               # Helper functions
└── README.md              # Project documentation
```

## 📸 Demo (Optional)

Include GIFs or screenshots here to showcase the anomaly detection and YOLO object recognition.
![screenshot](./screenshots/picture1.jpg)

## 🧑‍💻 Author

Pugal B – [LinkedIn](https://www.linkedin.com/in/pugal-balas-b256b5263?utm_source=share&utm_campaign=share_via&utm_content=profile&utm_medium=ios_app) | [GitHub](https://github.com/pugalbalasundaram)
